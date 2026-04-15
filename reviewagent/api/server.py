"""FastAPI app: synchronous review, async queue, metrics and audit APIs (cloud or on-prem)."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from reviewagent.config import (
    apply_llm_patch_to_yaml_file,
    apply_pipeline_report_locale_to_yaml_file,
    get_config_yaml_path,
    get_settings,
    reload_settings,
)
from reviewagent.pipeline.biz_context import biz_context_from_payload
from reviewagent.limits import LimitsExceededError, enforce_file_size, enforce_text_utf8_bytes
from reviewagent.observability.metrics import get_metrics
from reviewagent.storage.review import ReviewStore

_WEB_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def _public_single_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """Single-file response shape for legacy clients (no batch-only fields like filename)."""
    return {
        k: v
        for k, v in item.items()
        if k not in ("index", "path", "filename", "inferred_content_type")
    }


def _public_batch_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Batch item: keep filename / inferred_content_type; strip internal index and temp paths."""
    return {k: v for k, v in item.items() if k not in ("index", "path")}


class ModerateBody(BaseModel):
    content: str = Field(..., description="Content to review or media path description")
    content_type: str = Field(default="auto", description="auto | text | image | video | audio")
    session_id: Optional[str] = Field(
        default=None,
        description="Multi-turn session id; alternatively use X-Review-Session header; same id shares short-term memory",
    )
    biz_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Business context (biz_line, tenant_id, …) injected into system prompt and pipeline_trace",
    )


class TaskCreateBody(ModerateBody):
    pass


def _infer_content_type_from_input(content: str) -> str:
    from reviewagent.ingest import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

    raw = (content or "").strip()
    if not raw:
        return "text"
    if "://" in raw:
        # Local paths only for media; URLs are treated as plain text
        return "text"

    p = Path(raw)
    suf = p.suffix.lower()
    if suf in IMAGE_EXTENSIONS:
        return "image"
    if suf in VIDEO_EXTENSIONS:
        return "video"
    if suf in AUDIO_EXTENSIONS:
        return "audio"

    return "text"


class LLMConfigPublic(BaseModel):
    """GET /v1/config/llm: never returns raw api_key."""

    provider: str
    model: str
    api_base: str
    temperature: float
    max_tokens: int
    timeout: int
    minimax_group_id: str
    api_key_configured: bool


class LLMPatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    minimax_group_id: Optional[str] = None


def _llm_public_from_settings() -> LLMConfigPublic:
    s = get_settings().llm
    key = (s.api_key or "").strip()
    return LLMConfigPublic(
        provider=s.provider,
        model=s.model,
        api_base=s.api_base,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        minimax_group_id=s.minimax_group_id,
        api_key_configured=bool(key),
    )


class DisplayConfigPublic(BaseModel):
    """GET/PATCH /v1/config/display: report language (pipeline.image_dual_check.report_locale)."""

    report_locale: Literal["zh", "en"]


class DisplayPatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_locale: Literal["zh", "en"]


def _display_public_from_settings() -> DisplayConfigPublic:
    loc = get_settings().pipeline.image_dual_check.report_locale
    if loc not in ("zh", "en"):
        loc = "zh"
    return DisplayConfigPublic(report_locale=loc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import logging

    log = logging.getLogger(__name__)

    settings = get_settings()
    from reviewagent.pipeline.image_fingerprint import log_fingerprint_config_warnings

    log_fingerprint_config_warnings(settings)

    from reviewagent.observability.file_logging import configure_reviewagent_logging

    p = configure_reviewagent_logging(settings)
    if p is not None:
        log.info("reviewagent file log: %s", p)

    from reviewagent.review_queue.service import ReviewQueueService

    from reviewagent.agent import review_job_runner

    if settings.rag.enabled and settings.rag.auto_ingest_on_startup:
        try:
            from reviewagent.rag.store import ingest_configured_directories

            n = ingest_configured_directories()
            log.info("RAG: auto_ingest_on_startup finished, ~%s chunks written this batch", n)
        except Exception:
            log.exception("RAG: auto_ingest_on_startup failed")

    if settings.queue.enabled:
        svc = ReviewQueueService(review_job_runner)
        await svc.start()
        app.state.review_queue = svc
    else:
        app.state.review_queue = None
    yield
    if app.state.review_queue is not None:
        await app.state.review_queue.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Content Review Agent",
        version="0.2.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        loc = get_settings().pipeline.image_dual_check.report_locale
        if loc not in ("zh", "en"):
            loc = "zh"
        return {"status": "ok", "report_locale": loc}

    @app.get("/v1/metrics")
    def metrics_json() -> dict[str, Any]:
        s = get_settings()
        if not s.observability.metrics_enabled:
            return {"enabled": False}
        return {"enabled": True, **get_metrics().snapshot()}

    @app.get("/v1/tool-packs")
    def list_tool_packs() -> dict[str, Any]:
        """List registered tool packs (LangChain tool groups); not vendor \"Skills\" products."""
        from reviewagent.toolpacks.registry import default_registry

        return {"tool_packs": default_registry().list_tool_packs()}

    @app.get("/v1/config/llm", response_model=LLMConfigPublic)
    def get_llm_config() -> LLMConfigPublic:
        """Active LLM settings summary (no raw secret values)."""
        return _llm_public_from_settings()

    @app.patch("/v1/config/llm", response_model=LLMConfigPublic)
    def patch_llm_config(body: LLMPatchBody) -> LLMConfigPublic:
        """
        Merge patch into config.yaml llm section and hot-reload.
        Omitted fields unchanged; api_key may be cleared with an empty string.
        """
        patch = body.model_dump(exclude_none=True)
        if not patch:
            raise HTTPException(status_code=400, detail="no updatable fields in request body")
        path = get_config_yaml_path()
        try:
            apply_llm_patch_to_yaml_file(path, patch)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"validation or write failed: {e}") from e
        try:
            reload_settings()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"file written but reload failed; check YAML: {e}",
            ) from e
        return _llm_public_from_settings()

    @app.get("/v1/config/display", response_model=DisplayConfigPublic)
    def get_display_config() -> DisplayConfigPublic:
        return _display_public_from_settings()

    @app.patch("/v1/config/display", response_model=DisplayConfigPublic)
    def patch_display_config(body: DisplayPatchBody) -> DisplayConfigPublic:
        path = get_config_yaml_path()
        try:
            apply_pipeline_report_locale_to_yaml_file(path, body.report_locale)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"validation or write failed: {e}") from e
        try:
            reload_settings()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"file written but reload failed; check YAML: {e}",
            ) from e
        return _display_public_from_settings()

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        """Drop server-side short-term memory for this session (e.g. TUI /new)."""
        from reviewagent.memory import clear_session_memory

        clear_session_memory(session_id)
        return {"ok": True}

    @app.post("/v1/review")
    async def review_sync(
        body: ModerateBody,
        x_review_session: Optional[str] = Header(default=None, alias="X-Review-Session"),
    ) -> dict[str, Any]:
        from reviewagent.agent import create_review_orchestrator

        s = get_settings()
        try:
            enforce_text_utf8_bytes(body.content, s.limits.max_text_bytes, field="content to review")
        except LimitsExceededError as e:
            raise HTTPException(status_code=413, detail=str(e)) from e

        sid = (x_review_session or body.session_id or "").strip() or None
        biz = biz_context_from_payload(body.biz_context)
        orchestrator = create_review_orchestrator(session_id=sid, biz_context=biz)
        ct = body.content_type.strip().lower() if isinstance(body.content_type, str) else "auto"
        if ct not in ("auto", "text", "image", "video", "audio"):
            ct = "auto"
        resolved_ct = _infer_content_type_from_input(body.content) if ct == "auto" else ct
        return await orchestrator.review_payload_async(resolved_ct, body.content)

    @app.post("/v1/review/file")
    async def review_file_upload(
        file: Optional[UploadFile] = File(
            None,
            description="Single upload (use either file or files; field name file)",
        ),
        files: Optional[List[UploadFile]] = File(
            None,
            description="Multiple uploads (repeatable files field)",
        ),
        x_review_session: Optional[str] = Header(default=None, alias="X-Review-Session"),
    ) -> dict[str, Any]:
        """
        Upload one or more files and review. Multiple files run in order; same session keeps memory.
        A single file returns a flat JSON body (no batch wrapper).
        """
        from reviewagent.agent import create_review_orchestrator

        from reviewagent.ingest import load_local_file_for_review

        sid = (x_review_session or "").strip() or None
        uploads: List[UploadFile] = []
        if files:
            uploads.extend(files)
        elif file is not None:
            uploads.append(file)
        else:
            raise HTTPException(status_code=422, detail="upload file or files (at least one)")

        tmp_paths: list[str] = []
        meta: list[str] = []
        s = get_settings()
        try:
            for up in uploads:
                data = await up.read()
                orig = up.filename or "upload.bin"
                try:
                    enforce_file_size(
                        len(data),
                        s.limits.max_file_bytes,
                        name=f"upload:{orig}",
                    )
                except LimitsExceededError as e:
                    raise HTTPException(status_code=413, detail=str(e)) from e
                meta.append(orig)
                # Write under sanitized original filename in temp dir for readable logs
                safe_name = Path(orig).name.replace("\x00", "").strip() or "upload.bin"
                if safe_name in (".", ".."):
                    safe_name = "upload.bin"
                tmp_dir = tempfile.mkdtemp(prefix="review_upload_")
                tmp_path = str(Path(tmp_dir) / safe_name)
                Path(tmp_path).write_bytes(data)
                tmp_paths.append(tmp_path)

            orchestrator = create_review_orchestrator(session_id=sid)
            results: List[Dict[str, Any]] = []
            for orig_name, tmp_path in zip(meta, tmp_paths):
                item: Dict[str, Any] = {"index": len(results), "filename": orig_name}
                try:
                    ct, payload = load_local_file_for_review(Path(tmp_path))
                    item["inferred_content_type"] = ct
                    r = await orchestrator.review_payload_async(ct, payload)
                    item.update(r)
                except Exception as e:
                    item["success"] = False
                    item["error"] = str(e)
                    item.setdefault("response", "")
                results.append(item)

            if len(results) == 1:
                return _public_single_result(results[0])
            return {
                "batch": True,
                "count": len(results),
                "results": [_public_batch_item(x) for x in results],
            }
        finally:
            for tmp_path in tmp_paths:
                try:
                    p = Path(tmp_path)
                    if p.exists():
                        p.unlink()
                    # Remove our temp dir if empty
                    parent = p.parent
                    if parent.name.startswith("review_upload_"):
                        parent.rmdir()
                except OSError:
                    pass

    @app.post("/v1/tasks", status_code=202)
    async def enqueue_task(body: TaskCreateBody) -> dict[str, str]:
        q: Optional[Any] = getattr(app.state, "review_queue", None)
        if q is None:
            raise HTTPException(status_code=503, detail="task queue disabled (config.queue.enabled=false)")
        s = get_settings()
        try:
            enforce_text_utf8_bytes(body.content, s.limits.max_text_bytes, field="task content")
        except LimitsExceededError as e:
            raise HTTPException(status_code=413, detail=str(e)) from e
        ct = body.content_type.strip().lower() if isinstance(body.content_type, str) else "auto"
        if ct not in ("auto", "text", "image", "video", "audio"):
            ct = "auto"
        resolved_ct = _infer_content_type_from_input(body.content) if ct == "auto" else ct
        task_id = await q.enqueue(content_type=resolved_ct, content=body.content)
        return {"task_id": task_id, "status": "accepted"}

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> dict[str, Any]:
        q: Optional[Any] = getattr(app.state, "review_queue", None)
        if q is None:
            raise HTTPException(status_code=503, detail="task queue disabled")
        row = await q.get_task(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        return dict(row)

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        settings = get_settings()
        store = ReviewStore(settings.storage.review_db_path)
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        return row

    @app.get("/v1/runs")
    def list_runs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        settings = get_settings()
        store = ReviewStore(settings.storage.review_db_path)
        return {"runs": store.list_runs(limit=min(limit, 500), offset=offset)}

    @app.get("/")
    def root_redirect() -> RedirectResponse:
        """Local web console (static UI calling same-origin API)."""
        return RedirectResponse(url="/ui/")

    if _WEB_STATIC_DIR.is_dir():
        app.mount(
            "/ui",
            StaticFiles(directory=str(_WEB_STATIC_DIR), html=True),
            name="ui",
        )

    return app


app = create_app()
