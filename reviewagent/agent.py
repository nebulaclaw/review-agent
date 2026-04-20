"""Content review orchestrator (``ReviewOrchestrator``): LLM tool loop, tool-pack registry, audit and metrics.

Architecture
------------
Per-content-type pipeline logic lives in ``reviewagent/reviewers/``:

    text  → TextReviewer   (wordlist fast-path → LLM agent)
    image → ImageReviewer  (fingerprint → OCR wordlist → dual-LLM / single-agent)
    video → VideoReviewer  (VideoDetector → agent fallback)
    audio → AudioReviewer  (VideoDetector audio-only → agent fallback)

``ReviewOrchestrator`` owns only the shared infrastructure:
  * LLM client + tool binding
  * Short-term memory
  * ``run()``  – the core async LLM agent loop
  * ``build_messages()`` / ``get_system_prompt()``
  * ``_finalize_*`` helpers (audit + metrics)
  * Session / memory helpers
  * ``review_payload_async()`` – thin dispatcher to the reviewer registry
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from reviewagent.adapters.errors import enrich_agent_error
from reviewagent.adapters.llm_factory import create_chat_model
from reviewagent.config import Settings, get_settings
from reviewagent.limits import enforce_file_size, enforce_text_utf8_bytes
from reviewagent.memory import get_memory, get_session_review_staging_paths
from reviewagent.observability.metrics import get_metrics
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.rag.store import retrieve_knowledge_context
from reviewagent.review_report import enrich_result_response_violation_types, parse_review_json_from_llm_output
from reviewagent.reviewers import get_reviewer
from reviewagent.reviewers.utils import (
    attach_pipeline_review_domain,
    image_readable_for_vision,
    local_image_data_url,
    response_verdict_hint,
    vision_attachment_eligible,
)
from reviewagent.storage.review import ReviewStore
from reviewagent.toolpacks.registry import default_registry

logger = logging.getLogger(__name__)

# Log prefix: grep "[review.core]" to trace one request through the core path
_MC = "[review.core]"

# One-shot JSON normalization when the main agent turn omits parseable verdict JSON.
_VERDICT_REPAIR_SYSTEM = """你是内容审核流水线中的「输出规范化」步骤。
你的唯一任务：阅读下面「助手输出」全文（可能含自然语言、工具结果摘录、或误印的 XML），产出**一条且仅一条**合法 JSON 对象。

硬性要求：
- 不要 Markdown 代码围栏；不要输出 ```；不要输出 <tool_call> 等伪工具标签；不要任何解释性前后缀。
- 顶层字段必须包含：verdict、confidence、violations、summary。
- verdict 只能是 PASS、WARN、BLOCK、UNKNOWN 之一；UNKNOWN 表示无法从给定材料做出可靠合规判断。
- confidence 为 0 到 1 的数值；violations 为数组（无则 []）；summary 为简短中文说明。

若上文明显在规划调用工具却未给出结论，请结合「用户请求节选」给出最合理的 UNKNOWN 或 WARN，并在 summary 中说明信息不足或需重试。"""


# ---------------------------------------------------------------------------
# Module-level helpers private to the orchestrator
# ---------------------------------------------------------------------------

def _payload_brief(content_type: str, content: str) -> str:
    if content_type in ("image", "video", "audio"):
        p = Path(content)
        return f"file={p.name!r} path_len={len(content)}"
    return f"text_len={len(content)}"


async def _maybe_repair_verdict_json(llm: Any, final_text: str, user_input: str) -> str:
    """If assistant content has no review-shaped JSON, ask base LLM once (no tools) to emit one."""
    s = (final_text or "").strip()
    if not s:
        return final_text or ""
    if parse_review_json_from_llm_output(final_text) is not None:
        return final_text
    try:
        repair_messages = [
            SystemMessage(content=_VERDICT_REPAIR_SYSTEM),
            HumanMessage(
                content="【用户请求节选】\n"
                + ((user_input or "")[:4000] or "(无)")
                + "\n\n【需规范为 JSON 的助手输出】\n"
                + s[:14000]
            ),
        ]
        ai = await llm.ainvoke(repair_messages)
        if isinstance(ai, AIMessage):
            candidate = (ai.content or "").strip()
        else:
            candidate = str(getattr(ai, "content", ai) or "").strip()
        if not candidate:
            return final_text
        if parse_review_json_from_llm_output(candidate) is not None:
            logger.info("%s verdict_repair ok (replaced non-JSON assistant output)", _MC)
            return candidate
        logger.info("%s verdict_repair skipped (repair output still not JSON)", _MC)
        return final_text
    except Exception as e:
        logger.warning("%s verdict_repair failed: %s", _MC, e)
        return final_text


# ---------------------------------------------------------------------------
# ReviewOrchestrator
# ---------------------------------------------------------------------------

class ReviewOrchestrator:
    """Central orchestrator: owns shared infra, dispatches to per-type reviewers."""

    def __init__(
        self,
        *,
        registry=None,
        review_store: Optional[ReviewStore] = None,
        session_id: Optional[str] = None,
        biz_context: Optional[BizContext] = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._session_id = session_id
        self._biz_context = biz_context or BizContext()
        self.llm = create_chat_model()
        # Inject backend-specific LangChain callbacks (LangFuse needs an explicit
        # CallbackHandler; LangSmith auto-traces via LANGCHAIN_TRACING_V2).
        from reviewagent.observability import tracing  # noqa: PLC0415
        _cbs = tracing.get_llm_callbacks()
        if _cbs:
            self.llm = self.llm.with_config({"callbacks": _cbs})
        self.memory = get_memory(session_id)
        self.max_iterations = settings.agent.max_iterations
        self._registry = registry or default_registry()
        self._review_store = review_store
        if self._review_store is None and settings.storage.review_db_path:
            self._review_store = ReviewStore(settings.storage.review_db_path)

        ctx = {"memory": self.memory}
        self.tools = self._registry.resolve_tools(ctx)
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _latest_staged_media_path_and_type(self) -> tuple[Optional[str], Optional[str]]:
        from reviewagent.ingest import load_local_file_for_review

        sid = (self._session_id or "").strip()
        if not sid:
            return None, None
        for p in reversed(get_session_review_staging_paths(sid)):
            pp = Path(p)
            if not pp.is_file():
                continue
            ct, _ = load_local_file_for_review(pp)
            if ct in ("video", "audio", "image"):
                return str(pp), ct
        return None, None

    def has_staged_media_for_session_followup(self) -> bool:
        mp, mt = self._latest_staged_media_path_and_type()
        return bool(mp and mt)

    def record_file_upload_turn_for_session(
        self,
        *,
        orig_name: str,
        content_type: str,
        staging_path: str,
        result: dict[str, Any],
    ) -> None:
        """Persist upload + outcome into short-term memory (early pipelines skip ``run()`` otherwise)."""
        if not self._session_id or not str(self._session_id).strip():
            return
        for m in reversed(self.memory.short_term.get_messages()[-12:]):
            if getattr(m, "type", "") == "human" and staging_path in str(
                getattr(m, "content", "") or ""
            ):
                return
        user_turn = (
            f"[Uploaded file] original_name={orig_name} content_type={content_type} "
            f"server_staging_path={staging_path}\n"
            "For follow-up or re-analysis, use this exact path with video_detector, "
            "audio_detector, or image_detector as appropriate."
        )
        resp = str(result.get("response") or "").strip()
        if not resp:
            err = result.get("error")
            if err is not None:
                resp = str(err)
        if not resp:
            resp = "(empty response)"
        self.memory.short_term.add_user_message(user_turn)
        self.memory.short_term.add_ai_message(resp[:80000])

    async def review_session_text_followup_async(self, user_text: str) -> dict[str, Any]:
        """Re-check the latest same-session staged media via the stable media pipeline."""
        self._enforce_payload_raw("text", user_text)
        media_path, media_ct = self._latest_staged_media_path_and_type()
        if not media_path or not media_ct:
            return await self.review_payload_async("text", user_text)

        # Avoid agent tool-loop dead-ends for media re-check; reuse deterministic media pipeline.
        out = await self.review_payload_async(media_ct, media_path)
        pt = dict(out.get("pipeline_trace") or {})
        pt["session_staged_followup"] = True
        note = (user_text or "").strip()
        if note:
            pt["session_followup_note"] = note[:500]
        out["pipeline_trace"] = pt
        return out

    def prior_substantive_user_text_for_text_recheck(self) -> Optional[str]:
        """Prior user message to re-run as text review: skip upload stubs and short re-check phrases."""
        from reviewagent.api.followup_text_heuristic import text_suggests_recheck_same_media

        for m in reversed(self.memory.short_term.get_messages()):
            if getattr(m, "type", "") != "human":
                continue
            c = str(getattr(m, "content", "") or "").strip()
            if not c or c.startswith("[Uploaded file]"):
                continue
            if text_suggests_recheck_same_media(c):
                continue
            return c
        return None

    def no_staged_media_recheck_result(self, *, user_text: str) -> dict[str, Any]:
        """Fast path when the user asks to re-check but there is no prior text or file to re-run."""
        summary = (
            "未找到上一轮可复检的内容：多轮「再检」需要会话里已有用户发送过的待审正文，"
            "或先通过 /file 上传媒体。请先发送要审核的文字或使用 /file。"
        )
        payload = {
            "verdict": "WARN",
            "confidence": 1.0,
            "violations": [
                {
                    "type": "no_recheck_target",
                    "content": summary,
                    "severity": "low",
                    "position": "—",
                }
            ],
            "summary": summary,
        }
        t0 = time.perf_counter()
        rr = {
            "success": True,
            "response": json.dumps(payload, ensure_ascii=False),
            "iterations": 0,
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "pipeline_trace": {
                "mode": "no_staged_recheck",
                "continued_to_llm": False,
            },
        }
        return self._finalize_early_pipeline_result(
            rr,
            content_type="text",
            input_summary=user_text,
            early_block_log_path="no_staged_recheck",
        )

    # ------------------------------------------------------------------
    # Payload validation
    # ------------------------------------------------------------------

    def _enforce_payload_raw(self, content_type: str, content: str) -> None:
        lim = self._settings.limits
        if content_type == "text":
            enforce_text_utf8_bytes(content, lim.max_text_bytes, field="待审文本")
        elif content_type in ("image", "video", "audio"):
            p = Path(content)
            if p.is_file():
                enforce_file_size(
                    p.stat().st_size,
                    lim.max_file_bytes,
                    name=f"媒体文件「{p.name}」",
                )
            enforce_text_utf8_bytes(content, lim.max_text_bytes, field="路径或说明")
        else:
            enforce_text_utf8_bytes(content, lim.max_text_bytes, field="待审内容")

    def _enforce_user_message(self, user_input: str) -> None:
        enforce_text_utf8_bytes(
            user_input,
            self._settings.limits.max_user_message_bytes,
            field="用户消息（含提示模板）",
        )

    # ------------------------------------------------------------------
    # Prompt / message builders
    # ------------------------------------------------------------------

    def _biz_context_system_suffix(self) -> str:
        b = self._biz_context
        if not any((b.biz_line, b.tenant_id, b.trust_tier, b.audience, b.policy_pack_id)):
            return ""
        lines = ["\n## 业务上下文（请求注入，仅供审核参考）"]
        if b.biz_line:
            lines.append(f"- 业务线：{b.biz_line}")
        if b.tenant_id:
            lines.append(f"- 租户：{b.tenant_id}")
        if b.trust_tier:
            lines.append(f"- 信誉档位：{b.trust_tier}")
        if b.audience:
            lines.append(f"- 可见性：{b.audience}")
        if b.policy_pack_id:
            lines.append(f"- 策略包：{b.policy_pack_id}")
        return "\n".join(lines)

    def get_system_prompt(self, *, mode: str) -> str:
        """Build the system prompt for a given run mode.

        Modes:
        - ``subtask``: dual OCR/vision sub-prompt (no tool-use section)
        - ``text``:    text content safety review
        - ``media``:   multimodal path review (image/video/audio)
        """
        standards = """你是一个专业的内容审核助手。你的职责是判断输入的内容是否合规。

## 审核标准
1. 敏感词检测：检测是否包含敏感词汇
2. 违规内容检测：检测是否包含违规内容（如暴力、色情、赌博等）
3. 禁用词检测：检测是否包含禁用词汇

## 审核结果
根据内容的违规程度，返回以下结果：
- PASS: 内容合规，可以发布
- WARN: 内容可能违规，需要人工复核
- BLOCK: 内容严重违规，阻止发布
"""
        output_fmt = f"""
## 输出格式
请以 JSON 格式返回审核结果：
```json
{{
  "verdict": "PASS|WARN|BLOCK",
  "confidence": 0.95,
  "violations": [
    {{
      "type": "porn|illegal|spam|violence|…",
      "content": "违规内容",
      "severity": "high|medium|low",
      "position": "具体位置"
    }}
  ],
  "summary": "简要说明"
}}
```
每条 violations[].type 使用稳定英文 key（如 porn、illegal、spam）；**违规分类**由服务端根据 violations 自动汇总展示，无需单独输出字段。
- **禁止**在最终对用户可见的回复中书写 `<tool_call>`、`</tool_call>`、`<arg_key>` 等伪工具 XML；真实工具调用由系统自动处理；你的最后一轮回复**必须**是一段可直接解析的 JSON 对象（可含简短中文自然语言前缀时仍须在同条消息内给出完整 JSON）。
"""
        multi = """
## 多轮对话
- 对话历史中若已有上一轮审核结论（JSON），用户后续消息可能是**追问、异议或要求解释**，请结合历史上下文回应，不要当作一条全新的、无上下文的待审正文。
- 仅当用户**明确粘贴了新稿件/新图片路径**要求重新审核时，再按新输入执行完整审核流程（含必要时的工具调用）。
"""
        kb = """
## 知识库摘录
- 若系统提供了「知识库中与当前请求相关的摘录」，请结合其中的政策说明、内部规范或案例辅助判断，但仍须以工具检测与最终合规要求为准。
"""
        suffix = self._biz_context_system_suffix()

        if mode == "subtask":
            return standards + output_fmt + multi + kb + suffix

        if mode == "text":
            tools = """
## 工具使用（文本）
- 使用 text_detector 检测敏感词
- 结合工具结果与待审全文输出最终 JSON 裁决
"""
            return standards + output_fmt + tools + multi + kb + suffix

        if mode == "media":
            tools = """
## 工具使用（子流程：图片/视频/音频）
- 按任务说明使用 image_detector、video_detector 或 audio_detector（路径须与用户提供一致）
- 可结合 text_detector 处理用户附加的文字说明
- 结合工具与多模态判读输出最终 JSON 裁决
"""
            return standards + output_fmt + tools + multi + kb + suffix

        raise ValueError(f"unknown prompt mode: {mode!r}")

    def _dual_llm_rag_messages(self, rag_seed: str) -> list:
        """RAG excerpt injected into dual-LLM branches (not mixed into conversational memory)."""
        out: list = []
        if self._settings.rag.enabled:
            kb = retrieve_knowledge_context((rag_seed or "")[:4000])
            if kb.strip():
                out.append(
                    SystemMessage(
                        content="以下是知识库中与当前请求相关的摘录（供审核参考）：\n\n"
                        + kb.strip()
                    )
                )
        return out

    def dual_llm_messages_ocr_branch(self, ocr_text: str, prefix: str) -> list:
        """Subtask 1: verdict from OCR text only (no tools, no image)."""
        messages: list = [SystemMessage(content=self.get_system_prompt(mode="subtask"))]
        messages.extend(self._dual_llm_rag_messages(ocr_text))
        body = (
            (prefix or "")
            + "## 当前子任务：OCR 文本审核（与视觉分支独立）\n"
            "以下为同一张图片经**本地 OCR** 得到的文字（可能有漏识、错字）。请**仅依据这段文字**给出合规裁决。\n"
            "只输出一段 JSON，字段含 verdict、confidence、violations、summary；不要使用 Markdown 代码围栏。\n\n"
            "--- OCR 文本 ---\n"
            + ocr_text
        )
        messages.append(HumanMessage(content=body))
        return messages

    def dual_llm_messages_vision_branch(self, image_path: str) -> list:
        """Subtask 2: verdict from raw pixels only (no tools)."""
        messages: list = [SystemMessage(content=self.get_system_prompt(mode="subtask"))]
        messages.extend(self._dual_llm_rag_messages(image_path))
        text = (
            "## 当前子任务：原图视觉审核（与 OCR 文本分支独立）\n"
            "请根据本消息中的**原始图片**综合判断画面与图中文字（含 OCR 可能漏检的内容）。\n"
            "只输出一段 JSON，字段含 verdict、confidence、violations、summary；不要使用 Markdown 代码围栏。\n"
            f"文件路径（记录用）：{image_path}"
        )
        data_url = local_image_data_url(Path(image_path))
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
        )
        return messages

    def build_messages(
        self,
        user_input: str,
        *,
        vision_image_path: Optional[str] = None,
        prompt_mode: str = "media",
    ) -> list:
        messages: list = [SystemMessage(content=self.get_system_prompt(mode=prompt_mode))]
        if self._settings.rag.enabled:
            kb = retrieve_knowledge_context(user_input)
            if kb.strip():
                messages.append(
                    SystemMessage(
                        content="以下是知识库中与当前请求相关的摘录（供审核参考，最终以法规、平台规则及工具检测结果为准）：\n\n"
                        + kb.strip()
                    )
                )
        messages.extend(self.memory.get_context())
        attach = vision_attachment_eligible(self._settings, vision_image_path)
        if attach and vision_image_path is not None:
            p = Path(vision_image_path)
            try:
                data_url = local_image_data_url(p)
                logger.info(
                    "review_vision_inline_image file=%s bytes=%d provider=%s",
                    p.name,
                    p.stat().st_size,
                    self._settings.llm.provider,
                )
                messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                )
            except OSError as e:
                logger.warning("review_vision_inline_image failed: %s", e)
                messages.append(HumanMessage(content=user_input))
        else:
            messages.append(HumanMessage(content=user_input))
        return messages

    # ------------------------------------------------------------------
    # Finalizer helpers (called by reviewers + orchestrator internals)
    # ------------------------------------------------------------------

    def _finalize_early_pipeline_result(
        self,
        result: dict[str, Any],
        *,
        content_type: str,
        input_summary: str,
        early_block_log_path: str = "early_wordlist",
    ) -> dict[str, Any]:
        """Early pipeline exit (wordlist / fingerprint / detector): audit, metrics, no LLM."""
        settings = self._settings
        enrich_result_response_violation_types(result)
        result.setdefault("review_domain", "content_safety")
        pt_early = dict(result.get("pipeline_trace") or {})
        pt_early.setdefault("review_domain", "content_safety")
        result["pipeline_trace"] = pt_early
        if settings.observability.metrics_enabled:
            get_metrics().inc("pipeline.early_block_total")
        if self._review_store:
            rid = self._review_store.append_run(
                status="completed",
                content_type=content_type,
                input_summary=input_summary[:2000],
                result=result,
                error=None,
                iterations=0,
                model_provider=settings.llm.provider,
                model_name=settings.llm.model,
                duration_ms=float(result.get("duration_ms", 0)),
                task_id=None,
            )
            result["run_id"] = rid
        if settings.observability.metrics_enabled:
            get_metrics().observe(
                "review.end_to_end_ms", float(result.get("duration_ms", 0))
            )
            get_metrics().inc("review.requests_total")
        vh = response_verdict_hint(result.get("response"))
        logger.info(
            "%s path=%s content_type=%s verdict=%s duration_ms=%s run_id=%s",
            _MC,
            early_block_log_path,
            content_type,
            vh,
            result.get("duration_ms"),
            result.get("run_id"),
        )
        return result

    def _finalize_dual_llm_image_result(
        self,
        result: dict[str, Any],
        *,
        content_type: str,
        input_summary: str,
    ) -> dict[str, Any]:
        """Dual image LLM path: audit and metrics (same wrap-up as run())."""
        settings = self._settings
        enrich_result_response_violation_types(result)
        err = result.get("error")
        ok = bool(result.get("success")) and not err
        duration_ms = float(result.get("duration_ms", 0))
        if self._review_store:
            rid = self._review_store.append_run(
                status="completed" if ok else "failed",
                content_type=content_type,
                input_summary=input_summary[:2000],
                result=result if ok else None,
                error=err if not ok else None,
                iterations=int(result.get("iterations", 0)),
                model_provider=settings.llm.provider,
                model_name=settings.llm.model,
                duration_ms=duration_ms,
                task_id=None,
            )
            result["run_id"] = rid
        if settings.observability.metrics_enabled:
            get_metrics().observe("review.end_to_end_ms", duration_ms)
            get_metrics().inc("review.requests_total")
        vh = response_verdict_hint(result.get("response"))
        logger.info(
            "%s path=dual_image_llm content_type=%s verdict=%s llm_calls=%s duration_ms=%s run_id=%s ok=%s",
            _MC,
            content_type,
            vh,
            result.get("iterations"),
            result.get("duration_ms"),
            result.get("run_id"),
            ok,
        )
        return result

    # ------------------------------------------------------------------
    # Main review entry points
    # ------------------------------------------------------------------

    async def review_payload_async(self, content_type: str, content: str) -> dict[str, Any]:
        """Validate, dispatch to the registered reviewer, return a result dict."""
        from reviewagent.observability import tracing  # noqa: PLC0415

        self._enforce_payload_raw(content_type, content)
        logger.info(
            "%s moderate_begin content_type=%s brief=%s",
            _MC,
            content_type,
            _payload_brief(content_type, content),
        )

        # Named parent span — both LangSmith and LangFuse attach LLM child calls here.
        @tracing.span(f"review.{content_type}")
        async def _traced() -> dict[str, Any]:
            tracing.update_session(self._session_id)

            reviewer = get_reviewer(content_type)
            if reviewer is not None:
                return await reviewer.review(content, self)

            # Unknown content type: direct agent-only path
            logger.info(
                "%s branch=agent_only reason=unknown_type content_type=%s", _MC, content_type
            )
            ui = f"请审核以下内容（类型 {content_type}）：\n\n{content}"
            rr = await self.run(ui, content_type=content_type)
            out = dict(rr)
            attach_pipeline_review_domain(out)
            logger.info(
                "%s moderate_end path=agent verdict=%s iterations=%s duration_ms=%s",
                _MC,
                response_verdict_hint(out.get("response")),
                out.get("iterations"),
                out.get("duration_ms"),
            )
            return out

        return await _traced()

    def moderate_payload(self, content_type: str, content: str) -> dict[str, Any]:
        import asyncio

        self._enforce_payload_raw(content_type, content)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.review_payload_async(content_type, content))
        raise RuntimeError("在异步上下文中请使用 review_payload_async()")

    def run_sync(self, user_input: str, *, content_type: str = "text") -> dict[str, Any]:
        import asyncio

        return asyncio.run(self.run(user_input, content_type=content_type))

    # ------------------------------------------------------------------
    # Core LLM agent loop
    # ------------------------------------------------------------------

    async def run(
        self,
        user_input: str,
        *,
        content_type: str = "text",
        vision_image_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the LLM tool loop for one review request."""
        metrics = get_metrics()
        settings = self._settings
        t0 = time.perf_counter()
        run_id: Optional[str] = None
        iterations = 0
        final_text = ""
        err: Optional[str] = None

        # Derive system-prompt mode from content type
        prompt_mode = "text" if content_type == "text" else "media"
        active_tools = list(self.tools)

        if settings.observability.metrics_enabled:
            metrics.inc("review.requests_total")

        try:
            self._enforce_user_message(user_input)
            vpath = vision_image_path if content_type == "image" else None
            vision_inline = (
                vpath is not None
                and vision_attachment_eligible(self._settings, vpath)
                and image_readable_for_vision(vpath)
            )

            tool_map_run = {t.name: t for t in active_tools}
            messages = self.build_messages(
                user_input,
                vision_image_path=vpath if vision_inline else None,
                prompt_mode=prompt_mode,
            )
            # Multimodal + tools together: many vision models cannot handle them reliably;
            # inline image → chat-only (no bind_tools).
            chat = self.llm if vision_inline else self.llm.bind_tools(active_tools)
            logger.info(
                "%s agent_run begin content_type=%s prompt_mode=%s vision_inline=%s "
                "bind_tools=%s tools=%s max_iter=%s provider=%s",
                _MC,
                content_type,
                prompt_mode,
                vision_inline,
                not vision_inline,
                [t.name for t in active_tools],
                self.max_iterations,
                settings.llm.provider,
            )

            for iteration in range(self.max_iterations):
                iterations = iteration + 1
                if settings.observability.metrics_enabled:
                    with metrics.time_block("llm.invoke_ms"):
                        ai_msg = await chat.ainvoke(messages)
                else:
                    ai_msg = await chat.ainvoke(messages)

                if not isinstance(ai_msg, AIMessage):
                    messages.append(ai_msg)
                    final_text = str(getattr(ai_msg, "content", ai_msg))
                    final_text = await _maybe_repair_verdict_json(self.llm, final_text, user_input)
                    break

                tool_calls = getattr(ai_msg, "tool_calls", None) or []
                if tool_calls:
                    messages.append(ai_msg)
                    for tc in tool_calls:
                        name = tc.get("name")
                        logger.debug(
                            "%s agent_run tool_round=%s tool=%s", _MC, iteration, name
                        )
                        tid = tc.get("id") or name or "call"
                        raw_args = tc.get("args")
                        if raw_args is None and "function" in tc:
                            fn = tc["function"]
                            name = fn.get("name", name)
                            raw = fn.get("arguments", "{}")
                            raw_args = json.loads(raw) if isinstance(raw, str) else raw
                        if raw_args is None and isinstance(tc.get("arguments"), str):
                            raw_args = json.loads(tc["arguments"])
                        args_dict = raw_args if isinstance(raw_args, dict) else {}
                        tool = tool_map_run.get(name or "")
                        if tool is None:
                            out = f"unknown tool: {name}"
                        else:
                            try:
                                out = tool.invoke(args_dict)
                            except Exception as e:
                                out = f"Error: {e}"
                        if isinstance(out, (dict, list)):
                            tool_body = json.dumps(out, ensure_ascii=False)
                        else:
                            tool_body = str(out)
                        messages.append(ToolMessage(content=tool_body, tool_call_id=str(tid)))
                    continue

                messages.append(ai_msg)
                final_text = ai_msg.content or ""
                final_text = await _maybe_repair_verdict_json(self.llm, final_text, user_input)
                self.memory.short_term.add_user_message(user_input)
                self.memory.short_term.add_ai_message(final_text)
                break
            else:
                # Hit max_iterations; persist last assistant-visible output if any.
                final_text = ""
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and (m.content or "").strip():
                        final_text = m.content or ""
                        break
                if not final_text and messages:
                    final_text = str(getattr(messages[-1], "content", "") or "")
                if not err:
                    final_text = await _maybe_repair_verdict_json(
                        self.llm, final_text or "", user_input
                    )
                    self.memory.short_term.add_user_message(user_input)
                    self.memory.short_term.add_ai_message(
                        final_text or "（已达最大工具轮次，请用户简化请求或提高 max_iterations）"
                    )
        except Exception as e:
            err = enrich_agent_error(e, settings)
            if settings.observability.metrics_enabled:
                metrics.inc("review.errors_total")

        duration_ms = (time.perf_counter() - t0) * 1000.0

        result: dict[str, Any] = {
            "success": err is None,
            "response": final_text,
            "iterations": iterations,
            "duration_ms": round(duration_ms, 2),
            "review_domain": "content_safety",
        }
        if err:
            result["error"] = err

        if not err:
            enrich_result_response_violation_types(result)

        if self._review_store:
            status = "failed" if err else "completed"
            run_id = self._review_store.append_run(
                status=status,
                content_type=content_type,
                input_summary=user_input[:2000],
                result=result if not err else None,
                error=err,
                iterations=iterations,
                model_provider=settings.llm.provider,
                model_name=settings.llm.model,
                duration_ms=duration_ms,
                task_id=task_id,
            )
            result["run_id"] = run_id

        if settings.observability.metrics_enabled:
            metrics.observe("review.end_to_end_ms", duration_ms)

        attach_pipeline_review_domain(result)

        logger.info(
            "%s agent_run end content_type=%s iterations=%s verdict_hint=%s "
            "success=%s duration_ms=%s run_id=%s",
            _MC,
            content_type,
            iterations,
            response_verdict_hint(result.get("response")),
            result.get("success"),
            result.get("duration_ms"),
            result.get("run_id"),
        )
        return result

    def clear_memory(self) -> None:
        self.memory.clear()


# ---------------------------------------------------------------------------
# Public factory helpers
# ---------------------------------------------------------------------------

def create_review_orchestrator(
    *,
    session_id: Optional[str] = None,
    biz_context: Optional[BizContext] = None,
    **kwargs: Any,
) -> ReviewOrchestrator:
    return ReviewOrchestrator(session_id=session_id, biz_context=biz_context, **kwargs)


def review_job_runner(content_type: str, content: str) -> dict[str, Any]:
    """Run one job synchronously (queue worker thread)."""
    from reviewagent.observability.file_logging import configure_reviewagent_logging

    configure_reviewagent_logging(get_settings())
    orch = create_review_orchestrator()
    return orch.moderate_payload(content_type, content)
