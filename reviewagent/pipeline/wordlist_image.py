"""Image: OCR first, then run the same wordlist recall as plain text."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from reviewagent.config import Settings
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.pipeline.wordlist_text import run_text_wordlist

logger = logging.getLogger(__name__)

_MC = "[review.core]"
_OCR_LOG_PREVIEW = 400


def _short_preview(text: str, limit: int = _OCR_LOG_PREVIEW) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


@dataclass
class ImageWordlistOutcome:
    """If early_result is set, no further LLM; else llm_prompt_prefix / ocr_text for dual LLM."""

    early_result: Optional[dict[str, Any]]
    llm_prompt_prefix: str
    pipeline_trace: dict[str, Any]
    ocr_text: str = ""


def run_image_wordlist(
    image_path: str,
    biz: BizContext,
    settings: Settings,
) -> Optional[ImageWordlistOutcome]:
    """
    Returns None when disabled or path is not a file (full-image review path).
    On OCR failure or empty text, returns outcome without early_result and empty prefix.
    """
    fc = settings.pipeline.wordlist
    if not fc.scan_image_ocr_for_wordlist:
        logger.debug(
            "wordlist_image_skip reason=scan_image_ocr_for_wordlist_false file=%s",
            Path(image_path).name,
        )
        return None
    path = Path(image_path)
    if not path.is_file():
        logger.info("wordlist_image_skip reason=not_a_file path=%s", image_path[:200])
        return None

    t_all0 = time.perf_counter()
    trace: dict[str, Any] = {
        "mode": "wordlist_image",
        "biz_line": biz.biz_line,
        "tenant_id": biz.tenant_id,
        "trust_tier": biz.trust_tier,
        "audience": biz.audience,
        "policy_pack_id": biz.policy_pack_id,
        "stages": [],
    }

    logger.info(
        "%s wordlist_image_start file=%s path_len=%s",
        _MC,
        path.name,
        len(image_path),
    )

    t0 = time.perf_counter()
    from reviewagent.toolpacks.image_detector import ImageDetector

    det = ImageDetector()
    res = det.detect_sync(str(path))
    ocr_ms = (time.perf_counter() - t0) * 1000.0
    trace["stages"].append(
        {
            "name": "ocr",
            "ms": round(ocr_ms, 3),
            "success": res.get("success"),
            "verdict": res.get("verdict"),
        }
    )

    if not res.get("success"):
        trace["duration_ms"] = round((time.perf_counter() - t_all0) * 1000.0, 3)
        logger.warning(
            "wordlist_image_ocr_failed file=%s error=%s",
            path.name,
            (res.get("error") or "")[:300],
        )
        return ImageWordlistOutcome(None, "", trace, ocr_text="")

    details = res.get("details") or {}
    ocr_text = (details.get("detected_text") or "").strip()
    trace["stages"][-1]["ocr_text_chars"] = len(ocr_text)
    trace["stages"][-1]["ocr_raw_pass_count"] = details.get("ocr_raw_pass_count")

    logger.info(
        "wordlist_image_ocr file=%s text_chars=%d ocr_passes=%s ocr_hint=%s",
        path.name,
        len(ocr_text),
        details.get("ocr_raw_pass_count"),
        details.get("ocr_hint"),
    )
    if ocr_text:
        logger.debug(
            "wordlist_image_ocr_preview file=%s text=%r",
            path.name,
            _short_preview(ocr_text),
        )

    if not ocr_text:
        trace["stages"].append({"name": "ocr_text_empty", "decision": "CONTINUE_TO_LLM"})
        trace["duration_ms"] = round((time.perf_counter() - t_all0) * 1000.0, 3)
        logger.info(
            "wordlist_image_wordlist_skip reason=ocr_empty file=%s "
            "(wordlist applies to OCR text only; continuing to LLM)",
            path.name,
        )
        return ImageWordlistOutcome(None, "", trace, ocr_text="")

    txt_out = run_text_wordlist(
        ocr_text,
        biz,
        settings,
        trace=trace,
        duration_anchor=t_all0,
        stage_suffix="_ocr",
        compose_full_text_review_message=False,
        ocr_excerpt_for_prefix=ocr_text[:4000],
        image_ocr=True,
    )

    if txt_out.early_result is not None:
        er = dict(txt_out.early_result)
        er["pipeline_trace"] = txt_out.pipeline_trace
        logger.info(
            "wordlist_image_early_block file=%s ocr_chars=%d (wordlist hit; vision LLM skipped)",
            path.name,
            len(ocr_text),
        )
        return ImageWordlistOutcome(
            er, "", txt_out.pipeline_trace, ocr_text=ocr_text
        )

    prefix = (txt_out.image_llm_prefix or "").strip()
    if prefix:
        prefix = prefix + "\n\n"
    logger.info(
        "wordlist_image_continue_llm file=%s ocr_chars=%d llm_prefix_chars=%d (no hard wordlist block)",
        path.name,
        len(ocr_text),
        len(prefix),
    )
    return ImageWordlistOutcome(None, prefix, txt_out.pipeline_trace, ocr_text=ocr_text)


__all__ = ["ImageWordlistOutcome", "run_image_wordlist"]
