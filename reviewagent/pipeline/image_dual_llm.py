"""Image: after local OCR + wordlist, call LLM twice (OCR text / raw pixels) and merge."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import AIMessage

from reviewagent.config import Settings

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator

from reviewagent.pipeline.image_dual_consistency import (
    apply_disagreement_to_merged,
    dual_branch_consistency,
)
from reviewagent.pipeline.image_dual_merge import merge_dual_verdicts
from reviewagent.pipeline.wordlist_image import ImageWordlistOutcome

logger = logging.getLogger(__name__)

_MC = "[review.core]"
_RANK = {"BLOCK": 3, "WARN": 2, "PASS": 1, "REJECT": 3}


def parse_llm_json_verdict(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object containing verdict from an assistant message."""
    t = (text or "").strip()
    if not t:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "verdict" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict) and "verdict" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _norm_verdict(v: Any) -> str:
    s = str(v or "PASS").strip().upper()
    return s if s in _RANK else "PASS"


async def run_image_dual_llm(
    orchestrator: ReviewOrchestrator,
    image_path: str,
    img_out: ImageWordlistOutcome,
) -> Optional[dict[str, Any]]:
    """
    Order: (optional) OCR text + LLM; (optional) raw image + vision LLM; merge JSON.
    Never uses bind_tools.
    If both LLM paths are disabled in config, returns None (caller falls back to single run()).
    """
    from reviewagent.reviewers.utils import image_readable_for_vision, vision_attachment_eligible

    settings: Settings = orchestrator._settings
    fc = settings.pipeline.wordlist
    if not fc.image_llm_review_ocr_text and not fc.image_llm_review_pixels:
        return None
    t0 = time.perf_counter()
    trace: dict[str, Any] = {"mode": "image_dual_llm", "stages": []}

    ocr_text = (img_out.ocr_text or "").strip()
    prefix = img_out.llm_prompt_prefix or ""

    ocr_parsed: Optional[dict[str, Any]] = None
    vision_parsed: Optional[dict[str, Any]] = None
    llm_calls = 0
    soft_errors: list[str] = []

    do_ocr = bool(fc.image_llm_review_ocr_text and ocr_text)
    do_vis = bool(
        fc.image_llm_review_pixels
        and vision_attachment_eligible(settings, image_path)
        and image_readable_for_vision(image_path)
    )
    logger.info(
        "%s dual_llm_start do_ocr=%s do_vis=%s ocr_chars=%s prefix_chars=%s",
        _MC,
        do_ocr,
        do_vis,
        len(ocr_text),
        len(prefix),
    )
    if do_ocr:
        t1 = time.perf_counter()
        try:
            msgs = orchestrator.dual_llm_messages_ocr_branch(ocr_text, prefix)
            ai = await orchestrator.llm.ainvoke(msgs)
            raw = (ai.content or "").strip() if isinstance(ai, AIMessage) else str(ai)
            ocr_parsed = parse_llm_json_verdict(raw)
            llm_calls += 1
            trace["stages"].append(
                {
                    "name": "llm_ocr_text",
                    "ms": round((time.perf_counter() - t1) * 1000.0, 3),
                    "parsed_ok": bool(ocr_parsed),
                }
            )
            if not ocr_parsed:
                soft_errors.append("ocr_text_llm_no_json")
        except Exception as e:
            soft_errors.append(f"ocr_text_llm:{e!s}")
            trace["stages"].append(
                {
                    "name": "llm_ocr_text",
                    "error": str(e)[:300],
                    "ms": round((time.perf_counter() - t1) * 1000.0, 3),
                }
            )
    else:
        trace["stages"].append(
            {
                "name": "llm_ocr_text",
                "skipped": True,
                "reason": "disabled" if not fc.image_llm_review_ocr_text else "empty_ocr",
            }
        )

    if do_vis:
        t2 = time.perf_counter()
        try:
            msgs = orchestrator.dual_llm_messages_vision_branch(image_path)
            ai = await orchestrator.llm.ainvoke(msgs)
            raw = (ai.content or "").strip() if isinstance(ai, AIMessage) else str(ai)
            vision_parsed = parse_llm_json_verdict(raw)
            llm_calls += 1
            trace["stages"].append(
                {
                    "name": "llm_vision_pixels",
                    "ms": round((time.perf_counter() - t2) * 1000.0, 3),
                    "parsed_ok": bool(vision_parsed),
                }
            )
            if not vision_parsed:
                soft_errors.append("vision_llm_no_json")
        except Exception as e:
            soft_errors.append(f"vision_llm:{e!s}")
            trace["stages"].append(
                {
                    "name": "llm_vision_pixels",
                    "error": str(e)[:300],
                    "ms": round((time.perf_counter() - t2) * 1000.0, 3),
                }
            )
    else:
        trace["stages"].append(
            {
                "name": "llm_vision_pixels",
                "skipped": True,
                "reason": "disabled"
                if not fc.image_llm_review_pixels
                else "not_eligible_or_unreadable",
            }
        )

    fp_cfg = settings.pipeline.fingerprint
    dual_cfg = settings.pipeline.image_dual_check
    pol = fp_cfg.image_dual_merge_policy
    loc = dual_cfg.report_locale
    merged = merge_dual_verdicts(
        ocr_parsed,
        vision_parsed,
        do_ocr,
        do_vis,
        policy=pol,
        report_locale=loc,
    )
    merge_meta = merged.pop("_merge_meta", {})
    consistency = dual_branch_consistency(
        ocr_parsed,
        vision_parsed,
        do_ocr,
        do_vis,
        enabled=dual_cfg.image_dual_consistency_enabled,
    )
    trace["consistency"] = consistency
    if (
        dual_cfg.image_dual_consistency_enabled
        and consistency.get("disagreed")
        and settings.observability.metrics_enabled
    ):
        from reviewagent.observability.metrics import get_metrics

        get_metrics().inc("pipeline.image_dual_disagreement_total")
    apply_disagreement_to_merged(
        merged,
        consistency,
        dual_cfg.image_dual_disagreement_action,
    )
    trace["merge"] = merge_meta
    trace["branch_verdicts"] = {
        "ocr_text_llm": _norm_verdict(ocr_parsed.get("verdict")) if ocr_parsed else None,
        "vision_llm": _norm_verdict(vision_parsed.get("verdict")) if vision_parsed else None,
    }
    duration_ms = (time.perf_counter() - t0) * 1000.0
    trace["duration_ms"] = round(duration_ms, 3)
    trace["llm_calls"] = llm_calls
    if soft_errors:
        trace["soft_warnings"] = soft_errors

    logger.info(
        "%s dual_llm_end calls=%d final=%s ocr_ok=%s vision_ok=%s duration_ms=%s",
        _MC,
        llm_calls,
        merged.get("verdict"),
        bool(ocr_parsed),
        bool(vision_parsed),
        round(duration_ms, 2),
    )

    return {
        "success": True,
        "response": json.dumps(merged, ensure_ascii=False),
        "iterations": llm_calls,
        "duration_ms": round(duration_ms, 2),
        "pipeline_trace": trace,
        "error": None,
    }


__all__ = [
    "merge_dual_verdicts",
    "parse_llm_json_verdict",
    "run_image_dual_llm",
]
