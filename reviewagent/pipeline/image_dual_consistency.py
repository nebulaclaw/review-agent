"""OCR-text vs vision branch verdict consistency check and optional escalation."""

from __future__ import annotations

from typing import Any, Literal, Optional

_RANK = {"BLOCK": 3, "WARN": 2, "PASS": 1, "REJECT": 3}

DisagreementAction = Literal["none", "elevate_warn"]


def _norm_verdict(v: Any) -> str:
    s = str(v or "PASS").strip().upper()
    return s if s in _RANK else "PASS"


def dual_branch_consistency(
    ocr_parsed: Optional[dict[str, Any]],
    vision_parsed: Optional[dict[str, Any]],
    ran_ocr_llm: bool,
    ran_vision_llm: bool,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """
    Compare verdicts when both branches have valid JSON; otherwise applicable=false.
    """
    if not enabled:
        return {"enabled": False, "applicable": False}
    if not (ran_ocr_llm and ran_vision_llm and ocr_parsed and vision_parsed):
        return {"enabled": True, "applicable": False, "reason": "single_branch_or_no_json"}

    ov = _norm_verdict(ocr_parsed.get("verdict"))
    vv = _norm_verdict(vision_parsed.get("verdict"))
    agreed = ov == vv
    return {
        "enabled": True,
        "applicable": True,
        "ocr_verdict": ov,
        "vision_verdict": vv,
        "agreed": agreed,
        "disagreed": not agreed,
    }


def apply_disagreement_to_merged(
    merged: dict[str, Any],
    consistency: dict[str, Any],
    action: DisagreementAction,
) -> bool:
    """
    If action=elevate_warn and branches disagree, bump final verdict to at least WARN (PASS→WARN).
    Does not add a "branch disagreement" row to violations/summary (only verdict/confidence;
    see pipeline_trace.consistency). Returns whether merged was modified.
    """
    if action != "elevate_warn":
        return False
    if not consistency.get("applicable") or not consistency.get("disagreed"):
        return False

    fv = _norm_verdict(merged.get("verdict"))
    if _RANK[fv] >= _RANK["WARN"]:
        return False

    merged["verdict"] = "WARN"
    prev_c = merged.get("confidence")
    try:
        merged["confidence"] = min(float(prev_c) if prev_c is not None else 0.85, 0.75)
    except (TypeError, ValueError):
        merged["confidence"] = 0.75
    return True


__all__ = [
    "DisagreementAction",
    "apply_disagreement_to_merged",
    "dual_branch_consistency",
]
