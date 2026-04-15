"""Merge dual image LLM results (OCR-text branch vs vision/pixels branch)."""

from __future__ import annotations

from typing import Any, Literal, Optional

_RANK = {"BLOCK": 3, "WARN": 2, "PASS": 1, "REJECT": 3}

ImageDualMergePolicy = Literal["max_severity", "vision_primary", "ocr_primary"]
ReportLocale = Literal["zh", "en"]


def dual_merge_branch_labels(locale: ReportLocale) -> tuple[str, str]:
    if locale == "en":
        return ("OCR text", "Vision")
    return ("读字分支", "看图分支")


def empty_dual_merge_summary(locale: ReportLocale) -> str:
    if locale == "en":
        return "Dual-branch merge verdict"
    return "双分支合并裁决"


def _no_valid_branch_summary(locale: ReportLocale) -> str:
    if locale == "en":
        return (
            "Neither the OCR-text branch nor the vision branch produced a valid "
            "review result; manual review is recommended."
        )
    return "读字分支与看图分支均未产生有效审核结果，建议人工复核。"


def _norm_verdict(v: Any) -> str:
    s = str(v or "PASS").strip().upper()
    return s if s in _RANK else "PASS"


def merge_dual_verdicts(
    ocr_parsed: Optional[dict[str, Any]],
    vision_parsed: Optional[dict[str, Any]],
    ran_ocr_llm: bool,
    ran_vision_llm: bool,
    *,
    policy: ImageDualMergePolicy = "max_severity",
    report_locale: ReportLocale = "zh",
) -> dict[str, Any]:
    """
    Merge JSON verdicts from the OCR-text and vision branches.

    - max_severity: take the strictest verdict across branches (default, conservative).
    - vision_primary: when both branches have valid JSON, use vision; else fall back.
    - ocr_primary: same, OCR-text branch leads when both valid.
    violations / summary still combine evidence from both for audit.

    report_locale controls the language of branch labels and fallback summaries.
    """
    branches: list[tuple[str, str]] = []
    if ran_ocr_llm and ocr_parsed:
        branches.append(("ocr_text_llm", _norm_verdict(ocr_parsed.get("verdict"))))
    if ran_vision_llm and vision_parsed:
        branches.append(("vision_llm", _norm_verdict(vision_parsed.get("verdict"))))

    if not branches:
        return {
            "verdict": "WARN",
            "confidence": 0.5,
            "violations": [],
            "summary": _no_valid_branch_summary(report_locale),
            "_merge_meta": {
                "branches": [],
                "reason": "no_valid_json",
                "policy": policy,
            },
        }

    final_v: str
    merge_source: str

    if policy == "vision_primary":
        if ran_vision_llm and vision_parsed:
            final_v = _norm_verdict(vision_parsed.get("verdict"))
            merge_source = "vision_primary"
        elif ran_ocr_llm and ocr_parsed:
            final_v = _norm_verdict(ocr_parsed.get("verdict"))
            merge_source = "ocr_fallback_no_vision_json"
        else:
            final_v = max((b[1] for b in branches), key=lambda x: _RANK.get(x, 0))
            merge_source = "fallback_max"
    elif policy == "ocr_primary":
        if ran_ocr_llm and ocr_parsed:
            final_v = _norm_verdict(ocr_parsed.get("verdict"))
            merge_source = "ocr_primary"
        elif ran_vision_llm and vision_parsed:
            final_v = _norm_verdict(vision_parsed.get("verdict"))
            merge_source = "vision_fallback_no_ocr_json"
        else:
            final_v = max((b[1] for b in branches), key=lambda x: _RANK.get(x, 0))
            merge_source = "fallback_max"
    else:
        final_v = max((b[1] for b in branches), key=lambda x: _RANK.get(x, 0))
        merge_source = "max_severity"

    violations: list[dict[str, Any]] = []
    confidences: list[float] = []
    summaries: list[str] = []
    seen: set[tuple[str, str]] = set()

    ocr_label, vis_label = dual_merge_branch_labels(report_locale)
    for label, parsed in ((ocr_label, ocr_parsed), (vis_label, vision_parsed)):
        if not parsed:
            continue
        for v in parsed.get("violations") or []:
            if not isinstance(v, dict):
                continue
            key = (str(v.get("type", "")), str(v.get("content", "")))
            if key in seen:
                continue
            seen.add(key)
            violations.append(v)
        c = parsed.get("confidence")
        if c is not None:
            try:
                confidences.append(float(c))
            except (TypeError, ValueError):
                pass
        s = parsed.get("summary")
        if s:
            summaries.append(f"[{label}] {s}")

    # Primary branch confidence wins (per merge policy)
    primary_conf: list[float] = []
    if policy == "vision_primary" and vision_parsed:
        c = vision_parsed.get("confidence")
        if c is not None:
            try:
                primary_conf.append(float(c))
            except (TypeError, ValueError):
                pass
    elif policy == "ocr_primary" and ocr_parsed:
        c = ocr_parsed.get("confidence")
        if c is not None:
            try:
                primary_conf.append(float(c))
            except (TypeError, ValueError):
                pass
    conf_out = (
        primary_conf[0]
        if primary_conf
        else (max(confidences) if confidences else 0.85)
    )

    return {
        "verdict": final_v,
        "confidence": conf_out,
        "violations": violations,
        "summary": "\n".join(summaries)
        if summaries
        else empty_dual_merge_summary(report_locale),
        "_merge_meta": {
            "branches": branches,
            "policy": policy,
            "merge_source": merge_source,
        },
    }


__all__ = [
    "ImageDualMergePolicy",
    "ReportLocale",
    "dual_merge_branch_labels",
    "empty_dual_merge_summary",
    "merge_dual_verdicts",
]
