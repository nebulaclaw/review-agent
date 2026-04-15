"""Review JSON helpers: derive violation_type_labels from violations (category labels)."""

from __future__ import annotations

import json
from typing import Any, Optional

from reviewagent.content_violation import (
    ReportDisplayLocale,
    label_for_violation_type,
    violation_category_labels,
)


def _effective_report_locale(locale: Optional[ReportDisplayLocale]) -> ReportDisplayLocale:
    if locale in ("zh", "en"):
        return locale
    from reviewagent.config import get_settings

    loc = get_settings().pipeline.image_dual_check.report_locale
    return loc if loc in ("zh", "en") else "zh"

_NON_REPORT_VIOLATION_TYPES = frozenset({"dual_branch_disagreement"})


def violations_for_report_display(violations: Any) -> list[dict[str, Any]]:
    """For TUI/web display: drop non-user-facing items (e.g. dual-branch consistency notes)."""
    if not isinstance(violations, list):
        return []
    return [
        x
        for x in violations
        if isinstance(x, dict)
        and str(x.get("type", "")).strip().lower() not in _NON_REPORT_VIOLATION_TYPES
    ]


def _strip_non_report_violations(obj: dict[str, Any]) -> None:
    """Remove pipeline-only entries from violations before user-facing reports."""
    obj["violations"] = violations_for_report_display(obj.get("violations"))


def label_for_violation_kind(kind: str) -> str:
    """Map violations[].type to a display label (legacy wrapper around label_for_violation_type)."""
    return label_for_violation_type(kind)


def compute_violation_type_labels(
    obj: dict[str, Any], *, locale: ReportDisplayLocale = "zh"
) -> list[str]:
    """
    Aggregate top-level content categories from violations.
    For WARN/BLOCK with nothing categorized, return unspecified label.
    """
    violations = obj.get("violations")
    labels = violation_category_labels(violations, locale=locale)
    if labels:
        return labels
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("WARN", "BLOCK"):
        return []
    return ["未标明"] if locale == "zh" else ["Unspecified"]


def enrich_review_json_in_response(
    response: str, *, locale: Optional[ReportDisplayLocale] = None
) -> Optional[str]:
    """Parse review JSON, refresh violation_type_labels; drop model-supplied violation_types to avoid confusion."""
    s = (response or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "verdict" not in obj:
        return None
    obj.pop("violation_types", None)
    _strip_non_report_violations(obj)
    eff = _effective_report_locale(locale)
    obj["violation_type_labels"] = compute_violation_type_labels(obj, locale=eff)
    return json.dumps(obj, ensure_ascii=False)


def batch_item_source_label(item: dict[str, Any], *, locale: str = "zh") -> str:
    """Human-readable source for a batch item: path, then filename, then index (locale-specific)."""
    loc = locale if locale in ("zh", "en") else "zh"
    p = str(item.get("path") or "").strip()
    if p:
        return p
    fn = str(item.get("filename") or "").strip()
    if fn:
        return fn
    idx = item.get("index")
    if isinstance(idx, int) and idx >= 0:
        if loc == "en":
            return f"Item {idx + 1}"
        return f"第 {idx + 1} 项"
    return "Unknown source" if loc == "en" else "未知来源"


def batch_item_verdict(item: dict[str, Any]) -> str:
    """Parse verdict from a batch item: PASS / WARN / BLOCK / ERROR / UNKNOWN."""
    if item.get("success") is False:
        return "ERROR"
    err = item.get("error")
    if err is not None and str(err).strip():
        return "ERROR"
    raw = (item.get("response") or "").strip()
    if not raw:
        return "UNKNOWN"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "UNKNOWN"
    if not isinstance(obj, dict):
        return "UNKNOWN"
    v = str(obj.get("verdict", "")).strip().upper()
    if v in ("PASS", "WARN", "BLOCK"):
        return v
    return "UNKNOWN"


def format_batch_summary(results: list[dict[str, Any]], *, locale: str = "zh") -> str:
    """
    One-line batch summary: per-verdict counts and overall outcome.
    Rejected: any BLOCK or ERROR; needs review: WARN/UNKNOWN with no BLOCK/ERROR; else passed.
    """
    loc = locale if locale in ("zh", "en") else "zh"
    n = len(results)
    b = w = p = e = u = 0
    for it in results:
        v = batch_item_verdict(it)
        if v == "BLOCK":
            b += 1
        elif v == "WARN":
            w += 1
        elif v == "PASS":
            p += 1
        elif v == "ERROR":
            e += 1
        else:
            u += 1
    tallies: list[str] = []
    if b:
        tallies.append(f"BLOCK×{b}")
    if w:
        tallies.append(f"WARN×{w}")
    if p:
        tallies.append(f"PASS×{p}")
    if e:
        tallies.append(f"failed×{e}" if loc == "en" else f"失败×{e}")
    if u:
        tallies.append(f"other×{u}" if loc == "en" else f"其它×{u}")
    sep = ", " if loc == "en" else "、"
    if loc == "en":
        none_tally = "none"
        if b > 0 or e > 0:
            overall = "Rejected"
        elif w > 0 or u > 0:
            overall = "Needs review"
        else:
            overall = "Passed"
        return f"[Batch] {n} item(s): {sep.join(tallies) if tallies else none_tally} → {overall}"
    detail = sep.join(tallies) if tallies else "无分项"
    if b > 0 or e > 0:
        overall = "不通过"
    elif w > 0 or u > 0:
        overall = "待复核"
    else:
        overall = "通过"
    return f"【批量检测】共 {n} 项：{detail} → 整体：{overall}"


def format_batch_summary_zh(results: list[dict[str, Any]]) -> str:
    """Backward compat: same as format_batch_summary with locale fixed to zh."""
    return format_batch_summary(results, locale="zh")


def enrich_result_response_violation_types(result: dict[str, Any]) -> None:
    """In-place update of result['response'] when success and body is review JSON."""
    if result.get("error"):
        return
    if result.get("success") is False:
        return
    raw = result.get("response")
    if not isinstance(raw, str):
        return
    new_s = enrich_review_json_in_response(raw, locale=None)
    if new_s is not None:
        result["response"] = new_s


__all__ = [
    "batch_item_source_label",
    "batch_item_verdict",
    "compute_violation_type_labels",
    "enrich_result_response_violation_types",
    "enrich_review_json_in_response",
    "format_batch_summary_zh",
    "label_for_violation_kind",
    "violations_for_report_display",
]
