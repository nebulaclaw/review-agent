"""Review JSON helpers: derive violation_type_labels from violations (category labels)."""

from __future__ import annotations

import json
import re
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

# Strip in-band chain-of-thought before extracting JSON (e.g. DeepSeek-style).
_COT_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>[\s\S]*?</thinking>", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<redacted_reasoning>[\s\S]*?</redacted_reasoning>",
        re.IGNORECASE | re.DOTALL,
    ),
)

# Models sometimes print pseudo tool-call XML in plain text instead of returning JSON.
_HALLUCINATED_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<tool_call>[\s\S]*?</tool_call>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"<tool_calls>[\s\S]*?</tool_calls>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"<arg_key>[\s\S]*?</arg_key>\s*<arg_value>[\s\S]*?</arg_value>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?m)^\s*</tool_call>\s*$", re.IGNORECASE),
)
# Whole-line tool name leaked into content (no JSON tool_calls)
_STANDALONE_TOOL_LINE = re.compile(
    r"(?m)^\s*(?:text_detector|image_detector|video_detector|audio_detector)\s*$",
    re.IGNORECASE,
)


def strip_llm_reasoning_sections(text: str) -> str:
    """Remove common model reasoning wrappers that are not part of the review JSON."""
    t = text or ""
    for pat in _COT_BLOCK_PATTERNS:
        t = pat.sub("", t)
    return t


def strip_llm_hallucinated_tool_markup(text: str) -> str:
    """Remove pseudo tool-call blocks occasionally emitted in assistant *content* (not real tool_calls)."""
    t = text or ""
    for pat in _HALLUCINATED_TOOL_PATTERNS:
        t = pat.sub("", t)
    # Orphan opening tag without a full block
    t = re.sub(r"<tool_calls>\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<tool_call>\s*", "", t, flags=re.IGNORECASE)
    t = _STANDALONE_TOOL_LINE.sub("", t)
    return t


def _strip_standalone_markdown_fence_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*```(?:json)?\s*$", ln, re.IGNORECASE)]
    return "\n".join(lines)


def parse_review_json_from_llm_output(text: str) -> Optional[dict[str, Any]]:
    """
    Extract the content-safety review JSON from raw LLM output.

    Handles markdown fences, multiple JSON objects, and reasoning tags. When several
    dict objects include a top-level "verdict" key, returns the **last** one (draft then revise).
    """
    t = _strip_standalone_markdown_fence_lines(
        strip_llm_hallucinated_tool_markup(strip_llm_reasoning_sections(text))
    )
    dec = json.JSONDecoder()
    last: Optional[dict[str, Any]] = None
    i = 0
    n = len(t)
    while i < n:
        while i < n and t[i].isspace():
            i += 1
        if i >= n:
            break
        if t[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(t, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            last = obj
        i = end
    return last


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
    obj = parse_review_json_from_llm_output(s)
    if obj is None:
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
    obj = parse_review_json_from_llm_output(raw)
    if not isinstance(obj, dict):
        return "UNKNOWN"
    v = str(obj.get("verdict", "")).strip().upper()
    if v in ("PASS", "WARN", "BLOCK", "UNKNOWN"):
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
    "parse_review_json_from_llm_output",
    "strip_llm_hallucinated_tool_markup",
    "strip_llm_reasoning_sections",
    "violations_for_report_display",
]
