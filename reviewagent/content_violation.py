"""Violation taxonomy: violations[].type / position → display labels (summary and rows; zh/en)."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, NamedTuple, Optional, Union

ReportDisplayLocale = Literal["zh", "en"]


class LocalizedLabel(NamedTuple):
    """A single display label in both locales."""

    zh: str
    en: str


# Content category (what the violation is about) 
class ContentViolationType(str, Enum):
    """Violation subject — what policy the content violates."""

    PORN = "porn"
    VIOLENCE = "violence"
    HATE = "hate"
    ILLEGAL = "illegal"
    HARASS = "harass"
    MISINFO = "misinfo"
    PRIVACY = "privacy"
    SPAM = "spam"
    DANGEROUS = "dangerous"
    SELF_HARM = "self_harm"
    NORMAL = "normal"


_CONTENT_TYPE_LABELS: dict[str, LocalizedLabel] = {
    "porn":      LocalizedLabel("色情内容", "Pornographic content"),
    "violence":  LocalizedLabel("暴力血腥", "Violence / gore"),
    "hate":      LocalizedLabel("仇恨言论", "Hate speech"),
    "illegal":   LocalizedLabel("违法信息", "Illegal content"),
    "harass":    LocalizedLabel("骚扰霸凌", "Harassment"),
    "misinfo":   LocalizedLabel("虚假信息", "Misinformation"),
    "privacy":   LocalizedLabel("隐私侵犯", "Privacy violation"),
    "spam":      LocalizedLabel("垃圾广告", "Spam"),
    "dangerous": LocalizedLabel("危险行为", "Dangerous acts"),
    "self_harm": LocalizedLabel("自残自杀", "Self-harm / suicide"),
    "normal":    LocalizedLabel("正常内容", "Normal content"),
}


# Detection method (how the violation was found) 
class DetectionMethod(str, Enum):
    """Pipeline stage / detection technique that flagged the content."""

    WORDLIST = "wordlist"
    TEXT_DETECTOR = "text_detector"
    IMAGE_PHASH = "image_phash"
    VIOLATION = "violation"


_DETECTION_METHOD_LABELS: dict[str, LocalizedLabel] = {
    "wordlist":       LocalizedLabel("敏感词表命中",   "Sensitive wordlist hit"),
    "text_detector":  LocalizedLabel("文本检测工具命中", "Text detector hit"),
    "image_phash":    LocalizedLabel("图像指纹封禁",   "Image fingerprint block"),
    "violation":      LocalizedLabel("违规内容",       "Policy violation"),
}


# Merged label table (for lookup functions that accept either dimension)
_ALL_TYPE_LABELS: dict[str, LocalizedLabel] = {
    **_DETECTION_METHOD_LABELS,
    **_CONTENT_TYPE_LABELS,
}

VIOLATION_TYPE_LABELS: dict[str, str] = {k: v.zh for k, v in _ALL_TYPE_LABELS.items()}
VIOLATION_TYPE_LABELS_EN: dict[str, str] = {k: v.en for k, v in _ALL_TYPE_LABELS.items()}


# Position taxonomy (detection branch / modality) 
_OCR_BRANCH = LocalizedLabel("读字分支", "OCR-text branch")
_VISION_BRANCH = LocalizedLabel("看图分支", "Vision branch")

_POSITION_LABELS: dict[str, LocalizedLabel] = {
    "ocr_text":          _OCR_BRANCH,
    "ocr_text_llm":      _OCR_BRANCH,
    "llm_ocr_text":      _OCR_BRANCH,
    "vision":            _VISION_BRANCH,
    "vision_llm":        _VISION_BRANCH,
    "llm_vision_pixels": _VISION_BRANCH,
    "image":             LocalizedLabel("图片", "Image"),
    "text":              LocalizedLabel("文本", "Text"),
}


# ── Severity labels ─────────────────────────────────────────────────────

_SEVERITY_LABELS: dict[str, LocalizedLabel] = {
    "high":   LocalizedLabel("高", "High"),
    "medium": LocalizedLabel("中", "Medium"),
    "low":    LocalizedLabel("低", "Low"),
}


# Summary roll-up config 
_SKIP_SUMMARY_TYPES: frozenset[str] = frozenset({
    "dual_branch_disagreement",
})


# Video-frame position patterns
_RE_VIDEO_FRAMES_MERGED = re.compile(
    r"^frames\s+(\d+)-(\d+)\s+\((\d+)\s+sampled-frame hits merged\)\s*$",
    re.IGNORECASE,
)
_RE_VIDEO_FRAME_ONE = re.compile(r"^frame\s+(\d+)\s*$", re.IGNORECASE)



# Label lookup helpers
def _pick_locale(label: LocalizedLabel, locale: ReportDisplayLocale) -> str:
    return label.en if locale == "en" else label.zh


def label_for_violation_type(
    type_str: str, *, locale: ReportDisplayLocale = "zh"
) -> str:
    """Map one violations[].type to a display label; unknown values return the raw type string."""
    raw = (type_str or "").strip()
    if not raw:
        return "未标明" if locale == "zh" else "Unspecified"
    label = _ALL_TYPE_LABELS.get(raw.lower())
    if label is None:
        return raw
    return _pick_locale(label, locale)


def label_for_violation_position(
    position_str: str, *, locale: ReportDisplayLocale = "zh"
) -> str:
    """Map violations[].position to a display label; unknown values return the original string."""
    raw = (position_str or "").strip()
    if not raw:
        return ""
    label = _POSITION_LABELS.get(raw.lower())
    if label is None:
        return raw
    return _pick_locale(label, locale)


def label_for_severity(
    severity: Union[str, int, float, None], *, locale: ReportDisplayLocale = "zh"
) -> str:
    """Map violations[].severity to a short report label for the current locale."""
    raw = str(severity or "").strip().lower()
    if not raw:
        return ""
    label = _SEVERITY_LABELS.get(raw)
    if label is None:
        return str(severity).strip()
    return _pick_locale(label, locale)


# Report formatting
def format_violation_position_for_report(
    position: Union[str, int, float, None], *, locale: ReportDisplayLocale = "zh"
) -> str:
    """
    Format position for violation detail rows: branch names, video frame ranges, etc.
    Non-enumerated values (e.g. numeric offsets) are left as-is.
    """
    if position is None:
        return ""
    s = str(position).strip()
    if not s:
        return ""
    m = _RE_VIDEO_FRAMES_MERGED.match(s)
    if m:
        a, b, n = m.group(1), m.group(2), m.group(3)
        if locale == "zh":
            return f"第{a}–{b} 帧（{n} 次采样命中已合并）"
        return f"Frames {a}–{b} ({n} sampled hits merged)"
    m = _RE_VIDEO_FRAME_ONE.match(s)
    if m:
        i = m.group(1)
        if locale == "zh":
            return f"第 {i} 帧"
        return f"Frame {i}"
    return label_for_violation_position(s, locale=locale)


def format_violation_row_for_report(
    item: dict, *, locale: ReportDisplayLocale = "zh"
) -> str:
    """One violation as a report/audit line (type and position localized to report_locale)."""
    sev = label_for_severity(item.get("severity"), locale=locale)
    t_raw = str(item.get("type", "") or "")
    c = item.get("content", "")
    pos_raw = item.get("position")
    type_l = label_for_violation_type(t_raw, locale=locale)
    extra = ""
    if pos_raw is not None and str(pos_raw).strip():
        pos_l = format_violation_position_for_report(pos_raw, locale=locale)
        if pos_l and pos_l.casefold() != type_l.casefold():
            extra = f" · {pos_l}"
    return f"  · [{sev}] {type_l}{extra}: {c}"


# Category aggregation
def violation_category_labels(
    violations: object, *, locale: ReportDisplayLocale = "zh"
) -> list[str]:
    """Deduplicated type labels in first-seen order for the report header category list."""
    if not isinstance(violations, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in violations:
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if t is None:
            continue
        key = str(t).strip().lower()
        if not key or key in _SKIP_SUMMARY_TYPES or key in seen:
            continue
        seen.add(key)
        out.append(label_for_violation_type(str(t), locale=locale))
    return out



__all__ = [
    "ContentViolationType",
    "DetectionMethod",
    "LocalizedLabel",
    "ReportDisplayLocale",
    "VIOLATION_TYPE_LABELS",
    "VIOLATION_TYPE_LABELS_EN",
    "format_violation_position_for_report",
    "format_violation_row_for_report",
    "label_for_severity",
    "label_for_violation_position",
    "label_for_violation_type",
    "violation_category_labels",
]
