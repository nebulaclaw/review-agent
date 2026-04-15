"""Generate pinyin variants for CJK wordlist entries to catch romanized evasion."""

from __future__ import annotations

import re

from reviewagent.config import PipelineWordlistConfig
from reviewagent.pipeline.preprocess import normalize_text_for_recall

# Basic CJK block (incl. common Extension A); can extend with more unified ideographs
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _has_cjk(s: str) -> bool:
    return _CJK_RE.search(s) is not None


def pinyin_variants_for_phrase(phrase: str) -> list[str]:
    """
    Toneless pinyin for a phrase: no-space and space-separated syllable variants.
    Requires pypinyin; returns [] if not installed.
    """
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return []

    if not phrase.strip():
        return []

    parts = lazy_pinyin(phrase, style=Style.NORMAL, errors="ignore")
    parts = [p for p in parts if p]
    if not parts:
        return []

    nospace = "".join(parts).lower()
    spaced = " ".join(parts).lower()
    out: list[str] = []
    if nospace:
        out.append(nospace)
    if spaced and spaced != nospace:
        out.append(spaced)
    # Common separators: dot, hyphen (e.g. lian.tong)
    if len(parts) >= 2 and nospace:
        out.append(".".join(parts).lower())
        out.append("-".join(parts).lower())
    return list(dict.fromkeys(out))


def expand_patterns_with_pinyin(
    normalized_patterns: list[tuple[str, str]],
    fc: PipelineWordlistConfig,
) -> list[tuple[str, str]]:
    """
    After normalization, append pinyin variants for entries that contain CJK (via same preprocess).
    Each item is (pattern, category); pinyin variants inherit the original category.
    """
    if not fc.expand_cjk_pinyin:
        return list(normalized_patterns)

    seen: set[str] = {p for p, _ in normalized_patterns}
    out: list[tuple[str, str]] = list(normalized_patterns)

    for raw, cat in normalized_patterns:
        if not _has_cjk(raw):
            continue
        for variant in pinyin_variants_for_phrase(raw):
            nv = normalize_text_for_recall(variant, fc).text
            if nv and nv not in seen:
                seen.add(nv)
                out.append((nv, cat))
    return out


__all__ = [
    "expand_patterns_with_pinyin",
    "pinyin_variants_for_phrase",
    "_has_cjk",
]
