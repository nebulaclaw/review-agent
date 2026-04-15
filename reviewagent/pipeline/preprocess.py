"""Text normalization before wordlist matching."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from reviewagent.config import PipelineWordlistConfig


# Common zero-width chars and BOM (evade invisibly split tokens around wordlists)
_ZERO_WIDTH_AND_BOM = frozenset(
    "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\ufeff"
)


@dataclass
class PreprocessResult:
    text: str
    removed_cf_count: int
    nfkc_applied: bool
    lowercased: bool


def normalize_text_for_recall(text: str, cfg: PipelineWordlistConfig) -> PreprocessResult:
    """Apply optional NFKC, strip zero-width, optional lowercasing for recall."""
    t = text
    removed = 0
    if cfg.strip_zero_width:
        out = []
        for ch in t:
            if ch in _ZERO_WIDTH_AND_BOM:
                removed += 1
                continue
            if unicodedata.category(ch) == "Cf" and ch not in "\t\n\r":
                removed += 1
                continue
            out.append(ch)
        t = "".join(out)
    nfkc = False
    if cfg.preprocess_nfkc:
        t2 = unicodedata.normalize("NFKC", t)
        nfkc = t2 != t
        t = t2
    low = False
    if cfg.preprocess_lowercase:
        t2 = t.lower()
        low = t2 != t
        t = t2
    return PreprocessResult(
        text=t,
        removed_cf_count=removed,
        nfkc_applied=nfkc,
        lowercased=low,
    )


__all__ = ["PreprocessResult", "normalize_text_for_recall"]
