"""Heuristic: **short** plain-text follow-ups that clearly ask to re-run staged media review.

We intentionally **do not** try to classify long pasted content: those always go through normal
text review to avoid false positives when phrases like 「再检测」 appear inside a document.
"""

from __future__ import annotations

import re

# Only one–two conversational lines; longer input is treated as new material, not a re-check cue.
_DEFAULT_MAX_CHARS = 160

_RE_RECHECK = re.compile(
    r"(?:"
    r"再\s*检|重新\s*检测|再\s*检测|重\s*检|复\s*检|"
    r"再\s*审核|重新\s*审核|再\s*审|重\s*审|"
    r"再\s*跑\s*一?\s*次|再\s*来\s*一?\s*次|再\s*看\s*一?\s*次|再\s*查\s*一?\s*次|"
    r"核\s*实|不\s*信|搞\s*错|有\s*误|不\s*对|异\s*议|"
    r"确定.{0,12}(?:正常|准确|靠谱)|"
    r"re[\s-]*check|re[\s-]*analy[sz]e|re[\s-]*run|"
    r"check\s*again|run\s*again|detect\s*again|review\s*again"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def text_suggests_recheck_same_media(text: str, *, max_chars: int = _DEFAULT_MAX_CHARS) -> bool:
    """True only for short messages matching obvious re-check / dispute phrasing (zh + en)."""
    t = (text or "").strip()
    if not t or len(t) > max_chars:
        return False
    return bool(_RE_RECHECK.search(t))
