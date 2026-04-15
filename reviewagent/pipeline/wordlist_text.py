"""Text: preprocess → wordlist (AC) → early exit on hit, else compose content for the LLM."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from reviewagent.config import Settings
from reviewagent.pipeline.ac_matcher import ACMatch, AhoCorasickAutomaton
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.pipeline.pinyin_expand import expand_patterns_with_pinyin
from reviewagent.pipeline.preprocess import normalize_text_for_recall

logger = logging.getLogger(__name__)
_MC = "[review.core]"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_automaton_cache: dict[str, tuple[AhoCorasickAutomaton, list[str]]] = {}


def _resolve_wordlist_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p.resolve() if p.is_absolute() else (root / rel).resolve()


def _cache_key(settings: Settings) -> str:
    root = _project_root()
    parts: list[str] = []
    for rel in settings.pipeline.wordlist.wordlist_paths:
        p = _resolve_wordlist_path(root, rel)
        m = p.stat().st_mtime if p.is_file() else -1.0
        parts.append(f"{p}:{m}")
    cfg = settings.pipeline.wordlist
    parts.append(
        f"nfkc={cfg.preprocess_nfkc};lower={cfg.preprocess_lowercase};zw={cfg.strip_zero_width}"
        f";pinyin={cfg.expand_cjk_pinyin}"
    )
    return "|".join(parts)


_CATEGORY_DIRECTIVE_PREFIX = "# @category "
_DEFAULT_CATEGORY = "illegal"


def load_wordlist_patterns(settings: Settings) -> list[tuple[str, str]]:
    """Load wordlist files; normalize entries like body text.

    Supports ``# @category <ContentViolationType>`` directives — all
    subsequent entries inherit that category until the next directive.
    Lines starting with ``#`` (without ``@category``) are plain comments.
    Default category is ``illegal``.
    """
    root = _project_root()
    fc = settings.pipeline.wordlist
    patterns: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rel in fc.wordlist_paths:
        p = _resolve_wordlist_path(root, rel)
        if not p.is_file():
            continue
        current_category = _DEFAULT_CATEGORY
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith(_CATEGORY_DIRECTIVE_PREFIX):
                current_category = stripped[len(_CATEGORY_DIRECTIVE_PREFIX):].strip().lower() or _DEFAULT_CATEGORY
                continue
            if stripped.startswith("#"):
                continue
            norm = normalize_text_for_recall(stripped, fc).text
            if norm and norm not in seen:
                seen.add(norm)
                patterns.append((norm, current_category))
    return patterns


def get_automaton(settings: Settings) -> tuple[AhoCorasickAutomaton, list[tuple[str, str]]]:
    key = _cache_key(settings)
    hit = _automaton_cache.get(key)
    if hit is not None:
        return hit
    base = load_wordlist_patterns(settings)
    pats = expand_patterns_with_pinyin(base, settings.pipeline.wordlist)
    ac = AhoCorasickAutomaton()
    for w, cat in pats:
        ac.add(w, cat)
    ac.build()
    _automaton_cache[key] = (ac, pats)
    return ac, pats


def _matches_to_violations(matches: list[ACMatch], cap: int = 50) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in matches:
        if m.pattern in seen:
            continue
        seen.add(m.pattern)
        out.append(
            {
                "type": m.category or "illegal",
                "detection_method": "wordlist",
                "content": m.pattern,
                "severity": "high",
                "position": f"char[{m.start}:{m.end}]",
            }
        )
        if len(out) >= cap:
            break
    return out


def _wordlist_block_summary(locale: str = "zh", *, image_ocr: bool = False) -> str:
    if locale == "en":
        if image_ocr:
            return "Text extracted from the image matched the sensitive wordlist; the request was auto-blocked."
        return "Matched the sensitive wordlist; the request was auto-blocked."
    if image_ocr:
        return "图片中提取的文字命中敏感词表，已被系统自动拦截。"
    return "命中敏感词表，已被系统自动拦截。"


def _block_response_text(matches: list[ACMatch], *, locale: str = "zh", image_ocr: bool = False) -> str:
    body = {
        "verdict": "BLOCK",
        "confidence": 1.0,
        "violations": _matches_to_violations(matches),
        "summary": _wordlist_block_summary(locale, image_ocr=image_ocr),
    }
    return json.dumps(body, ensure_ascii=False)


@dataclass
class TextWordlistOutcome:
    """Wordlist text path result; early_result set means the LLM was not called."""

    early_result: Optional[dict[str, Any]]
    user_input_for_llm: str
    pipeline_trace: dict[str, Any]
    # Image path: when not prefixing with text-review wording, use this note before image instructions
    image_llm_prefix: str = ""


def run_text_wordlist(
    original_text: str,
    biz: BizContext,
    settings: Settings,
    *,
    trace: Optional[dict[str, Any]] = None,
    duration_anchor: Optional[float] = None,
    stage_suffix: str = "",
    compose_full_text_review_message: bool = True,
    ocr_excerpt_for_prefix: Optional[str] = None,
    image_ocr: bool = False,
) -> TextWordlistOutcome:
    """
    :param trace: if set, append stages (e.g. reuse after image OCR).
    :param duration_anchor: start time for trace/early duration (includes OCR, etc.).
    :param compose_full_text_review_message: if False, only fill image_llm_prefix (image OCR wordlist path).
    :param image_ocr: if True, summary mentions text was extracted from an image.
    """
    fc = settings.pipeline.wordlist
    t_all0 = duration_anchor if duration_anchor is not None else time.perf_counter()
    if trace is None:
        trace = {
            "mode": "wordlist_text",
            "biz_line": biz.biz_line,
            "tenant_id": biz.tenant_id,
            "trust_tier": biz.trust_tier,
            "audience": biz.audience,
            "policy_pack_id": biz.policy_pack_id,
            "stages": [],
        }

    sp = stage_suffix or ""
    t0 = time.perf_counter()
    pr = normalize_text_for_recall(original_text, fc)
    pre_ms = (time.perf_counter() - t0) * 1000.0
    trace["stages"].append(
        {
            "name": f"preprocess{sp}",
            "ms": round(pre_ms, 3),
            "removed_cf_count": pr.removed_cf_count,
            "nfkc_applied": pr.nfkc_applied,
            "lowercased": pr.lowercased,
        }
    )

    t0 = time.perf_counter()
    ac, loaded_patterns = get_automaton(settings)
    matches = ac.find_all(pr.text)
    recall_ms = (time.perf_counter() - t0) * 1000.0
    trace["stages"].append(
        {
            "name": f"recall_ac{sp}",
            "ms": round(recall_ms, 3),
            "pattern_count": len(loaded_patterns),
            "match_count": len(matches),
        }
    )

    duration_ms = (time.perf_counter() - t_all0) * 1000.0
    trace["duration_ms"] = round(duration_ms, 3)

    locale = settings.pipeline.image_dual_check.report_locale or "zh"

    if matches and fc.early_exit_on_match:
        logger.info(
            "%s wordlist_text decision=EARLY_BLOCK match_count=%s stage_suffix=%s",
            _MC,
            len(matches),
            sp or "(none)",
        )
        early = {
            "success": True,
            "response": _block_response_text(matches, locale=locale, image_ocr=image_ocr),
            "iterations": 0,
            "duration_ms": round(duration_ms, 2),
            "pipeline_trace": trace,
        }
        return TextWordlistOutcome(
            early_result=early,
            user_input_for_llm="",
            pipeline_trace=trace,
            image_llm_prefix="",
        )

    if matches and not fc.early_exit_on_match:
        logger.info(
            "%s wordlist_text decision=CONTINUE_HINT match_count=%s stage_suffix=%s",
            _MC,
            len(matches),
            sp or "(none)",
        )
        vpreview = ", ".join(sorted({m.pattern for m in matches})[:20])
        hint = (
            f"[system-wordlist] Wordlist scan found these patterns "
            f"(hard block is off; judge using the full text): {vpreview}\n\n"
        )
        if compose_full_text_review_message:
            ui = hint + f"Please review the following text:\n\n{original_text}"
            return TextWordlistOutcome(None, ui, trace, "")
        img_pre = _image_prefix_from_ocr_hint(
            hint, ocr_excerpt_for_prefix or original_text, fc.inject_recall_hint
        )
        return TextWordlistOutcome(None, "", trace, img_pre)

    if fc.inject_recall_hint:
        hint = (
            "[system-wordlist] Normalization and wordlist scan completed with no wordlist hit. "
            f"(removed {pr.removed_cf_count} invisible/control formatting character(s))\n\n"
        )
    else:
        hint = ""

    if compose_full_text_review_message:
        ui = (
            hint + f"Please review the following text:\n\n{original_text}"
            if hint
            else f"Please review the following text:\n\n{original_text}"
        )
        trace["stages"].append({"name": f"recall_decision{sp}", "decision": "CONTINUE"})
        logger.info(
            "%s wordlist_text decision=CONTINUE_LLM match_count=0 inject_hint=%s stage_suffix=%s",
            _MC,
            bool(hint),
            sp or "(none)",
        )
        return TextWordlistOutcome(None, ui, trace, "")

    img_pre = _image_prefix_from_ocr_hint(
        hint
        or "[system-wordlist] OCR text wordlist scan did not match a hard-block pattern.\n\n",
        ocr_excerpt_for_prefix or original_text,
        fc.inject_recall_hint,
    )
    trace["stages"].append({"name": f"recall_decision{sp}", "decision": "CONTINUE"})
    logger.info(
        "%s wordlist_text decision=CONTINUE_IMAGE_PREFIX match_count=0 inject_hint=%s stage_suffix=%s",
        _MC,
        bool(hint),
        sp or "(none)",
    )
    return TextWordlistOutcome(None, "", trace, img_pre)


def _image_prefix_from_ocr_hint(hint: str, ocr_excerpt: str, inject: bool) -> str:
    if not inject:
        return ""
    excerpt = (ocr_excerpt or "").strip()
    if len(excerpt) > 3500:
        excerpt = excerpt[:3500] + "…"
    return f"{hint.strip()}\n[OCR excerpt]\n{excerpt}\n"


def clear_automaton_cache() -> None:
    """Clear automaton cache (for tests)."""
    _automaton_cache.clear()


__all__ = [
    "TextWordlistOutcome",
    "run_text_wordlist",
    "load_wordlist_patterns",
    "clear_automaton_cache",
    "get_automaton",
]
