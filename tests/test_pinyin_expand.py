"""Unit tests for pinyin variant generation."""

from __future__ import annotations

from reviewagent.config import PipelineWordlistConfig
from reviewagent.pipeline.pinyin_expand import (
    expand_patterns_with_pinyin,
    pinyin_variants_for_phrase,
)


def test_variants_liantong() -> None:
    v = pinyin_variants_for_phrase("恋童")
    assert "liantong" in v
    assert "lian tong" in v


def test_expand_adds_ascii_patterns() -> None:
    fc = PipelineWordlistConfig()
    base = [("恋童", "illegal")]
    out = expand_patterns_with_pinyin(base, fc)
    patterns = [p for p, _ in out]
    assert "恋童" in patterns
    assert "liantong" in patterns
