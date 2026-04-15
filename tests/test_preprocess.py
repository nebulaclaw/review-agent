"""Text preprocessing self-tests."""

from __future__ import annotations

from reviewagent.config import PipelineWordlistConfig
from reviewagent.pipeline.preprocess import normalize_text_for_recall


def test_strip_zero_width() -> None:
    cfg = PipelineWordlistConfig(
        preprocess_nfkc=False, preprocess_lowercase=False, strip_zero_width=True
    )
    raw = "微\u200b信"
    pr = normalize_text_for_recall(raw, cfg)
    assert pr.text == "微信"
    assert pr.removed_cf_count == 1


def test_nfkc_and_lower() -> None:
    cfg = PipelineWordlistConfig(
        preprocess_nfkc=True, preprocess_lowercase=True, strip_zero_width=False
    )
    pr = normalize_text_for_recall("ＡＢｃ", cfg)
    assert pr.text == "abc"
