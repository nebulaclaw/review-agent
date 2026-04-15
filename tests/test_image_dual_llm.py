"""Dual-branch image LLM: unit tests for parse and merge logic."""

from __future__ import annotations

from reviewagent.pipeline.image_dual_llm import (
    merge_dual_verdicts,
    parse_llm_json_verdict,
)


def test_parse_llm_json_verdict_fence() -> None:
    raw = '```json\n{"verdict": "BLOCK", "confidence": 1, "violations": [], "summary": "x"}\n```'
    d = parse_llm_json_verdict(raw)
    assert d is not None
    assert d["verdict"] == "BLOCK"


def test_merge_dual_block_wins() -> None:
    ocr = {"verdict": "PASS", "confidence": 0.9, "violations": [], "summary": "t"}
    vis = {"verdict": "BLOCK", "confidence": 0.95, "violations": [{"type": "x", "content": "y"}], "summary": "v"}
    m = merge_dual_verdicts(ocr, vis, True, True)
    assert m["verdict"] == "BLOCK"
    assert len(m["violations"]) >= 1


def test_merge_neither_json() -> None:
    m = merge_dual_verdicts(None, None, True, True)
    assert m["verdict"] == "WARN"
