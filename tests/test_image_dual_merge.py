"""Merge policy for dual-branch image LLM results."""

from __future__ import annotations

from reviewagent.pipeline.image_dual_merge import merge_dual_verdicts


def test_merge_max_severity_block_wins() -> None:
    ocr = {"verdict": "PASS", "confidence": 0.9, "violations": [], "summary": "t"}
    vis = {"verdict": "BLOCK", "confidence": 0.95, "violations": [{"type": "x", "content": "y"}], "summary": "v"}
    m = merge_dual_verdicts(ocr, vis, True, True, policy="max_severity")
    assert m["verdict"] == "BLOCK"
    assert m["_merge_meta"]["merge_source"] == "max_severity"


def test_merge_vision_primary_prefers_pass_over_ocr_block() -> None:
    ocr = {"verdict": "BLOCK", "confidence": 0.99, "violations": [], "summary": "bad text"}
    vis = {"verdict": "PASS", "confidence": 0.8, "violations": [], "summary": "ok pic"}
    m = merge_dual_verdicts(ocr, vis, True, True, policy="vision_primary")
    assert m["verdict"] == "PASS"
    assert m["_merge_meta"]["merge_source"] == "vision_primary"


def test_merge_ocr_primary_prefers_block_over_vision_pass() -> None:
    ocr = {"verdict": "BLOCK", "confidence": 0.9, "violations": [], "summary": "bad"}
    vis = {"verdict": "PASS", "confidence": 0.95, "violations": [], "summary": "ok"}
    m = merge_dual_verdicts(ocr, vis, True, True, policy="ocr_primary")
    assert m["verdict"] == "BLOCK"
    assert m["_merge_meta"]["merge_source"] == "ocr_primary"


def test_merge_neither_json() -> None:
    m = merge_dual_verdicts(None, None, True, True, policy="max_severity")
    assert m["verdict"] == "WARN"
    assert m["_merge_meta"]["reason"] == "no_valid_json"
    assert "读字分支" in m["summary"] and "看图分支" in m["summary"]


def test_merge_summary_ocr_and_vision_newlines_zh() -> None:
    ocr = {"verdict": "PASS", "confidence": 0.9, "violations": [], "summary": "ocr branch note"}
    vis = {"verdict": "PASS", "confidence": 0.8, "violations": [], "summary": "vision branch note"}
    m = merge_dual_verdicts(ocr, vis, True, True, policy="max_severity", report_locale="zh")
    assert m["summary"] == "[读字分支] ocr branch note\n[看图分支] vision branch note"


def test_merge_summary_locale_en() -> None:
    ocr = {"verdict": "PASS", "confidence": 0.9, "violations": [], "summary": "Text is clean."}
    vis = {"verdict": "PASS", "confidence": 0.8, "violations": [], "summary": "Image is clean."}
    m = merge_dual_verdicts(ocr, vis, True, True, policy="max_severity", report_locale="en")
    assert m["summary"] == "[OCR text] Text is clean.\n[Vision] Image is clean."
