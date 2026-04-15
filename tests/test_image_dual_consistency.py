"""Dual-branch image LLM consistency checks and escalation."""

from __future__ import annotations

from reviewagent.pipeline.image_dual_consistency import (
    apply_disagreement_to_merged,
    dual_branch_consistency,
)


def test_consistency_applicable_disagreed() -> None:
    ocr = {"verdict": "PASS", "violations": [], "summary": "a"}
    vis = {"verdict": "BLOCK", "violations": [], "summary": "b"}
    c = dual_branch_consistency(ocr, vis, True, True, enabled=True)
    assert c["applicable"] is True
    assert c["disagreed"] is True
    assert c["ocr_verdict"] == "PASS"
    assert c["vision_verdict"] == "BLOCK"


def test_consistency_disabled() -> None:
    c = dual_branch_consistency({"verdict": "PASS"}, {"verdict": "BLOCK"}, True, True, enabled=False)
    assert c["applicable"] is False


def test_elevate_warn_pass_to_warn() -> None:
    ocr = {"verdict": "PASS"}
    vis = {"verdict": "WARN"}
    c = dual_branch_consistency(ocr, vis, True, True, enabled=True)
    merged = {"verdict": "PASS", "violations": [], "summary": "x", "confidence": 0.9}
    assert apply_disagreement_to_merged(merged, c, "elevate_warn") is True
    assert merged["verdict"] == "WARN"
    assert merged["summary"] == "x"
    assert merged["violations"] == []


def test_elevate_warn_no_op_when_already_block() -> None:
    ocr = {"verdict": "BLOCK"}
    vis = {"verdict": "PASS"}
    c = dual_branch_consistency(ocr, vis, True, True, enabled=True)
    merged = {"verdict": "BLOCK", "violations": [], "summary": "x"}
    assert apply_disagreement_to_merged(merged, c, "elevate_warn") is False
    assert merged["verdict"] == "BLOCK"
    assert merged["violations"] == []


def test_none_action() -> None:
    c = dual_branch_consistency(
        {"verdict": "PASS"}, {"verdict": "BLOCK"}, True, True, enabled=True
    )
    merged = {"verdict": "PASS", "violations": [], "summary": "x"}
    assert apply_disagreement_to_merged(merged, c, "none") is False
    assert merged["verdict"] == "PASS"
