"""review_report: populate violation_type_labels from violations."""

import json

from reviewagent.review_report import (
    compute_violation_type_labels,
    enrich_review_json_in_response,
    enrich_result_response_violation_types,
)


def test_labels_wordlist_violations_carry_content_type():
    obj = {
        "verdict": "BLOCK",
        "violations": [
            {"type": "illegal", "detection_method": "wordlist", "content": "恋童", "severity": "high"},
            {"type": "illegal", "detection_method": "wordlist", "content": "枪支", "severity": "high"},
        ],
    }
    assert compute_violation_type_labels(obj) == ["违法信息"]


def test_labels_mixed_content_types_from_wordlist():
    obj = {
        "verdict": "BLOCK",
        "violations": [
            {"type": "illegal", "detection_method": "wordlist", "content": "恋童", "severity": "high"},
            {"type": "porn", "detection_method": "wordlist", "content": "色情", "severity": "high"},
        ],
    }
    labels = compute_violation_type_labels(obj)
    assert labels == ["违法信息", "色情内容"]


def test_warn_block_fallback_unspecified():
    obj = {"verdict": "BLOCK", "violations": []}
    assert compute_violation_type_labels(obj) == ["未标明"]
    assert compute_violation_type_labels(obj, locale="en") == ["Unspecified"]


def test_pass_empty():
    obj = {"verdict": "PASS", "violations": []}
    assert compute_violation_type_labels(obj) == []


def test_enrich_strips_violation_types_and_sets_labels():
    raw = json.dumps(
        {
            "verdict": "BLOCK",
            "violation_types": ["noise"],
            "violations": [{"type": "image_phash", "content": "ab", "severity": "high"}],
            "summary": "指纹",
        },
        ensure_ascii=False,
    )
    new_s = enrich_review_json_in_response(raw)
    assert new_s is not None
    d = json.loads(new_s)
    assert "violation_types" not in d
    assert d["violation_type_labels"] == ["图像指纹封禁"]


def test_enrich_dual_branch_only_yields_unspecified():
    obj = {
        "verdict": "WARN",
        "violations": [
            {"type": "dual_branch_disagreement", "content": "x", "severity": "low"},
        ],
    }
    raw = json.dumps(obj, ensure_ascii=False)
    d = json.loads(enrich_review_json_in_response(raw) or "{}")
    assert d["violation_type_labels"] == ["未标明"]
    assert d["violations"] == []


def test_enrich_wordlist_violation_carries_content_type():
    raw = json.dumps(
        {
            "verdict": "BLOCK",
            "violations": [{"type": "illegal", "detection_method": "wordlist", "content": "恋童", "severity": "high"}],
            "summary": "命中敏感词表",
        },
        ensure_ascii=False,
    )
    d = json.loads(enrich_review_json_in_response(raw) or "{}")
    assert d["violation_type_labels"] == ["违法信息"]


def test_enrich_content_type_labels_respect_locale_en():
    raw = json.dumps(
        {
            "verdict": "BLOCK",
            "violations": [
                {"type": "illegal", "detection_method": "text_detector", "content": "x", "severity": "high"},
            ],
        },
        ensure_ascii=False,
    )
    d = json.loads(enrich_review_json_in_response(raw, locale="en") or "{}")
    assert d["violation_type_labels"] == ["Illegal content"]


def test_enrich_result_mutates():
    result = {
        "success": True,
        "response": json.dumps(
            {"verdict": "BLOCK", "violations": [{"type": "illegal", "detection_method": "wordlist", "content": "a"}]},
            ensure_ascii=False,
        ),
    }
    enrich_result_response_violation_types(result)
    d = json.loads(result["response"])
    assert d["violation_type_labels"] == ["违法信息"]


def test_enrich_skips_non_json():
    assert enrich_review_json_in_response("not json") is None
    r = {"success": True, "response": "plain text"}
    enrich_result_response_violation_types(r)
    assert r["response"] == "plain text"
