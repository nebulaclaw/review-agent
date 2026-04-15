"""Batch review summary and source labels (shared by TUI / Web and review_report)."""

from __future__ import annotations

from reviewagent.review_report import (
    batch_item_source_label,
    batch_item_verdict,
    format_batch_summary_zh,
)


def test_batch_item_source_label_prefers_path() -> None:
    assert batch_item_source_label({"path": "/abs/x.png", "filename": "x.png"}) == "/abs/x.png"


def test_batch_item_source_label_filename_fallback() -> None:
    assert batch_item_source_label({"filename": "doc.md"}) == "doc.md"


def test_batch_item_source_label_index_fallback() -> None:
    assert batch_item_source_label({"index": 2}) == "第 3 项"


def test_batch_item_verdict_error_on_failure() -> None:
    assert batch_item_verdict({"success": False, "error": "x"}) == "ERROR"


def test_format_batch_summary_block_and_pass() -> None:
    results = [
        {"success": True, "response": '{"verdict":"BLOCK"}'},
        {"success": True, "response": '{"verdict":"PASS"}'},
    ]
    s = format_batch_summary_zh(results)
    assert "共 2 项" in s
    assert "BLOCK×1" in s
    assert "PASS×1" in s
    assert "整体：不通过" in s


def test_format_batch_summary_warn_only() -> None:
    results = [{"success": True, "response": '{"verdict":"WARN"}'}]
    assert "整体：待复核" in format_batch_summary_zh(results)


def test_format_batch_summary_all_pass() -> None:
    results = [
        {"success": True, "response": '{"verdict":"PASS"}'},
        {"success": True, "response": '{"verdict":"PASS"}'},
    ]
    assert "整体：通过" in format_batch_summary_zh(results)
