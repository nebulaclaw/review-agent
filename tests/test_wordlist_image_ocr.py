"""Image + OCR + wordlist (mock OCR, no real image engine)."""

from __future__ import annotations

import json
from unittest.mock import patch

from reviewagent.config import PipelineConfig, PipelineWordlistConfig, Settings
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.pipeline.wordlist_image import run_image_wordlist
from reviewagent.pipeline.wordlist_text import clear_automaton_cache


def _settings(wordlist: str) -> Settings:
    wl = PipelineWordlistConfig(wordlist_paths=[wordlist], scan_image_ocr_for_wordlist=True)
    return Settings(pipeline=PipelineConfig(wordlist=wl))


@patch("reviewagent.toolpacks.image_detector.ImageDetector")
def test_image_ocr_liantong_triggers_early_block(mock_cls, tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("恋童\n", encoding="utf-8")
    img = tmp_path / "fake.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    inst = mock_cls.return_value
    inst.detect_sync.return_value = {
        "success": True,
        "verdict": "PASS",
        "violations": [],
        "details": {"detected_text": "画面里有恋童字样", "has_text": True},
    }

    s = _settings(str(wl))
    out = run_image_wordlist(str(img), BizContext(), s)
    assert out is not None
    assert out.early_result is not None
    body = json.loads(out.early_result["response"])
    assert body["verdict"] == "BLOCK"
    assert out.ocr_text == "画面里有恋童字样"
    assert "wordlist" in body["summary"].lower() or any(
        "恋" in str(v.get("content", "")) for v in body.get("violations", [])
    )


@patch("reviewagent.toolpacks.image_detector.ImageDetector")
def test_image_ocr_empty_skips_wordlist_early(mock_cls, tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("恋童\n", encoding="utf-8")
    img = tmp_path / "fake.png"
    img.write_bytes(b"x")

    mock_cls.return_value.detect_sync.return_value = {
        "success": True,
        "verdict": "PASS",
        "details": {"detected_text": "   ", "has_text": False},
    }

    s = _settings(str(wl))
    out = run_image_wordlist(str(img), BizContext(), s)
    assert out is not None
    assert out.early_result is None
    assert out.llm_prompt_prefix == ""
    assert out.ocr_text == ""
