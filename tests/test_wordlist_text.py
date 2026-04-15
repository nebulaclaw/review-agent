"""Wordlist-on-text path self-tests (no LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewagent.config import PipelineConfig, PipelineWordlistConfig, Settings
from reviewagent.pipeline.biz_context import BizContext
from reviewagent.pipeline.wordlist_text import (
    clear_automaton_cache,
    load_wordlist_patterns,
    run_text_wordlist,
)


def _settings_with_wordlist(path: str, **wl_kw) -> Settings:
    wl = PipelineWordlistConfig(wordlist_paths=[path], **wl_kw)
    return Settings(pipeline=PipelineConfig(wordlist=wl))


def test_settings_ignores_legacy_pipeline_keys() -> None:
    """Legacy keys like m1 lack model fields and are ignored; valid sub-sections still parse."""
    s = Settings.model_validate(
        {
            "pipeline": {
                "m1": {"wordlist_paths": ["config/wordlists/default.txt"]},
                "wordlist": {},
            }
        }
    )
    assert "config/wordlists/default.txt" in s.pipeline.wordlist.wordlist_paths


def test_settings_ignores_legacy_pipeline_mode() -> None:
    """pipeline.mode is no longer a switch; dropped before load."""
    s = Settings.model_validate({"pipeline": {"mode": "legacy", "wordlist": {}}})
    assert s.pipeline.wordlist is not None


def test_shipped_default_wordlist_has_at_least_one_entry() -> None:
    """If shipped default.txt is comment-only, wordlist is empty and image OCR is not AC-blocked."""
    clear_automaton_cache()
    repo_root = Path(__file__).resolve().parents[1]
    wl = repo_root / "config/wordlists/default.txt"
    assert wl.is_file()
    s = _settings_with_wordlist(str(wl))
    pats = load_wordlist_patterns(s)
    assert len(pats) >= 1, "config/wordlists/default.txt 需至少一条非注释词条"


def test_early_block_on_match(tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("违禁词alpha\n", encoding="utf-8")
    s = _settings_with_wordlist(str(wl))
    out = run_text_wordlist("这句话含违禁词alpha结尾", BizContext(), s)
    assert out.early_result is not None
    assert out.early_result["iterations"] == 0
    body = json.loads(out.early_result["response"])
    assert body["verdict"] == "BLOCK"
    assert any("违禁词alpha" in v.get("content", "") for v in body["violations"])


def test_continue_when_no_match(tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("onlythis\n", encoding="utf-8")
    s = _settings_with_wordlist(str(wl))
    out = run_text_wordlist("完全不相干正文", BizContext(), s)
    assert out.early_result is None
    assert "Please review the following text" in out.user_input_for_llm
    assert "[system-wordlist]" in out.user_input_for_llm
    assert out.pipeline_trace["stages"][0]["name"] == "preprocess"


def test_biz_context_in_trace(tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("# empty\n", encoding="utf-8")
    s = _settings_with_wordlist(str(wl))
    biz = BizContext(biz_line="ugc", tenant_id="t1")
    out = run_text_wordlist("hello", biz, s)
    assert out.pipeline_trace["biz_line"] == "ugc"
    assert out.pipeline_trace["tenant_id"] == "t1"


def test_pinyin_evasion_caught_when_wordlist_is_cjk(tmp_path) -> None:
    """CJK wordlist entries get pinyin variants to catch romanized evasion (e.g. liantong / lian tong)."""
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("恋童\n", encoding="utf-8")
    s = _settings_with_wordlist(str(wl))
    for body in (
        "用户发了liantong",
        "这是 lian tong 内容",
        "lian-tong",
        "lian.tong",
    ):
        out = run_text_wordlist(body, BizContext(), s)
        assert out.early_result is not None, f"应拦截: {body!r}"
        assert out.early_result["iterations"] == 0


def test_pinyin_expand_disabled_no_liantong_match(tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("恋童\n", encoding="utf-8")
    wl_cfg = PipelineWordlistConfig(wordlist_paths=[str(wl)], expand_cjk_pinyin=False)
    s = Settings(pipeline=PipelineConfig(wordlist=wl_cfg))
    out = run_text_wordlist("只有liantong没有汉字", BizContext(), s)
    assert out.early_result is None


def test_no_early_exit_but_hint_when_configured(tmp_path) -> None:
    clear_automaton_cache()
    wl = tmp_path / "wl.txt"
    wl.write_text("badword\n", encoding="utf-8")
    s = _settings_with_wordlist(str(wl), early_exit_on_match=False)
    out = run_text_wordlist("含有badword的内容", BizContext(), s)
    assert out.early_result is None
    assert "badword" in out.user_input_for_llm
    assert "hard block is off" in out.user_input_for_llm
