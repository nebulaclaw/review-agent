"""Rule-based phrasing for staged-media re-check (no LLM)."""

from __future__ import annotations

import json

from reviewagent.agent import create_review_orchestrator
from reviewagent.api.followup_text_heuristic import text_suggests_recheck_same_media


def test_recheck_zh_short_phrases() -> None:
    assert text_suggests_recheck_same_media("确定正常检测吗，再检测一次")
    assert text_suggests_recheck_same_media("重新检测")
    assert text_suggests_recheck_same_media(" 复检  ")
    assert text_suggests_recheck_same_media("我不信，重审一下")


def test_recheck_en() -> None:
    assert text_suggests_recheck_same_media("Please check again")
    assert text_suggests_recheck_same_media("re-analyze this")


def test_not_recheck() -> None:
    assert not text_suggests_recheck_same_media("今天天气不错，随便聊聊")
    assert not text_suggests_recheck_same_media("")
    # Long body must not trigger even if it ends with a re-check phrase (only short follow-ups).
    assert not text_suggests_recheck_same_media("x" * 200 + "再检测")


def test_no_staged_recheck_fast_result() -> None:
    orch = create_review_orchestrator()
    r = orch.no_staged_media_recheck_result(user_text="再检一次")
    assert r.get("success") is True
    body = json.loads(r["response"])
    assert body["verdict"] == "WARN"
    assert any(v.get("type") == "no_recheck_target" for v in body.get("violations", []))


def test_prior_substantive_user_text_skips_recheck_lines() -> None:
    from reviewagent.memory import clear_session_memory, get_memory

    sid = "test-prior-substantive"
    clear_session_memory(sid)
    mem = get_memory(sid)
    mem.short_term.add_user_message("健康健康健康的大健康")
    mem.short_term.add_ai_message('{"verdict":"PASS"}')
    orch = create_review_orchestrator(session_id=sid)
    assert orch.prior_substantive_user_text_for_text_recheck() == "健康健康健康的大健康"
    mem.short_term.add_user_message("再检一次")
    assert orch.prior_substantive_user_text_for_text_recheck() == "健康健康健康的大健康"
    clear_session_memory(sid)
