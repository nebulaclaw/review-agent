"""Aho-Corasick multi-pattern matcher self-tests."""

from __future__ import annotations

from reviewagent.pipeline.ac_matcher import AhoCorasickAutomaton


def test_ac_overlapping_patterns() -> None:
    ac = AhoCorasickAutomaton()
    ac.add("he")
    ac.add("she")
    ac.add("his")
    ac.build()
    m = ac.find_all("ushers")
    patterns = sorted({x.pattern for x in m})
    assert "he" in patterns
    assert "she" in patterns


def test_ac_unicode_chinese() -> None:
    ac = AhoCorasickAutomaton()
    ac.add("敏感")
    ac.add("测试词")
    ac.build()
    m = ac.find_all("这是一条敏感测试词条目")
    ps = {x.pattern for x in m}
    assert "敏感" in ps
    assert "测试词" in ps


def test_ac_duplicate_report_per_pattern_instance() -> None:
    ac = AhoCorasickAutomaton()
    ac.add("aa")
    ac.build()
    m = ac.find_all("aaaa")
    assert len(m) >= 2
