"""LLM api_key: vendor env per provider + LLM_API_KEY fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from reviewagent.config import Settings, reload_settings


@pytest.fixture(autouse=True)
def _reload_config():
    yield
    reload_settings()


def test_llm_api_key_from_llm_api_key_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "one-key")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: glm\n  model: glm-4\n  api_key: ${LLM_API_KEY}\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "one-key"


def test_llm_api_key_empty_yaml_uses_vendor_env_for_glm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPUAI_API_KEY", "legacy-zhipu")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: glm\n  model: glm-4\n  api_key: ${LLM_API_KEY}\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "legacy-zhipu"


def test_llm_api_key_openai_uses_openai_not_zhipu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "zh")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: openai\n  model: m\n  api_key: ''\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "oa"


def test_llm_api_key_glm_prefers_zhipu_over_llm_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "uni")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "zh")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: glm\n  model: m\n  api_key: ''\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "zh"


def test_llm_api_key_glm_does_not_use_openai_only_llm_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    monkeypatch.setenv("LLM_API_KEY", "uni")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: glm\n  model: m\n  api_key: ''\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "uni"


def test_llm_api_key_explicit_yaml_wins_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  provider: openai\n  model: m\n  api_key: plain-in-yaml\n",
        encoding="utf-8",
    )
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "plain-in-yaml"
