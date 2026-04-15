"""config.yaml llm section merge, write, and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from reviewagent.config import apply_llm_patch_to_yaml_file


def test_apply_llm_patch_merges_llm_section() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        p = f.name
        yaml.safe_dump(
            {
                "llm": {"provider": "openai", "model": "old"},
                "other": {"x": 1},
            },
            f,
            allow_unicode=True,
            default_flow_style=False,
        )
    try:
        apply_llm_patch_to_yaml_file(p, {"model": "new-model", "temperature": 0.42})
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        assert data["other"]["x"] == 1
        assert data["llm"]["provider"] == "openai"
        assert data["llm"]["model"] == "new-model"
        assert data["llm"]["temperature"] == 0.42
    finally:
        Path(p).unlink(missing_ok=True)


def test_apply_llm_patch_rejects_invalid_llm() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        p = f.name
        yaml.safe_dump({"llm": {"provider": "openai", "model": "m"}}, f)
    try:
        try:
            apply_llm_patch_to_yaml_file(p, {"max_tokens": "not-int"})
        except Exception:
            return
        raise AssertionError("expected validation error")
    finally:
        Path(p).unlink(missing_ok=True)
