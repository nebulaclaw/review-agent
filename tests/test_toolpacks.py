"""Tool pack registry and agent config validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from reviewagent.config import AgentConfig
from reviewagent.toolpacks.registry import ToolPackRegistry


def test_registry_has_rules_only_no_contract_pack() -> None:
    reg = ToolPackRegistry()
    ids = {p["id"] for p in reg.list_tool_packs()}
    assert ids == {"review.rules"}
    assert "review.contract" not in ids


def test_agent_config_forbids_legacy_skill_ids_key() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"skill_ids": ["review.rules"]})


def test_resolve_tools_has_four_detectors() -> None:
    reg = ToolPackRegistry()
    names = {t.name for t in reg.resolve_tools({})}
    assert names == {"text_detector", "image_detector", "video_detector", "audio_detector"}
    assert "contract_review" not in names
