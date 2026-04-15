from __future__ import annotations

import json

from click.testing import CliRunner

from reviewagent import cli as cli_module


def test_check_lets_agent_decide_content_type(monkeypatch) -> None:
    called: dict = {}

    class _FakeOrchestrator:
        def run_sync(self, user_input: str, *, content_type: str = "text") -> dict:
            called["user_input"] = user_input
            called["content_type"] = content_type
            return {"success": True, "response": "ok"}

    monkeypatch.setattr(cli_module, "create_review_orchestrator", lambda: _FakeOrchestrator())
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["check", "demo.mp4"])

    assert result.exit_code == 0
    assert called["content_type"] == "text"
    assert "First infer the content type (text/image/video/audio)" in called["user_input"]
    assert "Input to review: demo.mp4" in called["user_input"]
    out = json.loads(result.output)
    assert out["success"] is True


def test_check_rejects_legacy_type_option() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["check", "--type", "video", "demo.mp4"])
    assert result.exit_code != 0
    assert "No such option: --type" in result.output
