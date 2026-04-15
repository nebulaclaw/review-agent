"""
LLM model settings modal: PATCH /v1/config/llm when online; otherwise write local config.yaml.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from reviewagent.clients.review_api import ReviewAPIClient
from reviewagent.config import apply_llm_patch_to_yaml_file, get_config_yaml_path, reload_settings
from reviewagent.tui.i18n import tt


def _parse_float(s: str, default: float) -> float:
    t = s.strip()
    if not t:
        return default
    return float(t)


def _parse_int(s: str, default: int) -> int:
    t = s.strip()
    if not t:
        return default
    return int(t)


class ModelConfigScreen(ModalScreen[bool]):
    """dismiss(True) if saved; False if cancelled."""

    CSS = """
    ModelConfigScreen {
        align: center middle;
    }
    #mc-dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        background: #1a1d28;
        border: solid #4a7ac7;
        padding: 1 2;
    }
    #mc-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #mc-scroll {
        height: 22;
        border: solid #2d3348;
        padding: 0 1;
    }
    .mc-row {
        height: auto;
        margin-bottom: 1;
    }
    .mc-label {
        width: 18;
        color: #8b93a8;
    }
    .mc-field {
        width: 1fr;
    }
    #mc-hint {
        color: #8b93a8;
        margin-top: 1;
        height: auto;
    }
    #mc-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(
        self,
        *,
        client: Optional[ReviewAPIClient],
        api_ok: bool,
        initial: dict[str, Any],
    ) -> None:
        super().__init__()
        self._client = client
        self._api_ok = api_ok
        self._initial = initial
        self._api_key_configured = bool(initial.get("api_key_configured"))

    def compose(self) -> ComposeResult:
        with Vertical(id="mc-dialog"):
            yield Static(tt("mc_title"), id="mc-title")
            with ScrollableContainer(id="mc-scroll"):
                with Horizontal(classes="mc-row"):
                    yield Static("provider", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("provider", "")),
                        placeholder="ollama / openai / glm …",
                        classes="mc-field",
                        id="mc-provider",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("model", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("model", "")),
                        classes="mc-field",
                        id="mc-model",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("api_base", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("api_base", "")),
                        placeholder=tt("mc_api_base_ph"),
                        classes="mc-field",
                        id="mc-api-base",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("temperature", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("temperature", "")),
                        classes="mc-field",
                        id="mc-temperature",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("max_tokens", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("max_tokens", "")),
                        classes="mc-field",
                        id="mc-max-tokens",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("timeout", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("timeout", "")),
                        classes="mc-field",
                        id="mc-timeout",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("minimax_group", classes="mc-label")
                    yield Input(
                        value=str(self._initial.get("minimax_group_id", "")),
                        classes="mc-field",
                        id="mc-minimax",
                    )
                with Horizontal(classes="mc-row"):
                    yield Static("api_key", classes="mc-label")
                    yield Input(
                        value="",
                        password=True,
                        placeholder=(
                            tt("mc_api_key_ph_ok")
                            if self._api_key_configured
                            else tt("mc_api_key_ph")
                        ),
                        classes="mc-field",
                        id="mc-api-key",
                    )
            yield Static(tt("mc_hint"), id="mc-hint")
            with Horizontal(id="mc-buttons"):
                yield Button(tt("btn_cancel"), variant="default", id="mc-btn-cancel")
                yield Button(tt("btn_save"), variant="primary", id="mc-btn-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mc-btn-cancel":
            self.dismiss(False)
            return
        if event.button.id != "mc-btn-save":
            return

        try:
            patch = self._build_llm_patch()
        except ValueError as e:
            self.app.bell()
            self.query_one("#mc-hint", Static).update(tt("mc_input_err").format(e=e))
            return

        if not patch:
            self.query_one("#mc-hint", Static).update(tt("mc_no_change"))
            return

        try:
            if self._api_ok and self._client is not None:
                self._client.patch_llm_config_sync(patch)
            else:
                apply_llm_patch_to_yaml_file(get_config_yaml_path(), patch)
            reload_settings()
        except Exception as e:
            self.app.bell()
            self.query_one("#mc-hint", Static).update(tt("mc_save_fail").format(e=e))
            return

        self.dismiss(True)

    def _build_llm_patch(self) -> dict[str, Any]:
        prov = self.query_one("#mc-provider", Input).value.strip()
        model = self.query_one("#mc-model", Input).value.strip()
        api_base = self.query_one("#mc-api-base", Input).value.strip()
        temp_s = self.query_one("#mc-temperature", Input).value
        max_s = self.query_one("#mc-max-tokens", Input).value
        to_s = self.query_one("#mc-timeout", Input).value
        mm = self.query_one("#mc-minimax", Input).value.strip()
        key = self.query_one("#mc-api-key", Input).value.strip()

        patch: dict[str, Any] = {}
        if prov != str(self._initial.get("provider", "")):
            patch["provider"] = prov
        if model != str(self._initial.get("model", "")):
            patch["model"] = model
        if api_base != str(self._initial.get("api_base", "")):
            patch["api_base"] = api_base

        try:
            t0 = float(self._initial.get("temperature", 0.3))
            m0 = int(self._initial.get("max_tokens", 8192))
            s0 = int(self._initial.get("timeout", 60))
            nt = _parse_float(temp_s, t0)
            nm = _parse_int(max_s, m0)
            ns = _parse_int(to_s, s0)
        except ValueError as e:
            raise ValueError(tt("mc_err_numbers")) from e

        if nt != t0:
            patch["temperature"] = nt
        if nm != m0:
            patch["max_tokens"] = nm
        if ns != s0:
            patch["timeout"] = ns

        if mm != str(self._initial.get("minimax_group_id", "")):
            patch["minimax_group_id"] = mm

        if key:
            patch["api_key"] = key

        return patch
