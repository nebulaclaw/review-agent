"""
Modal for display / locale settings (separate from LLM model config for easier extension).

Updates the server via PATCH /v1/config/display; falls back to local config.yaml when the API is down.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from reviewagent.clients.review_api import ReviewAPIClient
from reviewagent.config import (
    apply_pipeline_report_locale_to_yaml_file,
    get_config_yaml_path,
    reload_settings,
)
from reviewagent.tui.i18n import tt


class DisplayConfigScreen(ModalScreen[bool]):
    """dismiss(True) if saved; False if cancelled."""

    CSS = """
    DisplayConfigScreen {
        align: center middle;
    }
    #dc-dialog {
        width: 58;
        height: auto;
        max-height: 90%;
        background: #1a1d28;
        border: solid #3d8a7a;
        padding: 1 2;
    }
    #dc-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .dc-row {
        height: auto;
        margin-bottom: 1;
    }
    .dc-label {
        width: 18;
        color: #8b93a8;
    }
    .dc-field {
        width: 1fr;
    }
    #dc-hint {
        color: #8b93a8;
        margin-top: 1;
        height: auto;
    }
    #dc-buttons {
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

    def _initial_locale(self) -> str:
        r = str(self._initial.get("report_locale", "zh")).strip().lower()
        return r if r in ("zh", "en") else "zh"

    def compose(self) -> ComposeResult:
        with Vertical(id="dc-dialog"):
            yield Static(tt("dc_title"), id="dc-title")
            with Horizontal(classes="dc-row"):
                yield Static(tt("dc_locale"), classes="dc-label")
                yield Select(
                    [(tt("dc_opt_zh"), "zh"), (tt("dc_opt_en"), "en")],
                    value=self._initial_locale(),
                    allow_blank=False,
                    classes="dc-field",
                    id="dc-report-locale",
                )
            yield Static(tt("dc_hint"), id="dc-hint")
            with Horizontal(id="dc-buttons"):
                yield Button(tt("btn_cancel"), variant="default", id="dc-btn-cancel")
                yield Button(tt("btn_save"), variant="primary", id="dc-btn-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dc-btn-cancel":
            self.dismiss(False)
            return
        if event.button.id != "dc-btn-save":
            return

        locale_new = self._current_locale()
        if locale_new == self._initial_locale():
            self.query_one("#dc-hint", Static).update(tt("dc_no_change"))
            return

        try:
            if self._api_ok and self._client is not None:
                self._client.patch_display_config_sync({"report_locale": locale_new})
            else:
                apply_pipeline_report_locale_to_yaml_file(
                    get_config_yaml_path(), locale_new
                )
            reload_settings()
        except Exception as e:
            self.app.bell()
            self.query_one("#dc-hint", Static).update(tt("dc_save_fail").format(e=e))
            return

        self.dismiss(True)

    def _current_locale(self) -> str:
        v = self.query_one("#dc-report-locale", Select).value
        s = str(v).strip().lower() if v is not None else "zh"
        return s if s in ("zh", "en") else "zh"
