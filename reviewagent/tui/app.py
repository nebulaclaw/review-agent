"""
Terminal TUI: HTTP client to the review API only; does not embed the agent.

Set REVIEW_AGENT_API_BASE_URL (default http://127.0.0.1:18080).
One-shot server + TUI: content-review tui --with-server
"""

from __future__ import annotations

import asyncio
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Footer, Header, Static, TextArea
from rich.text import Text

from reviewagent.clients.review_api import ReviewAPIClient, default_api_base
from reviewagent.config import get_settings
from reviewagent.tui.display_config_screen import DisplayConfigScreen
from reviewagent.tui.model_config_screen import ModelConfigScreen
from reviewagent.content_violation import format_violation_row_for_report
from reviewagent.review_report import (
    batch_item_source_label,
    compute_violation_type_labels,
    format_batch_summary,
    parse_review_json_from_llm_output,
    violations_for_report_display,
)
from reviewagent.tui.i18n import build_help_message, review_tui_bindings, tt, tui_ui_locale
from reviewagent.tui.kitty_keyboard import (
    KITTY_KBD_POP,
    KITTY_KBD_PUSH,
    driver_write_raw,
    kitty_kbd_enhance_enabled,
)
from reviewagent.limits import LimitsExceededError, enforce_file_size, enforce_text_utf8_bytes

# Match _dispatch_slash: keep space after /file for typing paths
SLASH_COMMAND_COMPLETIONS: tuple[str, ...] = (
    "/help",
    "/?",
    "/file ",
    "/again ",
    "/copy",
    "/toolpacks",
    "/model",
    "/config",
    "/lang",
    "/refresh",
    "/new",
)


class ReviewSubmitTextArea(TextArea):
    """Enter submits; Shift+Enter newline when the terminal reports it (Kitty CSI u, see kitty_keyboard)."""

    def on_focus(self) -> None:
        app = self.app
        if isinstance(app, ReviewTUI):
            app._kbd_enhance_on_input_focus()

    def on_blur(self) -> None:
        app = self.app
        if isinstance(app, ReviewTUI):
            app._kbd_enhance_on_input_blur()

    async def _on_key(self, event: events.Key) -> None:
        k = event.key
        if k in ("shift+enter", "shift+return"):
            event.stop()
            event.prevent_default()
            await super()._on_key(events.Key("enter", "\n"))
            return
        if k == "ctrl+j":
            event.stop()
            event.prevent_default()
            await super()._on_key(events.Key("enter", "\n"))
            return
        if k in ("enter", "return"):
            event.stop()
            event.prevent_default()
            self.app.action_submit_review()
            return
        await super()._on_key(event)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Prefer the last review-shaped JSON object (same rules as API enrichment)."""
    return parse_review_json_from_llm_output(text or "")


_VERDICT_LINE_PREFIXES: tuple[str, ...] = ("审核结果:", "Review result:")


def _verdict_token_from_response(response: str) -> str:
    data = _extract_json_object(response)
    if not data:
        return ""
    return str(data.get("verdict", "")).upper()


def verdict_style_from_response(response: str) -> str:
    v = _verdict_token_from_response(response)
    if not v:
        return "ai-plain"
    if v == "PASS":
        return "verdict-pass"
    if v == "WARN":
        return "verdict-warn"
    if v == "BLOCK":
        return "verdict-block"
    if v == "UNKNOWN":
        return "verdict-warn"
    return "ai-plain"


def _verdict_label_style(verdict: str) -> str:
    if verdict == "PASS":
        return "bold #2ecc71"
    if verdict == "WARN":
        return "bold #f1c40f"
    if verdict == "BLOCK":
        return "bold #e74c3c"
    if verdict == "UNKNOWN":
        return "bold #f39c12"
    return "bold #95a5a6"


def rich_review_report(block: str, response_text: str) -> Text:
    """Highlight PASS / WARN / BLOCK on the verdict line (Chinese or English prefix)."""
    verdict = _verdict_token_from_response(response_text)
    if not verdict:
        for line in block.split("\n"):
            for prefix in _VERDICT_LINE_PREFIXES:
                if line.startswith(prefix):
                    verdict = line.split(":", 1)[1].strip().upper().split()[0]
                    break
            if verdict:
                break
    idx = None
    lines = block.split("\n")
    for i, line in enumerate(lines):
        if any(line.startswith(p) for p in _VERDICT_LINE_PREFIXES):
            idx = i
            break
    if idx is None:
        return Text(block)

    before = "\n".join(lines[:idx])
    first = lines[idx]
    rest = "\n".join(lines[idx + 1 :])
    label_style = _verdict_label_style(verdict)
    out = Text()
    if before:
        out.append(before)
        out.append("\n")
    value_part = first.split(":", 1)[1].strip() if ":" in first else ""
    head_label = first.split(":", 1)[0] + ": " if ":" in first else ""
    out.append(head_label, style="bold")
    out.append(value_part, style=label_style)
    if rest:
        out.append("\n")
        out.append(rest)
    return out


def _report_chrome(locale: str) -> dict[str, str]:
    if locale == "en":
        return {
            "verdict": "Review result",
            "violation_types": "Violation types",
            "none": "None",
            "confidence": "Confidence",
            "summary": "Summary",
            "violations": "Violations",
            "more_hidden": "… {n} more not shown",
            "sep_types": ", ",
            "risk_level": "Risk level",
            "risk_none": "None",
            "risk_low": "Low",
            "risk_medium": "Medium",
            "risk_high": "High",
            "reasoning": "Reasoning",
            "modality_analysis": "Modality analysis",
            "recommendations": "Recommendations",
        }
    return {
        "verdict": "审核结果",
        "violation_types": "违规类型",
        "none": "无",
        "confidence": "置信度",
        "summary": "摘要",
        "violations": "违规项",
        "more_hidden": "… 另有 {n} 条未显示",
        "sep_types": "、",
        "risk_level": "风险等级",
        "risk_none": "无",
        "risk_low": "低",
        "risk_medium": "中",
        "risk_high": "高",
        "reasoning": "推理依据",
        "modality_analysis": "模态分析",
        "recommendations": "处置建议",
    }


def format_review_body(response: str) -> str:
    data = _extract_json_object(response)
    if not data:
        banner = tt("response_no_json_banner")
        raw = (response or "").strip()
        if not raw:
            return banner
        clipped = raw[:4000] + ("…" if len(raw) > 4000 else "")
        return f"{banner}\n\n{clipped}"

    rloc = get_settings().pipeline.image_dual_check.report_locale
    loc = rloc if rloc in ("zh", "en") else "zh"
    chrome = _report_chrome(loc)

    verdict = str(data.get("verdict", "UNKNOWN")).upper()
    conf = data.get("confidence")
    summary = (data.get("summary") or "").strip()
    violations = violations_for_report_display(data.get("violations") or [])
    data_labels = {**data, "violations": violations}

    risk_level = (data.get("risk_level") or "").strip().lower()
    reasoning = (data.get("reasoning") or "").strip()
    modality: dict = data.get("modality_analysis") or {}
    recommendations = (data.get("recommendations") or "").strip()

    lines: list[str] = []
    lines.append(f"{chrome['verdict']}: {verdict}")
    cats = compute_violation_type_labels(data_labels, locale=loc)
    if cats:
        lines.append(f"{chrome['violation_types']}: {chrome['sep_types'].join(cats)}")
    elif verdict == "PASS":
        lines.append(f"{chrome['violation_types']}: {chrome['none']}")
    if conf is not None:
        lines.append(f"{chrome['confidence']}: {conf}")
    if risk_level:
        risk_label = chrome.get(f"risk_{risk_level}", risk_level)
        lines.append(f"{chrome['risk_level']}: {risk_label}")
    if summary:
        lines.append("")
        lines.append(f"{chrome['summary']}:")
        lines.append(summary)
    if reasoning:
        lines.append("")
        lines.append(f"{chrome['reasoning']}:")
        lines.append(reasoning)
    if isinstance(modality, dict) and modality:
        lines.append("")
        lines.append(f"{chrome['modality_analysis']}:")
        for k, v in modality.items():
            lines.append(f"  · {k}: {v}")
    if recommendations:
        lines.append("")
        lines.append(f"{chrome['recommendations']}:")
        lines.append(recommendations)
    if violations:
        lines.append("")
        lines.append(f"{chrome['violations']} ({len(violations)}):")
        for item in violations[:12]:
            if isinstance(item, dict):
                lines.append(format_violation_row_for_report(item, locale=loc))
            else:
                lines.append(f"  · {item}")
        if len(violations) > 12:
            lines.append(chrome["more_hidden"].format(n=len(violations) - 12))
    return "\n".join(lines)


def _format_meta_for_copy(result: dict[str, Any]) -> str:
    parts: list[str] = []
    if result.get("run_id"):
        parts.append(f"run_id {result['run_id']}")
    if result.get("iterations") is not None:
        parts.append(tt("meta_iters").format(n=result["iterations"]))
    if result.get("duration_ms") is not None:
        parts.append(tt("meta_ms").format(n=result["duration_ms"]))
    sep = ", " if tui_ui_locale() == "en" else " · "
    return sep.join(parts)


def _format_one_report_for_copy(item: dict[str, Any], *, batch_label: str = "") -> str:
    """Plain text for the clipboard, including JSON when present (e.g. for tickets)."""
    lines: list[str] = []
    if batch_label:
        lines.append(batch_label)
    elif item.get("filename"):
        lines.append(tt("label_file").format(s=item["filename"]))
    elif item.get("path"):
        lines.append(tt("label_path").format(s=item["path"]))
    if not item.get("success", True):
        lines.append(tt("status_fail"))
        lines.append(item.get("error") or tt("err_unknown"))
        return "\n".join(lines)

    meta = _format_meta_for_copy(item)
    if meta:
        lines.append(tt("label_meta").format(s=meta))

    raw = item.get("response") or ""
    parsed = _extract_json_object(raw)
    if parsed:
        lines.append(tt("clip_structured"))
        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        lines.append(tt("clip_raw"))
        lines.append(raw[:12000] + ("…" if len(raw) > 12000 else ""))
    return "\n".join(lines)


def build_report_clipboard_text(api_result: dict[str, Any]) -> str:
    loc = tui_ui_locale()
    if api_result.get("batch"):
        results = api_result.get("results") or []
        chunks: list[str] = [format_batch_summary(results, locale=loc), ""]
        n = int(api_result.get("count") or len(results))
        chunks.append(tt("batch_detail").format(n=n))
        chunks.append("")
        for it in results:
            src = batch_item_source_label(it, locale=loc)
            chunks.append(
                _format_one_report_for_copy(it, batch_label=tt("batch_from").format(s=src))
            )
            chunks.append("")
        return "\n".join(chunks).rstrip()
    return _format_one_report_for_copy(api_result)


def copy_report_to_clipboard(app: App, text: str) -> None:
    """Textual OSC 52; on macOS also writes pbcopy for Terminal.app and similar."""
    app.copy_to_clipboard(text)
    if platform.system() == "Darwin":
        try:
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            pass


def format_agent_result_for_display(
    result: dict[str, Any], *, source_label: Optional[str] = None
) -> Tuple[str, str]:
    meta: list[str] = []
    if result.get("run_id"):
        rid = str(result["run_id"])
        rid_show = rid[:8] + "…" if len(rid) > 8 else rid
        meta.append(tt("meta_run").format(rid=rid_show))
    if result.get("iterations") is not None:
        meta.append(tt("meta_iters").format(n=result["iterations"]))
    if result.get("duration_ms") is not None:
        meta.append(tt("meta_ms").format(n=result["duration_ms"]))
    sep = ", " if tui_ui_locale() == "en" else " · "
    header = sep.join(meta) if meta else ""

    lead: list[str] = []
    if source_label:
        lead.append(tt("label_source").format(s=source_label))
    if header:
        lead.append(header)
    prefix = "\n".join(lead)

    if not result.get("success"):
        body = result.get("error") or tt("err_unknown")
        text = f"{prefix}\n\n{body}" if prefix else body
        return "msg ai-msg error-msg", text

    response = result.get("response") or ""
    style = verdict_style_from_response(response)
    body = format_review_body(response)
    block = f"{prefix}\n\n{body}" if prefix else body
    return f"msg ai-msg {style}", block


def format_any_review_result(data: dict[str, Any]) -> Tuple[str, str]:
    """Format one API result or a batch wrapper as one string (copy paths); use mount_review_api_result for TUI."""
    loc = tui_ui_locale()
    if data.get("batch"):
        results = data.get("results") or []
        chunks: list[str] = [format_batch_summary(results, locale=loc), ""]
        worst = "ai-plain"
        for it in results:
            src = batch_item_source_label(it, locale=loc)
            c, b = format_agent_result_for_display(it, source_label=src)
            if "verdict-block" in c:
                worst = "verdict-block"
            elif "verdict-warn" in c and worst == "ai-plain":
                worst = "verdict-warn"
            chunks.append(b)
        body = "\n\n".join(chunks)
        return f"msg ai-msg {worst}", body
    return format_agent_result_for_display(data)


def mount_review_api_result(msgs: ScrollableContainer, data: dict[str, Any]) -> None:
    """Mount review output: one panel per single result; batch shows count then each item with its own styling."""
    loc = tui_ui_locale()
    for w in msgs.query(".welcome"):
        w.remove()
    if data.get("batch"):
        results = data.get("results") or []
        msgs.mount(
            Static(format_batch_summary(results, locale=loc), classes="msg system-msg", markup=False)
        )
        for it in results:
            src = batch_item_source_label(it, locale=loc)
            classes, body = format_agent_result_for_display(it, source_label=src)
            resp = (it.get("response") or "") if it.get("success") else ""
            renderable = rich_review_report(body, resp)
            msgs.mount(Static(renderable, classes=classes, markup=False))
    else:
        classes, body = format_agent_result_for_display(data)
        resp = (data.get("response") or "") if data.get("success") else ""
        renderable = rich_review_report(body, resp)
        msgs.mount(Static(renderable, classes=classes, markup=False))
    msgs.scroll_end()


def build_sidebar_text(
    client: ReviewAPIClient,
    api_ok: bool,
    server_llm: Optional[dict[str, Any]] = None,
) -> str:
    """Sidebar: local / server LLM summary plus command list."""
    settings = get_settings()
    off = tt("sb_yes") if settings.offline_mode else tt("sb_no")
    lines = [
        tt("sb_header"),
        tt("sb_api").format(url=client.base_url),
        tt("sb_session").format(id=client.session_id[:8]),
        tt("sb_conn_ok") if api_ok else tt("sb_conn_bad"),
        "",
        tt("sb_local_yaml"),
        tt("sb_model").format(v=settings.llm.provider),
        tt("sb_name").format(v=settings.llm.model),
        tt("sb_offline").format(v=off),
    ]
    if api_ok and server_llm:
        key_ok = tt("sb_key_yes") if server_llm.get("api_key_configured") else tt("sb_key_no")
        lines.extend(
            [
                "",
                tt("sb_server_llm"),
                tt("sb_model").format(v=server_llm.get("provider", "")),
                tt("sb_name").format(v=server_llm.get("model", "")),
                tt("sb_key").format(v=key_ok),
                tt("sb_temp").format(
                    t=server_llm.get("temperature", ""),
                    m=server_llm.get("max_tokens", ""),
                ),
            ]
        )
    lines.extend(
        [
            "",
            tt("sb_cmds"),
            tt("sb_line_help"),
            tt("sb_line_submit"),
            tt("sb_line_file"),
            tt("sb_line_again"),
            tt("sb_line_refresh"),
            tt("sb_line_new"),
            tt("sb_line_model"),
            tt("sb_line_config"),
            tt("sb_line_lang"),
            tt("sb_line_copy"),
            tt("sb_line_clear"),
            tt("sb_line_quit"),
        ]
    )
    return "\n".join(lines)


class ReviewTUI(App):
    TITLE = "Content review"
    CSS = """
    Screen {
        background: #12141c;
    }
    #main-row {
        height: 1fr;
    }
    #messages {
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }
    #sidebar {
        width: 32;
        height: 100%;
        padding: 0 1;
        background: #1a1d28;
        border-left: solid #2d3348;
    }
    #sidebar-body {
        color: #a8b0c4;
    }
    .msg {
        margin: 1 0;
        padding: 1 2;
        min-height: 3;
    }
    .user-msg {
        background: #243656;
        border-left: outer heavy #4a7ac7;
    }
    .ai-msg {
        background: #1e2436;
    }
    .ai-plain {
        border-left: outer heavy #5c6370;
    }
    .verdict-pass {
        border-left: outer heavy #27ae60;
        background: #0d1a12;
        color: #c8ebd4;
        text-style: none;
    }
    .verdict-warn {
        border-left: outer heavy #f39c12;
        background: #1c1708;
        color: #f2e4b8;
        text-style: none;
    }
    .verdict-block {
        border-left: outer heavy #c0392b;
        background: #1a0c0c;
        color: #f0c4c0;
        text-style: none;
    }
    .error-msg {
        background: #4a1f28;
        border-left: outer heavy #e94560;
    }
    .system-msg {
        background: #252836;
        border-left: outer heavy #7f8c9a;
        color: #c5cad8;
    }
    .welcome {
        background: #1a1d28;
        color: #8b93a8;
        padding: 1 2;
        margin-bottom: 1;
    }
    #input-bar {
        dock: bottom;
        /* Fixed height: docked + height:auto collapses the bar in Textual (status invisible). */
        height: 7;
        background: #1a1d28;
        border-top: solid #2d3348;
        padding: 0 1;
    }
    #input-box {
        height: 4;
        margin-top: 0;
    }
    #input-field {
        width: 1fr;
        height: 3;
        min-height: 3;
        max-height: 4;
        background: #12141c;
    }
    #send-btn {
        width: 12;
        margin-left: 1;
    }
    #status {
        height: 3;
        margin-top: 0;
        background: #12141c;
        color: #8b93a8;
        padding: 0 1;
    }
    Footer {
        background: #12141c;
    }
    """

    def __init__(self, api_base: Optional[str] = None) -> None:
        super().__init__()
        self._api_base_override = api_base
        self.client: Optional[ReviewAPIClient] = None
        self._api_ok = False
        self.is_busy = False
        self._last_report_copy: str = ""
        self._server_llm: Optional[dict[str, Any]] = None
        self._kbd_enhance_active = False
        self.BINDINGS = review_tui_bindings()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with ScrollableContainer(id="messages"):
                yield Static(
                    tt("welcome"),
                    classes="welcome",
                    markup=False,
                    id="welcome-text",
                )
            with Vertical(id="sidebar"):
                yield Static("", id="sidebar-body", markup=False)
        with Vertical(id="input-bar"):
            with Horizontal(id="input-box"):
                yield ReviewSubmitTextArea(
                    placeholder=tt("input_placeholder"),
                    id="input-field",
                    compact=True,
                    soft_wrap=True,
                    tab_behavior="focus",
                )
                yield Button(tt("send"), id="send-btn", variant="primary")
            yield Static(
                tt("status_ready"),
                id="status",
            )
        yield Footer()

    def on_mount(self) -> None:
        base = self._api_base_override or default_api_base()
        self.client = ReviewAPIClient(base)
        self._api_ok = self.client.health_sync()
        self._refresh_server_llm_snapshot()
        self._update_subtitle()
        self._apply_ui_language()
        self.query_one("#input-field", ReviewSubmitTextArea).focus()

    def on_unmount(self) -> None:
        self._kbd_enhance_pop()

    def _kbd_enhance_pop(self) -> None:
        if not self._kbd_enhance_active:
            return
        driver_write_raw(self, KITTY_KBD_POP)
        self._kbd_enhance_active = False

    def _kbd_enhance_on_input_focus(self) -> None:
        """Called from ReviewSubmitTextArea.on_focus."""
        if self.is_busy or not kitty_kbd_enhance_enabled():
            return
        if self._kbd_enhance_active:
            return
        driver_write_raw(self, KITTY_KBD_PUSH)
        self._kbd_enhance_active = True

    def _kbd_enhance_on_input_blur(self) -> None:
        """Called from ReviewSubmitTextArea.on_blur."""
        self._kbd_enhance_pop()

    def _apply_ui_language(self) -> None:
        """Refresh copy, shortcut hints, and sidebar when display locale changes in config."""
        self.BINDINGS = review_tui_bindings()
        self.refresh_bindings()
        self.title = tt("app_title")
        try:
            self.query_one("#welcome-text", Static).update(tt("welcome"))
        except NoMatches:
            pass
        inp = self.query_one("#input-field", ReviewSubmitTextArea)
        inp.placeholder = tt("input_placeholder")
        self.query_one("#send-btn", Button).label = tt("send")
        if self.client:
            self.query_one("#sidebar-body", Static).update(
                build_sidebar_text(self.client, self._api_ok, self._server_llm)
            )
        if not self.is_busy:
            st = self.query_one("#status", Static)
            if not self._api_ok and self.client:
                st.update(tt("status_api_down").format(base=self.client.base_url))
            elif self._api_ok:
                self._update_input_status_line(inp.text)
            else:
                st.update(tt("status_ready"))

    @on(TextArea.Changed, "#input-field")
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self.is_busy:
            return
        self._update_input_status_line(event.text_area.text)

    def _update_input_status_line(self, v: str) -> None:
        st = self.query_one("#status", Static)
        if not self._api_ok:
            return
        raw = v
        if not raw.strip():
            st.update(tt("status_idle_cmds"))
            return
        if not raw.startswith("/"):
            st.update(tt("status_enter_review"))
            return
        if raw == "/":
            st.update(tt("status_slash_list"))
            return
        folded = raw.casefold()
        matches = [c for c in SLASH_COMMAND_COMPLETIONS if c.casefold().startswith(folded)]
        if not matches:
            st.update(tt("status_no_cmd"))
            return
        if len(matches) == 1:
            m = matches[0].rstrip()
            st.update(tt("status_complete").format(m=m))
            return
        heads = " ".join(m.split()[0] for m in matches[:8])
        st.update(tt("status_candidates").format(heads=heads))

    def action_clear_screen(self) -> None:
        if self.is_busy:
            return
        msgs = self.query_one("#messages", ScrollableContainer)
        msgs.remove_children()
        self._add_system(tt("cleared_msgs"))

    def action_show_help(self) -> None:
        if self.is_busy:
            return
        self._add_system(build_help_message())

    def action_copy_last_report(self) -> None:
        if self.is_busy:
            return
        if not self._last_report_copy.strip():
            self._add_system(tt("copy_none"))
            return
        copy_report_to_clipboard(self, self._last_report_copy)
        n = len(self._last_report_copy)
        self._add_system(tt("copy_done").format(n=n))
        if self._api_ok:
            self._update_input_status_line(self.query_one("#input-field", ReviewSubmitTextArea).text)

    def _refresh_server_llm_snapshot(self) -> None:
        self._server_llm = None
        if self.client and self._api_ok:
            self._server_llm = self.client.get_llm_config_sync()

    def _update_subtitle(self) -> None:
        if self._api_ok and self._server_llm:
            p = str(self._server_llm.get("provider", ""))
            m = str(self._server_llm.get("model", ""))
        else:
            s = get_settings().llm
            p, m = s.provider, s.model
        self.sub_title = f"{p} · {m} · API"

    def _llm_form_initial(self) -> dict[str, Any]:
        if self._api_ok and self.client:
            remote = self.client.get_llm_config_sync()
            if remote:
                return remote
        s = get_settings().llm
        return {
            "provider": s.provider,
            "model": s.model,
            "api_base": s.api_base,
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
            "timeout": s.timeout,
            "minimax_group_id": s.minimax_group_id,
            "api_key_configured": bool((s.api_key or "").strip()),
        }

    def _display_form_initial(self) -> dict[str, Any]:
        loc = get_settings().pipeline.image_dual_check.report_locale
        if loc not in ("zh", "en"):
            loc = "zh"
        if self._api_ok and self.client:
            disp = self.client.get_display_config_sync()
            if disp and str(disp.get("report_locale", "")) in ("zh", "en"):
                return {"report_locale": str(disp["report_locale"])}
        return {"report_locale": loc}

    def _on_model_config_closed(self, saved: Optional[bool]) -> None:
        if not saved:
            return
        self._refresh_server_llm_snapshot()
        self._update_subtitle()
        if self.client:
            self.query_one("#sidebar-body", Static).update(
                build_sidebar_text(self.client, self._api_ok, self._server_llm)
            )
        self._add_system(tt("model_saved"))
        self._apply_ui_language()
        if self._api_ok:
            self._update_input_status_line(self.query_one("#input-field", ReviewSubmitTextArea).text)

    def _on_display_config_closed(self, saved: Optional[bool]) -> None:
        if not saved:
            return
        self._add_system(tt("display_saved"))
        self._apply_ui_language()
        if self._api_ok:
            self._update_input_status_line(self.query_one("#input-field", ReviewSubmitTextArea).text)

    def action_configure_model(self) -> None:
        if self.is_busy:
            return
        assert self.client is not None
        initial = self._llm_form_initial()
        self.push_screen(
            ModelConfigScreen(client=self.client, api_ok=self._api_ok, initial=initial),
            self._on_model_config_closed,
        )

    def action_configure_display(self) -> None:
        if self.is_busy:
            return
        assert self.client is not None
        initial = self._display_form_initial()
        self.push_screen(
            DisplayConfigScreen(client=self.client, api_ok=self._api_ok, initial=initial),
            self._on_display_config_closed,
        )

    @on(Button.Pressed, "#send-btn")
    def on_send(self) -> None:
        self._handle_submit()

    def action_submit_review(self) -> None:
        """Submit current input (Enter in input, or Ctrl+S)."""
        self._handle_submit()

    def _handle_submit(self) -> None:
        if self.is_busy:
            return

        input_field = self.query_one("#input-field", ReviewSubmitTextArea)
        text = input_field.text.strip()
        if not text:
            return

        low = text.strip().lower()
        if low in ("quit", "exit", "q"):
            self.exit()
            return

        if low == "clear":
            self.query_one("#messages", ScrollableContainer).remove_children()
            self._add_system(tt("cleared_local"))
            self.query_one("Static#status", Static).update(tt("ready"))
            input_field.text = ""
            input_field.focus()
            return

        if text.startswith("/"):
            self._dispatch_slash(text)
            input_field.text = ""
            input_field.focus()
            return

        if not self._api_ok:
            self._add_system(tt("api_unavailable"))
            return

        input_field.text = ""
        self._add_message("user", text)
        self._set_busy(True, tt("busy_review"))
        self._run_moderate_text(text)

    def _dispatch_slash(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        if cmd in ("/help", "/?"):
            self._add_system(build_help_message())
        elif cmd == "/toolpacks":
            if self.client:
                self._api_ok = self.client.health_sync()
                self._refresh_server_llm_snapshot()
                self.query_one("#sidebar-body", Static).update(
                    build_sidebar_text(self.client, self._api_ok, self._server_llm)
                )
            self._add_system(tt("toolpacks_synced"))
        elif cmd == "/model":
            s = get_settings()
            lines = [
                tt("md_local_yaml"),
                f"  provider: {s.llm.provider}",
                f"  model: {s.llm.model}",
                f"  api_base: {s.llm.api_base or tt('md_default')}",
                f"  temperature: {s.llm.temperature}  max_tokens: {s.llm.max_tokens}  timeout: {s.llm.timeout}",
                f"  offline_mode: {s.offline_mode}",
            ]
            if self._api_ok and self._server_llm:
                lines.append("")
                lines.append(tt("md_server_llm"))
                for k in (
                    "provider",
                    "model",
                    "api_base",
                    "temperature",
                    "max_tokens",
                    "timeout",
                    "api_key_configured",
                ):
                    lines.append(f"  {k}: {self._server_llm.get(k)}")
            elif self._api_ok:
                lines.append("")
                lines.append(tt("md_fetch_fail"))
            self._add_system("\n".join(lines))
        elif cmd == "/config":
            self.action_configure_model()
        elif cmd == "/lang":
            self.action_configure_display()
        elif cmd == "/refresh":
            if self.client:
                self._api_ok = self.client.health_sync()
                self._refresh_server_llm_snapshot()
                self.query_one("#sidebar-body", Static).update(
                    build_sidebar_text(self.client, self._api_ok, self._server_llm)
                )
            self._add_system(tt("sidebar_refreshed"))
        elif cmd == "/new":
            if self.client and self._api_ok:
                old = self.client.new_session()
                self.client.delete_session_sync(old)
            elif self.client:
                self.client.new_session()
            if self.client:
                self.query_one("#sidebar-body", Static).update(
                    build_sidebar_text(self.client, self._api_ok, self._server_llm)
                )
            self._add_system(
                tt("new_session") + (f" session={self.client.session_id[:8]}…" if self.client else "")
            )
        elif cmd == "/copy":
            self.action_copy_last_report()
        elif cmd == "/again":
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not self._api_ok:
                self._add_system(tt("file_api_down"))
                return
            label = f"/again {rest}".strip() if rest else "/again"
            self._add_message("user", label)
            self._set_busy(True, tt("busy_review"))
            self._run_moderate_again(rest)
        elif cmd == "/file":
            import shlex

            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                self._add_system(tt("file_usage"))
                return
            if not self._api_ok:
                self._add_system(tt("file_api_down"))
                return
            tokens = shlex.split(rest, posix=True)
            paths: list[Path] = []
            for t in tokens:
                p = Path(t).expanduser()
                if not p.is_file():
                    self._add_system(tt("not_file").format(path=p))
                    return
                paths.append(p.resolve())
            self._add_message("user", f"[上传×{len(paths)}] " + " ".join(str(p) for p in paths))
            self._set_busy(True, tt("busy_upload"))
            self._run_moderate_paths(paths)
        else:
            self._add_system(tt("unknown_cmd").format(cmd=cmd))

    @work(exclusive=True)
    async def _run_moderate_again(self, text: str) -> None:
        """POST /v1/review with continue_last_upload (same session staged file)."""
        assert self.client is not None
        try:
            try:
                lim = get_settings().limits
                note = text.strip() or tt("again_default_note")
                enforce_text_utf8_bytes(note, lim.max_text_bytes, field=tt("field_pending_text"))
            except LimitsExceededError as e:
                self._add_message("error", str(e))
                return
            result = await self.client.moderate(
                note, "auto", continue_last_upload=True
            )
            self._last_report_copy = build_report_clipboard_text(result)
            msgs = self.query_one("#messages", ScrollableContainer)
            mount_review_api_result(msgs, result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_report_copy = f"{tt('clipboard_fail_title')}\n{str(e)}"
            self._add_message("error", str(e))
        finally:
            self._set_busy(False, tt("ready"))

    @work(exclusive=True)
    async def _run_moderate_text(self, text: str) -> None:
        assert self.client is not None
        try:
            try:
                lim = get_settings().limits
                enforce_text_utf8_bytes(text, lim.max_text_bytes, field=tt("field_pending_text"))
            except LimitsExceededError as e:
                self._add_message("error", str(e))
                return
            result = await self.client.moderate(text, "auto")
            self._last_report_copy = build_report_clipboard_text(result)
            msgs = self.query_one("#messages", ScrollableContainer)
            mount_review_api_result(msgs, result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_report_copy = f"{tt('clipboard_fail_title')}\n{str(e)}"
            self._add_message("error", str(e))
        finally:
            self._set_busy(False, tt("ready"))

    @work(exclusive=True)
    async def _run_moderate_paths(self, paths: list[Path]) -> None:
        assert self.client is not None
        try:
            lim = get_settings().limits
            for p in paths:
                try:
                    enforce_file_size(
                        p.stat().st_size,
                        lim.max_file_bytes,
                        name=f"「{p.name}」",
                    )
                except LimitsExceededError as e:
                    self._add_message("error", str(e))
                    return
            if len(paths) == 1:
                result = await self.client.moderate_file(paths[0])
            else:
                result = await self.client.moderate_files(paths)
            self._last_report_copy = build_report_clipboard_text(result)
            msgs = self.query_one("#messages", ScrollableContainer)
            mount_review_api_result(msgs, result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._last_report_copy = f"{tt('clipboard_file_fail')}\n{str(e)}"
            self._add_message("error", str(e))
        finally:
            self._set_busy(False, tt("ready"))

    def _set_busy(self, busy: bool, status: str) -> None:
        self.is_busy = busy
        inp = self.query_one("#input-field", ReviewSubmitTextArea)
        inp.disabled = busy
        self.query_one("#send-btn", Button).disabled = busy
        self.query_one("Static#status", Static).update(status)
        if not busy:
            inp.focus()
            if self._api_ok:
                self._update_input_status_line(inp.text)

    def _add_message(self, msg_type: str, content: str) -> None:
        msgs = self.query_one("#messages", ScrollableContainer)
        for w in msgs.query(".welcome"):
            w.remove()
        cls = f"msg {msg_type}-msg"
        msgs.mount(Static(content, classes=cls, markup=False))
        msgs.scroll_end()

    def _add_system(self, content: str) -> None:
        msgs = self.query_one("#messages", ScrollableContainer)
        for w in msgs.query(".welcome"):
            w.remove()
        msgs.mount(Static(content, classes="msg system-msg", markup=False))
        msgs.scroll_end()


def run_tui(api_base: Optional[str] = None) -> None:
    ReviewTUI(api_base=api_base).run()


if __name__ == "__main__":
    run_tui()
