"""TUI copy: follows config pipeline.image_dual_check.report_locale (UI language)."""

from __future__ import annotations

from typing import Literal

from textual.binding import Binding

from reviewagent.config import get_settings

TuiUiLocale = Literal["zh", "en"]


def tui_ui_locale() -> TuiUiLocale:
    loc = get_settings().pipeline.image_dual_check.report_locale
    return loc if loc in ("zh", "en") else "zh"


def tt(key: str) -> str:
    """Short label for current UI locale."""
    loc = tui_ui_locale()
    table = _TUI_TEXT.get(loc) or _TUI_TEXT["zh"]
    return table.get(key, _TUI_TEXT["zh"].get(key, key))


def review_tui_bindings() -> list[Binding]:
    """Main-screen shortcut help (switches with locale)."""
    if tui_ui_locale() == "en":
        return [
            Binding("ctrl+c", "quit", "Quit", show=True),
            Binding("ctrl+l", "clear_screen", "Clear", show=True),
            Binding("f1", "show_help", "Help", show=True),
            Binding("f2", "copy_last_report", "Copy report", show=True),
            Binding("f3", "configure_model", "Model", show=True),
            Binding("f4", "configure_display", "Language", show=True),
        ]
    return [
        Binding("ctrl+c", "quit", "退出", show=True),
        Binding("ctrl+l", "clear_screen", "清屏", show=True),
        Binding("f1", "show_help", "帮助", show=True),
        Binding("f2", "copy_last_report", "复制报告", show=True),
        Binding("f3", "configure_model", "模型", show=True),
        Binding("f4", "configure_display", "界面语言", show=True),
    ]


_TUI_TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "app_title": "内容审核",
        "welcome": (
            "内容审核 TUI（HTTP 客户端）\n\n"
            "需先启动 API: content-review server\n"
            "或: content-review tui --with-server\n\n"
            "文本回车提交；F2 /copy 复制报告；F3 /config 模型；F4 /lang 界面语言。\n"
            "/ 命令可补全（Tab）；/file ./a.png 上传。"
        ),
        "input_placeholder": "审核文本回车 · / 开头命令自动提示 · Tab/→ 接受灰色补全",
        "send": "发送",
        "status_ready": "就绪 · 输入 / 查看命令列表；灰色字为可补全后缀",
        "status_api_down": "API 不可达: {base}",
        "status_idle_cmds": "就绪 · 输入 / 查看命令；灰色为补全后缀，Tab 或 → 接受",
        "status_enter_review": "Enter 发送文本审核 · / 使用命令（见侧栏）",
        "status_slash_list": "/help /? /file /copy /toolpacks /model /config /lang /refresh /new — Tab 或 → 补全",
        "status_no_cmd": "无匹配命令，输入 /help 查看说明",
        "status_complete": "补全 → {m}（Tab / →）",
        "status_candidates": "候选: {heads}",
        "busy_review": "审核中…",
        "busy_upload": "上传审核中…",
        "ready": "就绪",
        "cleared_msgs": "已清空消息区。",
        "cleared_local": "已清空本地消息区（服务端按请求无会话状态）。",
        "copy_none": "尚无审核结果可复制。先完成一条文本审核或 /file 后再试。",
        "copy_done": "已复制审核报告（{n} 字符）。若未进系统剪贴板，请用 iTerm2 / VS Code 终端，或在 macOS 上已同步写入 pbcopy。",
        "model_saved": "模型配置已保存。",
        "display_saved": "界面与展示配置已保存。",
        "api_unavailable": "API 不可用。请先: content-review server 或使用 tui --with-server",
        "unknown_cmd": "未知命令: {cmd}。输入 /help 查看列表。",
        "toolpacks_synced": "工具包列表已同步到侧栏（GET /v1/tool-packs）。说明见 docs/agent-tool-packs.md（与厂商 Skills 不同）。",
        "sidebar_refreshed": "侧栏已刷新。",
        "new_session": "已开始新会话。",
        "file_usage": "用法: /file <路径> [路径2 …]\n例: /file ./a.png ./b.txt  或  /file \"/path/with spaces/x.png\"",
        "file_api_down": "API 不可用，无法上传文件。",
        "not_file": "不是可读文件: {path}",
        "field_pending_text": "待审文本",
        "clipboard_fail_title": "审核请求失败",
        "clipboard_file_fail": "文件审核失败",
        "meta_run": "审计 run_id: {rid}",
        "meta_iters": "工具轮次: {n}",
        "meta_ms": "耗时: {n} ms",
        "label_source": "来源: {s}",
        "label_meta": "元数据: {s}",
        "label_file": "文件: {s}",
        "label_path": "路径: {s}",
        "status_fail": "状态: 失败",
        "err_unknown": "未知错误",
        "clip_structured": "结构化结果（JSON）",
        "clip_raw": "模型原文",
        "batch_detail": "批量审核明细，共 {n} 项",
        "batch_from": "来源: {s}",
        # sidebar
        "sb_header": "[ 审核 TUI ]",
        "sb_api": "API: {url}",
        "sb_session": "会话: {id}…（多轮记忆）",
        "sb_conn_ok": "连接: OK",
        "sb_conn_bad": "连接: × 请先启动服务",
        "sb_local_yaml": "[ 本地 config.yaml ]",
        "sb_model": "模型: {v}",
        "sb_name": "名称: {v}",
        "sb_offline": "离线: {v}",
        "sb_yes": "是",
        "sb_no": "否",
        "sb_server_llm": "[ 服务端当前 LLM ]",
        "sb_key": "密钥: {v}",
        "sb_key_yes": "已配置",
        "sb_key_no": "未配置",
        "sb_temp": "温度: {t} · max_tokens: {m}",
        "sb_toolpacks": "[ 工具包（GET /v1/tool-packs）]",
        "sb_toolpacks_fail": "（未能拉取列表）",
        "sb_toolpacks_retry": "（连接恢复后点 /refresh）",
        "sb_cmds": "[ 命令 ]",
        "sb_line_help": "/help   帮助（/ 后输入字母可补全，Tab）",
        "sb_line_file": "/file <路径…>  多路径空格分隔，可加引号",
        "sb_line_refresh": "/refresh  刷新侧栏",
        "sb_line_new": "/new    新会话（释放服务端记忆）",
        "sb_line_model": "/model   查看模型信息",
        "sb_line_config": "/config  编辑 LLM 模型（F3）",
        "sb_line_lang": "/lang    界面语言 zh/en（F4）",
        "sb_line_copy": "/copy   复制上次审核报告（F2）",
        "sb_line_clear": "clear    清空本地消息区",
        "sb_line_quit": "quit     退出",
        # /model output
        "md_local_yaml": "本地 config.yaml:",
        "md_default": "(默认)",
        "md_server_llm": "服务端当前 LLM（GET /v1/config/llm）:",
        "md_fetch_fail": "（未能拉取服务端配置，可 /refresh 重试）",
        "btn_cancel": "取消",
        "btn_save": "保存",
        "mc_title": "模型配置（llm）",
        "mc_hint": "API 可用时保存会更新服务端 config 并热加载；仅本地时写入本机 YAML。密钥留空表示不修改已有密钥。",
        "mc_api_key_ph_ok": "（已配置，留空不改）",
        "mc_api_key_ph": "可选",
        "mc_api_base_ph": "留空用各通道默认",
        "mc_err_numbers": "temperature / max_tokens / timeout 须为数字",
        "mc_no_change": "没有修改任何字段。",
        "mc_input_err": "输入有误: {e}",
        "mc_save_fail": "保存失败: {e}",
        "dc_title": "界面与展示",
        "dc_locale": "界面语言",
        "dc_opt_zh": "中文界面",
        "dc_opt_en": "English UI",
        "dc_hint": "审核报告用语与本 TUI 界面语言。API 可用时 PATCH /v1/config/display；否则写入本机 config.yaml。",
        "dc_no_change": "没有修改。",
        "dc_save_fail": "保存失败: {e}",
    },
    "en": {
        "app_title": "Content review",
        "welcome": (
            "Content review TUI (HTTP client)\n\n"
            "Start the API: content-review server\n"
            "or: content-review tui --with-server\n\n"
            "Enter to submit text; F2 / /copy for last report; F3 /config model; F4 /lang UI language.\n"
            "/ commands tab-complete; /file ./a.png to upload."
        ),
        "input_placeholder": "Type text, Enter to review · / for commands · Tab/→ accept suggestion",
        "send": "Send",
        "status_ready": "Ready · type / for commands; gray text is completable",
        "status_api_down": "API unreachable: {base}",
        "status_idle_cmds": "Ready · / for commands; Tab or → to accept completion",
        "status_enter_review": "Enter to submit text · / for commands (see sidebar)",
        "status_slash_list": "/help /? /file /copy /toolpacks /model /config /lang /refresh /new — Tab or →",
        "status_no_cmd": "No matching command; type /help",
        "status_complete": "Complete → {m} (Tab / →)",
        "status_candidates": "Choices: {heads}",
        "busy_review": "Reviewing…",
        "busy_upload": "Uploading & reviewing…",
        "ready": "Ready",
        "cleared_msgs": "Message area cleared.",
        "cleared_local": "Local messages cleared (server is stateless per request).",
        "copy_none": "No report to copy yet. Run a text review or /file first.",
        "copy_done": "Copied report ({n} chars). If the system clipboard is empty, use iTerm2/VS Code terminal, or macOS pbcopy sync.",
        "model_saved": "Model settings saved.",
        "display_saved": "UI & display settings saved.",
        "api_unavailable": "API unavailable. Start: content-review server or use tui --with-server",
        "unknown_cmd": "Unknown command: {cmd}. Type /help for a list.",
        "toolpacks_synced": "Tool packs synced to sidebar (GET /v1/tool-packs). See docs/agent-tool-packs.md.",
        "sidebar_refreshed": "Sidebar refreshed.",
        "new_session": "New session started.",
        "file_usage": "Usage: /file <path> [path2 …]\n e.g. /file ./a.png  or  /file \"/path with spaces/x.png\"",
        "file_api_down": "API unavailable; cannot upload.",
        "not_file": "Not a readable file: {path}",
        "field_pending_text": "Content to review",
        "clipboard_fail_title": "Review request failed",
        "clipboard_file_fail": "File review failed",
        "meta_run": "Audit run_id: {rid}",
        "meta_iters": "Tool rounds: {n}",
        "meta_ms": "Duration: {n} ms",
        "label_source": "Source: {s}",
        "label_meta": "Meta: {s}",
        "label_file": "File: {s}",
        "label_path": "Path: {s}",
        "status_fail": "Status: failed",
        "err_unknown": "Unknown error",
        "clip_structured": "Structured result (JSON)",
        "clip_raw": "Raw model output",
        "batch_detail": "Batch detail, {n} item(s)",
        "batch_from": "Source: {s}",
        "sb_header": "[ Review TUI ]",
        "sb_api": "API: {url}",
        "sb_session": "Session: {id}… (multi-turn memory)",
        "sb_conn_ok": "Connection: OK",
        "sb_conn_bad": "Connection: × start the server first",
        "sb_local_yaml": "[ Local config.yaml ]",
        "sb_model": "Provider: {v}",
        "sb_name": "Model: {v}",
        "sb_offline": "Offline: {v}",
        "sb_yes": "yes",
        "sb_no": "no",
        "sb_server_llm": "[ Server LLM ]",
        "sb_key": "API key: {v}",
        "sb_key_yes": "set",
        "sb_key_no": "not set",
        "sb_temp": "temp: {t} · max_tokens: {m}",
        "sb_toolpacks": "[ Tool packs (GET /v1/tool-packs) ]",
        "sb_toolpacks_fail": "(could not load list)",
        "sb_toolpacks_retry": "(use /refresh after connection)",
        "sb_cmds": "[ Commands ]",
        "sb_line_help": "/help   Help (Tab-complete after /)",
        "sb_line_file": "/file <paths…>  multiple paths, quoted OK",
        "sb_line_refresh": "/refresh  Refresh sidebar",
        "sb_line_new": "/new    New session (clear server memory)",
        "sb_line_model": "/model   Show model info",
        "sb_line_config": "/config  Edit LLM (F3)",
        "sb_line_lang": "/lang    UI language zh/en (F4)",
        "sb_line_copy": "/copy   Copy last report (F2)",
        "sb_line_clear": "clear    Clear local messages",
        "sb_line_quit": "quit     Exit",
        "md_local_yaml": "Local config.yaml:",
        "md_default": "(default)",
        "md_server_llm": "Server LLM (GET /v1/config/llm):",
        "md_fetch_fail": "(could not fetch server config; try /refresh)",
        "btn_cancel": "Cancel",
        "btn_save": "Save",
        "mc_title": "Model (LLM) settings",
        "mc_hint": "When API is up, Save updates server config and hot-reloads; otherwise writes local YAML. Leave API key empty to keep the current key.",
        "mc_api_key_ph_ok": "(set; leave empty to keep)",
        "mc_api_key_ph": "optional",
        "mc_api_base_ph": "empty = provider default",
        "mc_err_numbers": "temperature / max_tokens / timeout must be numbers",
        "mc_no_change": "No changes.",
        "mc_input_err": "Invalid input: {e}",
        "mc_save_fail": "Save failed: {e}",
        "dc_title": "Display & language",
        "dc_locale": "UI language",
        "dc_opt_zh": "Chinese UI",
        "dc_opt_en": "English UI",
        "dc_hint": "Report wording and this TUI language. PATCH /v1/config/display when API is up; else local config.yaml.",
        "dc_no_change": "No changes.",
        "dc_save_fail": "Save failed: {e}",
    },
}


def build_help_message() -> str:
    if tui_ui_locale() == "en":
        return (
            "[ TUI · HTTP client ]\n\n"
            "This UI only calls the review API (not embedded Agent).\n\n"
            "1) In another terminal: content-review server\n"
            "   or one command: content-review tui --with-server\n\n"
            "2) Env REVIEW_AGENT_API_BASE_URL can point to a remote API.\n\n"
            "This process keeps one X-Review-Session; follow-ups reuse context.\n"
            "/new starts a new session and clears server-side memory.\n\n"
            "Plain text + Enter → POST /v1/review\n"
            "/file paths… → POST /v1/review/file (multipart field files)\n\n"
            "F2 or /copy: copy last report (plain text + parsed JSON).\n"
            "/toolpacks: refresh tool-pack list in sidebar.\n"
            "F3 or /config: edit LLM (PATCH server or local YAML).\n"
            "F4 or /lang: UI language zh/en (PATCH /v1/config/display or local YAML).\n"
        )
    return (
        "[ TUI · HTTP 客户端模式 ]\n\n"
        "本界面不直接跑 Agent，只调用审核 API（TUI 与审核服务分离）。\n\n"
        "1) 另开终端: content-review server\n"
        "   或一条命令: content-review tui --with-server\n\n"
        "2) 环境变量 REVIEW_AGENT_API_BASE_URL 可指向远程服务。\n\n"
        "同一 TUI 进程固定 X-Review-Session，追问/异议会带上文。\n"
        "/new 可换会话并释放服务端记忆。\n\n"
        "输入文本回车 → POST /v1/review\n"
        "/file 路径… → POST /v1/review/file（多文件时 multipart 字段 files）\n\n"
        "F2 或 /copy：复制最近一次审核报告到剪贴板（纯文本 + 解析出的 JSON）。\n"
        "/toolpacks：刷新侧栏工具包列表（LangChain 工具分组，非 Claude SKILL 类 Skills）。\n"
        "F3 或 /config：编辑 LLM（API 可用时 PATCH 服务端并热加载；否则写本机 YAML）。\n"
        "F4 或 /lang：界面语言 zh/en（PATCH /v1/config/display 或本地 YAML）。\n"
    )


__all__ = [
    "build_help_message",
    "review_tui_bindings",
    "tt",
    "tui_ui_locale",
]
