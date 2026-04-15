"""Readable hints when LLM calls fail (network / misconfiguration)."""

from __future__ import annotations

from typing import Any


def is_connection_related_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "all connection attempts failed" in msg:
        return True
    if "connection refused" in msg:
        return True
    if "name or service not known" in msg or "nodename nor servname" in msg:
        return True
    if "network is unreachable" in msg:
        return True
    try:
        import httpx

        return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError))
    except ImportError:
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))


def llm_connection_hint(settings: Any) -> str:
    """Troubleshooting lines to append after the raw error, based on current llm config."""
    llm = settings.llm
    prov = (llm.provider or "").lower().strip()
    base = (llm.api_base or "").strip()

    lines = [
        "",
        "[Connection failed — troubleshooting]",
    ]

    if prov in ("ollama", "local"):
        ob = base or "http://localhost:11434"
        lines.append(f"- Current api_base: {ob}")
        lines.append("- Ensure Ollama is running: `ollama serve` (or the Ollama app)")
        lines.append(f"- Quick check: curl -s \"{ob.rstrip('/')}/api/tags\"")
        lines.append("- In config.yaml, `model` must match a name from `ollama list`")
    elif prov in ("glm", "zhipuai", "zhipu"):
        lines.append(
            "- Zhipu API needs outbound HTTPS to open.bigmodel.cn (or the api_base you set)"
        )
        lines.append(
            "- If provider is glm, do not point api_base at Ollama (e.g. http://127.0.0.1:11434); "
            "leave empty to use the official endpoint"
        )
        if base:
            lines.append(f"- Current api_base: {base}")
        lines.append("- Check proxies / corporate firewall for blocked HTTPS")
    elif prov in ("openai",):
        lines.append(f"- Current api_base: {base or '(default api.openai.com)'}")
        lines.append(
            "- For a custom gateway, ensure the URL is reachable, includes /v1 where needed, "
            "and check proxy/firewall rules"
        )
    elif prov in ("kimi", "moonshot", "qwen", "dashscope", "tongyi", "minimax", "mini_max"):
        lines.append(
            "- Cloud APIs need reachability to the vendor or compatible-mode host; check network and proxy"
        )
        if base:
            lines.append(f"- Current api_base: {base}")
    else:
        lines.append("- Verify api_base is reachable and review firewall/proxy settings")

    return "\n".join(lines)


def enrich_agent_error(exc: BaseException, settings: Any) -> str:
    base = str(exc)
    if is_connection_related_error(exc):
        return base + llm_connection_hint(settings)
    return base


__all__ = [
    "enrich_agent_error",
    "is_connection_related_error",
    "llm_connection_hint",
]
