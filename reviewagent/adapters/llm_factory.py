"""
Unified LLM factory: decouple orchestration from vendor SDKs.

Supports:
- ollama: local inference (offline-capable)
- openai / anthropic: general cloud
- glm / zhipuai: Zhipu GLM (langchain_community)
- kimi: Moonshot OpenAI-compatible API
- qwen: Alibaba DashScope compatible-mode (OpenAI-compatible)
- minimax: MiniMax OpenAI-compatible chat completions
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI

from reviewagent.config import get_settings


def _validate_provider_vs_api_base(provider: str, api_base: str) -> None:
    """Reject cloud providers with an Ollama :11434 api_base (causes connection failures)."""
    if not api_base:
        return
    low = api_base.lower()
    if "11434" in low and provider not in ("ollama", "local", "llamacpp"):
        raise ValueError(
            f"llm config mismatch: provider is '{provider}' (expects a cloud API), but api_base points to "
            f"Ollama ({api_base}).\n"
            "Set llm.api_base to empty to use the vendor default (e.g. Zhipu), or set the correct URL "
            "from their docs; for local Ollama only, set llm.provider to ollama."
        )


def _openai_compatible(
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: Optional[int] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if default_headers:
        kwargs["default_headers"] = default_headers
    return ChatOpenAI(**kwargs)


def create_chat_model(**kwargs: Any) -> BaseChatModel:
    settings = get_settings()
    provider = (kwargs.get("provider") or settings.llm.provider).lower().strip()
    model = kwargs.get("model") or settings.llm.model
    api_key = kwargs.get("api_key") or settings.llm.api_key
    api_base = (kwargs.get("api_base") or settings.llm.api_base or "").strip()
    temperature = float(kwargs.get("temperature", settings.llm.temperature))
    max_tokens = int(kwargs.get("max_tokens", settings.llm.max_tokens))
    timeout = int(kwargs.get("timeout", settings.llm.timeout))

    _validate_provider_vs_api_base(provider, api_base)

    if settings.offline_mode and provider not in ("ollama", "local", "llamacpp"):
        raise RuntimeError(
            f"offline_mode is enabled: provider '{provider}' is not allowed. Use ollama (or local) only."
        )

    # --- Local: Ollama ---
    if provider in ("ollama", "local"):
        from langchain_community.chat_models import ChatOllama

        base = api_base or "http://localhost:11434"
        return ChatOllama(
            model=model,
            base_url=base.rstrip("/"),
            temperature=temperature,
            num_predict=max_tokens,
        )

    # --- OpenAI ---
    if provider == "openai":
        kw: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if api_key:
            kw["api_key"] = api_key
        if api_base:
            kw["base_url"] = api_base.rstrip("/")
        return ChatOpenAI(**kw)

    # --- Anthropic ---
    if provider == "anthropic":
        from langchain_community.chat_models import ChatAnthropic

        kw2: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if api_key:
            kw2["anthropic_api_key"] = api_key
        return ChatAnthropic(**kw2)

    # --- Zhipu GLM ---
    if provider in ("glm", "zhipuai", "zhipu"):
        from langchain_community.chat_models import ChatZhipuAI

        kw_z: dict[str, Any] = {
            "model_name": model,
            "zhipuai_api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_base:
            kw_z["zhipuai_api_base"] = api_base.rstrip("/")
        return ChatZhipuAI(**kw_z)

    # --- Kimi (Moonshot) OpenAI-compatible ---
    if provider in ("kimi", "moonshot"):
        base = api_base or "https://api.moonshot.cn/v1"
        return _openai_compatible(
            model=model,
            api_key=api_key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # --- Qwen compatible-mode ---
    if provider in ("qwen", "dashscope", "tongyi"):
        base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return _openai_compatible(
            model=model,
            api_key=api_key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    # --- MiniMax OpenAI-compatible ---
    if provider in ("minimax", "mini_max"):
        base = api_base or "https://api.minimax.chat/v1"
        headers = {}
        group_id = kwargs.get("minimax_group_id") or settings.llm.minimax_group_id
        if group_id:
            headers["Group-Id"] = group_id
        return _openai_compatible(
            model=model,
            api_key=api_key,
            base_url=base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            default_headers=headers or None,
        )

    raise ValueError(
        f"Unknown llm.provider: {provider}. "
        "Supported: ollama, openai, anthropic, glm, zhipuai, kimi, qwen, minimax"
    )


def _default_embedding_model(provider: str) -> str:
    p = provider.lower().strip()
    if p in ("ollama", "local"):
        return "nomic-embed-text"
    if p == "openai":
        return "text-embedding-3-small"
    if p in ("glm", "zhipuai", "zhipu"):
        return "embedding-2"
    if p in ("kimi", "moonshot"):
        return "text-embedding"
    if p in ("qwen", "dashscope", "tongyi"):
        return "text-embedding-v1"
    if p in ("minimax", "mini_max"):
        return "embo-01"
    return "text-embedding-3-small"


def create_embeddings_model(**kwargs: Any):
    """
    Embeddings for RAG; used by the knowledge module when rag.enabled is true.
    Returns None when chat provider is anthropic and rag.embedding_provider is not set explicitly.
    """
    settings = get_settings()
    if not settings.rag.enabled:
        return None

    explicit = (kwargs.get("embedding_provider") or settings.rag.embedding_provider or "").strip()
    provider = (explicit or settings.llm.provider).lower().strip()
    api_key = str(kwargs.get("api_key") or settings.llm.api_key or "")
    api_base = (kwargs.get("api_base") or settings.llm.api_base or "").strip()
    model = (kwargs.get("embedding_model") or settings.rag.embedding_model or "").strip()
    if not model:
        model = _default_embedding_model(provider)

    if provider == "anthropic" and not explicit:
        return None

    if settings.offline_mode and provider not in ("ollama", "local", "llamacpp"):
        return None

    _validate_provider_vs_api_base(provider, api_base)

    try:
        if provider in ("ollama", "local", "llamacpp"):
            from langchain_community.embeddings import OllamaEmbeddings

            base = api_base or "http://localhost:11434"
            return OllamaEmbeddings(model=model, base_url=base.rstrip("/"))

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            kw: dict[str, Any] = {"model": model}
            if api_key:
                kw["api_key"] = api_key
            if api_base:
                kw["base_url"] = api_base.rstrip("/")
            return OpenAIEmbeddings(**kw)

        if provider in ("glm", "zhipuai", "zhipu"):
            from langchain_community.embeddings import ZhipuAIEmbeddings

            return ZhipuAIEmbeddings(model=model, api_key=api_key)

        if provider in ("kimi", "moonshot"):
            from langchain_openai import OpenAIEmbeddings

            base = api_base or "https://api.moonshot.cn/v1"
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base.rstrip("/"),
            )

        if provider in ("qwen", "dashscope", "tongyi"):
            from langchain_openai import OpenAIEmbeddings

            base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base.rstrip("/"),
            )

        if provider in ("minimax", "mini_max"):
            from langchain_openai import OpenAIEmbeddings

            base = api_base or "https://api.minimax.chat/v1"
            headers: dict[str, str] = {}
            group_id = kwargs.get("minimax_group_id") or settings.llm.minimax_group_id
            if group_id:
                headers["Group-Id"] = group_id
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base.rstrip("/"),
                default_headers=headers or None,
            )
    except Exception:
        return None

    return None


__all__ = ["create_chat_model", "create_embeddings_model"]
