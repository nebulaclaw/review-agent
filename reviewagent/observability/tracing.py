"""Unified LLM tracing facade.

A single, backend-agnostic API for instrumenting LLM calls.  Configure once at
startup; every call site uses the same interface regardless of the underlying
platform.

Supported backends (set via ``observability.tracing.backend`` in config.yaml):

- ``none``       — tracing disabled, all calls are no-ops (default)
- ``langsmith``  — LangSmith cloud / on-prem
- ``langfuse``   — LangFuse cloud or self-hosted

Startup (once)::

    from reviewagent.observability import tracing
    tracing.configure(get_settings())

Instrument an async function::

    @tracing.span("review.video")
    async def _traced() -> dict:
        tracing.update_session(session_id)
        return await do_review()

    result = await _traced()

Inject tracing into a LangChain model::

    llm = llm.with_config({"callbacks": tracing.get_llm_callbacks()})

The ``@span`` decorator is a no-op when tracing is disabled, so call sites
need no conditional logic.  The function's return value is automatically
captured as the span output by both backends.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
_MC = "[tracing]"

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Module state — set once by configure()
# ---------------------------------------------------------------------------

_backend: str = "none"       # active backend: "langsmith" | "langfuse" | "none"
_tracing_active: bool = False
_hide_inputs: bool = False
_hide_outputs: bool = False
_extra_tags: list[str] = []


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def configure(settings: Any) -> str:
    """Configure the selected tracing backend from *settings*.

    Reads ``settings.observability.tracing`` — specifically ``enabled``,
    ``backend``, ``hide_inputs``, ``hide_outputs``, ``tags``, and the nested
    ``langsmith`` / ``langfuse`` sub-configs.

    Returns the name of the active backend (``"langsmith"``, ``"langfuse"``,
    or ``"none"``).  Idempotent; last call wins.
    """
    global _backend, _tracing_active, _hide_inputs, _hide_outputs, _extra_tags

    obs = getattr(settings, "observability", None)
    tracing_cfg = getattr(obs, "tracing", None)
    offline = getattr(settings, "offline_mode", False)

    if not tracing_cfg or not tracing_cfg.enabled:
        _backend = "none"
        _tracing_active = False
        logger.debug("%s tracing disabled (enabled=false)", _MC)
        return "none"

    backend = (tracing_cfg.backend or "none").lower().strip()
    _hide_inputs = bool(tracing_cfg.hide_inputs)
    _hide_outputs = bool(tracing_cfg.hide_outputs)
    _extra_tags = list(tracing_cfg.tags or [])

    ok = False
    if backend == "langsmith":
        from reviewagent.observability.langsmith_tracer import _activate  # noqa: PLC0415
        ok = _activate(tracing_cfg, offline)
    elif backend == "langfuse":
        from reviewagent.observability.langfuse_tracer import _activate  # noqa: PLC0415
        ok = _activate(tracing_cfg, offline)
    elif backend != "none":
        logger.warning("%s unknown backend=%r — tracing disabled", _MC, backend)

    _backend = backend if ok else "none"
    _tracing_active = ok
    logger.info("%s backend=%s active=%s", _MC, _backend, _tracing_active)
    return _backend


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def span(name: str) -> Callable[[F], F]:
    """Return a decorator that wraps an async function with a named trace span.

    Both LangSmith (``@traceable``) and LangFuse (``@observe``) automatically
    capture the function's return value as the span output.

    When tracing is disabled the decorator is an identity function — zero
    overhead, no external connections.

    Usage::

        @tracing.span("review.video")
        async def _traced() -> dict:
            tracing.update_session(session_id)
            return await do_review()

        result = await _traced()
    """
    if not _tracing_active:
        return lambda f: f  # type: ignore[return-value]

    if _backend == "langsmith":
        try:
            from langsmith import traceable  # noqa: PLC0415
            return traceable(name=name, run_type="chain")  # type: ignore[return-value]
        except Exception as exc:
            logger.debug("%s span langsmith import error=%s", _MC, exc)

    elif _backend == "langfuse":
        try:
            from langfuse import observe  # noqa: PLC0415
            return observe(  # type: ignore[return-value]
                name=name,
                capture_input=not _hide_inputs,
                capture_output=not _hide_outputs,
            )
        except Exception as exc:
            logger.debug("%s span langfuse import error=%s", _MC, exc)

    return lambda f: f  # type: ignore[return-value]


def get_llm_callbacks() -> list[Any]:
    """Return LangChain callback handlers for the active backend.

    Inject the result into a LangChain model so every ``ainvoke`` is
    automatically recorded as a child generation span::

        llm = llm.with_config({"callbacks": tracing.get_llm_callbacks()})

    - **LangFuse**: returns ``[CallbackHandler()]`` (reads env vars).
    - **LangSmith**: LangChain auto-traces via ``LANGCHAIN_TRACING_V2=true``;
      no explicit callback required — returns ``[]``.
    - **none / disabled**: returns ``[]``.
    """
    if not _tracing_active:
        return []

    if _backend == "langfuse":
        try:
            from langfuse.langchain import CallbackHandler  # noqa: PLC0415
            return [CallbackHandler()]
        except Exception as exc:
            logger.debug("%s get_llm_callbacks langfuse error=%s", _MC, exc)

    return []


def update_session(session_id: Optional[str]) -> None:
    """Associate the current span with a *session_id*.

    Call inside a ``@tracing.span``-decorated function.  A no-op when
    tracing is disabled or no session ID is provided.

    - **LangFuse**: sets the top-level ``session_id`` on the current trace
      (groups all spans for the same review session in the UI).
    - **LangSmith**: session grouping works via project; this call is a no-op.
    """
    if not _tracing_active or not session_id:
        return

    if _backend == "langfuse":
        try:
            from langfuse.decorators import langfuse_context  # noqa: PLC0415
            langfuse_context.update_current_trace(session_id=session_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def is_active() -> bool:
    """Return ``True`` when a tracing backend is configured and active."""
    return _tracing_active


def active_backend() -> str:
    """Return the active backend name: ``'langsmith'``, ``'langfuse'``, or ``'none'``."""
    return _backend


__all__ = [
    "configure",
    "span",
    "get_llm_callbacks",
    "update_session",
    "is_active",
    "active_backend",
]
