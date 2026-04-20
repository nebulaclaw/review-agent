"""LangSmith backend activator (internal).

This module is used exclusively by ``reviewagent.observability.tracing``.
Call sites should import from ``reviewagent.observability.tracing`` instead.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reviewagent.config import TracingConfig

logger = logging.getLogger(__name__)
_MC = "[tracing]"


def _activate(tracing_cfg: "TracingConfig", offline: bool) -> bool:
    """Set LangSmith env vars and return True if the backend is now active.

    Called once at startup by ``tracing.configure()``.  Reads credentials
    from ``tracing_cfg.langsmith`` with fallback to existing env vars so
    operators can override via container environment without touching
    ``config.yaml``.
    """
    if offline:
        logger.debug("%s langsmith skipped (offline_mode=true)", _MC)
        return False

    ls = tracing_cfg.langsmith
    api_key = (ls.api_key or "").strip() or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    env_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in ("true", "1")

    if not api_key:
        # Honour LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY set externally
        if not env_tracing:
            logger.debug("%s langsmith skipped (no api_key)", _MC)
            return False
        api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
        if not api_key:
            logger.debug("%s langsmith skipped (LANGCHAIN_TRACING_V2=true but no key)", _MC)
            return False

    project = (ls.project or "reviewagent").strip()
    endpoint = (ls.endpoint or "").strip()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    logger.info("%s langsmith active project=%s", _MC, project)
    return True
