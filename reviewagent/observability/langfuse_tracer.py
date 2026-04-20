"""LangFuse backend activator (internal).

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
    """Set LangFuse env vars and return True if the backend is now active.

    Called once at startup by ``tracing.configure()``.  Reads credentials
    from ``tracing_cfg.langfuse`` with fallback to existing env vars so
    operators can override via container environment without touching
    ``config.yaml``.
    """
    if offline:
        logger.debug("%s langfuse skipped (offline_mode=true)", _MC)
        return False

    lf = tracing_cfg.langfuse
    pk = (lf.public_key or "").strip() or os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = (lf.secret_key or "").strip() or os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

    if not pk or not sk:
        logger.debug("%s langfuse skipped (missing public_key or secret_key)", _MC)
        return False

    host = (
        (lf.host or "").strip()
        or os.environ.get("LANGFUSE_HOST", "").strip()
        or "https://cloud.langfuse.com"
    )

    os.environ["LANGFUSE_PUBLIC_KEY"] = pk
    os.environ["LANGFUSE_SECRET_KEY"] = sk
    os.environ["LANGFUSE_HOST"] = host

    logger.info("%s langfuse active host=%s", _MC, host)
    return True
