"""Append reviewagent.* logs to a file (path from observability.log_file_path)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from reviewagent.config import Settings

_TAG = "_reviewagent_file_handler"
_STREAM_TAG = "_reviewagent_stderr_handler"

# Installed log file abs path (str) to avoid duplicate addHandler
_installed_path: str | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_log_file_path(raw: str) -> Path:
    """Resolve relative paths against the repo root; normalize absolute paths as-is."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("log_file_path is empty")
    p = Path(s)
    return p.resolve() if p.is_absolute() else (project_root() / p).resolve()


def configure_reviewagent_logging(settings: Settings) -> Path | None:
    """
    Set reviewagent log level and attach handlers.

    - **REVIEWAGENT_LOG_LEVEL**: defaults to INFO (use DEBUG for tool rounds, etc.).
    - **stderr**: by default ``reviewagent.*`` also goes to stderr alongside uvicorn access logs;
      set ``REVIEWAGENT_LOG_CONSOLE=0`` to disable console output and log to file only.
    - **File**: controlled by ``observability.log_file_path`` (non-empty → UTF-8 append).

    If no stderr handler is configured, INFO bubbles to the root logger; Python's default root
    only shows WARNING+, so "file has logs but the terminal shows no INFO" is expected.
    """
    _rl = (os.environ.get("REVIEWAGENT_LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, _rl, logging.INFO)
    if not isinstance(lvl, int):
        lvl = logging.INFO
    log = logging.getLogger("reviewagent")
    log.setLevel(lvl)

    _console = (os.environ.get("REVIEWAGENT_LOG_CONSOLE") or "1").strip().lower()
    if _console not in ("0", "false", "no", "off") and not any(
        getattr(h, _STREAM_TAG, False) for h in log.handlers
    ):
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(lvl)
        setattr(sh, _STREAM_TAG, True)
        sh.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s"),
        )
        log.addHandler(sh)

    try:
        return setup_reviewagent_file_logging(settings.observability.log_file_path)
    except OSError as e:
        logging.getLogger(__name__).warning("reviewagent file logging not enabled: %s", e)
        return None


def setup_reviewagent_file_logging(log_file_path: str) -> Path | None:
    """
    Append a UTF-8 FileHandler to logger ``reviewagent``; events still propagate to parents
    (console unchanged if present). Installs once per process per path; changing path requires
    a process restart.
    """
    global _installed_path
    s = (log_file_path or "").strip()
    if not s:
        return None

    path = resolve_log_file_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    if _installed_path == key:
        return path

    log = logging.getLogger("reviewagent")

    # Replace prior app-level file handler (e.g. hot reload; still one primary path)
    for h in list(log.handlers):
        if getattr(h, _TAG, False):
            log.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    setattr(fh, _TAG, True)
    log.addHandler(fh)
    _installed_path = key
    return path


__all__ = [
    "project_root",
    "resolve_log_file_path",
    "setup_reviewagent_file_logging",
    "configure_reviewagent_logging",
]
