"""observability.file_logging path resolution and idempotent install."""

from __future__ import annotations

import logging

import pytest

from reviewagent.config import ObservabilityConfig, Settings
from reviewagent.observability import file_logging as file_logging_mod
from reviewagent.observability.file_logging import (
    configure_reviewagent_logging,
    resolve_log_file_path,
    setup_reviewagent_file_logging,
)


@pytest.fixture(autouse=True)
def _reset_reviewagent_file_handlers() -> None:
    yield
    log = logging.getLogger("reviewagent")
    for h in list(log.handlers):
        if getattr(h, "_reviewagent_file_handler", False):
            log.removeHandler(h)
            h.close()
    file_logging_mod._installed_path = None


def test_resolve_relative_to_repo(tmp_path, monkeypatch) -> None:
    from reviewagent.observability import file_logging as fl

    monkeypatch.setattr(fl, "project_root", lambda: tmp_path)
    p = resolve_log_file_path("logs/x.log")
    assert p == tmp_path / "logs" / "x.log"
    assert p.is_absolute()


def test_setup_idempotent(tmp_path) -> None:
    path = tmp_path / "a.log"
    p1 = setup_reviewagent_file_logging(str(path))
    p2 = setup_reviewagent_file_logging(str(path))
    assert p1 == p2 == path.resolve()
    root_ra = logging.getLogger("reviewagent")
    n = len([h for h in root_ra.handlers if getattr(h, "_reviewagent_file_handler", False)])
    assert n == 1


def test_configure_reviewagent_logging_empty_path() -> None:
    s = Settings(observability=ObservabilityConfig(log_file_path=""))
    assert configure_reviewagent_logging(s) is None
