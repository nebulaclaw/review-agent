"""Pytest plugins and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def reset_settings_singleton():
    """Tests that need config.yaml reloaded may request this fixture."""
    import reviewagent.config as c

    prev = c._settings
    c._settings = None
    yield
    c._settings = prev
