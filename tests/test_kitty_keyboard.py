"""kitty_keyboard: env and Apple Terminal gating."""

import os

import pytest

from reviewagent.tui import kitty_keyboard as kk


def test_kitty_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_TUI_DISABLE_KBD_ENHANCE", "1")
    assert kk.kitty_kbd_enhance_enabled() is False


def test_kitty_disabled_on_apple_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEW_TUI_DISABLE_KBD_ENHANCE", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert kk.kitty_kbd_enhance_enabled() is False


def test_kitty_push_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEW_TUI_DISABLE_KBD_ENHANCE", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert kk.kitty_kbd_enhance_enabled() is True
    assert kk.KITTY_KBD_PUSH == "\x1b[>25u"
    assert kk.KITTY_KBD_POP == "\x1b[<1u"
