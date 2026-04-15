"""Optional Kitty keyboard protocol push so Enter vs Shift+Enter can be distinguished (CSI u).

Textual already pushes CSI > 1 u at startup; that keeps Enter as legacy CR. We push an extra stack
frame while the review input is focused: flags 1|8|16 so Enter is reported as CSI with modifiers,
and associated text is still delivered for printable keys (per kitty spec).

Disable with env ``REVIEW_TUI_DISABLE_KBD_ENHANCE=1``. Skipped for Apple Terminal (limited support).
"""

from __future__ import annotations

import os
from typing import Any

# disambiguate (1) + report_all_keys_as_escape (8) + report_associated_text (16)
_KITTY_FLAGS_SHIFT_ENTER: int = 1 + 8 + 16
KITTY_KBD_PUSH: str = f"\x1b[>{_KITTY_FLAGS_SHIFT_ENTER}u"
KITTY_KBD_POP: str = "\x1b[<1u"


def kitty_kbd_enhance_enabled() -> bool:
    if os.environ.get("REVIEW_TUI_DISABLE_KBD_ENHANCE", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("TERM_PROGRAM", "") == "Apple_Terminal":
        return False
    return True


def driver_write_raw(app: Any, data: str) -> None:
    drv = getattr(app, "_driver", None)
    if drv is None or not hasattr(drv, "write"):
        return
    try:
        drv.write(data)
        drv.flush()
    except Exception:
        pass


__all__ = [
    "KITTY_KBD_POP",
    "KITTY_KBD_PUSH",
    "driver_write_raw",
    "kitty_kbd_enhance_enabled",
]
