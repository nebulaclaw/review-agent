"""Input size limits (aligned with config.limits; 0 means unlimited)."""

from __future__ import annotations


class LimitsExceededError(ValueError):
    """Raised when text or file size exceeds configured limits."""


def enforce_text_utf8_bytes(content: str, max_bytes: int, *, field: str = "text") -> None:
    if max_bytes <= 0:
        return
    n = len(content.encode("utf-8"))
    if n > max_bytes:
        cap_kb = max_bytes // 1024
        got_kb = max(1, (n + 1023) // 1024)
        raise LimitsExceededError(
            f"{field} exceeds the configured limit: cap {cap_kb} KiB "
            f"(current ~{got_kb} KiB, UTF-8). "
            "Adjust limits.max_text_kb in config.yaml."
        )


def enforce_file_size(size: int, max_bytes: int, *, name: str = "file") -> None:
    if max_bytes <= 0:
        return
    if size > max_bytes:
        cap_kb = max_bytes // 1024
        got_kb = max(1, (size + 1023) // 1024)
        raise LimitsExceededError(
            f"{name} exceeds the configured limit: cap {cap_kb} KiB (current ~{got_kb} KiB). "
            "Adjust limits.max_file_kb in config.yaml."
        )


__all__ = [
    "LimitsExceededError",
    "enforce_text_utf8_bytes",
    "enforce_file_size",
]
