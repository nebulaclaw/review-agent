"""Load local paths for review: infer text / image / video / audio from extension (shared by CLI, API, TUI)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from reviewagent.config import get_settings
from reviewagent.limits import enforce_file_size, enforce_text_utf8_bytes

# Aligned with image_detector and common formats
IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".ico",
    }
)

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".wmv",
    }
)

# Audio-only: audio pipeline (ASR + text rules), separate from VIDEO to avoid frame-extraction path
AUDIO_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".m4a",
        ".ogg",
        ".oga",
        ".opus",
        ".weba",
        ".wma",
    }
)

# Clearly binary; do not read as plain text
UNSUPPORTED_AS_TEXT = frozenset({".pdf", ".zip", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib"})


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


def read_text_with_fallback(path: Path) -> str:
    """Read a text file trying common encodings."""
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise ValueError(f"Could not decode file as utf-8/gb18030/etc.: {path}") from last_err
    raise ValueError(f"Could not read file: {path}")


def load_local_file_for_review(path: str | Path) -> Tuple[str, str]:
    """
    Return (content_type, content) for a path.
    - text: content is file body
    - image / video / audio: content is absolute path string for the matching detector
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a regular file: {p}")

    lim = get_settings().limits
    enforce_file_size(
        p.stat().st_size,
        lim.max_file_bytes,
        name=f"local file '{p.name}'",
    )

    suf = p.suffix.lower()
    if suf in IMAGE_EXTENSIONS:
        return "image", str(p)
    if suf in VIDEO_EXTENSIONS:
        return "video", str(p)
    if suf in AUDIO_EXTENSIONS:
        return "audio", str(p)
    if suf in UNSUPPORTED_AS_TEXT:
        raise ValueError(
            f"This type is not supported as plain text ({suf}). "
            f"Convert to text or use a dedicated tool; for images use a supported extension such as {IMAGE_EXTENSIONS}."
        )
    if _is_probably_binary(p):
        raise ValueError(
            "File looks binary and the extension is not a known image/video/audio type. "
            "Check the path or use a supported image/video/audio extension."
        )

    text = read_text_with_fallback(p)
    enforce_text_utf8_bytes(
        text,
        lim.max_text_bytes,
        field=f"text file '{p.name}'",
    )
    return "text", text


__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "load_local_file_for_review",
    "read_text_with_fallback",
]
