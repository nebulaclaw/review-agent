"""Lightweight image signals: no ML model; metadata and pixel size (for pipeline_trace)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def collect_image_signals(image_path: str) -> dict[str, Any]:
    """
    Read width, height, rough size, mode, etc. On failure returns ok=false without raising.
    """
    p = Path(image_path)
    t0 = time.perf_counter()
    if not p.is_file():
        return {
            "ok": False,
            "reason": "not_a_file",
            "ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
    try:
        from PIL import Image
    except ImportError:
        return {
            "ok": False,
            "reason": "pillow_missing",
            "ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    try:
        st = p.stat()
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
            fmt = im.format
        ms = (time.perf_counter() - t0) * 1000.0
        megapixels = (w * h) / 1_000_000.0
        aspect = round(w / h, 4) if h else None
        return {
            "ok": True,
            "ms": round(ms, 3),
            "width": w,
            "height": h,
            "megapixels": round(megapixels, 4),
            "aspect_ratio": aspect,
            "mode": mode,
            "format": fmt,
            "file_bytes": st.st_size,
            "tiny_side_max_32": min(w, h) <= 32,
            "extreme_aspect": aspect is not None and (aspect >= 10.0 or aspect <= 0.1),
        }
    except Exception as e:
        return {
            "ok": False,
            "reason": "read_error",
            "error": str(e)[:200],
            "ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


__all__ = ["collect_image_signals"]
