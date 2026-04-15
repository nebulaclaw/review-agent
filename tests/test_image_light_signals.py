"""Image lightweight signals (metadata)."""

from __future__ import annotations

import pytest

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    Image is None,
    reason="Pillow unavailable (arch mismatch or not installed)",
)

from reviewagent.pipeline.image_light_signals import collect_image_signals


def test_collect_signals_ok(tmp_path) -> None:
    assert Image is not None
    p = tmp_path / "s.png"
    Image.new("RGB", (100, 50), color="red").save(p)
    s = collect_image_signals(str(p))
    assert s["ok"] is True
    assert s["width"] == 100
    assert s["height"] == 50
    assert s["aspect_ratio"] == 2.0
    assert s["tiny_side_max_32"] is False


def test_collect_not_file(tmp_path) -> None:
    s = collect_image_signals(str(tmp_path / "nope.png"))
    assert s["ok"] is False
    assert s["reason"] == "not_a_file"
