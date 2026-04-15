from __future__ import annotations

from reviewagent.api.server import _infer_content_type_from_input


def test_infer_content_type_by_local_path_suffix() -> None:
    assert _infer_content_type_from_input("/tmp/a.jpg") == "image"
    assert _infer_content_type_from_input("/tmp/b.mp4") == "video"
    assert _infer_content_type_from_input("/tmp/speech.mp3") == "audio"
    assert _infer_content_type_from_input("/tmp/c.txt") == "text"


def test_infer_content_type_by_url_suffix() -> None:
    # Local paths only for media; URLs stay text
    assert _infer_content_type_from_input("https://x/y/z.png?token=1") == "text"
    assert _infer_content_type_from_input("https://x/y/z.webm") == "text"
    assert _infer_content_type_from_input("https://x/y/z") == "text"


def test_infer_content_type_plain_text() -> None:
    assert _infer_content_type_from_input("这是一段普通待审文本") == "text"
