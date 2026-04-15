from __future__ import annotations

from pathlib import Path

from reviewagent.ingest import AUDIO_EXTENSIONS, load_local_file_for_review


def test_audio_extensions_nonempty() -> None:
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS


def test_load_local_file_for_review_audio(tmp_path: Path) -> None:
    f = tmp_path / "clip.mp3"
    f.write_bytes(b"id3" + b"\x00" * 64)
    ct, payload = load_local_file_for_review(f)
    assert ct == "audio"
    assert payload == str(f.resolve())
