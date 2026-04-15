"""Session-scoped upload staging paths (multi-turn file review)."""

from __future__ import annotations

from pathlib import Path

from reviewagent.memory import (
    clear_session_memory,
    clear_session_review_staging,
    register_session_review_staging_paths,
)


def test_register_then_clear_session_removes_files(tmp_path: Path) -> None:
    sid = "test-staging-sid-1"
    f1 = tmp_path / "a.bin"
    d = tmp_path / "review_upload_xyz"
    d.mkdir()
    f1 = d / "a.bin"
    f1.write_bytes(b"1")
    paths = [str(f1)]

    register_session_review_staging_paths(sid, paths)
    assert f1.is_file()

    clear_session_review_staging(sid)
    assert not f1.exists()


def test_register_replaces_previous_staging(tmp_path: Path) -> None:
    sid = "test-staging-sid-2"
    d1 = tmp_path / "review_upload_1"
    d1.mkdir()
    p1 = d1 / "x.bin"
    p1.write_bytes(b"a")
    d2 = tmp_path / "review_upload_2"
    d2.mkdir()
    p2 = d2 / "y.bin"
    p2.write_bytes(b"b")

    register_session_review_staging_paths(sid, [str(p1)])
    assert p1.is_file()
    register_session_review_staging_paths(sid, [str(p2)])
    assert not p1.exists()
    assert p2.is_file()
    clear_session_review_staging(sid)
    assert not p2.exists()


def test_clear_session_memory_also_clears_staging(tmp_path: Path) -> None:
    sid = "test-staging-sid-3"
    d = tmp_path / "review_upload_3"
    d.mkdir()
    p = d / "z.bin"
    p.write_bytes(b"z")
    register_session_review_staging_paths(sid, [str(p)])
    assert p.is_file()

    clear_session_memory(sid)
    assert not p.exists()
