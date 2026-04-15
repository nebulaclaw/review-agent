"""PhashBlocklistStore: SQLite table without ImageHash (counts / disable)."""

from __future__ import annotations

import sqlite3

import pytest

from reviewagent.storage.phash_blocklist import PhashBlocklistStore


def test_active_count_empty(tmp_path) -> None:
    db = tmp_path / "p.db"
    s = PhashBlocklistStore(str(db))
    assert s.active_count() == 0


def test_disable(tmp_path) -> None:
    db = tmp_path / "p.db"
    s = PhashBlocklistStore(str(db))
    hx = "a" + "b" * 15
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            INSERT INTO image_phash_blocklist (phash_hex, note, created_at, disabled)
            VALUES (?, NULL, 0.0, 0)
            """,
            (hx,),
        )
        conn.commit()
    assert s.active_count() == 1
    assert s.disable(hx) is True
    assert s.active_count() == 0
