"""SQLite table image_phash_blocklist: perceptual-hash blocklist (may share DB file with audit DB)."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class PhashBlocklistStore:
    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS image_phash_blocklist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        phash_hex TEXT NOT NULL,
                        note TEXT,
                        created_at REAL NOT NULL,
                        disabled INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(phash_hex)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_phash_blocklist_active "
                    "ON image_phash_blocklist(disabled, phash_hex)"
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _validate_hex(hx: str) -> str:
        s = hx.strip().lower().replace(" ", "")
        if not s:
            raise ValueError("phash 为空")
        try:
            import imagehash

            imagehash.hex_to_hash(s)
        except ImportError:
            if len(s) != 16 or any(c not in "0123456789abcdef" for c in s):
                raise ValueError("phash 须为 16 位十六进制（默认 pHash），或安装 ImageHash 以校验")
        except Exception as e:
            raise ValueError(f"无效的 pHash 十六进制: {e}") from e
        return s

    def add(self, phash_hex: str, *, note: str = "") -> int:
        """Insert a blocked fingerprint; returns row id."""
        hx = self._validate_hex(phash_hex)
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO image_phash_blocklist (phash_hex, note, created_at, disabled)
                    VALUES (?, ?, ?, 0)
                    ON CONFLICT(phash_hex) DO UPDATE SET
                        note = excluded.note,
                        disabled = 0,
                        created_at = excluded.created_at
                    """,
                    (hx, (note or "").strip()[:2000] or None, now),
                )
                conn.commit()
                cur = conn.execute(
                    "SELECT id FROM image_phash_blocklist WHERE phash_hex = ?",
                    (hx,),
                )
                row = cur.fetchone()
                return int(row["id"]) if row else 0
            finally:
                conn.close()

    def disable(self, phash_hex: str) -> bool:
        """Soft-delete: set disabled=1."""
        hx = phash_hex.strip().lower().replace(" ", "")
        if not hx:
            return False
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE image_phash_blocklist SET disabled = 1 WHERE phash_hex = ?",
                    (hx,),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def active_count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS c FROM image_phash_blocklist WHERE disabled = 0"
                )
                row = cur.fetchone()
                return int(row["c"]) if row else 0
            finally:
                conn.close()

    def list_active_parsed_hashes(self, imagehash_mod: Any) -> list[tuple[str, Any]]:
        """Return [(normalized hex, ImageHash), ...] for disabled=0 only; skip unparseable rows."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT phash_hex FROM image_phash_blocklist
                    WHERE disabled = 0
                    ORDER BY id
                    """
                )
                rows = cur.fetchall()
            finally:
                conn.close()

        out: list[tuple[str, Any]] = []
        for row in rows:
            raw = (row["phash_hex"] or "").strip().lower()
            if not raw:
                continue
            try:
                h = imagehash_mod.hex_to_hash(raw)
            except Exception:
                continue
            out.append((raw, h))
        return out


__all__ = ["PhashBlocklistStore"]
