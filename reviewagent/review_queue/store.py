"""Durable task rows for async review jobs (survives restarts)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class TaskStore:
    def __init__(self, db_path: str = "data/queue.db") -> None:
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
                    CREATE TABLE IF NOT EXISTS moderation_tasks (
                        task_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        status TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        result_json TEXT,
                        error TEXT,
                        run_id TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mtasks_status ON moderation_tasks(status)"
                )
                conn.commit()
            finally:
                conn.close()

    def insert_pending(
        self, *, content_type: str, content: str, task_id: Optional[str] = None
    ) -> str:
        tid = task_id or str(uuid.uuid4())
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO moderation_tasks (
                        task_id, created_at, updated_at, status,
                        content_type, content, result_json, error, run_id
                    ) VALUES (?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL)
                    """,
                    (tid, now, now, content_type, content),
                )
                conn.commit()
            finally:
                conn.close()
        return tid

    def update_running(self, task_id: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE moderation_tasks SET updated_at = ?, status = 'running'
                    WHERE task_id = ?
                    """,
                    (now, task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def update_done(
        self,
        task_id: str,
        *,
        result: dict[str, Any],
        run_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        payload = json.dumps(result, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE moderation_tasks SET
                        updated_at = ?, status = 'done',
                        result_json = ?, error = NULL, run_id = COALESCE(?, run_id)
                    WHERE task_id = ?
                    """,
                    (now, payload, run_id, task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def update_failed(self, task_id: str, error: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE moderation_tasks SET
                        updated_at = ?, status = 'failed', error = ?
                    WHERE task_id = ?
                    """,
                    (now, error[:8000], task_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM moderation_tasks WHERE task_id = ?", (task_id,)
                )
                row = cur.fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        d = dict(row)
        if d.get("result_json"):
            try:
                d["result"] = json.loads(d["result_json"])
            except json.JSONDecodeError:
                d["result"] = None
        return d

    def list_pending_ids(self) -> list[str]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT task_id FROM moderation_tasks
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    """
                )
                return [r[0] for r in cur.fetchall()]
            finally:
                conn.close()


__all__ = ["TaskStore"]
