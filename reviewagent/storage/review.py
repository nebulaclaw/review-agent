"""SQLite persistence for review runs (traceability, compliance)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ReviewRecord:
    run_id: str
    created_at: float
    status: str
    content_type: str
    task_id: Optional[str]
    input_summary: str
    result_json: Optional[str]
    error: Optional[str]
    iterations: Optional[int]
    model_provider: Optional[str]
    model_name: Optional[str]
    duration_ms: Optional[float]


class ReviewStore:
    def __init__(self, db_path: str = "data/review.db") -> None:
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
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('moderation_runs', 'review_runs')"
                )
                tables = {row[0] for row in cur.fetchall()}
                if "moderation_runs" in tables and "review_runs" not in tables:
                    conn.execute("ALTER TABLE moderation_runs RENAME TO review_runs")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS review_runs (
                        run_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        status TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        task_id TEXT,
                        input_summary TEXT NOT NULL,
                        result_json TEXT,
                        error TEXT,
                        iterations INTEGER,
                        model_provider TEXT,
                        model_name TEXT,
                        duration_ms REAL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_created ON review_runs(created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_task ON review_runs(task_id)"
                )
                conn.commit()
            finally:
                conn.close()

    def append_run(
        self,
        *,
        status: str,
        content_type: str,
        input_summary: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        iterations: Optional[int] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        rid = run_id or str(uuid.uuid4())
        now = time.time()
        result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
        summary = input_summary[:4000]

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO review_runs (
                        run_id, created_at, status, content_type, task_id,
                        input_summary, result_json, error, iterations,
                        model_provider, model_name, duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rid,
                        now,
                        status,
                        content_type,
                        task_id,
                        summary,
                        result_json,
                        error,
                        iterations,
                        model_provider,
                        model_name,
                        duration_ms,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return rid

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM review_runs WHERE run_id = ?", (run_id,)
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

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT run_id, created_at, status, content_type, task_id,
                           input_summary, error, iterations, model_provider,
                           model_name, duration_ms
                    FROM review_runs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
        return rows


__all__ = ["ReviewStore", "ReviewRecord"]
