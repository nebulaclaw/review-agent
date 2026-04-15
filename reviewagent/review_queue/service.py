"""Bounded-concurrency workers + durable queue."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from reviewagent.config import get_settings

logger = logging.getLogger(__name__)


class ReviewQueueService:
    """
    Enqueue review jobs; workers pull from an asyncio queue.
    Pending rows in SQLite are re-queued on startup for at-least-once processing.
    """

    def __init__(
        self,
        runner: Callable[[str, str], dict[str, Any]],
        *,
        store: Optional[Any] = None,
    ) -> None:
        from reviewagent.review_queue.store import TaskStore

        settings = get_settings()
        self._store = store or TaskStore(settings.queue.persist_path)
        self._runner = runner
        self._max = max(1, settings.queue.max_concurrent)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[Any]] = []
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True
            for tid in self._store.list_pending_ids():
                await self._queue.put(tid)
            for i in range(self._max):
                self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self) -> None:
        async with self._lock:
            for w in self._workers:
                w.cancel()
            self._workers.clear()
            self._started = False

    async def enqueue(self, *, content_type: str, content: str) -> str:
        task_id = self._store.insert_pending(content_type=content_type, content=content)
        await self._queue.put(task_id)
        return task_id

    async def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return self._store.get(task_id)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            task_id = await self._queue.get()
            try:
                self._store.update_running(task_id)
                row = self._store.get(task_id)
                if row is None:
                    continue
                ct = row["content_type"]
                content = row["content"]

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: self._runner(ct, content)
                )
                run_id = result.get("run_id") if isinstance(result, dict) else None
                self._store.update_done(task_id, result=result, run_id=run_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("worker %s task %s failed", worker_id, task_id)
                self._store.update_failed(task_id, str(e))
            finally:
                self._queue.task_done()


__all__ = ["ReviewQueueService"]
