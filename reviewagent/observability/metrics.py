"""Lightweight in-process metrics (no external collector required for dev / air-gapped)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Optional


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self._max_samples = 1000

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            bucket = self.histograms[name]
            bucket.append(value)
            if len(bucket) > self._max_samples:
                del bucket[: len(bucket) - self._max_samples]

    def time_block(self, name: str):
        class _Timer:
            def __init__(self, coll: MetricsCollector, metric: str) -> None:
                self._coll = coll
                self._metric = metric
                self._t0 = 0.0

            def __enter__(self) -> None:
                self._t0 = time.perf_counter()

            def __exit__(self, *args: Any) -> None:
                self._coll.observe(self._metric, (time.perf_counter() - self._t0) * 1000.0)

        return _Timer(self, name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hist_summary: dict[str, dict[str, float]] = {}
            for k, vals in self.histograms.items():
                if not vals:
                    hist_summary[k] = {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
                    continue
                s = sorted(vals)
                n = len(s)
                hist_summary[k] = {
                    "count": n,
                    "p50_ms": s[n // 2],
                    "p95_ms": s[int(n * 0.95)] if n > 1 else s[0],
                }
            return {
                "counters": dict(self.counters),
                "histograms": hist_summary,
            }


_metrics_singleton: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    global _metrics_singleton
    with _metrics_lock:
        if _metrics_singleton is None:
            _metrics_singleton = MetricsCollector()
        return _metrics_singleton


__all__ = ["MetricsCollector", "get_metrics"]
