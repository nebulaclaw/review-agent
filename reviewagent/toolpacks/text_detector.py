"""Text detector tool: scan text against the shared pipeline wordlist (AC automaton)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class DetectionResult(BaseModel):
    success: bool = True
    verdict: str = "PASS"
    confidence: float = 1.0
    violations: list[dict] = []
    details: dict = {}


class TextDetector:
    """Scan text for sensitive terms using the pipeline AC automaton.

    Shares the same wordlist files and preprocessing as the pipeline
    (``config.yaml`` → ``pipeline.wordlist.wordlist_paths``).
    """

    def __init__(self, settings: Optional[Any] = None) -> None:
        self._settings = settings

    def _get_settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        from reviewagent.config import get_settings

        return get_settings()

    def detect(self, text: str) -> dict:
        from reviewagent.pipeline.ac_matcher import ACMatch
        from reviewagent.pipeline.preprocess import normalize_text_for_recall
        from reviewagent.pipeline.wordlist_text import get_automaton

        settings = self._get_settings()
        fc = settings.pipeline.wordlist

        pr = normalize_text_for_recall(text, fc)
        ac, _ = get_automaton(settings)
        matches: list[ACMatch] = ac.find_all(pr.text)

        violations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for m in matches:
            if m.pattern in seen:
                continue
            seen.add(m.pattern)
            start = m.start
            context = text[max(0, start - 10):start + len(m.pattern) + 10]
            violations.append({
                "type": m.category or "illegal",
                "detection_method": "text_detector",
                "content": m.pattern,
                "severity": "high",
                "position": start,
                "context": context,
            })

        if any(v["severity"] == "high" for v in violations):
            verdict = "BLOCK"
        elif violations:
            verdict = "WARN"
        else:
            verdict = "PASS"

        return {
            "total_violations": len(violations),
            "violations": violations,
            "verdict": verdict,
            "text_length": len(text),
        }

    def __call__(self, text: str) -> dict:
        return self.detect(text)
