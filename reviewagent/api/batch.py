"""Sequential multi-file review locally: CLI uses ReviewOrchestrator; batch shape matches the HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def moderate_paths_sync(
    paths: List[Path],
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Review multiple local files in path order; the same session_id accumulates short-term memory.

    Returns {"batch": True, "count": n, "results": [...]}.
    For a single file, use results[0] directly; its fields match the flat sync review API response.
    """
    from reviewagent.agent import create_review_orchestrator
    from reviewagent.ingest import load_local_file_for_review

    orchestrator = create_review_orchestrator(session_id=session_id)
    results: List[Dict[str, Any]] = []

    for i, raw in enumerate(paths):
        p = Path(raw).expanduser().resolve()
        item: Dict[str, Any] = {"index": i, "path": str(p), "filename": p.name}
        try:
            ct, payload = load_local_file_for_review(p)
            item["inferred_content_type"] = ct
            r = orchestrator.moderate_payload(ct, payload)
            item.update(r)
        except Exception as e:
            # One failed file does not stop the rest
            item["success"] = False
            item["error"] = str(e)
            item.setdefault("response", "")
        results.append(item)

    return {"batch": True, "count": len(paths), "results": results}


__all__ = ["moderate_paths_sync"]
