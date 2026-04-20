"""Shared utility functions used by reviewers and the ReviewOrchestrator.

Keeping these here (rather than in ``agent.py``) prevents reviewers from
importing private ``_``-prefixed names across module boundaries.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, FrozenSet, Optional

if TYPE_CHECKING:
    from reviewagent.config import Settings

from reviewagent.review_report import parse_review_json_from_llm_output

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def response_verdict_hint(response: Any) -> str:
    """Return a short verdict string for logging ('PASS', 'BLOCK', 'non_json', …)."""
    if response is None:
        return "?"
    s = str(response).strip()
    if not s:
        return "?"
    d = parse_review_json_from_llm_output(s)
    if isinstance(d, dict):
        return str(d.get("verdict", "?"))
    return "non_json"


def attach_pipeline_review_domain(result: dict[str, Any]) -> None:
    """Copy top-level ``review_domain`` into ``pipeline_trace`` when absent."""
    rd = result.get("review_domain")
    if not rd:
        return
    pt = dict(result.get("pipeline_trace") or {})
    if "review_domain" not in pt:
        pt["review_domain"] = rd
        result["pipeline_trace"] = pt


# ---------------------------------------------------------------------------
# Vision / inline-image helpers  (image reviewer + orchestrator message builder)
# ---------------------------------------------------------------------------

#: LLM providers that accept ``image_url`` with a ``data:`` base-64 URI.
VISION_INLINE_IMAGE_PROVIDERS: FrozenSet[str] = frozenset(
    {
        "glm",
        "zhipuai",
        "zhipu",
        "openai",
        "kimi",
        "moonshot",
        "qwen",
        "dashscope",
        "tongyi",
        "minimax",
        "mini_max",
    }
)


def provider_supports_inline_vision_image(provider: str) -> bool:
    return (provider or "").lower().strip() in VISION_INLINE_IMAGE_PROVIDERS


def local_image_data_url(path: Path) -> str:
    """Read *path* and return a ``data:<mime>;base64,…`` URL string."""
    raw = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_readable_for_vision(path: str) -> bool:
    """Return ``True`` if *path* can be read and base-64 encoded without error."""
    try:
        local_image_data_url(Path(path))
        return True
    except OSError:
        return False


def vision_attachment_eligible(settings: "Settings", image_path: Optional[str]) -> bool:
    """Return ``True`` when the current provider + config allow inline image attachment."""
    if not image_path:
        return False
    if not getattr(settings.agent, "attach_local_image_to_vision_llm", True):
        return False
    if not provider_supports_inline_vision_image(settings.llm.provider):
        return False
    return Path(image_path).is_file()
