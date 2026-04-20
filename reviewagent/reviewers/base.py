"""Abstract base class for per-content-type review pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator


class ContentReviewer(ABC):
    """Each content type (text / image / video / audio) gets one concrete subclass.

    The subclass owns:
    - The full pipeline logic for that content type
    - The LLM prompt string for the agent loop
    - The prompt mode forwarded to ``ReviewOrchestrator.get_system_prompt``
    """

    #: Must be overridden in every subclass (e.g. "text", "image").
    content_type: ClassVar[str]

    @abstractmethod
    async def review(
        self,
        content: str,
        orchestrator: "ReviewOrchestrator",
    ) -> dict[str, Any]:
        """Run the full review pipeline and return a result dict.

        The result dict must contain at least:
        ``success``, ``response``, ``iterations``, ``duration_ms``.
        """

    def build_user_input(self, content: str, **kwargs: Any) -> str:
        """Return the user-facing prompt string for the LLM agent loop."""
        return f"请审核以下内容（类型 {self.content_type}）：\n\n{content}"

    def prompt_mode(self) -> str:
        """Return the system-prompt mode passed to ``get_system_prompt``.

        Override in ``TextReviewer`` to return ``"text"``; all others return ``"media"``.
        """
        return "media"
