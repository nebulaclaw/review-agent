"""TextReviewer: wordlist fast-path → LLM agent loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from reviewagent.reviewers.base import ContentReviewer

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator

logger = logging.getLogger(__name__)
_MC = "[review.core]"


class TextReviewer(ContentReviewer):
    """Review plain-text content.

    Pipeline:
    1. Wordlist / AC-automaton fast-path → early block if hit.
    2. Otherwise feed the (possibly enriched) prompt to the LLM agent loop.
    """

    content_type = "text"

    def prompt_mode(self) -> str:
        return "text"

    def build_user_input(self, content: str, **kwargs: Any) -> str:
        return f"请审核以下文本内容：\n\n{content}"

    async def review(self, content: str, orchestrator: "ReviewOrchestrator") -> dict[str, Any]:
        from reviewagent.pipeline.wordlist_text import run_text_wordlist
        from reviewagent.reviewers.utils import response_verdict_hint

        settings = orchestrator._settings
        out = run_text_wordlist(content, orchestrator._biz_context, settings)

        if out.early_result is not None:
            logger.info("%s branch=text_wordlist decision=early_block", _MC)
            return orchestrator._finalize_early_pipeline_result(
                dict(out.early_result),
                content_type=self.content_type,
                input_summary=content,
            )

        logger.info("%s branch=text_wordlist decision=continue_llm", _MC)
        rr = await orchestrator.run(out.user_input_for_llm, content_type=self.content_type)
        merged = dict(rr)
        pt_m: dict[str, Any] = {**out.pipeline_trace, "continued_to_llm": True}
        if merged.get("review_domain"):
            pt_m["review_domain"] = merged["review_domain"]
        merged["pipeline_trace"] = pt_m
        logger.info(
            "%s moderate_end path=text_wordlist+agent verdict=%s iterations=%s duration_ms=%s",
            _MC,
            response_verdict_hint(merged.get("response")),
            merged.get("iterations"),
            merged.get("duration_ms"),
        )
        return merged
