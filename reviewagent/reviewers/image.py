"""ImageReviewer: fingerprint → OCR wordlist → dual-LLM / single-agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from reviewagent.reviewers.base import ContentReviewer

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator

logger = logging.getLogger(__name__)
_MC = "[review.core]"


class ImageReviewer(ContentReviewer):
    """Review image content.

    Pipeline:
    1. Collect lightweight signals (hash, metadata) if enabled.
    2. Perceptual-hash fingerprint early block.
    3. OCR → wordlist fast-path; early block if hit.
    4. Dual-LLM path (OCR branch + vision branch → merge) when enabled.
    5. Fallback: single LLM agent run (with optional inline image attachment).
    """

    content_type = "image"

    def build_user_input(self, content: str, *, vision_direct: bool = False, **kwargs: Any) -> str:
        if vision_direct:
            return (
                "请根据本消息中的 **原始图片**（多模态）完成内容审核，**只输出一段 JSON**："
                "verdict、confidence、violations、summary；不要输出 Markdown 代码围栏。\n"
                "必须逐字辨认图中所有可见文字、水印、标语；不得以「未提供图像」「无法看到图」为由给 PASS。\n"
                f"（文件路径仅作记录：{content}）\n\n"
                "若画面或文字涉及未成年人性剥削、恋童倾向宣传、性化儿童等违法不良内容，verdict 必须为 BLOCK。"
            )
        return (
            "请审核这张图片。同一条用户消息中已附带 **原始图片像素**（多模态），请直接阅读图中文字、标语与画面语义；"
            "OCR 工具若未识别出字，仍以你肉眼从像素中读到的内容为准。\n"
            "请调用工具 image_detector，参数 image_path 必须使用下面这一行绝对路径（逐字一致、不要改写）：\n"
            f"{content}\n\n"
            "结合工具结果、像素判读与平台政策输出最终 JSON（verdict/summary/violations）。"
            "若画面或文字涉及未成年人性剥削、恋童宣传等违法不良内容，必须 verdict=BLOCK。"
        )

    async def review(self, content: str, orchestrator: "ReviewOrchestrator") -> dict[str, Any]:
        from reviewagent.reviewers.utils import (
            attach_pipeline_review_domain,
            image_readable_for_vision,
            response_verdict_hint,
            vision_attachment_eligible,
        )

        settings = orchestrator._settings

        # --- Step 1: collect lightweight image signals (non-blocking) ---
        image_light_signals: Optional[dict[str, Any]] = None
        if settings.pipeline.fingerprint.image_collect_light_signals:
            from reviewagent.pipeline.image_light_signals import collect_image_signals

            image_light_signals = collect_image_signals(content)

        # --- Step 2: perceptual-hash fingerprint early block ---
        from reviewagent.pipeline.image_fingerprint import try_fingerprint_early_block

        fp_early = try_fingerprint_early_block(content, orchestrator._biz_context, settings)
        if fp_early is not None:
            if settings.observability.metrics_enabled:
                from reviewagent.observability.metrics import get_metrics

                get_metrics().inc("pipeline.image_phash_block_total")
            logger.info("%s branch=image_phash decision=early_block", _MC)
            fe = dict(fp_early)
            if image_light_signals is not None:
                pt = dict(fe.get("pipeline_trace") or {})
                pt["image_light_signals"] = image_light_signals
                fe["pipeline_trace"] = pt
            return orchestrator._finalize_early_pipeline_result(
                fe,
                content_type=self.content_type,
                input_summary=content,
                early_block_log_path="early_fingerprint",
            )

        # --- Step 3: OCR → wordlist ---
        from reviewagent.pipeline.wordlist_image import run_image_wordlist

        img_out = run_image_wordlist(content, orchestrator._biz_context, settings)

        if img_out is not None:
            if image_light_signals is not None:
                img_out.pipeline_trace["image_light_signals"] = image_light_signals

            # 3a. Wordlist early block
            if img_out.early_result is not None:
                logger.info(
                    "%s branch=image_wordlist decision=early_block ocr_chars=%s",
                    _MC,
                    len(img_out.ocr_text or ""),
                )
                return orchestrator._finalize_early_pipeline_result(
                    dict(img_out.early_result),
                    content_type=self.content_type,
                    input_summary=content,
                )

            # 3b. Dual-LLM path (OCR branch + vision branch)
            from reviewagent.pipeline.image_dual_llm import run_image_dual_llm

            dual = await run_image_dual_llm(orchestrator, content, img_out)
            if dual is not None:
                merged = dict(dual)
                merged.setdefault("review_domain", "content_safety")
                dual_sub = merged.pop("pipeline_trace", {})
                pt: dict[str, Any] = dict(img_out.pipeline_trace)
                pt["continued_to_llm"] = True
                pt["image_dual_llm"] = True
                merged["pipeline_trace"] = {
                    **pt,
                    "review_domain": merged["review_domain"],
                    "dual_llm": dual_sub,
                }
                logger.info(
                    "%s branch=image_wordlist decision=dual_llm ocr_chars=%s",
                    _MC,
                    len(img_out.ocr_text or ""),
                )
                return orchestrator._finalize_dual_llm_image_result(
                    merged,
                    content_type=self.content_type,
                    input_summary=content,
                )

            # 3c. Single-agent fallback (dual LLM disabled)
            logger.info(
                "%s branch=image_wordlist decision=single_agent_run (dual_llm disabled)", _MC
            )
            vdirect = vision_attachment_eligible(settings, content) and image_readable_for_vision(
                content
            )
            ui = img_out.llm_prompt_prefix + self.build_user_input(content, vision_direct=vdirect)
            rr = await orchestrator.run(
                ui,
                content_type=self.content_type,
                vision_image_path=content,
            )
            merged2 = dict(rr)
            pt2: dict[str, Any] = dict(img_out.pipeline_trace)
            pt2["continued_to_llm"] = True
            pt2["vision_image_attached"] = vdirect
            pt2["vision_toolless_llm"] = vdirect
            if merged2.get("review_domain"):
                pt2["review_domain"] = merged2["review_domain"]
            merged2["pipeline_trace"] = pt2
            logger.info(
                "%s moderate_end path=image_wordlist+agent verdict=%s iterations=%s duration_ms=%s",
                _MC,
                response_verdict_hint(merged2.get("response")),
                merged2.get("iterations"),
                merged2.get("duration_ms"),
            )
            return merged2

        # --- Step 4: agent-only (no wordlist module / img_out is None) ---
        logger.info("%s branch=agent_only reason=direct_review content_type=%s", _MC, self.content_type)
        vdirect2 = vision_attachment_eligible(settings, content) and image_readable_for_vision(
            content
        )
        ui2 = self.build_user_input(content, vision_direct=vdirect2)
        rr2 = await orchestrator.run(
            ui2,
            content_type=self.content_type,
            vision_image_path=content,
        )
        out = dict(rr2)
        attach_pipeline_review_domain(out)
        if image_light_signals is not None:
            pt3 = dict(out.get("pipeline_trace") or {})
            pt3["image_light_signals"] = image_light_signals
            out["pipeline_trace"] = pt3
        logger.info(
            "%s moderate_end path=agent verdict=%s iterations=%s duration_ms=%s",
            _MC,
            response_verdict_hint(out.get("response")),
            out.get("iterations"),
            out.get("duration_ms"),
        )
        return out
