"""VideoReviewer: VideoDetector extraction → parallel sub-agents → judge synthesis.

Detection flow
--------------
1. ``VideoDetector.detect()`` extracts frames (OCR), subtitles, ASR transcript.
2. **Wordlist hit** → early block (fast, deterministic — no LLM needed).
3. **No wordlist hit, text available** → multi-agent review:
   a. Spawn specialist sub-agents *in parallel* for each available text surface
      (ASR / subtitle / visual OCR), each making an independent verdict.
   b. A single Judge agent synthesizes all sub-verdicts into the final decision.
4. **No text extracted** (fully degraded) → return detector result as-is.
5. **Detector failed** → fall back to the LLM agent loop with ``video_detector`` tool.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from reviewagent.reviewers.base import ContentReviewer
from reviewagent.reviewers.multi_agent import (
    SubAgentTask,
    run_judge_agent,
    run_sub_agents_parallel,
)

if TYPE_CHECKING:
    from reviewagent.agent import ReviewOrchestrator

logger = logging.getLogger(__name__)
_MC = "[review.core]"

_DEGRADED_LABELS: dict[str, str] = {
    "probe_failed": "探测失败（缺少 ffprobe 或探测异常）",
    "no_frames_extracted": "关键帧抽取失败（缺少 ffmpeg 或解码异常）",
    "visual_skipped": "画面审核未执行",
    "subtitle_extract_failed": "字幕提取失败",
    "text_skipped": "字幕文本审核未执行",
    "audio_review_unavailable": "音频审核不可用（ASR 依赖缺失或转写失败）",
}


def _modal(details: dict[str, Any], key: str) -> dict[str, Any]:
    """Safely navigate ``details → modality_results → key``."""
    mr = details.get("modality_results") or {}
    v = mr.get(key)
    return v if isinstance(v, dict) else {}


class VideoReviewer(ContentReviewer):
    """Review video content via multi-agent orchestration.

    See module docstring for the full pipeline description.
    """

    content_type = "video"

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def build_user_input(self, content: str, **kwargs: Any) -> str:
        """Fallback prompt when the detector fails entirely (tool-based agent)."""
        return (
            "请审核这段视频。必须先调用工具 video_detector，参数 video_path 必须使用下面这一行"
            "绝对路径（逐字一致）：\n"
            f"{content}\n\n"
            "结合工具结果输出最终 JSON 裁决（含 verdict、violations、summary）。"
        )

    @staticmethod
    def _asr_prompt(asr_text: str) -> str:
        return (
            "你负责审核以下**语音转写（ASR）文本**，判断口语内容是否含违规信息"
            "（煽动性言论、违禁内容、恶意诱导等）：\n\n"
            + asr_text
        )

    @staticmethod
    def _subtitle_prompt(subtitle_text: str) -> str:
        return (
            "你负责审核以下**视频字幕文本**，判断字幕内容是否含违规信息：\n\n"
            + subtitle_text
        )

    @staticmethod
    def _ocr_prompt(ocr_text: str) -> str:
        return (
            "你负责审核以下**视频画面 OCR 提取文字**，判断画面中的文字内容是否含违规信息：\n\n"
            + ocr_text
        )

    @staticmethod
    def _visual_frames_prompt(frame_count: int) -> str:
        return (
            f"以下是视频中均匀抽取的 {frame_count} 张关键帧，请逐一检查每张画面，"
            "判断是否含有违规视觉内容（色情/裸露、暴力/血腥、恐怖图像、仇恨符号、违禁物品等）。"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _degraded_reason_labels(reasons: list[Any]) -> list[str]:
        return [_DEGRADED_LABELS.get(str(r), str(r)) for r in reasons or []]

    @staticmethod
    def _extract_text_surfaces(detector_out: dict[str, Any]) -> dict[str, str]:
        """Pull all human-readable text surfaces from a successful detector result."""
        details = detector_out.get("details") or {}
        audio_mod = _modal(details, "audio")
        text_mod = _modal(details, "text")
        visual_mod = _modal(details, "visual")

        asr = (audio_mod.get("detected_text") or "").strip()
        subtitle = (text_mod.get("detected_text") or "").strip()
        # Use all OCR text from visual frames (not just wordlist-hit snippets),
        # so the LLM sub-agent can do semantic review even with no wordlist match.
        ocr = (visual_mod.get("detected_text") or "").strip()

        return {"asr": asr, "subtitle": subtitle, "ocr": ocr}

    @staticmethod
    def _has_text(surfaces: dict[str, str]) -> bool:
        return any(surfaces.get(k, "").strip() for k in ("asr", "subtitle", "ocr"))

    def _build_sub_agent_tasks(
        self,
        surfaces: dict[str, str],
        frame_samples_b64: list[str] | None = None,
        supports_vision: bool = False,
    ) -> list[SubAgentTask]:
        """Create one SubAgentTask per non-empty text surface, plus a visual
        frames task when the provider supports vision and frames are available."""
        tasks: list[SubAgentTask] = []
        if surfaces.get("asr"):
            tasks.append(SubAgentTask("audio_asr", self._asr_prompt(surfaces["asr"])))
        if surfaces.get("subtitle"):
            tasks.append(SubAgentTask("subtitle", self._subtitle_prompt(surfaces["subtitle"])))
        if surfaces.get("ocr"):
            tasks.append(SubAgentTask("visual_ocr", self._ocr_prompt(surfaces["ocr"])))
        if supports_vision and frame_samples_b64:
            tasks.append(
                SubAgentTask(
                    "visual_frames",
                    self._visual_frames_prompt(len(frame_samples_b64)),
                    images=frame_samples_b64,
                )
            )
        return tasks

    def _build_early_block_result(
        self,
        orchestrator: "ReviewOrchestrator",
        content: str,
        detector_out: dict[str, Any],
        dur_ms: float,
    ) -> dict[str, Any]:
        """Assemble a finalized early-block result (wordlist hit or degraded pipeline)."""
        details = detector_out.get("details") or {}
        trace = details.get("pipeline_trace") or {} if isinstance(details, dict) else {}
        trace = trace if isinstance(trace, dict) else {}

        diagnosis = str(trace.get("diagnosis") or "").strip()
        degraded_reasons: list[Any] = trace.get("degraded_reasons") or []
        if not isinstance(degraded_reasons, list):
            degraded_reasons = []
        degraded_labels = self._degraded_reason_labels(degraded_reasons)
        violations: list[Any] = detector_out.get("violations") or []

        if diagnosis:
            summary = diagnosis
        elif violations:
            summary = f"规则引擎检测到 {len(violations)} 条疑似违规证据。"
        else:
            summary = "未发现明确违规证据（检测能力受限，请人工复核）。"

        payload = {
            "verdict": detector_out.get("verdict", "UNKNOWN"),
            "confidence": detector_out.get("confidence"),
            "violations": violations,
            "summary": summary,
            "degraded_reasons": degraded_reasons,
            "degraded_labels": degraded_labels,
            "capability_status": (
                "完整" if not degraded_labels else "降级：" + "；".join(degraded_labels)
            ),
        }
        rr = {
            "success": True,
            "response": json.dumps(payload, ensure_ascii=False),
            "iterations": 0,
            "duration_ms": dur_ms,
            "pipeline_trace": {
                "mode": "video_pipeline",
                "continued_to_llm": False,
                "video_detector": detector_out.get("details", {}),
            },
        }
        logger.info(
            "%s branch=video_pipeline decision=early_block verdict=%s duration_ms=%s",
            _MC, detector_out.get("verdict"), dur_ms,
        )
        return orchestrator._finalize_early_pipeline_result(
            rr, content_type=self.content_type, input_summary=content,
            early_block_log_path="video_pipeline",
        )

    # ------------------------------------------------------------------
    # Main review
    # ------------------------------------------------------------------

    async def review(self, content: str, orchestrator: "ReviewOrchestrator") -> dict[str, Any]:
        from reviewagent.reviewers.utils import (
            attach_pipeline_review_domain,
            provider_supports_inline_vision_image,
            response_verdict_hint,
        )
        from reviewagent.toolpacks.video_detector import VideoDetector

        t0 = time.perf_counter()
        vd_out = await VideoDetector().detect(content)
        detector_dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Detector failed: tool-based agent fallback ---
        if not vd_out.get("success"):
            logger.info(
                "%s branch=video_pipeline decision=fallback_agent reason=%s",
                _MC, vd_out.get("error"),
            )
            rr = await orchestrator.run(self.build_user_input(content), content_type=self.content_type)
            out = dict(rr)
            attach_pipeline_review_domain(out)
            logger.info(
                "%s moderate_end path=video_agent verdict=%s iterations=%s duration_ms=%s",
                _MC, response_verdict_hint(out.get("response")),
                out.get("iterations"), out.get("duration_ms"),
            )
            return out

        violations = vd_out.get("violations") or []
        surfaces = self._extract_text_surfaces(vd_out)
        frame_samples_b64: list[str] = vd_out.get("frame_samples_b64") or []
        _llm_cfg = getattr(getattr(orchestrator, "_settings", None), "llm", None)
        provider = getattr(_llm_cfg, "provider", "") or ""
        supports_vision = provider_supports_inline_vision_image(provider)

        # --- Wordlist hit: early block, no LLM needed ---
        if violations:
            logger.info(
                "%s branch=video_pipeline decision=early_block violations=%d",
                _MC, len(violations),
            )
            return self._build_early_block_result(orchestrator, content, vd_out, detector_dur_ms)

        # --- Has text or frames: multi-agent review ---
        has_visual = supports_vision and bool(frame_samples_b64)
        if self._has_text(surfaces) or has_visual:
            tasks = self._build_sub_agent_tasks(
                surfaces,
                frame_samples_b64=frame_samples_b64,
                supports_vision=supports_vision,
            )
            logger.info(
                "%s branch=video_pipeline decision=multi_agent "
                "sub_agents=%d asr_chars=%d subtitle_chars=%d ocr_chars=%d visual_frames=%d",
                _MC, len(tasks),
                len(surfaces.get("asr", "")),
                len(surfaces.get("subtitle", "")),
                len(surfaces.get("ocr", "")),
                len(frame_samples_b64) if has_visual else 0,
            )

            # Step 1: run specialist sub-agents in parallel
            sub_results = await run_sub_agents_parallel(tasks, orchestrator)

            # Step 2: judge synthesizes all sub-verdicts into final decision
            judge_rr = await run_judge_agent(sub_results, orchestrator, context=content)

            merged = dict(judge_rr)
            merged.setdefault("review_domain", "content_safety")
            pt: dict[str, Any] = {
                "mode": "video_pipeline+multi_agent",
                "continued_to_llm": True,
                "detector_dur_ms": detector_dur_ms,
                "video_detector": vd_out.get("details", {}),
                "review_domain": "content_safety",
                "sub_agents": [
                    {
                        "name": r.name,
                        "verdict": r.verdict,
                        "confidence": r.confidence,
                        "duration_ms": r.duration_ms,
                        "error": r.error,
                    }
                    for r in sub_results
                ],
            }
            merged["pipeline_trace"] = pt
            logger.info(
                "%s moderate_end path=video+multi_agent verdict=%s "
                "sub_agents=%d total_dur_ms=%s",
                _MC, response_verdict_hint(merged.get("response")),
                len(sub_results), merged.get("duration_ms"),
            )
            return merged

        # --- Fully degraded (no text extracted, no vision): return detector result directly ---
        logger.info(
            "%s branch=video_pipeline decision=degraded_no_text verdict=%s duration_ms=%s",
            _MC, vd_out.get("verdict"), detector_dur_ms,
        )
        return self._build_early_block_result(orchestrator, content, vd_out, detector_dur_ms)
