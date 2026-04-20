"""AudioReviewer: AudioDetector (ASR) extraction → sub-agent → judge synthesis.

Detection flow
--------------
1. ``VideoDetector.detect_audio_only()`` runs ffprobe + Whisper ASR.
2. **Wordlist hit** → early block (fast, deterministic — no LLM needed).
3. **No wordlist hit, ASR transcript available** → multi-agent review:
   a. An ASR specialist sub-agent semantically analyzes the transcript.
   b. A Judge agent issues the final binding decision.
4. **No transcript** (ASR unavailable / degraded) → return detector result as-is.
5. **Detector failed** → fall back to the LLM agent loop with ``audio_detector`` tool.
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
    "no_audio_stream": "未检测到音频流",
    "audio_extract_failed": "音频标准化失败（缺少 ffmpeg 或解码异常）",
    "audio_review_unavailable": "语音审核不可用（ASR 依赖缺失或转写失败）",
}


class AudioReviewer(ContentReviewer):
    """Review audio content via ASR sub-agent + judge synthesis.

    See module docstring for the full pipeline description.
    """

    content_type = "audio"

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def build_user_input(self, content: str, **kwargs: Any) -> str:
        """Fallback prompt when the detector fails entirely (tool-based agent)."""
        return (
            "请审核该音频文件。必须先调用工具 audio_detector，参数 audio_path 必须使用下面这一行"
            "绝对路径（逐字一致）：\n"
            f"{content}\n\n"
            "结合工具结果输出最终 JSON 裁决（含 verdict、violations、summary）。"
        )

    @staticmethod
    def _asr_prompt(asr_text: str) -> str:
        return (
            "你负责审核以下**音频语音转写（ASR）文本**，判断口语内容是否含违规信息"
            "（煽动性言论、诈骗话术、违禁内容、恶意诱导等）：\n\n"
            + asr_text
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _degraded_reason_labels(reasons: list[Any]) -> list[str]:
        return [_DEGRADED_LABELS.get(str(r), str(r)) for r in reasons or []]

    @staticmethod
    def _extract_asr_text(detector_out: dict[str, Any]) -> str:
        """Extract the ASR transcript from a successful detector result."""
        details = detector_out.get("details") or {}
        if not isinstance(details, dict):
            return ""
        mr = details.get("modality_results") or {}
        audio_mod = mr.get("audio") if isinstance(mr, dict) else None
        if not isinstance(audio_mod, dict):
            return ""
        return (audio_mod.get("detected_text") or "").strip()

    def _build_early_block_result(
        self,
        orchestrator: "ReviewOrchestrator",
        content: str,
        detector_out: dict[str, Any],
        dur_ms: float,
    ) -> dict[str, Any]:
        """Assemble a finalized early-block result (wordlist hit or degraded pipeline)."""
        details = detector_out.get("details") or {}
        trace = (details.get("pipeline_trace") or {}) if isinstance(details, dict) else {}
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
                "mode": "audio_pipeline",
                "continued_to_llm": False,
                "audio_detector": detector_out.get("details", {}),
            },
        }
        logger.info(
            "%s branch=audio_pipeline decision=early_block verdict=%s duration_ms=%s",
            _MC, detector_out.get("verdict"), dur_ms,
        )
        return orchestrator._finalize_early_pipeline_result(
            rr, content_type=self.content_type, input_summary=content,
            early_block_log_path="audio_pipeline",
        )

    # ------------------------------------------------------------------
    # Main review
    # ------------------------------------------------------------------

    async def review(self, content: str, orchestrator: "ReviewOrchestrator") -> dict[str, Any]:
        from reviewagent.reviewers.utils import attach_pipeline_review_domain, response_verdict_hint
        from reviewagent.toolpacks.video_detector import VideoDetector

        t0 = time.perf_counter()
        ad_out = await VideoDetector().detect_audio_only(content)
        detector_dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # --- Detector failed: tool-based agent fallback ---
        if not ad_out.get("success"):
            logger.info(
                "%s branch=audio_pipeline decision=fallback_agent reason=%s",
                _MC, ad_out.get("error"),
            )
            rr = await orchestrator.run(self.build_user_input(content), content_type=self.content_type)
            out = dict(rr)
            attach_pipeline_review_domain(out)
            logger.info(
                "%s moderate_end path=audio_agent verdict=%s iterations=%s duration_ms=%s",
                _MC, response_verdict_hint(out.get("response")),
                out.get("iterations"), out.get("duration_ms"),
            )
            return out

        violations = ad_out.get("violations") or []
        asr_text = self._extract_asr_text(ad_out)

        # --- Wordlist hit: early block, no LLM needed ---
        if violations:
            logger.info(
                "%s branch=audio_pipeline decision=early_block violations=%d",
                _MC, len(violations),
            )
            return self._build_early_block_result(orchestrator, content, ad_out, detector_dur_ms)

        # --- ASR transcript available: sub-agent + judge ---
        if asr_text:
            logger.info(
                "%s branch=audio_pipeline decision=multi_agent asr_chars=%d",
                _MC, len(asr_text),
            )
            tasks = [SubAgentTask("audio_asr", self._asr_prompt(asr_text))]

            # Step 1: ASR specialist sub-agent
            sub_results = await run_sub_agents_parallel(tasks, orchestrator)

            # Step 2: judge issues final binding decision
            judge_rr = await run_judge_agent(sub_results, orchestrator, context=content)

            merged = dict(judge_rr)
            merged.setdefault("review_domain", "content_safety")
            merged["pipeline_trace"] = {
                "mode": "audio_pipeline+multi_agent",
                "continued_to_llm": True,
                "detector_dur_ms": detector_dur_ms,
                "audio_detector": ad_out.get("details", {}),
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
            logger.info(
                "%s moderate_end path=audio+multi_agent verdict=%s "
                "sub_agents=%d total_dur_ms=%s",
                _MC, response_verdict_hint(merged.get("response")),
                len(sub_results), merged.get("duration_ms"),
            )
            return merged

        # --- Degraded (no ASR text): return detector result directly ---
        logger.info(
            "%s branch=audio_pipeline decision=degraded_no_text verdict=%s duration_ms=%s",
            _MC, ad_out.get("verdict"), detector_dur_ms,
        )
        return self._build_early_block_result(orchestrator, content, ad_out, detector_dur_ms)
