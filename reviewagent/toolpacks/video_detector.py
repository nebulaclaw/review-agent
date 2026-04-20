from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VideoDetector:
    def __init__(
        self,
        *,
        frame_interval: int = 1,
        max_frames: int = 30,
        frame_concurrency: int = 4,
        max_vision_frames: int = 12,
        scene_threshold: float = 0.35,
        max_ocr_frames: int = 10,
    ) -> None:
        self.frame_interval = max(1, int(frame_interval))
        self.max_frames = max(1, int(max_frames))
        self.frame_concurrency = max(1, int(frame_concurrency))
        # Vision LLM frame budget and scene-detection sensitivity.
        # Raise max_vision_frames for longer/riskier content; lower
        # scene_threshold to catch subtler transitions (0.1–0.5 range).
        self.max_vision_frames = max(1, int(max_vision_frames))
        self.scene_threshold = max(0.05, min(1.0, float(scene_threshold)))
        # OCR is CPU-intensive; adjacent frames contain nearly identical text.
        # This cap sub-samples the dense frame pool before running EasyOCR so
        # that OCR cost scales with content complexity, not raw frame count.
        self.max_ocr_frames = max(1, int(max_ocr_frames))

    async def detect(self, video_path: str) -> dict:
        path = Path(video_path)
        if not path.exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        probe = await asyncio.to_thread(self._probe_video, str(path))

        with tempfile.TemporaryDirectory(prefix="reviewagent_video_") as temp_dir:
            frame_refs = await asyncio.to_thread(self._extract_keyframes, str(path), temp_dir)
            audio_ref = await asyncio.to_thread(self._extract_audio_track, str(path), temp_dir)
            subtitles = await asyncio.to_thread(self._extract_subtitles, str(path), temp_dir)

            # Run scene-change extraction *in parallel* with modality analysis.
            # Both produce files inside temp_dir, which stays alive until all
            # tasks complete and base64 encoding finishes below.
            scene_dir = str(Path(temp_dir) / "scenes")
            scene_task = asyncio.create_task(
                asyncio.to_thread(
                    self._extract_scene_frames, str(path), scene_dir, self.scene_threshold
                )
            )
            visual_task = asyncio.create_task(self._run_visual_review(frame_refs))
            text_task = asyncio.create_task(self._run_text_review(subtitles))
            audio_task = asyncio.create_task(self._run_audio_review(audio_ref))

            scene_refs, visual_out, text_out, audio_out = await asyncio.gather(
                scene_task, visual_task, text_task, audio_task
            )

            # Encode representative frames as base64 **before** the temp dir is
            # deleted so the vision sub-agent can analyse the actual visuals.
            frame_samples_b64 = await asyncio.to_thread(
                self._sample_frames_b64, frame_refs, scene_refs, self.max_vision_frames
            )

        raw_violations = (
            visual_out.get("violations", [])
            + text_out.get("violations", [])
            + audio_out.get("violations", [])
        )
        all_violations = self._dedupe_violations(raw_violations)
        degraded_reasons = self._collect_degraded_reasons(
            frame_refs=frame_refs,
            subtitles=subtitles,
            audio_ref=audio_ref,
            preprocess_probe=probe,
            visual=visual_out,
            text=text_out,
            audio=audio_out,
        )
        degraded = len(degraded_reasons) > 0
        verdict = self._aggregate_verdict(
            all_violations=all_violations,
            degraded=degraded,
            degraded_reasons=degraded_reasons,
            critical_degraded=None,
        )

        timeline = self._build_timeline(
            visual_hits=visual_out.get("timeline", []),
            text_hits=text_out.get("timeline", []),
            audio_hits=audio_out.get("timeline", []),
        )

        return {
            "success": True,
            "verdict": verdict,
            "confidence": 0.7 if not degraded else 0.4,
            "violations": all_violations,
            "frame_samples_b64": frame_samples_b64,
            "details": {
                "frames_analyzed": len(frame_refs),
                "text_detected": text_out.get("detected_text", ""),
                "timeline_highlights": timeline,
                "preprocess": {
                    "probe": probe,
                    "frame_count": len(frame_refs),
                    "audio_extracted": bool(audio_ref),
                    "subtitle_segments": len(subtitles),
                },
                "modality_results": {
                    "visual": visual_out,
                    "audio": audio_out,
                    "text": text_out,
                },
                "pipeline_trace": {
                    "degraded": degraded,
                    "degraded_reasons": degraded_reasons,
                    "diagnosis": self._diagnosis_text(
                        verdict=verdict,
                        degraded_reasons=degraded_reasons,
                    ),
                },
            },
        }

    def _collect_degraded_reasons_audio_only(
        self,
        *,
        audio_ref: str | None,
        preprocess_probe: dict[str, Any],
        audio: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if not preprocess_probe.get("ok", False):
            reasons.append("probe_failed")
        if not preprocess_probe.get("has_audio"):
            reasons.append("no_audio_stream")
        elif not audio_ref:
            reasons.append("audio_extract_failed")
        if audio_ref and audio.get("status") != "success":
            reasons.append("audio_review_unavailable")
        return reasons

    def _diagnosis_text_audio_only(self, *, degraded_reasons: list[str]) -> str:
        reason_map = {
            "probe_failed": "Missing ffprobe or probe failed",
            "no_audio_stream": "No audio stream detected",
            "audio_extract_failed": "Audio decode or normalize failed (ffmpeg may be missing)",
            "audio_review_unavailable": "Speech transcription unavailable (e.g. ASR missing)",
        }
        critical = {"probe_failed", "no_audio_stream", "audio_extract_failed"}
        critical_reasons = [r for r in degraded_reasons if r in critical]
        if not critical_reasons:
            return ""
        labels = [reason_map.get(r, r) for r in critical_reasons]
        return "Audio pipeline degraded: " + "; ".join(labels)

    async def detect_audio_only(self, audio_path: str) -> dict:
        """Audio-only path: ffprobe + normalized WAV + ASR + text rules; no video/subtitles."""
        path = Path(audio_path)
        if not path.exists():
            return {"success": False, "error": f"Audio not found: {audio_path}"}

        probe = await asyncio.to_thread(self._probe_video, str(path))

        with tempfile.TemporaryDirectory(prefix="reviewagent_audio_") as temp_dir:
            audio_ref = await asyncio.to_thread(self._extract_audio_track, str(path), temp_dir)
            visual_out: dict[str, Any] = {
                "status": "skipped",
                "reason": "audio_only_no_video",
                "violations": [],
                "timeline": [],
            }
            text_out: dict[str, Any] = {
                "status": "skipped",
                "reason": "audio_only_no_subtitles",
                "detected_text": "",
                "violations": [],
                "timeline": [],
            }
            audio_out = await self._run_audio_review(audio_ref)

        raw_violations = list(audio_out.get("violations", []) or [])
        all_violations = self._dedupe_violations(raw_violations)
        degraded_reasons = self._collect_degraded_reasons_audio_only(
            audio_ref=audio_ref,
            preprocess_probe=probe,
            audio=audio_out,
        )
        degraded = len(degraded_reasons) > 0
        audio_critical = frozenset({"probe_failed", "no_audio_stream", "audio_extract_failed"})
        verdict = self._aggregate_verdict(
            all_violations=all_violations,
            degraded=degraded,
            degraded_reasons=degraded_reasons,
            critical_degraded=audio_critical,
        )
        timeline = self._build_timeline(
            visual_hits=[],
            text_hits=[],
            audio_hits=audio_out.get("timeline", []) or [],
        )
        return {
            "success": True,
            "verdict": verdict,
            "confidence": 0.7 if not degraded else 0.4,
            "violations": all_violations,
            "details": {
                "frames_analyzed": 0,
                "text_detected": audio_out.get("detected_text", "") or "",
                "timeline_highlights": timeline,
                "preprocess": {
                    "probe": probe,
                    "frame_count": 0,
                    "audio_extracted": bool(audio_ref),
                    "subtitle_segments": 0,
                },
                "modality_results": {
                    "visual": visual_out,
                    "audio": audio_out,
                    "text": text_out,
                },
                "pipeline_trace": {
                    "degraded": degraded,
                    "degraded_reasons": degraded_reasons,
                    "diagnosis": self._diagnosis_text_audio_only(degraded_reasons=degraded_reasons),
                },
            },
        }

    def _probe_video(self, video_path: str) -> dict[str, Any]:
        if shutil.which("ffprobe") is None:
            return {"ok": False, "error": "ffprobe_not_found"}
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        out = self._run_cmd(cmd)
        if not out["ok"]:
            return {"ok": False, "error": out["error"]}
        try:
            payload = json.loads(out["stdout"] or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "ffprobe_invalid_json"}
        streams = payload.get("streams", []) or []
        has_audio = any((s.get("codec_type") == "audio") for s in streams)
        has_subtitle = any((s.get("codec_type") == "subtitle") for s in streams)
        fmt = payload.get("format", {}) or {}
        return {
            "ok": True,
            "duration_sec": self._to_float(fmt.get("duration")),
            "format_name": fmt.get("format_name"),
            "has_audio": has_audio,
            "has_subtitle_stream": has_subtitle,
        }

    def _extract_scene_frames(
        self,
        video_path: str,
        scene_dir: str,
        threshold: float = 0.35,
    ) -> list[str]:
        """Extract frames at scene-change boundaries using ffmpeg's built-in detector.

        The ``select=gt(scene,T)`` filter scores each frame by its pixel-level
        difference from the previous one (0 = identical, 1 = completely different).
        A threshold around 0.3–0.4 captures hard cuts and major transitions while
        ignoring minor camera movement.

        Frames are scaled to 640 px wide to keep base64 payloads manageable.
        """
        if shutil.which("ffmpeg") is None:
            return []
        os.makedirs(scene_dir, exist_ok=True)
        pattern = str(Path(scene_dir) / "scene_%04d.jpg")
        # \, escapes the comma inside the filter expression so ffmpeg does not
        # interpret it as a filter-chain separator.
        vf = f"select=gt(scene\\,{threshold}),scale=640:-2"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vf",
            vf,
            "-vsync",
            "vfr",
            pattern,
        ]
        out = self._run_cmd(cmd)
        if not out["ok"]:
            return []
        return [str(p) for p in sorted(Path(scene_dir).glob("scene_*.jpg"))]

    def _sample_frames_b64(
        self,
        frame_paths: list[str],
        scene_frame_paths: list[str] | None = None,
        max_samples: int = 12,
    ) -> list[str]:
        """Build a representative frame sample for the vision sub-agent.

        Sampling strategy (combines two complementary sources):

        1. **Scene-change keyframes** (from ``_extract_scene_frames``):
           capture semantic boundaries — hard cuts, scene transitions, abrupt
           content changes.  These are the highest-risk frames and take priority.
           If there are more than *max_samples* scene frames they are sub-sampled
           evenly so no transition is over-represented.

        2. **Uniform temporal frames** (from ``_extract_keyframes``):
           fill the remaining budget with evenly-spaced frames to ensure no
           section of the video goes unchecked (in-scene content, gradual fades).

        The union is deduplicated by filename (scene and uniform frames live in
        different directories so collisions are impossible) and capped at
        *max_samples*.
        """
        scene_paths = list(scene_frame_paths or [])
        uniform_paths = list(frame_paths or [])

        # --- Scene frames: take up to max_samples, evenly spread if needed ---
        if len(scene_paths) > max_samples:
            step = max(1, len(scene_paths) // max_samples)
            scene_paths = scene_paths[::step][:max_samples]

        # --- Uniform frames: fill remaining budget ---
        remaining = max_samples - len(scene_paths)
        fill: list[str] = []
        if uniform_paths and remaining > 0:
            step = max(1, len(uniform_paths) // remaining)
            fill = uniform_paths[::step][:remaining]

        all_paths = scene_paths + fill

        # --- Encode to base64 data URLs ---
        result: list[str] = []
        for fp in all_paths:
            try:
                with open(fp, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode("ascii")
                # Scene frames are already scaled to 640 px; uniform frames
                # keep their original resolution (acceptable for most videos;
                # production deployments with 4K sources should add a resize step).
                mime = "image/jpeg"
                result.append(f"data:{mime};base64,{data}")
            except OSError:
                pass
        return result

    def _extract_keyframes(self, video_path: str, temp_dir: str) -> list[str]:
        if shutil.which("ffmpeg") is None:
            return []
        fps = 1.0 / float(self.frame_interval)
        pattern = str(Path(temp_dir) / "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(self.max_frames),
            pattern,
        ]
        out = self._run_cmd(cmd)
        if not out["ok"]:
            return []
        return [str(p) for p in sorted(Path(temp_dir).glob("frame_*.jpg"))]

    def _extract_audio_track(self, video_path: str, temp_dir: str) -> str | None:
        if shutil.which("ffmpeg") is None:
            return None
        out_audio = Path(temp_dir) / "audio.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_audio),
        ]
        out = self._run_cmd(cmd)
        if not out["ok"] or not out_audio.exists():
            return None
        return str(out_audio)

    def _extract_subtitles(self, video_path: str, temp_dir: str) -> list[dict[str, Any]]:
        if shutil.which("ffmpeg") is None:
            return []
        out_srt = Path(temp_dir) / "subtitle.srt"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_path,
            "-map",
            "0:s:0",
            str(out_srt),
        ]
        out = self._run_cmd(cmd)
        if not out["ok"] or not out_srt.exists():
            return []
        return self._parse_srt(out_srt)

    def _sample_ocr_frames(self, frame_paths: list[str]) -> list[str]:
        """Sub-sample *frame_paths* for OCR to avoid redundant work.

        Adjacent frames from a continuous shot share nearly identical text
        content.  Capping at ``max_ocr_frames`` evenly-spaced frames gives the
        same violation-detection quality at a fraction of the EasyOCR cost.

        Examples with default max_ocr_frames=10:
        - 26 s @ 1 fps → 26 frames extracted, 10 sent to OCR  (~2.6 s interval)
        - 60 s @ 1 fps → 30 frames extracted, 10 sent to OCR  (~3 s interval)
        - 10 s @ 1 fps → 10 frames extracted, all 10 sent to OCR (under cap)
        """
        if len(frame_paths) <= self.max_ocr_frames:
            return frame_paths
        step = max(1, len(frame_paths) // self.max_ocr_frames)
        return frame_paths[::step][: self.max_ocr_frames]

    async def _run_visual_review(self, frame_paths: list[str]) -> dict[str, Any]:
        if not frame_paths:
            return {"status": "skipped", "reason": "no_frames", "violations": [], "timeline": []}
        # Sub-sample before OCR: adjacent frames share nearly identical text.
        ocr_frames = self._sample_ocr_frames(frame_paths)
        sem = asyncio.Semaphore(self.frame_concurrency)
        detector = self._new_image_detector()

        async def _one(idx: int, fp: str) -> dict[str, Any]:
            async with sem:
                result = await detector.detect(fp)
            return {"index": idx, "frame_path": fp, "result": result}

        rows = await asyncio.gather(*[_one(i, fp) for i, fp in enumerate(ocr_frames)])
        violations: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        # Collect ALL OCR text, not just wordlist hits — LLM sub-agent needs the
        # full text for semantic review even when the wordlist finds nothing.
        all_ocr_snippets: list[str] = []
        seen_snippets: set[str] = set()
        for row in rows:
            r = row["result"] or {}
            # detected_text lives inside the "details" sub-dict of ImageDetector results
            frame_text = (r.get("details", {}).get("detected_text") or "").strip()
            if frame_text and frame_text not in seen_snippets:
                seen_snippets.add(frame_text)
                all_ocr_snippets.append(frame_text)
            vs = r.get("violations", []) or []
            if vs:
                timeline.append(
                    {
                        "modality": "visual",
                        "frame_index": row["index"],
                        "evidence_ref": row["frame_path"],
                        "severity": "high" if any(v.get("severity") == "high" for v in vs) else "medium",
                    }
                )
                for v in vs:
                    item = dict(v)
                    item["video_frame_index"] = row["index"]
                    violations.append(item)
        timeline = self._compress_visual_timeline(timeline)
        return {
            "status": "success",
            "frames_analyzed": len(ocr_frames),
            "detected_text": "\n---\n".join(all_ocr_snippets),
            "violations": violations,
            "timeline": timeline,
        }

    async def _run_text_review(self, subtitles: list[dict[str, Any]]) -> dict[str, Any]:
        if not subtitles:
            return {
                "status": "skipped",
                "reason": "no_subtitles",
                "detected_text": "",
                "violations": [],
                "timeline": [],
            }
        text = "\n".join(seg.get("text", "") for seg in subtitles if seg.get("text"))
        detector = self._new_text_detector()
        out = detector.detect(text)
        violations = out.get("violations", []) or []
        timeline = [
            {
                "modality": "text",
                "t_start": seg.get("start_sec"),
                "t_end": seg.get("end_sec"),
                "severity": "high",
                "evidence_ref": "subtitle",
            }
            for seg in subtitles
            if violations
        ]
        return {
            "status": "success",
            "source": "embedded_subtitle",
            "detected_text": text,
            "violations": violations,
            "timeline": timeline,
        }

    async def _run_audio_review(self, audio_ref: str | None) -> dict[str, Any]:
        if not audio_ref:
            logger.info("video_audio_review skipped reason=audio_not_extracted")
            return {
                "status": "skipped",
                "reason": "audio_not_extracted",
                "violations": [],
                "timeline": [],
            }
        logger.info("video_audio_review begin audio_ref=%s", Path(audio_ref).name)
        asr = await asyncio.to_thread(self._transcribe_audio, audio_ref)
        if not asr.get("ok", False):
            logger.warning(
                "video_audio_review skipped reason=%s audio_ref=%s",
                asr.get("reason", "asr_unavailable"),
                Path(audio_ref).name,
            )
            return {
                "status": "skipped",
                "reason": asr.get("reason", "asr_unavailable"),
                "audio_ref": audio_ref,
                "violations": [],
                "timeline": [],
            }
        text = asr.get("text", "").strip()
        logger.info(
            "video_audio_review asr_done model=%s text_chars=%s audio_ref=%s",
            asr.get("model"),
            len(text),
            Path(audio_ref).name,
        )
        if not text:
            return {
                "status": "success",
                "audio_ref": audio_ref,
                "detected_text": "",
                "violations": [],
                "timeline": [],
            }
        detector = self._new_text_detector()
        out = detector.detect(text)
        violations = out.get("violations", []) or []
        timeline = [
            {
                "modality": "audio",
                "severity": "high" if any(v.get("severity") == "high" for v in violations) else "medium",
                "evidence_ref": "asr_text",
            }
        ] if violations else []
        return {
            "status": "success",
            "audio_ref": audio_ref,
            "detected_text": text,
            "violations": violations,
            "timeline": timeline,
            "asr_model": asr.get("model"),
        }

    def _collect_degraded_reasons(
        self,
        *,
        frame_refs: list[str],
        subtitles: list[dict[str, Any]],
        audio_ref: str | None,
        preprocess_probe: dict[str, Any],
        visual: dict[str, Any],
        text: dict[str, Any],
        audio: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if not preprocess_probe.get("ok", False):
            reasons.append("probe_failed")
        if len(frame_refs) == 0:
            reasons.append("no_frames_extracted")
        if visual.get("status") != "success":
            reasons.append("visual_skipped")
        # Subtitles optional: degrade only when a subtitle stream exists but extraction is empty
        if preprocess_probe.get("has_subtitle_stream") and len(subtitles) == 0:
            reasons.append("subtitle_extract_failed")
        if preprocess_probe.get("has_subtitle_stream") and text.get("status") != "success":
            reasons.append("text_skipped")
        # Audio optional: degrade only when audio stream exists (probe) but review did not succeed
        if preprocess_probe.get("has_audio") and audio.get("status") != "success":
            reasons.append("audio_review_unavailable")
        return reasons

    def _diagnosis_text(self, *, verdict: str, degraded_reasons: list[str]) -> str:
        if not degraded_reasons:
            return ""
        reason_map = {
            "probe_failed": "Missing ffprobe or probe failed",
            "no_frames_extracted": "Could not extract key frames (ffmpeg missing or decode error)",
            "visual_skipped": "Visual review not run",
            "subtitle_extract_failed": "Subtitle stream present but extraction failed",
            "text_skipped": "Subtitle text review not run",
            "audio_review_unavailable": "Audio review unavailable (e.g. ASR missing)",
        }
        critical = {"probe_failed", "no_frames_extracted", "visual_skipped"}
        critical_reasons = [r for r in degraded_reasons if r in critical]
        if not critical_reasons:
            return ""
        labels = [reason_map.get(r, r) for r in critical_reasons]
        return "Critical video pipeline degraded; full review incomplete: " + "; ".join(labels)

    def _aggregate_verdict(
        self,
        *,
        all_violations: list[dict[str, Any]],
        degraded: bool,
        degraded_reasons: list[str],
        critical_degraded: frozenset[str] | None = None,
    ) -> str:
        if any(v.get("severity") == "high" for v in all_violations):
            return "BLOCK"
        if all_violations:
            return "WARN"
        if degraded:
            critical = critical_degraded or frozenset(
                {"probe_failed", "no_frames_extracted", "visual_skipped"}
            )
            if any(r in critical for r in degraded_reasons):
                return "UNKNOWN"
            # Optional modality (subtitle/audio) degraded → WARN instead of UNKNOWN
            return "WARN"
        return "PASS"

    def _dedupe_violations(self, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge duplicate visual hits that share OCR across frames; only entries with
        video_frame_index are merged so subtitle/audio hits with the same word stay separate."""
        framed: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for v in violations:
            if v.get("video_frame_index") is not None:
                framed.append(v)
            else:
                rest.append(v)
        if not framed:
            return list(violations)

        order: list[tuple[str, str, str]] = []
        buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
        for v in framed:
            key = (
                str(v.get("type", "")),
                str(v.get("content", "")),
                str(v.get("severity", "")),
            )
            fi = v.get("video_frame_index")
            if key not in buckets:
                order.append(key)
                base = {k: val for k, val in v.items() if k != "video_frame_index"}
                base.pop("video_frame_indices", None)
                base.pop("video_duplicate_hits", None)
                frames: list[int] = []
                if fi is not None:
                    try:
                        frames.append(int(fi))
                    except (TypeError, ValueError):
                        pass
                buckets[key] = {
                    "_base": base,
                    "_frames": frames,
                    "_hits": 1,
                }
            else:
                b = buckets[key]
                b["_hits"] += 1
                if fi is not None:
                    try:
                        nfi = int(fi)
                        if nfi not in b["_frames"]:
                            b["_frames"].append(nfi)
                            b["_frames"].sort()
                    except (TypeError, ValueError):
                        pass
        merged_visual: list[dict[str, Any]] = []
        for merge_key in order:
            b = buckets[merge_key]
            merged = dict(b["_base"])
            hits = b["_hits"]
            frames = b["_frames"]
            if hits > 1:
                merged["video_duplicate_hits"] = hits
            if frames:
                merged["video_frame_indices"] = frames
                if len(frames) == 1:
                    merged["position"] = f"frame {frames[0]}"
                else:
                    merged["position"] = (
                        f"frames {frames[0]}-{frames[-1]} "
                        f"({hits} sampled-frame hits merged)"
                    )
            merged.pop("video_frame_index", None)
            merged_visual.append(merged)
        return merged_visual + rest

    def _compress_visual_timeline(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """When the scene is static, collapse multi-frame visual timeline into one summary row."""
        if len(items) < 2:
            return items
        if any(x.get("modality") != "visual" for x in items):
            return items
        sevs = {x.get("severity") for x in items}
        if len(sevs) != 1:
            return items
        idxs = sorted(int(x["frame_index"]) for x in items if x.get("frame_index") is not None)
        if len(idxs) != len(items):
            return items
        return [
            {
                "modality": "visual",
                "frame_index_start": idxs[0],
                "frame_index_end": idxs[-1],
                "sampled_frame_hits": len(idxs),
                "severity": items[0].get("severity"),
                "evidence_ref": "video_sampled_frames",
            }
        ]

    def _build_timeline(
        self,
        *,
        visual_hits: list[dict[str, Any]],
        text_hits: list[dict[str, Any]],
        audio_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out = [*visual_hits, *text_hits, *audio_hits]
        return out[:100]

    async def _detect_frame(self, frame_path: str) -> dict:
        detector = self._new_image_detector()
        result = await detector.detect(frame_path)
        return {
            "text": result.get("details", {}).get("detected_text", ""),
            "violations": result.get("violations", []),
        }

    def _new_image_detector(self):
        from reviewagent.toolpacks.image_detector import ImageDetector

        return ImageDetector()

    def _new_text_detector(self):
        from reviewagent.toolpacks.text_detector import TextDetector

        return TextDetector()

    def _transcribe_audio(self, audio_ref: str) -> dict[str, Any]:
        """
        Optional ASR: prefer openai-whisper when installed.
        On missing dependency or failure, returns a reason code without aborting the pipeline.
        """
        try:
            import whisper  # type: ignore
        except Exception:
            logger.warning("video_asr dependency_missing provider=openai-whisper")
            return {"ok": False, "reason": "asr_dependency_missing"}
        model_name = os.getenv("REVIEW_AGENT_ASR_MODEL", "tiny")
        try:
            import torch  # type: ignore

            logger.info("video_asr begin model=%s audio_ref=%s", model_name, Path(audio_ref).name)
            model = whisper.load_model(model_name)
            # Disable FP16 on CPU to avoid whisper UserWarning
            fp16 = bool(torch.cuda.is_available())
            lang_raw = os.getenv("REVIEW_AGENT_ASR_LANGUAGE", "").strip()
            language = lang_raw if lang_raw else None
            out = model.transcribe(
                audio_ref,
                language=language,
                fp16=fp16,
            )
            text = str((out or {}).get("text", "")).strip()
            logger.info(
                "video_asr success model=%s text_chars=%s audio_ref=%s",
                model_name,
                len(text),
                Path(audio_ref).name,
            )
            return {"ok": True, "text": text, "model": model_name}
        except Exception:
            logger.exception("video_asr failed model=%s audio_ref=%s", model_name, Path(audio_ref).name)
            return {"ok": False, "reason": "asr_transcribe_failed"}

    def _run_cmd(self, cmd: list[str]) -> dict[str, Any]:
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": False, "error": f"command_not_found:{cmd[0]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if cp.returncode != 0:
            err = (cp.stderr or "").strip() or (cp.stdout or "").strip() or "command_failed"
            return {"ok": False, "error": err[:300]}
        return {"ok": True, "stdout": cp.stdout, "stderr": cp.stderr}

    def _parse_srt(self, srt_path: Path) -> list[dict[str, Any]]:
        text = srt_path.read_text(encoding="utf-8", errors="ignore")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out: list[dict[str, Any]] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue
            time_line = lines[1] if "-->" in lines[1] else (lines[0] if "-->" in lines[0] else "")
            if "-->" not in time_line:
                continue
            start_raw, end_raw = [x.strip() for x in time_line.split("-->", 1)]
            content_lines = lines[2:] if "-->" in lines[1] else lines[1:]
            out.append(
                {
                    "start_sec": self._srt_time_to_seconds(start_raw),
                    "end_sec": self._srt_time_to_seconds(end_raw),
                    "text": " ".join(content_lines).strip(),
                }
            )
        return [seg for seg in out if seg.get("text")]

    def _srt_time_to_seconds(self, t: str) -> float:
        try:
            hhmmss, ms = t.split(",", 1)
            hh, mm, ss = [int(x) for x in hhmmss.split(":")]
            return float(hh * 3600 + mm * 60 + ss) + float(int(ms)) / 1000.0
        except Exception:
            return 0.0

    def _to_float(self, value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def detect_sync(self, video_path: str) -> dict:
        """Safe when an asyncio loop is already running (e.g. FastAPI); avoids nested asyncio.run."""
        import concurrent.futures

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.detect(video_path))

        def _run_in_fresh_loop() -> dict:
            return asyncio.run(self.detect(video_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_in_fresh_loop).result(timeout=600)

    def detect_audio_sync(self, audio_path: str) -> dict:
        """Run `detect_audio_only` from a thread when a loop is already running; no nested asyncio.run."""
        import concurrent.futures

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.detect_audio_only(audio_path))

        def _run_in_fresh_loop() -> dict:
            return asyncio.run(self.detect_audio_only(audio_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_in_fresh_loop).result(timeout=600)

    def __call__(self, video_path: str) -> dict:
        return self.detect_sync(video_path)