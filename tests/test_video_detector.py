from __future__ import annotations

from pathlib import Path

import pytest

from reviewagent.toolpacks.video_detector import VideoDetector


class _FakeImageDetector:
    async def detect(self, image_path: str) -> dict:
        if image_path.endswith("2.jpg"):
            return {
                "violations": [{"type": "violence", "severity": "high"}],
                "details": {"detected_text": "bad"},
            }
        return {"violations": [], "details": {"detected_text": ""}}


class _FakeTextDetector:
    def detect(self, text: str) -> dict:
        if "敏感词" in text:
            return {
                "violations": [{"type": "illegal", "detection_method": "text_detector", "severity": "high"}],
                "verdict": "BLOCK",
            }
        return {"violations": [], "verdict": "PASS"}


class _FakeImageDetectorSameHitEveryFrame:
    """Return the same violation every frame to verify multi-frame deduplication."""

    async def detect(self, image_path: str) -> dict:
        return {
            "violations": [
                {"type": "illegal", "content": "恋童", "severity": "high", "position": 0},
            ],
            "details": {"detected_text": "恋童"},
        }


class _FakeTextDetector恋童:
    def detect(self, text: str) -> dict:
        if "恋童" in text:
            return {
                "violations": [
                    {"type": "illegal", "detection_method": "text_detector", "content": "恋童", "severity": "high", "position": 0},
                ],
                "verdict": "BLOCK",
            }
        return {"violations": [], "verdict": "PASS"}


@pytest.mark.asyncio
async def test_detect_returns_unknown_when_all_modalities_skipped(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": False})
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "UNKNOWN"
    trace = out["details"]["pipeline_trace"]
    assert trace["degraded"] is True
    assert "no_frames_extracted" in trace["degraded_reasons"]


@pytest.mark.asyncio
async def test_probe_failure_does_not_hard_fail(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "probe_fail.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": False, "error": "ffprobe_not_found"})
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "UNKNOWN"
    assert "probe_failed" in out["details"]["pipeline_trace"]["degraded_reasons"]


@pytest.mark.asyncio
async def test_detect_blocks_when_visual_hits_high_severity(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "b.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": False})
    monkeypatch.setattr(
        detector,
        "_extract_keyframes",
        lambda _p, _t: ["frame_1.jpg", "frame_2.jpg"],
    )
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(
        detector,
        "_extract_subtitles",
        lambda _p, _t: [{"start_sec": 0.0, "end_sec": 1.0, "text": "普通文本"}],
    )
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetector())
    monkeypatch.setattr(detector, "_new_text_detector", lambda: _FakeTextDetector())

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "BLOCK"
    assert out["details"]["frames_analyzed"] == 2
    assert out["details"]["modality_results"]["visual"]["status"] == "success"


@pytest.mark.asyncio
async def test_detect_blocks_when_audio_asr_hits_high_severity(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "c.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": True})
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: ["frame_1.jpg"])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: "audio.wav")
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetector())
    monkeypatch.setattr(detector, "_new_text_detector", lambda: _FakeTextDetector())
    monkeypatch.setattr(
        detector,
        "_transcribe_audio",
        lambda _a: {"ok": True, "text": "这里有敏感词", "model": "tiny"},
    )

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "BLOCK"
    audio = out["details"]["modality_results"]["audio"]
    assert audio["status"] == "success"
    assert audio["asr_model"] == "tiny"


@pytest.mark.asyncio
async def test_no_subtitle_stream_does_not_force_unknown(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "no_subs.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(
        detector,
        "_probe_video",
        lambda _p: {"ok": True, "has_audio": False, "has_subtitle_stream": False},
    )
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: ["frame_1.jpg"])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetector())

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "PASS"
    assert out["details"]["pipeline_trace"]["degraded"] is False


@pytest.mark.asyncio
async def test_detect_audio_degrades_when_asr_missing(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "d.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(
        detector,
        "_probe_video",
        lambda _p: {"ok": True, "has_audio": True, "has_subtitle_stream": False},
    )
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: ["frame_1.jpg"])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: "audio.wav")
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetector())
    monkeypatch.setattr(detector, "_transcribe_audio", lambda _a: {"ok": False, "reason": "asr_dependency_missing"})

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "WARN"
    assert out["details"]["modality_results"]["audio"]["status"] == "skipped"
    reasons = out["details"]["pipeline_trace"]["degraded_reasons"]
    assert "audio_review_unavailable" in reasons


@pytest.mark.asyncio
async def test_detect_merges_identical_visual_violations_across_frames(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "static.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    frames = ["f0.jpg", "f1.jpg", "f2.jpg"]
    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": False})
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: frames)
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(detector, "_extract_subtitles", lambda _p, _t: [])
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetectorSameHitEveryFrame())
    monkeypatch.setattr(detector, "_new_text_detector", lambda: _FakeTextDetector())

    out = await detector.detect(str(video))

    assert out["success"] is True
    assert out["verdict"] == "BLOCK"
    vios = out["violations"]
    assert len(vios) == 1
    assert vios[0]["type"] == "illegal"
    assert vios[0]["content"] == "恋童"
    assert vios[0]["video_duplicate_hits"] == 3
    assert vios[0]["video_frame_indices"] == [0, 1, 2]

    tl = out["details"]["timeline_highlights"]
    assert len(tl) == 1
    assert tl[0]["modality"] == "visual"
    assert tl[0]["frame_index_start"] == 0
    assert tl[0]["frame_index_end"] == 2
    assert tl[0]["sampled_frame_hits"] == 3


@pytest.mark.asyncio
async def test_visual_dedupe_does_not_merge_subtitle_hit_same_word(
    tmp_path: Path, monkeypatch
) -> None:
    """Vision multi-frame merges into one item; same word in subtitles without frame stays separate."""
    video = tmp_path / "both.mp4"
    video.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": False})
    monkeypatch.setattr(detector, "_extract_keyframes", lambda _p, _t: ["a.jpg", "b.jpg"])
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)
    monkeypatch.setattr(
        detector,
        "_extract_subtitles",
        lambda _p, _t: [{"start_sec": 0.0, "end_sec": 1.0, "text": "恋童相关"}],
    )
    monkeypatch.setattr(detector, "_new_image_detector", lambda: _FakeImageDetectorSameHitEveryFrame())
    monkeypatch.setattr(detector, "_new_text_detector", lambda: _FakeTextDetector恋童())

    out = await detector.detect(str(video))

    assert len(out["violations"]) == 2
    with_frames = [v for v in out["violations"] if "video_frame_indices" in v]
    no_frames = [v for v in out["violations"] if "video_frame_indices" not in v]
    assert len(with_frames) == 1
    assert with_frames[0]["video_duplicate_hits"] == 2
    assert len(no_frames) == 1


@pytest.mark.asyncio
async def test_detect_audio_only_blocks_on_transcript(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "speech.mp3"
    audio.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": True})
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: "norm.wav")
    monkeypatch.setattr(detector, "_new_text_detector", lambda: _FakeTextDetector())
    monkeypatch.setattr(
        detector,
        "_transcribe_audio",
        lambda _a: {"ok": True, "text": "这里有敏感词", "model": "tiny"},
    )

    out = await detector.detect_audio_only(str(audio))

    assert out["success"] is True
    assert out["verdict"] == "BLOCK"
    assert out["details"]["frames_analyzed"] == 0
    assert out["details"]["modality_results"]["visual"]["status"] == "skipped"
    assert out["details"]["modality_results"]["text"]["reason"] == "audio_only_no_subtitles"
    trace = out["details"]["pipeline_trace"]
    assert "no_frames_extracted" not in trace["degraded_reasons"]


@pytest.mark.asyncio
async def test_detect_audio_only_unknown_without_audio_stream(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "empty.mp3"
    audio.write_bytes(b"stub")
    detector = VideoDetector()

    monkeypatch.setattr(detector, "_probe_video", lambda _p: {"ok": True, "has_audio": False})
    monkeypatch.setattr(detector, "_extract_audio_track", lambda _p, _t: None)

    out = await detector.detect_audio_only(str(audio))

    assert out["success"] is True
    assert out["verdict"] == "UNKNOWN"
    assert "no_audio_stream" in out["details"]["pipeline_trace"]["degraded_reasons"]


def test_parse_srt_segments(tmp_path: Path) -> None:
    srt = tmp_path / "demo.srt"
    srt.write_text(
        "1\n00:00:00,500 --> 00:00:01,200\n第一行\n\n"
        "2\n00:00:02,000 --> 00:00:03,100\n第二行\n",
        encoding="utf-8",
    )
    detector = VideoDetector()
    segs = detector._parse_srt(srt)
    assert len(segs) == 2
    assert segs[0]["start_sec"] == 0.5
    assert segs[1]["text"] == "第二行"
