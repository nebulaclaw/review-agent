"""Tests for the VideoReviewer / AudioReviewer multi-agent pipeline.

Coverage:
- Wordlist early-block path (detector finds violations → no LLM)
- Degraded path (no text extracted → no LLM)
- Multi-agent path (ASR / subtitle → parallel sub-agents → judge)
- Detector failure fallback (→ orchestrator.run() tool loop)
- multi_agent helpers: _fallback_merge, judge fast-path on empty results
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from reviewagent.agent import ReviewOrchestrator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(provider="openai", model="gpt-4o-mini"),
        pipeline=SimpleNamespace(
            fingerprint=SimpleNamespace(image_collect_light_signals=False)
        ),
        observability=SimpleNamespace(metrics_enabled=False, tracing=None),
    )


def _make_orch(llm=None, run_result: dict | None = None) -> Any:
    """Build a minimal fake orchestrator accepted by the reviewer."""

    class _Orch:
        _settings = _fake_settings()
        settings = _settings  # public alias used by VideoReviewer
        _session_id: str | None = None

        def _enforce_payload_raw(self, _ct: str, _content: str) -> None:
            return None

        def _finalize_early_pipeline_result(self, result: dict, **kwargs: Any) -> dict:
            result.setdefault("_finalized", True)
            return result

    if llm is not None:
        _Orch.llm = llm

    if run_result is not None:
        async def _run(self, user_input: str, *, content_type: str, **_kw: Any) -> dict:
            return run_result
        _Orch.run = _run

    return _Orch()


def _json_llm(responses: list[str]):
    """Fake LLM that returns *responses* in order."""
    call_idx = {"n": 0}

    class _LLM:
        async def ainvoke(self, _messages: list) -> Any:
            i = call_idx["n"]
            call_idx["n"] += 1
            return SimpleNamespace(content=responses[min(i, len(responses) - 1)])

    return _LLM()


# ---------------------------------------------------------------------------
# VideoReviewer — early block (wordlist hit)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_pipeline_early_block_on_violation(monkeypatch) -> None:
    """Detector finds a wordlist violation → early block, LLM never called."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "BLOCK",
                "confidence": 0.9,
                "violations": [{"type": "illegal", "content": "敏感词", "severity": "high"}],
                "details": {},
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)
    orch = _make_orch()

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "BLOCK"
    assert len(body["violations"]) == 1
    assert out["pipeline_trace"]["continued_to_llm"] is False
    assert out["pipeline_trace"]["mode"] == "video_pipeline"


# ---------------------------------------------------------------------------
# VideoReviewer — degraded (no text extracted, no violations)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_pipeline_degraded_no_text(monkeypatch) -> None:
    """Detector succeeds but extracts no text → returns degraded result without LLM."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.8,
                "violations": [],
                "details": {"frames_analyzed": 3},
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)
    orch = _make_orch()

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"
    # Degraded path uses a note about limited detection capability
    assert "检测能力受限" in body["summary"]
    assert out["pipeline_trace"]["continued_to_llm"] is False
    assert out["pipeline_trace"]["mode"] == "video_pipeline"


@pytest.mark.asyncio
async def test_video_pipeline_degraded_shows_capability_status(monkeypatch) -> None:
    """Degraded reasons are translated into human-readable labels."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "WARN",
                "confidence": 0.4,
                "violations": [],
                "details": {
                    "pipeline_trace": {
                        "degraded": True,
                        "degraded_reasons": ["audio_review_unavailable"],
                        "diagnosis": "",
                    }
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)
    orch = _make_orch()

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )
    body = json.loads(out["response"])
    assert body["capability_status"].startswith("降级：")
    assert isinstance(body.get("degraded_labels"), list)
    assert len(body["degraded_labels"]) > 0


# ---------------------------------------------------------------------------
# VideoReviewer — multi-agent path (ASR text available)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_pipeline_multi_agent_with_asr(monkeypatch) -> None:
    """ASR text available → parallel sub-agents → judge → final verdict."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.8,
                "violations": [],
                "details": {
                    "frames_analyzed": 5,
                    "modality_results": {
                        "audio": {
                            "detected_text": "今天天气不错，欢迎收看我们的节目",
                            "violations": [],
                        },
                        "text": {"detected_text": "", "violations": []},
                        "visual": {"violations": []},
                    },
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    # 1st call → ASR sub-agent;  2nd call → judge
    llm = _json_llm([
        '{"verdict":"PASS","confidence":0.92,"violations":[],"summary":"语音合规"}',
        '{"verdict":"PASS","confidence":0.91,"violations":[],"summary":"综合判断：合规"}',
    ])
    orch = _make_orch(llm=llm)

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"
    pt = out["pipeline_trace"]
    assert pt["continued_to_llm"] is True
    assert pt["mode"] == "video_pipeline+multi_agent"
    assert len(pt["sub_agents"]) == 1
    assert pt["sub_agents"][0]["name"] == "audio_asr"
    assert pt["sub_agents"][0]["verdict"] == "PASS"


@pytest.mark.asyncio
async def test_video_pipeline_multi_agent_judge_escalates_to_block(monkeypatch) -> None:
    """Sub-agent finds BLOCK; judge escalates → final verdict BLOCK."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.5,
                "violations": [],
                "details": {
                    "frames_analyzed": 2,
                    "modality_results": {
                        "audio": {
                            "detected_text": "煽动性违规内容示例",
                            "violations": [],
                        },
                        "text": {"detected_text": "违规字幕", "violations": []},
                        "visual": {"violations": []},
                    },
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    # Sub-agents: ASR→BLOCK, subtitle→WARN; judge→BLOCK
    llm = _json_llm([
        '{"verdict":"BLOCK","confidence":0.98,"violations":[{"type":"illegal","content":"煽动","severity":"high"}],"summary":"煽动违规"}',
        '{"verdict":"WARN","confidence":0.6,"violations":[],"summary":"字幕轻微疑似"}',
        '{"verdict":"BLOCK","confidence":0.95,"violations":[{"type":"illegal","content":"煽动","severity":"high"}],"summary":"Judge: 升级为BLOCK"}',
    ])
    orch = _make_orch(llm=llm)

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    body = json.loads(out["response"])
    assert body["verdict"] == "BLOCK"
    pt = out["pipeline_trace"]
    assert len(pt["sub_agents"]) == 2  # audio_asr + subtitle


# ---------------------------------------------------------------------------
# VideoReviewer — visual frames sub-agent (vision LLM path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_pipeline_visual_frames_sub_agent(monkeypatch) -> None:
    """When detector returns frame_samples_b64 and provider supports vision,
    a visual_frames sub-agent is spawned alongside text sub-agents."""

    _FAKE_FRAME = "data:image/jpeg;base64,/9j/fake"

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.8,
                "violations": [],
                "frame_samples_b64": [_FAKE_FRAME, _FAKE_FRAME],
                "details": {
                    "frames_analyzed": 2,
                    "modality_results": {
                        "audio": {"detected_text": "一段正常的旁白", "violations": []},
                        "text": {"detected_text": "", "violations": []},
                        "visual": {"violations": []},
                    },
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    # ASR sub-agent → PASS, visual_frames sub-agent → PASS, judge → PASS
    llm = _json_llm([
        '{"verdict":"PASS","confidence":0.9,"violations":[],"summary":"音频正常"}',
        '{"verdict":"PASS","confidence":0.95,"violations":[],"summary":"画面无违规"}',
        '{"verdict":"PASS","confidence":0.92,"violations":[],"summary":"综合正常"}',
    ])
    orch = _make_orch(llm=llm)

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"

    pt = out["pipeline_trace"]
    agent_names = [a["name"] for a in pt["sub_agents"]]
    assert "audio_asr" in agent_names, "ASR sub-agent should be present"
    assert "visual_frames" in agent_names, "visual_frames sub-agent should be present"


@pytest.mark.asyncio
async def test_video_pipeline_visual_frames_only_no_text(monkeypatch) -> None:
    """No text extracted but frames available + vision supported → visual_frames
    sub-agent still runs (provider=openai in _fake_settings)."""

    _FAKE_FRAME = "data:image/jpeg;base64,/9j/fake"

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.5,
                "violations": [],
                "frame_samples_b64": [_FAKE_FRAME],
                "details": {
                    "frames_analyzed": 1,
                    "modality_results": {
                        "audio": {"detected_text": "", "violations": []},
                        "text": {"detected_text": "", "violations": []},
                        "visual": {"violations": []},
                    },
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    llm = _json_llm([
        '{"verdict":"PASS","confidence":0.9,"violations":[],"summary":"画面正常"}',
        '{"verdict":"PASS","confidence":0.9,"violations":[],"summary":"综合正常"}',
    ])
    orch = _make_orch(llm=llm)

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    pt = out["pipeline_trace"]
    assert pt["mode"] == "video_pipeline+multi_agent"
    agent_names = [a["name"] for a in pt["sub_agents"]]
    assert "visual_frames" in agent_names


# ---------------------------------------------------------------------------
# VideoReviewer — detector failure fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_pipeline_fallbacks_to_agent_on_detector_error(monkeypatch) -> None:
    """Detector fails → orchestrator.run() tool-loop fallback."""

    class _VD:
        async def detect(self, _path: str) -> dict:
            return {"success": False, "error": "probe_failed"}

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    run_response = '{"verdict":"UNKNOWN","confidence":0.3,"violations":[],"summary":"探测失败"}'
    orch = _make_orch(run_result={
        "success": True,
        "response": run_response,
        "iterations": 1,
        "duration_ms": 10.0,
    })

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    # Falls through to orchestrator.run() — response is the run result
    assert "探测失败" in out["response"]


# ---------------------------------------------------------------------------
# AudioReviewer — early block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_pipeline_early_block_on_violation(monkeypatch) -> None:
    class _VD:
        async def detect_audio_only(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "BLOCK",
                "confidence": 0.95,
                "violations": [{"type": "illegal", "content": "违禁词", "severity": "high"}],
                "details": {},
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)
    orch = _make_orch()

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="audio", content="/tmp/demo.mp3"
    )

    body = json.loads(out["response"])
    assert body["verdict"] == "BLOCK"
    assert out["pipeline_trace"]["continued_to_llm"] is False
    assert out["pipeline_trace"]["mode"] == "audio_pipeline"


# ---------------------------------------------------------------------------
# AudioReviewer — multi-agent (ASR transcript)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_pipeline_multi_agent_with_asr(monkeypatch) -> None:
    """AudioReviewer: ASR transcript available → sub-agent + judge."""

    class _VD:
        async def detect_audio_only(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.8,
                "violations": [],
                "details": {
                    "modality_results": {
                        "audio": {
                            "detected_text": "欢迎收听今天的播客节目",
                            "violations": [],
                        }
                    }
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    llm = _json_llm([
        '{"verdict":"PASS","confidence":0.9,"violations":[],"summary":"音频内容合规"}',
        '{"verdict":"PASS","confidence":0.9,"violations":[],"summary":"综合：合规"}',
    ])
    orch = _make_orch(llm=llm)

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="audio", content="/tmp/demo.mp3"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"
    pt = out["pipeline_trace"]
    assert pt["continued_to_llm"] is True
    assert pt["mode"] == "audio_pipeline+multi_agent"
    assert pt["sub_agents"][0]["name"] == "audio_asr"


@pytest.mark.asyncio
async def test_audio_pipeline_degraded_no_asr(monkeypatch) -> None:
    """No ASR transcript → degraded result without LLM."""

    class _VD:
        async def detect_audio_only(self, _path: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.3,
                "violations": [],
                "details": {
                    "pipeline_trace": {
                        "degraded_reasons": ["audio_review_unavailable"],
                        "diagnosis": "",
                    }
                },
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)
    orch = _make_orch()

    out = await ReviewOrchestrator.review_payload_async(
        orch, content_type="audio", content="/tmp/demo.mp3"
    )

    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"
    assert out["pipeline_trace"]["continued_to_llm"] is False


# ---------------------------------------------------------------------------
# multi_agent.py internals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sub_agent_handles_llm_error_gracefully() -> None:
    """Sub-agent LLM error → SubAgentResult with error set, verdict=UNKNOWN."""
    from reviewagent.reviewers.multi_agent import SubAgentTask, run_sub_agent

    class _BadLLM:
        async def ainvoke(self, _msgs: list) -> Any:
            raise RuntimeError("LLM timeout")

    class _Orch:
        llm = _BadLLM()

    task = SubAgentTask("test_agent", "审核内容 ABC")
    result = await run_sub_agent(task, _Orch())  # type: ignore[arg-type]

    assert result.verdict == "UNKNOWN"
    assert result.error is not None
    assert "timeout" in result.error.lower()


def test_fallback_merge_picks_heaviest_verdict() -> None:
    """_fallback_merge returns the BLOCK result and merges all violations."""
    from reviewagent.reviewers.multi_agent import SubAgentResult, _fallback_merge

    results = [
        SubAgentResult("a", "PASS", 0.9, [], "pass", ""),
        SubAgentResult("b", "BLOCK", 0.95,
                       [{"type": "illegal", "content": "x", "severity": "high"}],
                       "block found", ""),
        SubAgentResult("c", "WARN", 0.7, [], "warn", ""),
    ]
    raw = _fallback_merge(results)
    merged = json.loads(raw)
    assert merged["verdict"] == "BLOCK"
    assert any(v["content"] == "x" for v in merged["violations"])


@pytest.mark.asyncio
async def test_judge_fast_path_all_errored() -> None:
    """judge returns UNKNOWN immediately when all sub-results have errors."""
    from reviewagent.reviewers.multi_agent import SubAgentResult, run_judge_agent

    class _Orch:
        class llm:
            @staticmethod
            async def ainvoke(_msgs: list) -> Any:
                raise AssertionError("should not be called")

    errored = [
        SubAgentResult("a", "UNKNOWN", 0.0, [], "", "", error="timeout"),
        SubAgentResult("b", "UNKNOWN", 0.0, [], "", "", error="timeout"),
    ]
    result = await run_judge_agent(errored, _Orch())  # type: ignore[arg-type]
    body = json.loads(result["response"])
    assert body["verdict"] == "UNKNOWN"
    assert result["iterations"] == 0


@pytest.mark.asyncio
async def test_judge_fallback_merge_on_llm_error() -> None:
    """Judge LLM errors → graceful fallback to _fallback_merge."""
    from reviewagent.reviewers.multi_agent import SubAgentResult, run_judge_agent

    class _BadLLM:
        async def ainvoke(self, _msgs: list) -> Any:
            raise RuntimeError("judge down")

    class _Orch:
        llm = _BadLLM()

    sub = [
        SubAgentResult("x", "WARN", 0.7, [], "warn", ""),
        SubAgentResult("y", "PASS", 0.9, [], "pass", ""),
    ]
    result = await run_judge_agent(sub, _Orch())  # type: ignore[arg-type]
    body = json.loads(result["response"])
    # Fallback picks WARN (heaviest here)
    assert body["verdict"] == "WARN"
    assert "error" in result
