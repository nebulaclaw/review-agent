from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reviewagent.agent import ReviewOrchestrator


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(fingerprint=SimpleNamespace(image_collect_light_signals=False)),
        observability=SimpleNamespace(metrics_enabled=False),
    )


@pytest.mark.asyncio
async def test_video_pipeline_finalize_without_llm(monkeypatch) -> None:
    class _VD:
        async def detect(self, _content: str) -> dict:
            return {
                "success": True,
                "verdict": "PASS",
                "confidence": 0.8,
                "violations": [],
                "details": {"frames_analyzed": 3},
            }

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    finalized: dict = {}

    class _FakeOrch:
        _settings = _fake_settings()

        def _enforce_payload_raw(self, _ct: str, _content: str) -> None:
            return None

        def _finalize_early_pipeline_result(self, result: dict, **kwargs):
            finalized["kwargs"] = kwargs
            finalized["result"] = result
            return result

    out = await ReviewOrchestrator.review_payload_async(
        _FakeOrch(), content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    body = json.loads(out["response"])
    assert body["verdict"] == "PASS"
    assert body["summary"] == "未发现明确违规证据。"
    assert out["pipeline_trace"]["mode"] == "video_pipeline"
    assert finalized["kwargs"]["content_type"] == "video"


@pytest.mark.asyncio
async def test_video_pipeline_summary_uses_diagnosis(monkeypatch) -> None:
    class _VD:
        async def detect(self, _content: str) -> dict:
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

    class _FakeOrch:
        _settings = _fake_settings()

        def _enforce_payload_raw(self, _ct: str, _content: str) -> None:
            return None

        def _finalize_early_pipeline_result(self, result: dict, **kwargs):
            return result

    out = await ReviewOrchestrator.review_payload_async(
        _FakeOrch(), content_type="video", content="/tmp/demo.mp4"
    )
    body = json.loads(out["response"])
    assert body["summary"] == "未发现明确违规证据。"
    assert body["capability_status"].startswith("降级：")
    assert isinstance(body.get("degraded_labels"), list)


@pytest.mark.asyncio
async def test_video_pipeline_fallbacks_to_agent(monkeypatch) -> None:
    class _VD:
        async def detect(self, _content: str) -> dict:
            return {"success": False, "error": "probe_failed"}

    monkeypatch.setattr("reviewagent.toolpacks.video_detector.VideoDetector", _VD)

    class _FakeOrch:
        _settings = _fake_settings()

        def _enforce_payload_raw(self, _ct: str, _content: str) -> None:
            return None

        def _build_user_input(self, content_type: str, content: str, vision_direct: bool = False) -> str:
            assert content_type == "video"
            assert vision_direct is False
            return f"请审核以下视频：{content}"

        async def run(self, user_input: str, *, content_type: str, vision_image_path=None) -> dict:
            assert content_type == "video"
            assert vision_image_path is None
            return {"success": True, "response": user_input, "iterations": 1, "duration_ms": 1.1}

    out = await ReviewOrchestrator.review_payload_async(
        _FakeOrch(), content_type="video", content="/tmp/demo.mp4"
    )

    assert out["success"] is True
    assert "请审核以下视频" in out["response"]
