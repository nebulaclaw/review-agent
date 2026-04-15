"""LangChain BaseTool implementations for tool-pack registration (not vendor Skills)."""

from __future__ import annotations

from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class TextDetectorInput(BaseModel):
    text: str = Field(description="Text to scan")


class ImageDetectorInput(BaseModel):
    image_path: str = Field(description="Path to image file")


class VideoDetectorInput(BaseModel):
    video_path: str = Field(description="Path to video file")


class AudioDetectorInput(BaseModel):
    audio_path: str = Field(description="Path to audio file")


class ReviewTool(BaseTool):
    name: str = "text_detector"
    description: str = "Scan text for sensitive, policy-violation, and banned terms"
    args_schema: Type[BaseModel] = TextDetectorInput
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, memory: Optional[object] = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "memory", memory)

    def _run(self, text: str) -> dict:
        from reviewagent.toolpacks.text_detector import TextDetector

        detector = TextDetector()
        return detector.detect(text)


class ImageTool(BaseTool):
    name: str = "image_detector"
    description: str = "Detect policy issues in images, including OCR over text"
    args_schema: Type[BaseModel] = ImageDetectorInput
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, image_path: str) -> dict:
        from reviewagent.toolpacks.image_detector import ImageDetector

        detector = ImageDetector()
        return detector.detect_sync(image_path)


class VideoTool(BaseTool):
    name: str = "video_detector"
    description: str = "Detect policy issues in video via sampled frames"
    args_schema: Type[BaseModel] = VideoDetectorInput
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, video_path: str) -> dict:
        from reviewagent.toolpacks.video_detector import VideoDetector

        detector = VideoDetector()
        return detector.detect_sync(video_path)


class AudioTool(BaseTool):
    name: str = "audio_detector"
    description: str = "Detect policy issues in audio (transcribe then text rules)"
    args_schema: Type[BaseModel] = AudioDetectorInput
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, audio_path: str) -> dict:
        from reviewagent.toolpacks.video_detector import VideoDetector

        detector = VideoDetector()
        return detector.detect_audio_sync(audio_path)


__all__ = [
    "ReviewTool",
    "ImageTool",
    "VideoTool",
    "AudioTool",
]
