"""Reviewer registry: maps content_type strings to ContentReviewer instances."""

from __future__ import annotations

from typing import Optional

from reviewagent.reviewers.audio import AudioReviewer
from reviewagent.reviewers.base import ContentReviewer
from reviewagent.reviewers.image import ImageReviewer
from reviewagent.reviewers.text import TextReviewer
from reviewagent.reviewers.video import VideoReviewer

_REGISTRY: dict[str, ContentReviewer] = {
    "text": TextReviewer(),
    "image": ImageReviewer(),
    "video": VideoReviewer(),
    "audio": AudioReviewer(),
}


def get_reviewer(content_type: str) -> Optional[ContentReviewer]:
    """Return the registered reviewer for *content_type*, or ``None`` if unknown."""
    return _REGISTRY.get(content_type)


__all__ = [
    "ContentReviewer",
    "TextReviewer",
    "ImageReviewer",
    "VideoReviewer",
    "AudioReviewer",
    "get_reviewer",
]
