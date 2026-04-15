"""Built-in tool packs (LangChain tool groups)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reviewagent.toolpacks.base import ToolPackPlugin
from reviewagent.toolpacks.tools import AudioTool, ImageTool, ReviewTool, VideoTool


class ReviewRulesToolPack(ToolPackPlugin):
    id = "review.rules"
    display_name = "Rules and heuristics"
    description = "Text/image/video/audio rule checks (iterate wordlists and models independently)"

    def get_tools(self, context: Optional[Dict[str, Any]] = None) -> List:
        mem = (context or {}).get("memory")
        t = ReviewTool(memory=mem) if mem is not None else ReviewTool()
        return [t, ImageTool(), VideoTool(), AudioTool()]


BUILTIN_TOOL_PACKS: list[type[ToolPackPlugin]] = [
    ReviewRulesToolPack,
]

__all__ = [
    "ReviewRulesToolPack",
    "BUILTIN_TOOL_PACKS",
]
