"""Tool pack plugins: a unit that registers a group of LangChain tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class ToolPackPlugin(ABC):
    """Binds LangChain tools to a pack; registered under a stable id in ToolPackRegistry."""

    id: str = "abstract"
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def get_tools(self, context: Optional[Dict[str, Any]] = None) -> List["BaseTool"]:
        raise NotImplementedError


__all__ = ["ToolPackPlugin"]
