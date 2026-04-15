"""Tool pack registry: discover plugins and materialize LangChain tool lists."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Type

from reviewagent.toolpacks.base import ToolPackPlugin
from reviewagent.toolpacks.builtins import BUILTIN_TOOL_PACKS


class ToolPackRegistry:
    def __init__(self, extra_plugins: Optional[Iterable[Type[ToolPackPlugin]]] = None) -> None:
        self._by_id: dict[str, ToolPackPlugin] = {}
        for cls in BUILTIN_TOOL_PACKS:
            inst = cls()
            self._by_id[inst.id] = inst
        if extra_plugins:
            for cls in extra_plugins:
                inst = cls()
                self._by_id[inst.id] = inst

    def register(self, plugin: ToolPackPlugin) -> None:
        self._by_id[plugin.id] = plugin

    def list_tool_packs(self) -> list[dict[str, str]]:
        return [
            {"id": p.id, "name": p.display_name, "description": p.description}
            for p in self._by_id.values()
        ]

    def resolve_tools(self, context: Optional[Dict[str, Any]] = None) -> list:
        tools: list = []
        for plugin in self._by_id.values():
            tools.extend(plugin.get_tools(context))
        return tools


def default_registry() -> ToolPackRegistry:
    return ToolPackRegistry()


__all__ = ["ToolPackRegistry", "default_registry"]
