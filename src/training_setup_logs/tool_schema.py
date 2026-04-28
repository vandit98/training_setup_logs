"""Optional tool schema validation for agent trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    required_args: set[str]


@dataclass(frozen=True)
class ToolRegistry:
    tools: dict[str, ToolSpec]

    @classmethod
    def from_path(cls, path: Path | None) -> "ToolRegistry":
        if path is None:
            return cls({})
        payload = json.loads(path.read_text(encoding="utf-8"))
        tools: dict[str, ToolSpec] = {}
        for name, spec in payload.get("tools", {}).items():
            tools[name] = ToolSpec(name=name, required_args=set(spec.get("required_args", [])))
        return cls(tools)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def missing_required_args(self, name: str, args: dict[str, object] | None) -> set[str]:
        spec = self.tools.get(name)
        if spec is None:
            return set()
        actual = set((args or {}).keys())
        return spec.required_args - actual
