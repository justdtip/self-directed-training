from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolSandbox:
    timeout_s: int
    memory_mb: int
    max_tool_turns: int

    def __post_init__(self):
        self._turns = 0

    def check(self):
        if self._turns >= self.max_tool_turns:
            raise RuntimeError("Tool turn limit exceeded")
        self._turns += 1

