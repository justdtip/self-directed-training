from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class DataExample:
    prompt: str
    tests: List[str]
    timeout_s: int
    memory_mb: int


def _coerce_tests(value: object) -> List[str]:
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return []


def load_dataset(path: str) -> List[DataExample]:
    """
    Load a JSONL dataset. Each line must contain at least 'prompt'.
    Optional keys: 'tests' (list[str]), 'timeout_s', 'memory_mb'.
    """

    items: List[DataExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            tests = _coerce_tests(obj.get("tests", []))
            timeout_s = int(obj.get("timeout_s", 2))
            memory_mb = int(obj.get("memory_mb", 256))
            items.append(
                DataExample(
                    prompt=prompt,
                    tests=tests,
                    timeout_s=timeout_s,
                    memory_mb=memory_mb,
                )
            )
    return items


__all__ = ["DataExample", "load_dataset"]
