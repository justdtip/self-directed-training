"""Reward functions for GSPO training."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

from tool_harness import run_python

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _last_python_block(text: str) -> Optional[str]:
    """Return the last fenced code block in ``text``."""

    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def score_code_tests(
    model_output: str,
    tests: List[str],
    timeout_s: int = 2,
    memory_mb: int = 256,
) -> Tuple[float, Dict[str, int]]:
    """Run tests against extracted code and compute the pass rate."""

    timeout_s = max(1, int(timeout_s))
    memory_mb = max(64, int(memory_mb))

    if not tests:
        base = 0.1 if model_output.strip() else 0.0
        return base, {"passes": 0, "total": 0}

    code = _last_python_block(model_output)
    if code is None:
        return 0.0, {"passes": 0, "total": len(tests), "reason": "no-code-block"}

    passes = 0
    for test in tests:
        snippet = f"{code}\n\n{test}"
        result = run_python(snippet, timeout_s=timeout_s, memory_mb=memory_mb)
        if result["returncode"] == 0 and "AssertionError" not in (result["stderr"] or ""):
            passes += 1
    total = len(tests)
    return passes / total if total else 0.0, {"passes": passes, "total": total}


def style_penalty(model_output: str) -> float:
    """Return a small bonus for well-formatted answers."""

    text = model_output or ""
    if "final answer" in text.lower():
        return 0.05

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "final_answer" in obj:
            return 0.05
    except json.JSONDecodeError:
        pass
    return 0.0


def timeout_penalty(stderr: str) -> float:
    """Return a penalty when a timeout is observed in ``stderr``."""

    return -0.05 if "TIMEOUT" in (stderr or "") else 0.0


def blended_reward(
    model_output: str,
    tests: List[str],
    extra: Optional[Dict[str, object]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Combine structural bonuses with unit test results."""

    extra = extra or {}
    timeout_s = int(extra.get("timeout_s", 2) or 2)
    memory_mb = int(extra.get("memory_mb", 256) or 256)

    base, stats = score_code_tests(model_output, tests, timeout_s, memory_mb)
    bonus = style_penalty(model_output) + timeout_penalty(str(extra.get("stderr", "")))
    score = max(0.0, min(1.0, base + bonus))

    details: Dict[str, float] = {"base": base, "bonus": bonus}
    details.update(stats)
    return score, details


__all__ = [
    "_last_python_block",
    "score_code_tests",
    "style_penalty",
    "timeout_penalty",
    "blended_reward",
]
