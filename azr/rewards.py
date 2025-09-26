from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .tools.python_tool import run_code


def extract_last_code_block(text: str) -> str | None:
    """
    Return the contents of the last fenced code block in ``text``.
    Supports ```...``` or ```python ...``` fences.
    """

    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return blocks[-1] if blocks else None


def score_code_tests(
    model_output: str,
    tests: List[str],
    timeout_s: int = 2,
    memory_mb: int = 256,
) -> Tuple[float, Dict[str, int]]:
    """
    Execute extracted code against ``tests`` and compute the pass rate.
    Returns (score, stats) where ``score`` ∈ [0, 1] and stats stores ``passes`` and ``total``.
    """

    total = len(tests)
    if total == 0:
        base = 0.1 if model_output and model_output.strip() else 0.0
        return base, {"passes": 0, "total": 0}

    code = extract_last_code_block(model_output)
    if code is None:
        return 0.0, {"passes": 0, "total": total}

    passes = 0
    for test in tests:
        snippet = f"{code}\n\n{test}"
        result = run_code(snippet, timeout_s=timeout_s, memory_mb=memory_mb)
        ok = result.returncode == 0 and "AssertionError" not in (result.stderr or "")
        passes += int(ok)

    return passes / total, {"passes": passes, "total": total}


def style_penalty(model_output: str) -> float:
    """Return a small bonus when the answer contains a final answer marker."""

    lower = model_output.lower()
    if "final answer" in lower:
        return 0.05
    if re.search(r"\{\s*\"?final_answer\"?", model_output):
        return 0.05
    return 0.0


def timeout_penalty(stderr: str) -> float:
    """Return a penalty when the python tool reports a timeout."""

    return -0.05 if "TIMEOUT" in (stderr or "") else 0.0


def blended_reward(
    model_output: str,
    tests: List[str],
    extra: Dict | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Combine code pass rate with minor style/timeout bonuses."""

    timeout_s = extra.get("timeout_s", 2) if extra else 2
    memory_mb = extra.get("memory_mb", 256) if extra else 256
    base, stats = score_code_tests(model_output, tests, timeout_s=timeout_s, memory_mb=memory_mb)

    bonus = style_penalty(model_output)
    if extra and "stderr" in extra:
        bonus += timeout_penalty(extra["stderr"])

    # Conciseness bonus: reward shorter, correct solutions.
    if base == 1.0:
        tokens = model_output.strip().split()
        token_count = len(tokens)
        if token_count <= 200:
            bonus += 0.2
        elif token_count < 500:
            # Linearly decay the bonus from 0.2 at 200 tokens down to 0 at 500 tokens.
            bonus += 0.2 * max(0.0, (500 - token_count) / 300.0)

    score = max(0.0, min(1.0, base + bonus))
    stats.update({"base": base, "bonus": bonus})
    return score, stats


__all__ = [
    "extract_last_code_block",
    "score_code_tests",
    "style_penalty",
    "timeout_penalty",
    "blended_reward",
]
