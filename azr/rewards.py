from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .tools.python_tool import run_code


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)````", re.DOTALL | re.IGNORECASE)


def _validate_answer_format(text: str) -> bool:
    """Ensure the completion contains exactly one python block and a final answer line."""

    if not text:
        return False
    scrubbed = _THINK_RE.sub("", text)
    blocks = list(re.finditer(r"```python\b[\s\S]*?```", scrubbed, flags=re.IGNORECASE))
    if len(blocks) != 1:
        return False
    tail = scrubbed[blocks[-1].end():]
    tail_lower = tail.lower()
    if "final answer:" in tail_lower:
        return True
    if '"final_answer"' in tail_lower:
        return True
    return False


def extract_last_code_block(text: str) -> str | None:
    """Return the last fenced code block, ignoring private thinking spans."""

    text = _THINK_RE.sub("", text)
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return blocks[-1] if blocks else None


def score_code_tests(
    model_output: str,
    tests: List[str],
    timeout_s: int = 2,
    memory_mb: int = 256,
    *,
    collect_exec: bool = False,
    exec_store: str = "last",
    exec_max_bytes: int = 4096,
) -> Tuple[float, Dict[str, int]]:
    """
    Execute extracted code against ``tests`` and compute the pass rate.
    Returns (score, stats) where ``score`` ∈ [0, 1] and stats stores ``passes`` and ``total``.
    """

    total = len(tests)
    if total == 0:
        base = 0.1 if model_output and model_output.strip() else 0.0
        return base, {"passes": 0, "total": 0}

    sanitized_output = _THINK_RE.sub("", model_output)
    if not _validate_answer_format(model_output):
        return 0.0, {"passes": 0, "total": total, "reason": "format_error"}
    code = extract_last_code_block(sanitized_output)
    if code is None:
        return 0.0, {"passes": 0, "total": total, "reason": "format_error"}

    def _truncate(text: str | None) -> str:
        if text is None:
            return ""
        if exec_max_bytes is None:
            return text
        data = text.encode("utf-8")
        if len(data) <= exec_max_bytes:
            return text
        return data[:exec_max_bytes].decode("utf-8", errors="ignore")

    traces: List[Dict[str, object]] | None = [] if collect_exec else None
    passes = 0
    for idx, test in enumerate(tests):
        snippet = f"{code}\n\n{test}"
        result = run_code(snippet, timeout_s=timeout_s, memory_mb=memory_mb)
        ok = result.returncode == 0 and "AssertionError" not in (result.stderr or "")
        passes += int(ok)

        if collect_exec and traces is not None:
            record = {
                "test_index": idx,
                "returncode": result.returncode,
                "stdout": _truncate(result.stdout),
                "stderr": _truncate(result.stderr),
            }
            traces.append(record)
            if exec_store == "fail_first" and not ok:
                traces = [record]
                break

    stats: Dict[str, object] = {"passes": passes, "total": total}
    if collect_exec and traces is not None:
        if exec_store == "last" and traces:
            traces = [traces[-1]]
        stats["exec_traces"] = traces

    return passes / total if total else 0.0, stats  # total>0 ensured above


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
    *,
    collect_exec: bool = False,
    exec_store: str = "last",
    exec_max_bytes: int = 4096,
) -> Tuple[float, Dict[str, float]]:
    """Combine code pass rate with minor style/timeout bonuses."""

    if not _validate_answer_format(model_output):
        return 0.0, {"reason": "format_error"}

    timeout_s = extra.get("timeout_s", 2) if extra else 2
    memory_mb = extra.get("memory_mb", 256) if extra else 256
    base, stats = score_code_tests(
        model_output,
        tests,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
        collect_exec=collect_exec,
        exec_store=exec_store,
        exec_max_bytes=exec_max_bytes,
    )

    sanitized_output = _THINK_RE.sub("", model_output)

    bonus = style_penalty(sanitized_output)
    if extra and "stderr" in extra:
        bonus += timeout_penalty(extra["stderr"])

    # Conciseness bonus: reward shorter, correct solutions.
    if base == 1.0:
        tokens = sanitized_output.strip().split()
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
