"""Utilities for validating teacher/policy code outputs before execution."""

from __future__ import annotations

import ast
import re
from typing import Iterable, Optional, Sequence, Set

_PY_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class CodeGateError(RuntimeError):
    """Raised when generated output fails mandatory code requirements."""


def extract_last_python_block(text: str) -> Optional[str]:
    """Return the final fenced Python block from ``text`` or ``None``."""

    if not text:
        return None
    blocks = _PY_BLOCK_RE.findall(text)
    if not blocks:
        return None
    return blocks[-1].strip() or None


def has_function_signature(code: str, func_name: str) -> bool:
    """Check whether ``code`` defines a top-level function named ``func_name``."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True
    return False


_FUNC_DENYLIST: Set[str] = {
    "assert",
    "print",
    "len",
    "range",
    "sum",
    "max",
    "min",
}


def infer_function_name(tests: Sequence[str], prompt: str = "") -> Optional[str]:
    """Attempt to infer the target function name from dataset tests or prompt.

    Returns ``None`` when no unambiguous candidate can be found.
    """

    names: Set[str] = set()
    for test in tests or []:
        if not isinstance(test, str):
            continue
        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test):
            candidate = match.group(1)
            if candidate in _FUNC_DENYLIST:
                continue
            names.add(candidate)
    if len(names) == 1:
        return next(iter(names))

    # Fallback: look for "Define <name>(" pattern in the prompt.
    if prompt:
        match = re.search(r"define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", prompt, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if len(names) <= 1 or candidate in names:
                return candidate
    return None


__all__ = [
    "CodeGateError",
    "extract_last_python_block",
    "has_function_signature",
    "infer_function_name",
]
