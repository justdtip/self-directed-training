"""Shared prompt scaffolding for teacher assist and retry prompts."""

from __future__ import annotations

from typing import Dict, List


_BASE_SYSTEM_PROMPT = (
    "You are a careful engineer.\n"
    "You solve programming puzzles only by writing Python code.\n\n"
    "RESPONSE FORMAT (hard requirements):\n"
    "1) Output exactly one fenced Python code block, nothing else.\n"
    "   - The opening line must be ```python and the closing line must be ``` on its own line.\n"
    "   - The code must define exactly the function requested by the user with the correct signature.\n"
    "   - Use only the Python standard library.\n"
    "2) After the code block, output one line: Final answer: <short summary>.\n"
    "3) Do NOT output explanations, analysis, comments, or any scratchpad text.\n"
)

# Teacher should NOT output code; it should give a compact, actionable hint only.
_TEACHER_HINT_SYSTEM = (
    "You are a mentor. Provide ONE short, actionable hint (≤30 words). "
    "No code. No final answers. No explanations. Plain English only."
)


def format_system_prompt(allow_thinking: bool) -> str:
    if allow_thinking:
        # Policy/opponent: thinking allowed internally, but never surfaced.
        return _BASE_SYSTEM_PROMPT + "4) Do NOT write out your scratchpad. Never include <think> or similar tags.\n"
    return _BASE_SYSTEM_PROMPT + "4) Do NOT output any hidden reasoning markup. Never emit tags such as <think> or similar.\n"

ASSIST_SYSTEM_STRICT = format_system_prompt(False)

CODE_GATE_SYSTEM_NUDGE = (
    "IMPORTANT: The previous attempt did not include valid Python code. Output exactly one ```python code block"
    " implementing the requested function, followed by 'Final answer: <short summary>'. Do not output prose or"
    " additional text."
)


def build_assist_messages(user_prompt: str, allow_thinking: bool = False) -> List[Dict[str, object]]:
    """Construct a Responses-style messages payload for the teacher model (short hint only)."""

    return [
        {"role": "system", "content": [{"type": "input_text", "text": _TEACHER_HINT_SYSTEM}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]


def build_retry_messages(user_prompt: str, hint: str, allow_thinking: bool = False) -> List[Dict[str, object]]:
    hint_text = hint.strip()
    if not hint_text:
        hint_text = "Please provide the required Python function."
    system_hint = (
        "Hint (code only): "
        + hint_text
        + "\nRemember: output exactly one Python code block implementing the requested function signature,"
        + " then a single 'Final answer:' line."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": format_system_prompt(allow_thinking)}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        },
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_hint}],
        },
    ]


__all__ = [
    "format_system_prompt",
    "CODE_GATE_SYSTEM_NUDGE",
    "build_assist_messages",
    "build_retry_messages",
]
