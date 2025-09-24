"""Tool orchestration loop used during rollouts."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from tool_harness import run_python
from tools.web_tool import fetch_url, search_ddg

SCHEMA_PATH = "/opt/azr/tools/schema.json"
_TOOLS_CACHE: Optional[Dict[str, Any]] = None


def _load_tools() -> Dict[str, Any]:
    """Load and cache the tool schema from disk."""

    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
            _TOOLS_CACHE = json.load(fh)
    return _TOOLS_CACHE


def _extract_last_json_line(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last valid JSON object embedded in ``text``."""

    depth = 0
    end_idx: Optional[int] = None
    for idx in range(len(text) - 1, -1, -1):
        char = text[idx]
        if char == '}':
            if depth == 0:
                end_idx = idx
            depth += 1
        elif char == '{':
            if end_idx is None:
                continue
            depth -= 1
            if depth == 0:
                candidate = text[idx : end_idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    end_idx = None
        else:
            continue
    return None


def tool_dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to its implementation."""

    if name == "python.run":
        code = args.get("code")
        if not isinstance(code, str):
            return {"error": "missing code"}
        timeout = int(args.get("timeout_s", 2) or 2)
        memory = int(args.get("memory_mb", 256) or 256)
        return run_python(code, timeout_s=timeout, memory_mb=memory)

    if name == "web.search":
        query = args.get("query")
        if not isinstance(query, str):
            return {"error": "missing query"}
        count = int(args.get("count", 5) or 5)
        return search_ddg(query, count)

    if name == "web.fetch":
        url = args.get("url")
        if not isinstance(url, str):
            return {"error": "missing url"}
        max_bytes = int(args.get("max_bytes", 800_000) or 800_000)
        return fetch_url(url, max_bytes=max_bytes)

    return {"error": f"unknown tool {name}"}


def roll_with_tools(
    model,
    tokenizer,
    system: str,
    user: str,
    max_turns: int = 6,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Conduct a tool-augmented dialogue with the model."""

    tools = _load_tools()["tools"]
    instructions = (
        f"{system}\n\n"
        "You may call at most one tool per reply by emitting JSON with either"
        " {'tool_call': {'name': '<tool>', 'arguments': {...}}} or"
        " {'final_answer': '...'} as the final line."
        " Available tools are:\n" + json.dumps(tools, indent=2)
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]

    tool_log: List[Dict[str, Any]] = []
    transcript: List[str] = []

    for _ in range(max_turns):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generation = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        output_text = tokenizer.decode(generation[0], skip_special_tokens=True)
        new_segment = output_text[len(prompt):].strip()
        transcript.append(new_segment)

        parsed = _extract_last_json_line(new_segment)
        if parsed is None:
            return new_segment, tool_log, transcript

        if "final_answer" in parsed:
            return str(parsed["final_answer"]), tool_log, transcript

        tool_call = parsed.get("tool_call")
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments"), dict) else {}
            if not isinstance(name, str):
                return new_segment, tool_log, transcript
            result = tool_dispatch(name, arguments)
            tool_log.append({"call": {"name": name, "args": arguments}, "result": result})
            messages.append({"role": "assistant", "content": new_segment})
            messages.append({"role": "tool", "content": json.dumps({"name": name, "result": result})})
            continue

        return new_segment, tool_log, transcript

    return transcript[-1] if transcript else "", tool_log, transcript


__all__ = [
    "_load_tools",
    "_extract_last_json_line",
    "tool_dispatch",
    "roll_with_tools",
]
