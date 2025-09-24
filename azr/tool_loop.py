from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from .tools import WebTool, run_code

_TOOLS_SPEC: dict | None = None


def load_tools_spec() -> dict:
    """Load and cache the tool schema from the JSON file."""

    global _TOOLS_SPEC
    if _TOOLS_SPEC is None:
        schema_path = os.path.join(os.path.dirname(__file__), "tools", "schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            _TOOLS_SPEC = json.load(f)
    return _TOOLS_SPEC


def extract_last_json_obj(text: str) -> dict | None:
    """Extract the last valid JSON object from ``text`` scanning from the end."""

    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for snippet in reversed(candidates):
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None


class ToolDispatcher:
    """Route tool calls to their implementations."""

    def __init__(self) -> None:
        self.web_tool = WebTool()

    def call(self, name: str, args: Dict) -> Dict:
        if name == "python.run":
            code = args["code"]
            timeout_s = int(args.get("timeout_s", 2))
            memory_mb = int(args.get("memory_mb", 256))
            result = run_code(code, timeout_s=timeout_s, memory_mb=memory_mb)
            return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        if name == "web.search":
            query = args["query"]
            count = int(args.get("count", 5))
            results = self.web_tool.search(query, count=count)
            return {"results": [res.__dict__ for res in results]}
        if name == "web.fetch":
            url = args["url"]
            max_bytes = int(args.get("max_bytes", self.web_tool.max_bytes))
            text = self.web_tool.fetch(url)[:max_bytes]
            return {"text": text}
        return {"error": f"Unknown tool {name}"}


def roll_with_tools(model, tokenizer, system: str, user: str, max_turns: int = 6) -> Tuple[str, List[Dict], List[str]]:
    """Run a multi-turn conversation where the model can call tools."""

    tools_spec = load_tools_spec()
    dispatcher = ToolDispatcher()
    context = (
        "You are a helpful agent. Tools are available.\n"
        f"TOOLS_SPEC = {json.dumps(tools_spec['tools'], ensure_ascii=False)}\n"
        "To call a tool, output a JSON object at the end: "
        "{'tool_call': {'name': '<tool name>', 'arguments': {...}}}. "
        "When done, output {'final_answer': '<your best answer>'}."
    )
    messages: List[Dict] = [
        {"role": "system", "content": context},
        {"role": "user", "content": user},
    ]
    transcript: List[str] = []
    tool_log: List[Dict] = []

    for _ in range(max_turns):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        text = tokenizer.decode(generation[0], skip_special_tokens=True)
        transcript.append(text)
        obj = extract_last_json_obj(text)
        if obj and "final_answer" in obj:
            return obj["final_answer"], tool_log, transcript
        if obj and "tool_call" in obj:
            call = obj["tool_call"]
            name = call.get("name")
            args = call.get("arguments", {})
            result = dispatcher.call(name, args)
            tool_log.append({"call": {"name": name, "args": args}, "result": result})
            messages.append({"role": "tool", "content": json.dumps({"name": name, "result": result}, ensure_ascii=False)})
            continue
        return text.strip(), tool_log, transcript

    return (transcript[-1] if transcript else ""), tool_log, transcript


__all__ = ["load_tools_spec", "extract_last_json_obj", "ToolDispatcher", "roll_with_tools"]
