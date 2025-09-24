from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from datasets import Dataset


def load_prompts_from_list(prompts: List[str]) -> Dataset:
    return Dataset.from_dict({"prompt": prompts})


def load_prompts_from_path(path: Optional[str]) -> Dataset:
    import json
    if not path:
        # tiny synthetic
        return load_prompts_from_list([
            "Explain what RLHF is in one sentence.",
            "List two benefits of LoRA for LLM fine-tuning.",
        ])
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "prompt" in obj:
                    rows.append(obj["prompt"])
            except Exception:
                rows.append(line)
    return load_prompts_from_list(rows)

