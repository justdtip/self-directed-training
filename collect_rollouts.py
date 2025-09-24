"""Collect rollouts with tool usage."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from datasets import load_dataset
from unsloth import FastLanguageModel

from opt_azr_params import PARAMS
from tool_loop import roll_with_tools
from rl_train_gspo import build_prompt, format_for_model


def main(out_path: str = "/opt/azr/data/rollouts.jsonl", max_samples: int = 100) -> None:
    """Generate tool-augmented rollouts and write them to ``out_path``."""

    params = PARAMS
    azr = params["azr"]
    rlhf = azr["rlhf"]

    max_seq = int(rlhf["max_prompt_length"]) + int(rlhf["max_completion_length"])
    model_id = azr["model_id"]
    load_in_4bit = azr.get("quantization") == "4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=max_seq,
        load_in_4bit=load_in_4bit,
    )
    lora = azr["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": "/opt/azr/data/train.jsonl"},
        split="train",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    max_turns = int(params["azr"]["sandbox"]["max_tool_turns"])

    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, item in enumerate(dataset):
            if idx >= max_samples:
                break
            messages = build_prompt(item)
            prompt = format_for_model(messages, tokenizer)
            final, tool_log, transcript = roll_with_tools(
                model,
                tokenizer,
                messages[0]["content"],
                item["prompt"],
                max_turns=max_turns,
            )
            record: Dict[str, Any] = {
                "prompt": item.get("prompt", ""),
                "final_answer": final,
                "tool_log": tool_log,
                "transcript": transcript,
                "tests": item.get("tests", []),
            }
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
