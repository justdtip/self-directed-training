"""GSPO training entrypoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer

from opt_azr_params import PARAMS
from rewards import blended_reward

SYSTEM_PROMPT = (
    "You are a careful engineer. Generate Python code that solves the task, "
    "followed by assert-based tests if applicable, and finish with 'Final answer: <answer>'."
)


@dataclass
class Task:
    prompt: str
    tests: List[str]
    timeout_s: int
    memory_mb: int


def build_prompt(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a chat message list for a dataset row."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(task.get("prompt", ""))},
    ]


def format_for_model(messages: List[Dict[str, str]], tokenizer) -> str:
    """Render messages to the model's prompt format."""

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def reward_fn(
    batch_prompts: List[str],
    policy_outputs: List[str],
    metadata: List[Dict[str, Any]],
) -> List[float]:
    """Compute blended rewards for each generated completion."""

    rewards: List[float] = []
    for output, meta in zip(policy_outputs, metadata):
        timeout_s = int(meta.get("timeout_s", 2) or 2)
        memory_mb = int(meta.get("memory_mb", 256) or 256)
        tests = meta.get("tests", [])
        score, _ = blended_reward(output, tests, {"timeout_s": timeout_s, "memory_mb": memory_mb})
        rewards.append(score)
    return rewards


class PromptDataset:
    """Dataset wrapper used by the GRPO trainer."""

    def __init__(self, hf_dataset, tokenizer) -> None:
        self._dataset = hf_dataset
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._dataset[idx]
        tests = list(row.get("tests", [])) if isinstance(row.get("tests"), list) else []
        messages = build_prompt(row)
        prompt = format_for_model(messages, self._tokenizer)
        return {
            "prompt": prompt,
            "tests": tests,
            "timeout_s": int(row.get("timeout_s", 2) or 2),
            "memory_mb": int(row.get("memory_mb", 256) or 256),
        }


def main() -> None:
    """Entry point for GSPO training."""

    params = PARAMS
    azr = params["azr"]
    rlhf = azr["rlhf"]
    training_cfg = params.get("training", {})

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

    wrapped = PromptDataset(dataset, tokenizer)

    per_device_bs = max(1, int(rlhf["num_generations"]))
    config = GRPOConfig(
        output_dir=training_cfg.get("log_dir", "/opt/azr/runs/rl"),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=int(rlhf["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg.get("lr", 1e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        logging_steps=10,
        save_steps=100,
        num_generations=int(rlhf["num_generations"]),
        max_prompt_length=int(rlhf["max_prompt_length"]),
        max_completion_length=int(rlhf["max_completion_length"]),
        importance_sampling_level=str(rlhf["importance_sampling_level"]),
        epsilon=float(rlhf.get("clip_range_ratio", 0.1)),
        bf16=bool(rlhf.get("bf16", False)),
        temperature=float(rlhf.get("temperature", 0.7)),
        top_p=float(rlhf.get("top_p", 0.95)),
        use_vllm=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=wrapped,
        reward_funcs=[reward_fn],
    )
    trainer.train()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
