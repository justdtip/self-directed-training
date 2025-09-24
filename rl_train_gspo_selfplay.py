"""GSPO training entrypoint with self-play reward blending."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import set_seed
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from opt_azr_params import PARAMS
from rewards import blended_reward
from selfplay_manager import SelfPlayManager

SYSTEM_PROMPT = (
    "You are a careful engineer. Provide a correct solution."
    " Include a Python code block if useful and conclude with 'Final answer: ...'."
)


@dataclass
class PromptRecord:
    prompt: str
    metadata: Dict[str, Any]


def build_prompt(task: Dict[str, Any], tokenizer) -> PromptRecord:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.get("prompt", "")},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metadata = {
        "tests": task.get("tests", []),
        "timeout_s": task.get("timeout_s", 2),
        "memory_mb": task.get("memory_mb", 256),
    }
    return PromptRecord(prompt=prompt, metadata=metadata)


def main() -> None:
    params = PARAMS
    azr = params["azr"]
    rlhf = azr["rlhf"]
    training_cfg = params.get("training", {})
    self_play_cfg = params.get("self_play", {})

    set_seed(42)

    max_seq = int(rlhf["max_prompt_length"]) + int(rlhf["max_completion_length"])
    model_id = azr["model_id"]
    load_in_4bit = azr.get("quantization") == "4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=max_seq,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=azr["lora"]["r"],
        target_modules=azr["lora"]["target_modules"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": "/opt/azr/data/train.jsonl"},
        split="train",
    )

    prompts: List[PromptRecord] = [build_prompt(row, tokenizer) for row in dataset]

    class PromptDataset:
        def __len__(self) -> int:
            return len(prompts)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            record = prompts[idx]
            return {"prompt": record.prompt, "metadata": record.metadata}

    per_device_batch = max(1, int(rlhf["num_generations"]))
    grpo_cfg = GRPOConfig(
        output_dir=training_cfg.get("log_dir", "/opt/azr/runs/rl"),
        per_device_train_batch_size=per_device_batch,
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

    sp_enabled = bool(self_play_cfg.get("enabled", False))
    sp_manager = SelfPlayManager(tokenizer=tokenizer, max_seq=max_seq, enabled=sp_enabled)
    sp_weight = float(self_play_cfg.get("weight", 0.2)) if sp_enabled else 0.0

    def reward_fn(batch_prompts, policy_outputs, metadata):
        base_scores: List[float] = []
        for output, meta in zip(policy_outputs, metadata):
            score, _ = blended_reward(
                output,
                meta.get("tests", []),
                {"timeout_s": meta.get("timeout_s", 2), "memory_mb": meta.get("memory_mb", 256)},
            )
            base_scores.append(score)

        if sp_weight > 0.0:
            sp_scores = sp_manager.compute_selfplay_scores(batch_prompts, policy_outputs, metadata)
            final_scores = [
                (1.0 - sp_weight) * base + sp_weight * sp
                for base, sp in zip(base_scores, sp_scores)
            ]
            sp_manager.step_and_maybe_update(model)
            return final_scores
        return base_scores

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=PromptDataset(),
        reward_funcs=[reward_fn],
    )
    trainer.train()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
