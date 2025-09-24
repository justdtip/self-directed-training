"""Supervised fine-tuning on collected rollouts."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from opt_azr_params import PARAMS

SYSTEM_PROMPT = "You are helping to solve tasks using tools. Provide concise final answers."


def build_pairs(example: Dict[str, object]) -> Dict[str, List[Dict[str, str]]]:
    """Convert rollout records into supervised pairs."""

    transcript = example.get("transcript") or []
    if not isinstance(transcript, list):
        transcript = []
    prompt = str(example.get("prompt", ""))
    final_answer = str(example.get("final_answer", ""))

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": prompt})
    for turn in transcript:
        messages.append({"role": "assistant", "content": str(turn)})
    messages.append({"role": "assistant", "content": final_answer})
    return {"text": messages}


def main() -> None:
    """Run a lightweight SFT loop."""

    params = PARAMS
    azr = params["azr"]
    model_id = azr["model_id"]
    load_in_4bit = azr.get("quantization") == "4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
    )
    lora = azr["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
    )

    dataset = load_dataset("json", data_files={"train": "/opt/azr/data/rollouts.jsonl"})["train"]
    prepared = dataset.map(build_pairs)

    def _tokenize(batch: Dict[str, List[List[Dict[str, str]]]]):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["text"]
        ]
        tokens = tokenizer(
            texts,
            max_length=4096,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    tokenized = prepared.map(_tokenize, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir="/opt/azr/runs/sft",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=bool(azr["rlhf"].get("bf16", False)),
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model("/opt/azr/runs/sft")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
