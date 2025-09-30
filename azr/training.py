from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer
from transformers.trainer_callback import TrainerCallback


class _EnsureLmHeadDtype(TrainerCallback):
    """Align the lm_head dtype with the rest of the model to avoid Float/BFloat16 mismatches."""

    def _align(self, model) -> None:
        if not hasattr(model, "lm_head"):
            return
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            return

        lm_head = model.lm_head
        weight = getattr(lm_head, "weight", None)
        if weight is not None and weight.dtype != param_dtype:
            weight.data = weight.data.to(param_dtype)
        bias = getattr(lm_head, "bias", None)
        if bias is not None and bias.dtype != param_dtype:
            bias.data = bias.data.to(param_dtype)

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is not None:
            self._align(model)
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is not None:
            self._align(model)
        return control

from .config import AzrModelCfg, AzrTrainingCfg
from .data import DataExample, load_dataset
from .modeling import load_tokenizer, setup_model
from .simple_rewards import combine_rewards, keyword_reward, length_penalty
from .codegate import infer_function_name
from .utils import console
from .prompts_assist import format_system_prompt


def _merge_rlhf(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "num_generations": 4,
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "importance_sampling_level": "sequence",
        "clip_range_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "bf16": False,
    }
    if cfg:
        defaults.update({k: cfg[k] for k in cfg if cfg[k] is not None})
    return defaults


def _default_examples() -> List[DataExample]:
    return [
        DataExample(
            prompt="Explain what RLHF is in one sentence.",
            tests=[],
            timeout_s=2,
            memory_mb=256,
        ),
        DataExample(
            prompt="List two benefits of LoRA for LLM fine-tuning.",
            tests=[],
            timeout_s=2,
            memory_mb=256,
        ),
    ]


def build_trainer(
    config: Dict[str, Any],
    dataset_path: Optional[str],
    output_dir: str,
    max_steps: Optional[int] = None,
) -> GRPOTrainer:
    """Build the standard GRPO trainer using the simplified configuration schema."""

    model_cfg = AzrModelCfg.from_dict(config.get("model", {}))
    training_cfg = AzrTrainingCfg.from_dict(config.get("training", {}))
    rlhf_cfg = _merge_rlhf(config.get("rlhf"))

    if not model_cfg.model_id:
        raise ValueError("model.model_id must be specified in the configuration")

    tokenizer = load_tokenizer(model_cfg.model_id)
    model = setup_model(
        model_cfg,
        bf16=bool(rlhf_cfg["bf16"]),
        ia3_cfg=config.get("ia3"),
    )

    # Debug information about parameters to confirm LoRA wiring
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = (trainable_params / max(1, total_params)) * 100.0
        console.print(
            f"[cyan]Params:[/] trainable={trainable_params:,} total={total_params:,} ({pct:.2f}% trainable)"
        )
    except Exception:
        pass

    if dataset_path:
        examples = load_dataset(dataset_path)
    else:
        examples = []

    if not examples:
        examples = _default_examples()

    class PromptDataset:
        def __init__(self, rows: List[DataExample]) -> None:
            self.rows = rows
            self._system_prompt = format_system_prompt(False)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            ex = self.rows[idx]
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": ex.prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "tests": ex.tests,
                "timeout_s": ex.timeout_s,
                "memory_mb": ex.memory_mb,
                "func_name": infer_function_name(ex.tests, ex.prompt),
            }

    dataset = PromptDataset(examples)

    per_device_train_bs = max(1, int(rlhf_cfg["num_generations"]))
    bf16_flag = bool(rlhf_cfg["bf16"]) and torch.cuda.is_available()

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=int(rlhf_cfg["gradient_accumulation_steps"]),
        learning_rate=training_cfg.lr,
        logging_steps=1,
        save_steps=10,
        max_steps=max_steps or 10,
        bf16=bf16_flag,
        report_to=[],
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        num_generations=int(rlhf_cfg["num_generations"]),
        max_prompt_length=int(rlhf_cfg["max_prompt_length"]),
        max_completion_length=int(rlhf_cfg["max_completion_length"]),
        importance_sampling_level=str(rlhf_cfg["importance_sampling_level"]),
        epsilon=float(rlhf_cfg["clip_range_ratio"]),
        temperature=float(rlhf_cfg["temperature"]),
        top_p=float(rlhf_cfg["top_p"]),
    )

    def reward_fn(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        kw = keyword_reward(completions)
        lp = length_penalty(completions, max_len=int(rlhf_cfg["max_completion_length"]))
        rewards = combine_rewards(kw, lp)

        if os.getenv("AZR_DEBUG_REWARDS"):
            try:
                n = len(rewards)
                mean = sum(rewards) / max(1, n)
                var = sum((r - mean) ** 2 for r in rewards) / max(1, n - 1)
                unique = len(set(round(r, 6) for r in rewards))
                console.print(
                    f"[magenta]Reward stats:[/] n={n} mean={mean:.4f} var={var:.6f} unique={unique}"
                )
            except Exception:
                pass
        return rewards

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=dataset,
    )
    # Align lm_head dtype with the model's parameters to avoid Float/BFloat16 errors.
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(_EnsureLmHeadDtype())
    return trainer


__all__ = ["build_trainer"]
