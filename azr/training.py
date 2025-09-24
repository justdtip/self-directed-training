from __future__ import annotations

import math
from dataclasses import asdict
from typing import List, Dict, Any
import os

import torch
from trl.trainer.grpo_trainer import GRPOTrainer, GRPOConfig

from .config import RootCfg
from .modeling import load_tokenizer, setup_model
from .data import load_prompts_from_path
from .rewards import keyword_reward, length_penalty, combine_rewards
from .utils import console


def build_trainer(cfg: RootCfg, dataset_path: str | None, output_dir: str, max_steps: int | None = None) -> GRPOTrainer:
    azr = cfg.azr
    tok = load_tokenizer(azr.model_id)
    model = setup_model(azr)

    # Debug: show trainable parameter counts to verify LoRA wiring
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = (trainable_params / max(1, total_params)) * 100.0
        console.print(
            f"[cyan]Params:[/] trainable={trainable_params:,} total={total_params:,} ({pct:.2f}% trainable)"
        )
    except Exception:
        pass

    ds = load_prompts_from_path(dataset_path)

    # Combine training args + algorithm args in a single GRPOConfig
    per_device_train_bs = max(1, azr.rlhf.num_generations)
    lr = getattr(cfg, "training", None).lr if getattr(cfg, "training", None) else 5e-5
    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=azr.rlhf.gradient_accumulation_steps,
        learning_rate=lr,
        logging_steps=1,
        save_steps=10,
        max_steps=max_steps or 10,
        bf16=azr.rlhf.bf16 and torch.cuda.is_available(),
        report_to=[],
        warmup_ratio=getattr(cfg.training, "warmup_ratio", 0.0) if getattr(cfg, "training", None) else 0.0,
        weight_decay=getattr(cfg.training, "weight_decay", 0.0) if getattr(cfg, "training", None) else 0.0,
        # Generation / sampling & algo params
        num_generations=azr.rlhf.num_generations,
        max_prompt_length=azr.rlhf.max_prompt_length,
        max_completion_length=azr.rlhf.max_completion_length,
        importance_sampling_level=azr.rlhf.importance_sampling_level,
        epsilon=azr.rlhf.clip_range_ratio,
        temperature=getattr(azr.rlhf, "temperature", 1.0),
        top_p=getattr(azr.rlhf, "top_p", 1.0),
    )

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Combine simple keyword and length rewards as a placeholder
        kw = keyword_reward(completions)
        lp = length_penalty(completions, max_len=azr.rlhf.max_completion_length)
        rewards = combine_rewards(kw, lp)

        # Optional debug: print reward stats to help diagnose zero-loss issues
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
        processing_class=tok,
        reward_funcs=[reward_fn],  # sequence-level rewards
        args=grpo_cfg,
        train_dataset=ds,
    )
    return trainer
