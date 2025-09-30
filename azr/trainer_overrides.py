from __future__ import annotations

from typing import Any

from trl.trainer.grpo_trainer import GRPOTrainer


class GRPOTrainerWithStop(GRPOTrainer):
    """GRPOTrainer override that respects gradient-checkpoint settings when generating."""

    def _prepare_generation_inputs(self, tokenizer, **gen_kwargs: Any):  # type: ignore[override]
        kwargs = super()._prepare_generation_inputs(tokenizer, **gen_kwargs)
        if getattr(self.model, "gradient_checkpointing", False):
            kwargs["use_cache"] = False
        else:
            kwargs.setdefault("use_cache", True)
        return kwargs


__all__ = ["GRPOTrainerWithStop"]
