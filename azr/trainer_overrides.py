from __future__ import annotations

from typing import Any

from trl.trainer.grpo_trainer import GRPOTrainer


class GRPOTrainerWithStop(GRPOTrainer):
    """GRPOTrainer override that enforces caching for HF generation path."""

    def _prepare_generation_inputs(self, tokenizer, **gen_kwargs: Any):  # type: ignore[override]
        kwargs = super()._prepare_generation_inputs(tokenizer, **gen_kwargs)
        kwargs["use_cache"] = True
        return kwargs


__all__ = ["GRPOTrainerWithStop"]
