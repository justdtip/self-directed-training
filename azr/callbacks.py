import json
import os
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from transformers.trainer_callback import TrainerCallback


class TrainingMetricsCallback(TrainerCallback):
    """Stream metrics to console and persist them to a JSONL file."""

    def __init__(self, out_dir: Optional[str] = None) -> None:
        self.out_dir = out_dir
        self._fp = None

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        run_dir = self.out_dir or getattr(args, "output_dir", "outputs/azr-run")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(run_dir, "metrics.jsonl")
        self._fp = open(path, "a", encoding="utf-8")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return control
        step = getattr(state, "global_step", None)
        keys = (
            "loss",
            "reward",
            "reward_std",
            "entropy",
            "completions/mean_length",
            "completions/clipped_ratio",
        )
        line = " | ".join(f"{k}={logs.get(k)}" for k in keys if k in logs)
        print(f"[Metrics] step={step} | {line}")

        record = {"ts": time.time(), "step": step, **logs}
        if self._fp:
            self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fp.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._fp:
            self._fp.close()
            self._fp = None
        return control


class UnifiedWarmupCallback(TrainerCallback):
    """Gradually scale adapter and gate parameters during warm-up."""

    _MIN_SCALE = 1e-6

    def __init__(self, model: nn.Module, warmup_steps: int) -> None:
        self._warmup_steps = max(1, int(warmup_steps))
        self._alpha_modules: List[nn.Module] = []
        self._scale_params: List[Tuple[nn.Parameter, torch.Tensor]] = []
        self._log_scale_params: List[
            Tuple[nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._layernorm_params: List[Tuple[nn.Parameter, torch.Tensor]] = []

        seen_params: Set[int] = set()

        for module in model.modules():
            if hasattr(module, "set_warmup_alpha"):
                self._alpha_modules.append(module)
                continue

            scale_param = getattr(module, "scale", None)
            if isinstance(scale_param, nn.Parameter):
                param_id = id(scale_param)
                if param_id not in seen_params:
                    self._scale_params.append((scale_param, scale_param.detach().clone()))
                    seen_params.add(param_id)
                continue

            log_scale_param = getattr(module, "log_scale", None)
            if isinstance(log_scale_param, nn.Parameter):
                param_id = id(log_scale_param)
                if param_id not in seen_params:
                    original_log = log_scale_param.detach().clone()
                    target_scale = torch.exp(original_log)
                    min_scale = torch.full_like(target_scale, self._MIN_SCALE)
                    min_log = torch.log(min_scale)
                    self._log_scale_params.append(
                        (log_scale_param, original_log, target_scale, min_log)
                    )
                    seen_params.add(param_id)
                continue

            class_name = module.__class__.__name__.lower()
            is_layernorm = isinstance(module, nn.LayerNorm) or "rmsnorm" in class_name
            if not is_layernorm:
                continue
            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter) and weight.requires_grad:
                param_id = id(weight)
                if param_id not in seen_params:
                    self._layernorm_params.append((weight, weight.detach().clone()))
                    seen_params.add(param_id)

    def _alpha_for_step(self, step: int) -> float:
        if self._warmup_steps <= 0:
            return 1.0
        return max(0.0, min(1.0, step / self._warmup_steps))

    def _apply(self, step: int) -> None:
        alpha = self._alpha_for_step(step)
        with torch.no_grad():
            for module in self._alpha_modules:
                module.set_warmup_alpha(alpha)
            for param, original in self._scale_params:
                param.data.copy_(original * alpha)
            for param, original_log, target_scale, min_log in self._log_scale_params:
                if alpha >= 1.0:
                    param.data.copy_(original_log)
                elif alpha <= 0.0:
                    param.data.copy_(min_log)
                else:
                    scaled = torch.clamp(target_scale * alpha, min=self._MIN_SCALE)
                    param.data.copy_(scaled.log())
            for weight, original in self._layernorm_params:
                weight.data.copy_(original * alpha + (1.0 - alpha))
            for module in self._alpha_modules:
                if hasattr(module, "set_warmup_alpha"):
                    try:
                        module.set_warmup_alpha(alpha)
                    except Exception:
                        continue

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._apply(getattr(state, "global_step", 0))
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._apply(getattr(state, "global_step", 0))
        return control


class SoftPromptSchedulerCallback(TrainerCallback):
    """Manage soft prompt freezing and evaluation toggling."""

    def __init__(self, model: nn.Module, freeze_steps: int) -> None:
        self._model = model
        self._freeze_steps = max(0, int(freeze_steps))

    def _prompts(self) -> Optional[nn.Module]:
        return getattr(self._model, "soft_prompt_embeddings", None)

    def _apply(self, step: int) -> None:
        prompts = self._prompts()
        if prompts is None:
            return
        requires = step >= self._freeze_steps
        prompts.embeds.requires_grad = requires
        if getattr(self._model, "_soft_prompts_eval_disabled", False) and self._model.training:
            self._model._soft_prompts_enabled = True
            self._model._soft_prompts_eval_disabled = False

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._apply(getattr(state, "global_step", 0))
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._apply(getattr(state, "global_step", 0))
        return control

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        prompts = self._prompts()
        if prompts is not None:
            self._model._soft_prompts_enabled = False
            self._model._soft_prompts_eval_disabled = True
        return control


__all__ = [
    "TrainingMetricsCallback",
    "UnifiedWarmupCallback",
    "SoftPromptSchedulerCallback",
]
