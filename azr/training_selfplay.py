from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from inspect import signature

from transformers import set_seed
from transformers.trainer_callback import TrainerCallback
import torch
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer

_GRPO_SUPPORTS_CLIP_RANGE = "clip_range_ratio" in signature(GRPOConfig).parameters
_GRPO_SUPPORTS_TOKENIZER_ARG = "tokenizer" in signature(GRPOTrainer.__init__).parameters

from .config import AzrModelCfg, AzrSelfPlayCfg, AzrTrainingCfg, load_config
from .data import load_dataset as load_data
from .modeling import load_tokenizer, setup_model
from .rewards import blended_reward
from .selfplay_manager import SelfPlayManager

_DEFAULT_DATA_PATH = "/opt/azr/data/train.jsonl"


class _EnsureLmHeadDtype(TrainerCallback):
    def _align(self, model):
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

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._align(model)
        if model is not None:
            self._align(model)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._align(model)
        return control



def _ensure_mapping(config: Any) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("Configuration must be a mapping")


def _merge_rlhf(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
    if config:
        defaults.update({k: config[k] for k in config if config[k] is not None})
    return defaults


def build_trainer(config: Any, *, max_steps: int | None = None) -> GRPOTrainer:
    """Build and return a GRPOTrainer using self-play and code-execution rewards."""

    cfg_map = _ensure_mapping(config)
    set_seed(42)

    model_cfg = AzrModelCfg.from_dict(cfg_map.get("model", {}))
    sp_cfg = AzrSelfPlayCfg.from_dict(cfg_map.get("self_play", {}))
    train_cfg = AzrTrainingCfg.from_dict(cfg_map.get("training", {}))
    rlhf_cfg = _merge_rlhf(cfg_map.get("rlhf"))

    dataset_path = cfg_map.get("data", {}).get("train_path", _DEFAULT_DATA_PATH)
    samples = load_data(dataset_path)
    if not samples:
        raise ValueError(
            f"No training prompts were loaded from {dataset_path}. Ensure the JSONL file has at least one entry."
        )

    tokenizer = load_tokenizer(model_cfg.model_id)
    bf16_requested = bool(rlhf_cfg["bf16"])
    has_cuda = torch.cuda.is_available()
    supports_bf16 = torch.cuda.is_bf16_supported() if has_cuda else False
    bf16_flag = bf16_requested and has_cuda and supports_bf16
    if bf16_requested and not bf16_flag:
        try:
            from .utils import console
            reason = "CUDA is unavailable" if not has_cuda else "CUDA device lacks bfloat16 support"
            console.print(f"[yellow]bf16 requested but {reason}; defaulting to float32.[/]")
        except Exception:
            pass
    model = setup_model(model_cfg, bf16=bf16_flag)

    max_prompt_len = int(rlhf_cfg["max_prompt_length"])
    max_completion_len = int(rlhf_cfg["max_completion_length"])

    sp_manager: SelfPlayManager | None = None
    log_intersteps = bool(cfg_map.get("log_intersteps"))

    if sp_cfg.enabled:
        sp_manager = SelfPlayManager(
            model_cfg,
            opponent_device=sp_cfg.device,
            log_intersteps=log_intersteps,
        )

    per_device_train_bs = max(1, int(rlhf_cfg["num_generations"]))

    grpo_kwargs = dict(
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=int(rlhf_cfg["gradient_accumulation_steps"]),
        learning_rate=train_cfg.lr,
        logging_steps=5,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        bf16=bf16_flag,
        num_generations=int(rlhf_cfg["num_generations"]),
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        importance_sampling_level=str(rlhf_cfg["importance_sampling_level"]),
        temperature=float(rlhf_cfg["temperature"]),
        top_p=float(rlhf_cfg["top_p"]),
    )
    if max_steps is not None:
        grpo_kwargs["max_steps"] = max(1, int(max_steps))
    clip_value = float(rlhf_cfg["clip_range_ratio"])
    if _GRPO_SUPPORTS_CLIP_RANGE:
        grpo_kwargs["clip_range_ratio"] = clip_value
    else:
        grpo_kwargs["epsilon"] = clip_value
    grpo_cfg = GRPOConfig(**grpo_kwargs)

    class PromptDataset:
        def __init__(self, rows) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            example = self.rows[index]
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful engineer. Provide a correct solution. If coding helps, "
                            "include a final Python code block implementing the solution, then state 'Final answer: ...'."
                        ),
                    },
                    {"role": "user", "content": example.prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            meta = {
                "tests": example.tests,
                "timeout_s": example.timeout_s,
                "memory_mb": example.memory_mb,
            }
            return {"prompt": prompt, "metadata": meta}

    train_dataset = PromptDataset(samples)

    def reward_fn(
        *,
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],  # unused but required by TRL
        metadata: List[Dict[str, Any]] | None = None,
        trainer_state=None,
        **_: Any,
    ) -> List[float]:
        del completion_ids, trainer_state  # not used in reward computation yet
        metadata = metadata or [{} for _ in completions]

        base_scores: List[float] = []
        for output, meta in zip(completions, metadata):
            score, _ = blended_reward(
                output,
                meta.get("tests", []),
                {
                    "timeout_s": meta.get("timeout_s", 2),
                    "memory_mb": meta.get("memory_mb", 256),
                    "stderr": meta.get("stderr", ""),
                },
            )
            base_scores.append(score)
        # --- BEGIN interstep logging ---
        if cfg_map.get("log_intersteps"):
            step = getattr(trainer_state, "global_step", None)
            print(f"[Interstep] global_step={step}")
            if prompts:
                prompt_preview = prompts[0][:120].replace("\n", " ")
                completion_preview = completions[0][:200].replace("\n", " ")
                print(f"[Prompt] {prompt_preview}")
                print(f"[Completion] {completion_preview}")
                print(f"[Score] {base_scores[0] if base_scores else None}")
        # --- END interstep logging ---
        if log_intersteps:
            step = getattr(trainer_state, "global_step", None)
            print(f"[Stage] reward_fn start | step={step} prompts={len(prompts)}")

        if sp_cfg.enabled and sp_manager is not None:
            opponent_outputs = sp_manager.generate_opponent(prompts, max_completion_len)
            tests_per_example = [meta.get("tests", []) for meta in metadata]
            sp_scores = sp_manager.compute_scores(
                completions,
                opponent_outputs,
                tests_per_example,
            )
            weight = float(sp_cfg.weight)
            combined = [(1.0 - weight) * base + weight * sp for base, sp in zip(base_scores, sp_scores)]

            if sp_cfg.update_every > 0:
                sp_manager.call_counter += 1
                if sp_manager.call_counter % sp_cfg.update_every == 0:
                    sp_manager.update_opponent(model)
            if log_intersteps:
                base_preview = base_scores[0] if base_scores else float("nan")
                sp_preview = sp_scores[0] if sp_scores else float("nan")
                combined_preview = combined[0] if combined else float("nan")
                print(
                    "[Stage] reward_fn blend | "
                    f"base={base_preview:.4f} sp={sp_preview:.4f} combined={combined_preview:.4f} "
                    f"weight={weight}"
                )
            return combined

        if log_intersteps and base_scores:
            print(f"[Stage] reward_fn return base score={base_scores[0]:.4f}")
        return base_scores

    # Pass the tokenizer through whichever argument this TRL version supports.
    trainer_kwargs = dict(
        model=model,
        args=grpo_cfg,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
    )
    if _GRPO_SUPPORTS_TOKENIZER_ARG:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)

    if log_intersteps:
        class _StageLoggerCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
                print(f"[Stage] on_step_begin | step={state.global_step} epoch={state.epoch}")
                return control

            def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
                last = state.log_history[-1] if state.log_history else {}
                print(f"[Stage] on_step_end   | step={state.global_step} metrics={last}")
                return control

        trainer.add_callback(_StageLoggerCallback())

        original_generate = trainer.model.generate

        def _logged_generate(*gen_args, **gen_kwargs):
            step = getattr(trainer.state, 'global_step', None)
            print(f"[Stage] Policy generate start | step={step}")
            output = original_generate(*gen_args, **gen_kwargs)
            print(f"[Stage] Policy generate done  | step={step}")
            return output

        trainer.model.generate = _logged_generate  # type: ignore[assignment]
    trainer.add_callback(_EnsureLmHeadDtype())
    return trainer


def main(config_path: str, *, max_steps: int | None = None) -> None:
    config = load_config(config_path)
    trainer = build_trainer(config, max_steps=max_steps)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "azr/config.json"
    main(path)
