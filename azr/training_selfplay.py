from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from transformers import set_seed
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer

from .config import AzrModelCfg, AzrSelfPlayCfg, AzrTrainingCfg, load_config
from .data import load_dataset as load_data
from .modeling import load_tokenizer, setup_model
from .rewards import blended_reward
from .selfplay_manager import SelfPlayManager

_DEFAULT_DATA_PATH = "/opt/azr/data/train.jsonl"


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


def build_trainer(config: Any) -> GRPOTrainer:
    """Build and return a GRPOTrainer using self-play and code-execution rewards."""

    cfg_map = _ensure_mapping(config)
    set_seed(42)

    model_cfg = AzrModelCfg.from_dict(cfg_map.get("model", {}))
    sp_cfg = AzrSelfPlayCfg.from_dict(cfg_map.get("self_play", {}))
    train_cfg = AzrTrainingCfg.from_dict(cfg_map.get("training", {}))
    rlhf_cfg = _merge_rlhf(cfg_map.get("rlhf"))

    dataset_path = cfg_map.get("data", {}).get("train_path", _DEFAULT_DATA_PATH)
    samples = load_data(dataset_path)

    tokenizer = load_tokenizer(model_cfg.model_id)
    model = setup_model(model_cfg, bf16=bool(rlhf_cfg["bf16"]))

    max_prompt_len = int(rlhf_cfg["max_prompt_length"])
    max_completion_len = int(rlhf_cfg["max_completion_length"])

    sp_manager: SelfPlayManager | None = None
    if sp_cfg.enabled:
        sp_manager = SelfPlayManager(model_cfg, opponent_device=sp_cfg.device)

    per_device_train_bs = max(1, int(rlhf_cfg["num_generations"]))

    grpo_cfg = GRPOConfig(
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=int(rlhf_cfg["gradient_accumulation_steps"]),
        learning_rate=train_cfg.lr,
        logging_steps=5,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        bf16=bool(rlhf_cfg["bf16"]),
        num_generations=int(rlhf_cfg["num_generations"]),
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        importance_sampling_level=str(rlhf_cfg["importance_sampling_level"]),
        epsilon=float(rlhf_cfg["clip_range_ratio"]),
        temperature=float(rlhf_cfg["temperature"]),
        top_p=float(rlhf_cfg["top_p"]),
    )

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

    def reward_fn(batch_prompts: List[str], policy_outputs: List[str], metadata: List[Dict[str, Any]]) -> List[float]:
        base_scores: List[float] = []
        for output, meta in zip(policy_outputs, metadata):
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

        if sp_cfg.enabled and sp_manager is not None:
            opponent_outputs = sp_manager.generate_opponent(batch_prompts, max_completion_len)
            sp_scores = sp_manager.compute_scores(
                policy_outputs,
                opponent_outputs,
                [meta.get("tests", []) for meta in metadata],
            )
            weight = float(sp_cfg.weight)
            combined = [(1.0 - weight) * base + weight * sp for base, sp in zip(base_scores, sp_scores)]

            if sp_cfg.update_every > 0:
                sp_manager.call_counter += 1
                if sp_manager.call_counter % sp_cfg.update_every == 0:
                    sp_manager.update_opponent(model)
            return combined

        return base_scores

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
    )
    return trainer


def main(config_path: str) -> None:
    config = load_config(config_path)
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "azr/config.json"
    main(path)
