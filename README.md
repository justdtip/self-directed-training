AZR RLHF Harness (GRPO + QLoRA)

Quickstart

- Install package: `pip install -e .` (already done in tests)
- Train (tiny smoke test):
  - Create a minimal params file (or use `examples/azr.params.yaml` but swap the model to a tiny one for local smoke):
    - `model_id: sshleifer/tiny-gpt2`
    - `quantization: null`
    - `lora.target_modules: ["c_attn"]`
  - Run: `azr-train --config examples/azr.params.yaml --dataset examples/sample_prompts.jsonl --output outputs/run --max-steps 1`

Notes

- The trainer uses TRL’s GRPOTrainer. The GRPOConfig carries both training args and algorithm params (e.g. `num_generations`, `max_prompt_length`, `importance_sampling_level`). The historical PPO fields like `kl_coeff` are ignored.
- For correctness, `per_device_train_batch_size` is set to `num_generations` so that `generation_batch_size` is divisible by `num_generations`.
- 4-bit quantization is used when `quantization: "4bit"` and bitsandbytes is available; otherwise falls back to full precision.
- The tool sandbox and a minimal web tool are included and tested. They can be extended for agent tool-calling setups.
- Thinking budgets (`policy_budget_tokens`, `opponent_budget_tokens`) consume completion budget. If you raise them, also bump `rlhf.max_completion_length` so policy and opponent answers are not truncated.

CLI

- `azr-train` – run GRPO RLHF according to `--config`.
- `azr-eval` – quick test of tool sandbox and web fetch.
- `azr-gen` – quick generation helper using current config’s model.

Testing

- Run `pytest -q` to validate config parsing, sandbox limits, web fetch, and a one-step GRPO training with a tiny model.
