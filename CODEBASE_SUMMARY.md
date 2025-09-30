# AZR RLHF Harness – Detailed Codebase Summary

## Project Overview
AZR provides a reinforcement-learning-from-human-feedback (RLHF) training harness centered on GRPO and QLoRA fine-tuning, plus a sandboxed tool ecosystem for code-evaluation rewards and web access. It ships CLI entry points, self-play infrastructure, tool execution sandboxes, and automation YAMLs for bootstrapping remote machines.

## Top-Level Files & Scripts
- `README.md` – Quickstart instructions, GRPO notes, CLI usage, test guidance.
- `pyproject.toml` – Build metadata, core dependencies, console scripts (`azr-train`, `azr-eval`, `azr-gen`).
- `requirements.txt` – Runtime dependency list for deployments.
- `pytest.ini` – Defines the `network` marker for web-dependent tests.
- `SPEC_FUNCTIONS.md` – Authoritative specification for tool sandboxing, web utilities, reward functions, and acceptance tests.
- `tool_harness.py` – nsjail-aware Python execution harness returning stdout/stderr/return code dictionaries.
- `tool_loop.py` (root) – Re-exports `azr.tool_loop` for backwards compatibility.
- `rewards.py` (root) – Lightweight reward helpers for GSPO scripts (code execution + bonuses).
- `collect_rollouts.py` – Generates tool-augmented rollouts using the harness and writes JSONL datasets.
- `rl_train_gspo.py` / `rl_train_gspo_selfplay.py` – GRPO entry points (with/without self-play) using Unsloth model loading.
- `run_smoke.py` – Basic sandbox/web tool smoke test.
- `sft_train.py` – Converts rollouts into supervised chat pairs and runs a PEFT-aware Trainer loop.
- `opt_azr_params.py` – Resolves `azr.params.yaml` from env or canonical paths for scripts/tests.
- `selfplay_manager.py`, `tool_loop.py` (root) – Compatibility wrappers exporting `azr` implementations.
- `tmp_config.json`, `tmp_tiny_config.json` – Tiny-model configs for smoke tests.
- `logs/` – Example runtime log output (not code).

## Core Package `azr/`
- `__init__.py` – Re-exports `load_config` for convenience.
- `callbacks.py` – `TrainingMetricsCallback` streams trainer logs to console and JSONL at runtime.
- `cli.py` – Click CLI that selects standard vs self-play training, handles log teeing, dataset overrides, checkpoint resume, and optional vLLM sidecar launch.
- `codegate.py` – Utilities to validate generated Python code (extract last block, ensure function signatures) used in teacher retries.
- `config.py` – Dataclasses for model, training, self-play, and vLLM configs; legacy schema normalization; YAML load/save helpers.
- `config.json` – Default full configuration (model ID, GRPO hyperparameters, thinking budgets, remote opponent/teacher setup, logging, dataset).
- `data.py` – JSONL loader returning `DataExample` objects (prompt, tests, resource limits).
- `failure_logger.py` – Asynchronous frontier logger writing `neither` and `opponent_only` cases to JSONL files.
- `generation_logger.py` – Background queue writing prompt/completion telemetry (optionally pretty-printed) with rotation/flush controls.
- `logging_io.py` – Durable file writes (`append_jsonl`, atomic JSON/text) used by loggers and scoreboard.
- `modeling.py` – Loads tokenizer (ensuring pad token), builds PEFT LoRA model, optionally prepares 4-bit quantization via bitsandbytes.
- `opponent_provider_together.py` – Async TogetherAI chat client supporting concurrency, retry, and optional thinking budgets.
- `openai_provider.py` – Async OpenAI Responses client with structured prompt support, usage stats caching, and fallback handling.
- `prompts_assist.py` – Shared prompt templates for policy/opponent modes, thinking flags, teacher hints, and retry instructions.
- `rewards.py` – Strict reward shaping: enforces single code block + final answer, executes tests via sandboxed python, records traces, applies bonuses/penalties.
- `sandbox.py` – `ToolSandbox` dataclass enforcing max tool turns plus simple resource limits metadata.
- `scoreboard.py` – Thread-safe scoreboard tracking policy vs opponent outcomes with Wilson interval reporting to JSON/text.
- `selfplay_manager.py` – Central orchestration for self-play: handles local/remote opponents, teacher hints, LoRA snapshot updates, logging hooks, and score computation.
- `simple_rewards.py` – Basic keyword and length-based reward components used by the standard trainer.
- `tool_loop.py` – Loads tool schema, dispatches calls to sandboxed python/web implementations, and runs multi-turn tool reasoning loops with JSON call detection.
- `tools/__init__.py` – Exposes python/web tool helpers (`run_code`, `WebTool`).
- `tools/python_tool.py` – POSIX-rlimit-based sandbox runner used inside pure-Python environments.
- `tools/web.py` – Streaming DuckDuckGo search and HTTP fetch with readability cleanup, respecting max byte limits.
- `training.py` – Standard GRPO trainer builder: loads model/tokenizer, wraps dataset, defines simple reward function, adds dtype-alignment callback.
- `training_selfplay.py` – Full self-play trainer: integrates thinking budgets, opponent blending, teacher assists, scoreboard updates, telemetry logging, retry gating, and LoRA updates.
- `trainer_overrides.py` – GRPOTrainer subclass disabling KV cache when gradient checkpointing is active.
- `utils.py` – Shared `rich` console instance and device/environment helpers.
- `vllm_launcher.py` – Context manager launching vLLM sidecars, polling health, and ensuring graceful teardown.

## Automation & Deployment YAML
- `azr.bootstrap.yaml` – System provisioning steps (apt packages, Python venv, CUDA PyTorch install, Unsloth stack, Playwright setup).
- `azr.files.yaml` – Declarative file materialization (nsjail wrapper, tool harness, web tool, schema, training script) for remote deployments.
- `azr.symlink.yaml` – Links `azr.params.yaml` into `/opt/azr`.
- `azr.apply-tools.yaml` – Workflow ensuring deps installed, files materialized, params copied, smoke tests run.
- `azr.apply-selfplay.yaml` & `azr.selfplay.impl.yaml` – Materials for deployed self-play scripts/managers.
- `azr.run.yaml` – Orchestrated bootstrap pipeline culminating in a demo training run.
- `azr.run-rl-selfplay.yaml` – One-step run command executing self-play training with environment exports.
- `azr.unsloth.yaml` – Default Unsloth hyperparameters and logging cadence.
- `azr.tools-train.impl.yaml` – Placeholder spec (currently empty `files: []`).

## Datasets & Artifacts
- `azr/data/train.jsonl` – Coding prompts with assert-based unit tests and resource budgets used for reward evaluation.
- `examples/azr.params.yaml` – Reference parameter bundle for remote jobs (model info, sandbox/web settings, hardware description).
- `examples/sample_prompts.jsonl` – Tiny chat dataset for CLI smoke runs.
- `trainer_output/` – Artifacts from sample training runs (model card, adapter weights, tokenizer, metrics, scoreboard, generation logs).

## Runtime Diagnostics Scripts
- `scripts/test_openai_teacher.py` – Samples a training problem and requests an OpenAI teacher hint via `OpenAIResponsesProvider`.
- `scripts/test_opponent_responses.py` – Instantiates `SelfPlayManager` and fetches a single opponent completion using the configured remote provider.

## Test Suite (`tests/`)
- `conftest.py` – Ensures repo and `/opt/azr` live on `sys.path` for imports.
- YAML structure tests (`test_apply_tools_yaml.py`, `test_bootstrap_yaml.py`, `test_run_yaml.py`, `test_symlink_yaml.py`, `test_unsloth_yaml.py`).
- Materialization tests (`test_files_yaml.py`) validating file descriptors, syntax, and schema contents.
- Config/data tests (`test_config.py`, `test_config_newshape.py`, `test_data.py`).
- CLI and trainer smoke tests (`test_cli.py`, `test_trainer_smoke.py`).
- Self-play module tests (`test_selfplay.py`, `test_training_selfplay.py`).
- Reward/sandbox/tool tests (`test_azr_rewards.py`, `test_rewards.py`, `test_tool_harness.py`, `test_python_tool.py`, `test_tool_loop.py`, `test_sandbox_web.py`, `test_web_tool.py`, `test_schema.py`).

## Miscellaneous
- `azr.egg-info/` – Packaging metadata emitted by `pip install -e .`.
- `.git`, `.gitignore`, `.pytest_cache/` – Git and pytest state.
- `logs/`, `outputs/`, `tmp_sb/`, `tmp_config*` – Sample run outputs and temporary configs.

## Key Interactions
1. CLI loads configuration (`azr.config`) and selects appropriate trainer (`azr.training` vs `azr.training_selfplay`), optionally managing vLLM via `azr.vllm_launcher`.
2. Trainers load models/tokenizers (`azr.modeling`), construct datasets (`azr.data`), and compute rewards (`azr.rewards`, `tool_harness.py`).
3. Self-play flows rely on `SelfPlayManager` for opponent completions, teacher hints, and LoRA updates; telemetry flows through `generation_logger`, `failure_logger`, and `scoreboard` helpers.
4. Tool dispatch uses `azr.tool_loop` guards and sandbox runners (`tool_harness.py`, `azr/tools/python_tool.py`, `azr/tools/web.py`) described in `SPEC_FUNCTIONS.md`.
5. Automation YAMLs plus scripts in `scripts/` deliver bootstrap + validation workflows matching test expectations.

