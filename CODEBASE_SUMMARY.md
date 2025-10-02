# Self‑Directed Training Repository Audit

This audit summarises the structure and behaviour of the justdtip/self‑directed‑training repository. The project implements a reinforcement‑learning‑from‑human‑feedback (RLHF) pipeline that fine‑tunes a base language model to solve coding and algorithmic tasks through self‑play and teacher hints. The summary below is organised file by file and highlights cross‑module interactions, training workflow, configuration, and proposed enhancements.

## 1\. Top‑level overview

The repository contains a single Python package azr and various scripts, configuration files and data:

| Type | File(s) | Role |
| --- | --- | --- |
| Documentation | CODEBASE_SUMMARY.md | Detailed narrative of the self‑play training pipeline, modules and future extensions. |
| Scripts | sft_train.py, collect_rollouts.py, rl_train_gspo.py, rl_train_gspo_selfplay.py | Entry points for fine‑tuning and RLHF training. They set up models via LoRA, load datasets, build reward functions, and run training loops. rl_train_gspo_selfplay.pyadds self‑play scoring and uses a remote opponent. |
| Configuration | azr/config.json | Defines model parameters, LoRA hyper‑parameters, RLHF settings, hidden‑reasoning budgets, self‑play weight, remote opponent/teacher providers, teacher‑assist probabilities and data paths[1]. This file is loaded by azr/cli.py and training scripts. |
| Dataset | azr/data/train.jsonl | JSONL dataset of coding/algorithmic tasks with Python unit tests, memory and time limits. Tasks cover arithmetic, dynamic programming, graph and geometry problems. |
| Workflow YAMLs | .github/workflows/*.yml, azr.run.yaml, azr.bootstrap.yamland related files | CI/CD for testing, packaging and environment setup. They install dependencies, bootstrap unsloth, copy parameter files and orchestrate smoke tests and RLHF runs. |
| Testing | tests/test_schema.py, scripts/test_opponent_responses.py, scripts/test_openai_teacher.py | Validate YAML schemas, verify remote opponent and teacher responses, and check that tasks and hints are formatted correctly. |

## 2\. Core package azr

The azr package contains the logic for model setup, prompting, self‑play, rewards and utilities.

### 2.1 Model and configuration

·      **config.py** defines dataclasses for configuration objects. AzrModelCfg stores LoRA rank, alpha, quantisation and target modules. AzrCfg aggregates model, RLHF, thinking, self‑play, teacher, data and logging sections.

·      **modeling.py** loads a causal language model, applies PEFT LoRA, and attaches optional adapters according to `ia3_cfg`. When features are enabled it wires in IA³ scalar/head gates, grouped FFN gates, attention-logit gates, projection and input gates, residual stream gates, RoPE scalers, LoRA-MLP down adapters, soft prompt embeddings, BitFit bias tuning, and LayerNorm γ fine-tuning before freezing the remaining weights via `trainable_markers`[[2]](https://github.com/justdtip/self-directed-training/blob/main/azr/modeling.py#L88-L239).

·      **adapters.py** implements those adapters: `IA3Gate`, `IA3HeadGate`, `ChannelGate`, `ResidualStreamGate`, `ProjectionDimGate`, `InputProjectionGate`, `AttentionLogitGate`, `RoPEScale`, `LoRAMLPDownAdapter`, and `SoftPromptEmbedding`, each exposing either `scale`/`log_scale` parameters or a `set_warmup_alpha` hook to integrate with warm-up[[3]](https://github.com/justdtip/self-directed-training/blob/main/azr/adapters.py#L7-L259).

·      **data.py** defines a TaskExample dataclass and functions to load JSONL tasks. Each example includes a prompt, unit tests, timeouts and memory limits.

### 2.2 Prompt and assist helpers

*   **prompts\_assist.py** defines a strict **system prompt** requiring the model to return exactly one fenced Python code block followed by a Final answer: line[\[4\]](https://github.com/justdtip/self-directed-training/blob/main/azr/prompts_assist.py#L8-L34). It also contains functions build\_assist\_messages and build\_retry\_messages to construct teacher hints and retry prompts. Hints must avoid code and the Final answer: prefix; non‑ASCII characters are stripped and truncated.
*   **cli.py** provides a Click‑based command‑line interface. The train command loads the configuration, chooses between standard RLHF and self‑play training, manages LoRA checkpoints and logging, and launches a vLLM sidecar when tool‑loop training is enabled.

### 2.3 Training modules

*   **training.py** builds a GRPOTrainer for standard RLHF. After model setup it registers `_EnsureLmHeadDtype`, `UnifiedWarmupCallback`, and (when soft prompts are enabled) `SoftPromptSchedulerCallback` before defining the reward function and dataset pipeline[[5]](https://github.com/justdtip/self-directed-training/blob/main/azr/training.py#L86-L218)[[6]](https://github.com/justdtip/self-directed-training/blob/main/azr/callbacks.py#L54-L173).
*   **training\_selfplay.py** orchestrates the self‑play training loop. Key steps include:
*   **Model setup:** The script reads the configuration, loads the base model via setup\_model, freezes base parameters, and enables input gradients before enabling gradient checkpointing[\[7\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L137-L181).

·      **Prompt dataset:** A PromptDataset renders tasks using the system prompt and optionally inserts hidden reasoning inside <think> tags when policy\_enable\_thinking or opponent\_enable\_thinking are true in the configuration[\[1\]](https://github.com/justdtip/self-directed-training/blob/main/azr/config.json#L23-L60).

·      **Self‑play loop:** For each batch, the policy model generates a completion with hidden reasoning tokens and returns a code solution. The reward function reward\_fn calls blended\_reward from rewards.py to score code execution and formatting; if self‑play is enabled, it also queries a remote opponent via SelfPlayManager, compares scores and adds a self‑play component (weight controlled by self\_play.weight). The reward function handles teacher hints and retry logic—calling the teacher on unsolved or opponent‑only cases and retraining if the hint leads to a correct solution[\[8\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L402-L515).
·      **Parameter filtering and warm-up:** Only parameters whose names include the known adapter markers (LoRA, IA³ gates, grouped FFN gates, attention-logit gates, projection/input gates, residual gates, RoPE, LayerNorm, LoRA-MLP, soft prompts, BitFit) remain trainable, and the unified warm-up/soft prompt callbacks are registered when applicable, keeping ramp schedules and count summaries consistent[[9]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L146-L285).

*   **Logging and scoreboard:** The script logs all generations (policy, opponent, teacher) to JSONL, updates the scoreboard via scoreboard.update\_batch and writes results at the end of training.
*   **selfplay\_manager.py** manages remote interactions. It instantiates a remote client (Together AI or OpenAI) from configuration, enforces concurrency limits, tracks token usage and provides generate\_opponent and teacher\_hint methods. The opponent is always called using the **original user question** rather than the policy’s prompt, ensuring outputs are not truncated and self‑play scores are fair.
*   **rewards.py** implements reward functions. It validates that completions contain one code block and a final answer line[\[10\]](https://github.com/justdtip/self-directed-training/blob/main/azr/rewards.py#L12-L28), extracts the last code block[\[11\]](https://github.com/justdtip/self-directed-training/blob/main/azr/rewards.py#L30-L58), executes it in a sandbox, and blends the pass/fail signal with formatting penalties and self‑play scores. The blended reward encourages concise, correct solutions and punishes format violations.
*   **scoreboard.py** maintains a thread‑safe scoreboard recording counts of tasks solved by the policy, solved by the opponent, solved by both, or solved by neither. When writing results it computes pass rates and Wilson confidence intervals[\[12\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L14-L28)[\[13\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L76-L125).
*   **logging\_io.py****,** **generation\_logger.py****,** **failure\_logger.py** implement asynchronous JSONL and plain‑text logging. They ensure logs are flushed even if the run crashes.
*   **codegate.py** checks that Python code blocks meet the required signature and uses nsjail or local execution to sandbox user code. It is called by rewards.py.

### 2.4 Remote providers and tool loop

*   **openai\_provider.py** and **opponent\_provider\_together.py** define asynchronous clients for remote models (OpenAI or Together AI). They build request payloads, handle timeouts and concurrency, and return completions or hints.
*   **tool\_loop.py** implements a tool‑calling agent that can call functions or the web during training. It is not used in the current self‑play pipeline but is used by collect\_rollouts.py to generate transcripts for unsloth training.

## 3\. Training workflow

1.  **Configuration:** The primary configuration file azr/config.json sets LoRA parameters (lora\_r, lora\_alpha, target modules), quantisation and device map. The thinking section controls hidden‑reasoning budgets and enables thinking for both policy and opponent models[\[1\]](https://github.com/justdtip/self-directed-training/blob/main/azr/config.json#L23-L60). The self\_play section enables self‑play and defines the opponent provider (e.g., Together AI with a specific model, concurrency and sampling temperature). The teacher block configures the teacher (e.g., OpenAI GPT‑5) and hint termination conditions; teacher\_assist defines when hints are sampled and the maximum hint length. This file is passed to azr/cli.py and training scripts.
2.  **Invocation:** Running python -m azr.cli train --config azr/config.json loads the configuration, sets seeds and logging. The CLI chooses between standard RLHF (training.py) and self‑play RLHF (training\_selfplay.py) based on the presence of the self\_play.enabled flag.
3.  **Model and dataset setup:** build\_trainer (in training.py or training\_selfplay.py) loads the base model via modeling.setup\_model, applies LoRA adapters, optionally prepares the model for k‑bit training, and loads the training data using data.load\_jsonl. For self‑play, gradient checkpointing is enabled after calling model.enable\_input\_require\_grads() to ensure at least one tensor has requires\_grad=True[\[7\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L137-L181).
4.  **Self‑play loop:** For each batch of prompts, the trainer calls the policy model to generate a code solution. If self‑play is enabled, the SelfPlayManager queries the opponent using the original prompt (with extended token budget) and obtains an answer. The reward function reward\_fn calls rewards.blended\_reward on the policy output and the opponent output; it may also call the teacher to generate a hint and perform a retry if the task is unsolved by both or solved only by the opponent[\[8\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L402-L515).
5.  **Optimisation:** Rewards from the policy, opponent and teacher are combined into a final loss. LoRA weights and any enabled adapters (IA³ gates, grouped FFN gates, attention-logit gates, projection/input gates, residual gates, RoPE scalers, LoRA-MLP adapters, soft prompts, BitFit biases, LayerNorm γ) remain trainable while base model weights stay frozen.
6.  **Logging and evaluation:** All generations are logged to trainer\_output/generations.\*.jsonl with metadata (prompt, completion, scores, thinking tokens). The scoreboard is updated after each batch and written at the end of training. Test scripts and CI workflows run smoke tests and check for formatting, concurrency and remote API errors.

## 4\. Gaps and blindspots relative to the provided overview

The user’s initial summary proposed new modules and behaviours not present in the repository. This audit clarifies the following points:

1.  **Adapter stack implemented:** The repository now provides `azr/adapters.py` and configuration-driven attachment of IA³ gates, grouped FFN gates, attention-logit gates, projection/input gates, residual gates, RoPE scaling, LoRA-MLP adapters, soft prompts, BitFit biases, and LayerNorm γ. Earlier reports that these modules were missing are outdated; future audits should verify the relevant config toggles instead of assuming absence[[2]](https://github.com/justdtip/self-directed-training/blob/main/azr/modeling.py#L88-L239)[[3]](https://github.com/justdtip/self-directed-training/blob/main/azr/adapters.py#L7-L259).
2.  **No** **train\_model.py****:** Model loading occurs in modeling.py and training\_selfplay.py; there is no file named train\_model.py. The earlier summary incorrectly referenced such a file.
3.  **Hidden‑reasoning budgets:** The configuration file includes budgets for policy\_budget\_tokens, opponent\_budget\_tokens and teacher\_budget\_tokens, and flags policy\_enable\_thinking and opponent\_enable\_thinking[\[1\]](https://github.com/justdtip/self-directed-training/blob/main/azr/config.json#L23-L60). These budgets control the number of tokens allowed for thinking segments and must be large enough to cover the hidden reasoning and final answers.
4.  **Opponent prompt construction:** training\_selfplay.py rebuilds the opponent prompt from the original user question rather than using the policy’s templated prompt. This ensures the opponent output is not truncated and yields non‑zero wins. The earlier summary mentioned this fix but failed to highlight that it is implemented in the repository.
5.  **LoRA parameter freezing and gradient checkpointing:** After loading LoRA, the code calls model.enable\_input\_require\_grads() and model.gradient\_checkpointing\_enable()[\[7\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L137-L181). Warnings about requires\_grad=False are resolved by ensuring at least one input requires gradients, and caches are disabled during generation.
6.  **Improved scoreboard:** scoreboard.py pads and truncates pass flag lists to match batch size and writes results in a finally block to prevent missing entries[\[12\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L14-L28)[\[13\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L76-L125).

## 5\. Proposed enhancements (based on user’s outline)

The upstream outline called for richer adapter stacks and diagnostic tooling. With IA³ gates, grouped FFN gates, attention-logit gates, projection/input gates, residual gates, RoPE scalers, LoRA-MLP adapters, soft prompts, BitFit, and unified warm-up now integrated, future work can focus on:

·      Expanding evaluation scripts to log adapter-specific metrics (for example, per-gate scales or soft-prompt drift) and correlating them with reward variance.
·      Experimenting with additional capacity boosters (e.g., gated micro-experts or specialised prompt libraries) while leveraging the existing trainable-marker and warm-up infrastructure.
·      Hardening CI smoke tests to cover multiple adapter configurations (baseline LoRA, IA³ only, combined IA³ + LoRA-MLP + soft prompts) to catch regressions early.

## 6\. Coding guidance and best practices

·      **Parameter freezing:** Never modify base model parameters directly. Use peft.get\_peft\_model to apply LoRA and freeze the original weights. Unfreeze only LoRA (and any future adapter) parameters.

·      **Enable input grads before gradient checkpointing:** Always call model.enable\_input\_require\_grads() before model.gradient\_checkpointing\_enable()[\[7\]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L137-L181).

·      **Avoid** **use\_cache=True** **during training:** Caches are disabled when gradient checkpointing is enabled; passing use\_cache=True will be ignored and may produce warnings.

·      **Prompt construction:** Always use the original user prompt for opponent calls and include the system prompt via format\_system\_prompt() from prompts\_assist.py to ensure the correct format.

·      **Teacher hints:** Sanitise hints by removing Final answer: prefixes, stripping non‑ASCII characters and truncating to the configured maximum length. Hints should be concise and avoid providing code blocks.

·      **Scoreboard consistency:** Ensure policy and opponent pass flags lists are padded/truncated to match batch sizes. Always write the scoreboard in a finally block to persist results even if training is interrupted[\[12\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L14-L28)[\[13\]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L76-L125).

·      **Testing:** Run smoke tests after adding new modules to ensure the pipeline still executes end‑to‑end, the scoreboard updates correctly and there are no warnings or crashes.

## 7\. Conclusion

The self‑directed‑training repository implements a sophisticated RLHF framework combining LoRA fine‑tuning, configurable IA³ and gating adapters, soft prompts, self-play against a remote opponent and teacher-assisted retries. The current codebase supports hidden reasoning via <think> tags and provides robust logging, scoreboard tracking and remote client management. With unified warm-up, adapter toggles and parameter filtering in place, engineers can enable new combinations of modules confidently while preserving pipeline stability.

* * *

[[1]](https://github.com/justdtip/self-directed-training/blob/main/azr/config.json#L23-L60) config.json

https://github.com/justdtip/self-directed-training/blob/main/azr/config.json

[[2]](https://github.com/justdtip/self-directed-training/blob/main/azr/modeling.py#L88-L239) modeling.py

https://github.com/justdtip/self-directed-training/blob/main/azr/modeling.py

[[3]](https://github.com/justdtip/self-directed-training/blob/main/azr/adapters.py#L7-L259) adapters.py

https://github.com/justdtip/self-directed-training/blob/main/azr/adapters.py

[[4]](https://github.com/justdtip/self-directed-training/blob/main/azr/prompts_assist.py#L8-L34) prompts_assist.py

https://github.com/justdtip/self-directed-training/blob/main/azr/prompts_assist.py

[[5]](https://github.com/justdtip/self-directed-training/blob/main/azr/training.py#L86-L218) training.py

https://github.com/justdtip/self-directed-training/blob/main/azr/training.py

[[6]](https://github.com/justdtip/self-directed-training/blob/main/azr/callbacks.py#L54-L173) callbacks.py

https://github.com/justdtip/self-directed-training/blob/main/azr/callbacks.py

[[7]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L137-L181) training_selfplay.py

[[8]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L402-L515) training_selfplay.py

[[9]](https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py#L146-L285) training_selfplay.py

https://github.com/justdtip/self-directed-training/blob/main/azr/training_selfplay.py

[[10]](https://github.com/justdtip/self-directed-training/blob/main/azr/rewards.py#L12-L28) rewards.py

[[11]](https://github.com/justdtip/self-directed-training/blob/main/azr/rewards.py#L30-L58) rewards.py

https://github.com/justdtip/self-directed-training/blob/main/azr/rewards.py

[[12]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L14-L28) scoreboard.py

[[13]](https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py#L76-L125) scoreboard.py

https://github.com/justdtip/self-directed-training/blob/main/azr/scoreboard.py