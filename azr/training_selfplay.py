from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from inspect import signature

from transformers import set_seed
from transformers.trainer_callback import TrainerCallback
import torch
import random
import re
from trl.trainer.grpo_trainer import GRPOConfig

from .config import AzrModelCfg, AzrSelfPlayCfg, AzrTrainingCfg, load_config
from .data import load_dataset as load_data
from .modeling import load_tokenizer, setup_model, set_soft_prompt_role, enable_soft_prompts
from .rewards import blended_reward, score_code_tests
from .selfplay_manager import SelfPlayManager
from .trainer_overrides import GRPOTrainerWithStop
from .callbacks import TrainingMetricsCallback, UnifiedWarmupCallback, SoftPromptSchedulerCallback
from .generation_logger import GenerationLogger
from .failure_logger import FailureLogger
from .scoreboard import ScoreBoard
from .prompts_assist import (
    ASSIST_SYSTEM_STRICT,
    CODE_GATE_SYSTEM_NUDGE,
    build_assist_messages,
    build_retry_messages,
    format_system_prompt,
)
from .codegate import (
    CodeGateError,
    extract_last_python_block,
    has_function_signature,
    infer_function_name,
)
from .logging_io import append_jsonl

_GRPO_SUPPORTS_CLIP_RANGE = "clip_range_ratio" in signature(GRPOConfig).parameters
_GRPO_SUPPORTS_TOKENIZER_ARG = "tokenizer" in signature(GRPOTrainerWithStop.__init__).parameters

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
    model = setup_model(
        model_cfg,
        bf16=bf16_flag,
        ia3_cfg=cfg_map.get("ia3"),
    )
    set_soft_prompt_role(model, 0)
    enable_soft_prompts(model, True)

    # Disable KV caching when gradient checkpointing is enabled later.
    model_config = getattr(model, "config", None)
    if model_config is not None and getattr(model_config, "use_cache", None):
        model_config.use_cache = False

    # Freeze the base model weights and leave only LoRA adapters trainable.
    named_params = getattr(model, "named_parameters", None)
    if callable(named_params):
        for name, param in named_params():
            if any(
                tag in name
                for tag in (
                    "lora_",
                    "loraA",
                    "loraB",
                    "ia3_gate",
                    "ia3_head_gate",
                    "channel_gate",
                    "logit_gate",
                    "attn_residual_gate",
                    "ffn_residual_gate",
                    "projection_gate",
                    "rope_scale",
                    "input_projection_gate",
                    "layernorm.weight",
                    "norm.weight",
                    "mlp_down_adapter",
                    "soft_prompt_embeddings",
                )
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Gradient checkpointing requires inputs/embeddings to retain gradients.
    embedding_layer = None
    try:
        model.enable_input_require_grads()
    except AttributeError:
        embeddings = getattr(model, "get_input_embeddings", None)
        if callable(embeddings):
            embedding_layer = embeddings()
            if embedding_layer is not None and hasattr(embedding_layer, "weight"):
                embedding_layer.weight.requires_grad = True
    else:
        embeddings = getattr(model, "get_input_embeddings", None)
        if callable(embeddings):
            embedding_layer = embeddings()

    if embedding_layer is not None:
        def _set_input_requires_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        handle = embedding_layer.register_forward_hook(_set_input_requires_grad)
        setattr(model, "_input_grad_hook", handle)

    # Enable gradient checkpointing after gradients are wired up.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                pass

    # Summarise parameter counts for visibility.
    total_params = 0
    trainable_params = 0
    per_lora_module: Dict[str, int] = {}
    ia3_scalar_total = 0
    ia3_scalar_per_module: Dict[str, int] = {}
    ia3_head_total = 0
    ia3_head_per_module: Dict[str, int] = {}
    channel_gate_total = 0
    logit_gate_total = 0
    residual_gate_total = 0
    projection_gate_total = 0
    rope_scale_total = 0
    layernorm_gamma_total = 0
    mlp_down_total = 0
    soft_prompt_total = 0
    bitfit_total = 0
    bitfit_per_kind: Dict[str, int] = {}
    bitfit_tokens = {
        ".q_proj.bias": "q_proj",
        ".k_proj.bias": "k_proj",
        ".v_proj.bias": "v_proj",
        ".o_proj.bias": "o_proj",
        ".gate_proj.bias": "gate_proj",
        ".up_proj.bias": "up_proj",
        ".down_proj.bias": "down_proj",
        "lm_head.bias": "lm_head",
    }
    ia3_cfg_local = cfg_map.get("ia3") or {}
    bitfit_cfg = ia3_cfg_local.get("bitfit") or {}
    bitfit_enabled = bool(bitfit_cfg.get("enabled"))
    include_lm_head_bias = bool(bitfit_cfg.get("include_lm_head_bias", False))
    if callable(named_params):
        lora_targets = tuple(getattr(model_cfg, "lora_target_modules", ()) or ())
        ia3_targets = ("q_proj", "k_proj", "v_proj", "o_proj")
        for name, param in named_params():
            if not hasattr(param, "numel"):
                continue
            count = param.numel()
            total_params += count
            if param.requires_grad:
                trainable_params += count
                if "lora" in name:
                    for target in lora_targets:
                        if target and target in name:
                            per_lora_module[target] = per_lora_module.get(target, 0) + count
                            break
                elif "ia3_gate" in name:
                    ia3_scalar_total += count
                    for target in ia3_targets:
                        if target in name:
                            ia3_scalar_per_module[target] = (
                                ia3_scalar_per_module.get(target, 0) + count
                            )
                            break
                elif "ia3_head_gate" in name:
                    ia3_head_total += count
                    for target in ia3_targets:
                        if target in name:
                            ia3_head_per_module[target] = (
                                ia3_head_per_module.get(target, 0) + count
                            )
                            break
                elif "channel_gate" in name:
                    channel_gate_total += count
                elif "logit_gate" in name:
                    logit_gate_total += count
                elif "residual_gate" in name:
                    residual_gate_total += count
                elif "projection_gate" in name:
                    projection_gate_total += count
                elif "rope_scale" in name:
                    rope_scale_total += count
                elif "norm.weight" in name:
                    layernorm_gamma_total += count
                elif "mlp_down_adapter" in name:
                    mlp_down_total += count
                elif "soft_prompt_embeddings" in name:
                    soft_prompt_total += count
                elif bitfit_enabled and param.requires_grad and name.endswith("bias"):
                    kind = None
                    for token, label in bitfit_tokens.items():
                        if token == "lm_head.bias" and not include_lm_head_bias:
                            continue
                        if token in name:
                            kind = label
                            break
                    if kind:
                        bitfit_total += count
                        bitfit_per_kind[kind] = bitfit_per_kind.get(kind, 0) + count
    if total_params:
        pct = (trainable_params / total_params) * 100.0
        print(
            f"[Adapters] Trainable parameters: {trainable_params:,} / {total_params:,} ({pct:.2f}% trainable)"
        )
    if per_lora_module:
        for module_name, count in sorted(per_lora_module.items()):
            print(f"[LoRA]   {module_name}: {count:,} trainable params")
    ia3_total = ia3_scalar_total + ia3_head_total
    if ia3_total:
        print(f"[IA3] Trainable parameters: {ia3_total:,}")
        if ia3_scalar_total:
            for module_name, count in sorted(ia3_scalar_per_module.items()):
                print(f"[IA3]   {module_name}: {count:,} trainable params (scalar)")
        if ia3_head_total:
            for module_name, count in sorted(ia3_head_per_module.items()):
                print(f"[IA3]   {module_name}: {count:,} trainable params (head)")
    if channel_gate_total:
        print(f"[FFN] Trainable parameters: {channel_gate_total:,}")
    if logit_gate_total:
        print(f"[AttnLogit] Trainable parameters: {logit_gate_total:,}")
    if residual_gate_total:
        print(f"[Residual] Trainable parameters: {residual_gate_total:,}")
    if projection_gate_total:
        print(f"[ProjDim] Trainable parameters: {projection_gate_total:,}")
    if rope_scale_total:
        print(f"[RoPEScale] Trainable parameters: {rope_scale_total:,}")
    if layernorm_gamma_total:
        print(f"[LayerNorm] Trainable parameters: {layernorm_gamma_total:,}")
    if mlp_down_total:
        print(f"[LoRA-MLP] Trainable parameters: {mlp_down_total:,}")
    if soft_prompt_total:
        print(f"[SoftPrompt] Trainable parameters: {soft_prompt_total:,}")
    if bitfit_total:
        print(f"[BitFit] Trainable bias parameters: {bitfit_total:,}")
        for kind, count in sorted(bitfit_per_kind.items()):
            print(f"[BitFit]   {kind}: {count:,}")

    max_prompt_len = int(rlhf_cfg["max_prompt_length"])
    base_completion_len = int(rlhf_cfg["max_completion_length"])
    thinking_cfg = cfg_map.get("thinking", {}) or {}
    policy_extra = int(thinking_cfg.get("policy_budget_tokens", 0))
    max_completion_len = base_completion_len + policy_extra
    rlhf_cfg["max_completion_length"] = max_completion_len
    opponent_extra = int(thinking_cfg.get("opponent_budget_tokens", 0))
    opponent_completion_len = base_completion_len + opponent_extra
    policy_enable_thinking = bool(thinking_cfg.get("policy_enable_thinking", False))
    opponent_enable_thinking = bool(thinking_cfg.get("opponent_enable_thinking", False))

    log_intersteps = bool(cfg_map.get("log_intersteps"))
    logging_cfg = cfg_map.get("logging", {}) or {}
    log_executor_output = bool(logging_cfg.get("log_executor_output", False))
    debug_full_logs = bool(logging_cfg.get("debug_full_logs", False))
    exec_store = str(logging_cfg.get("exec_store", "fail_first"))
    exec_max_bytes_val = logging_cfg.get("exec_max_bytes")
    exec_max_bytes = int(exec_max_bytes_val) if exec_max_bytes_val is not None else None
    if debug_full_logs:
        exec_max_bytes = None
    if exec_max_bytes is not None:
        exec_truncate_bytes = exec_max_bytes
    else:
        exec_truncate_bytes = None if debug_full_logs else 4096

    sp_manager: SelfPlayManager | None = None
    if sp_cfg.enabled:
        try:
            sp_manager = SelfPlayManager(
                model_cfg,
                sp_cfg,
                opponent_device=sp_cfg.device,
                log_intersteps=log_intersteps,
                config_map=cfg_map,
            )
        except TypeError:
            # Backwards compatibility for simplified stubs in unit tests.
            sp_manager = SelfPlayManager(model_cfg, opponent_device=sp_cfg.device)

    per_device_train_bs = max(1, int(rlhf_cfg["num_generations"]))

    output_dir = train_cfg.output_dir or cfg_map.get("output_dir") or cfg_map.get("training_output_dir") or "trainer_output"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grpo_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=int(rlhf_cfg["gradient_accumulation_steps"]),
        learning_rate=train_cfg.lr,
        logging_steps=1,
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

    gen_logger: Optional[GenerationLogger] = GenerationLogger(
        out_dir=str(output_path / "generations"),
        gzip_output=False,
        rotate_mb=128,
        flush_every=200,
        redact_prompt=False,
        max_bytes=exec_max_bytes,
        pretty_print=debug_full_logs,
        write_text_log=debug_full_logs,
    )
    metrics_path = output_path / "metrics.jsonl"
    scoreboard = ScoreBoard(out_dir=output_dir)
    failure_logger = FailureLogger(out_dir=output_dir)

    trainer_ref: Optional[GRPOTrainerWithStop] = None

    class PromptDataset:
        _thinking_flags_logged = False

        def __init__(self, rows) -> None:
            self.rows = rows
            system_prompt_text = ASSIST_SYSTEM_STRICT
            sys_msg = {"role": "system", "content": system_prompt_text}
            self._rendered: List[str] = []
            self._opponent_prompts: List[str] = []
            thinking_cfg_local = cfg_map.get("thinking", {}) or {}
            if not PromptDataset._thinking_flags_logged:
                print(
                    f"[Thinking] policy_enable_thinking={policy_enable_thinking} "
                    f"opponent_enable_thinking={opponent_enable_thinking}"
                )
                PromptDataset._thinking_flags_logged = True
            want_template_thinking = bool(thinking_cfg_local.get("enabled", False) or policy_enable_thinking)
            template_thinking_supported: Optional[bool] = None
            logged_status = False

            for ex in self.rows:
                msgs = [sys_msg, {"role": "user", "content": ex.prompt}]
                if want_template_thinking:
                    if template_thinking_supported is None:
                        try:
                            kwargs = {"tokenize": False, "add_generation_prompt": True}
                            if policy_enable_thinking:
                                kwargs["enable_thinking"] = True
                            rendered = tokenizer.apply_chat_template(msgs, **kwargs)
                            template_thinking_supported = True
                            if not logged_status:
                                state = "enabled" if policy_enable_thinking else "disabled"
                                print(f"[Thinking] chat_template.enable_thinking {state} (policy)")
                                logged_status = True
                        except TypeError:
                            template_thinking_supported = False
                            rendered = tokenizer.apply_chat_template(
                                msgs,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            if not logged_status:
                                print(
                                    "[Thinking] chat_template.enable_thinking not supported; continuing without it"
                                )
                                logged_status = True
                    else:
                        if template_thinking_supported:
                            kwargs = {"tokenize": False, "add_generation_prompt": True}
                            if policy_enable_thinking:
                                kwargs["enable_thinking"] = True
                            rendered = tokenizer.apply_chat_template(msgs, **kwargs)
                        else:
                            rendered = tokenizer.apply_chat_template(
                                msgs,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                else:
                    kwargs = {"tokenize": False, "add_generation_prompt": True}
                    if policy_enable_thinking:
                        kwargs["enable_thinking"] = True
                    rendered = tokenizer.apply_chat_template(msgs, **kwargs)
                self._rendered.append(rendered)

                opp_msgs = [
                    {"role": "system", "content": format_system_prompt(allow_thinking=opponent_enable_thinking)},
                    {"role": "user", "content": ex.prompt},
                ]
                opp_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if opponent_enable_thinking:
                    opp_kwargs["enable_thinking"] = True
                try:
                    opponent_prompt = tokenizer.apply_chat_template(opp_msgs, **opp_kwargs)
                except TypeError:
                    opp_kwargs.pop("enable_thinking", None)
                    opponent_prompt = tokenizer.apply_chat_template(opp_msgs, **opp_kwargs)
                self._opponent_prompts.append(opponent_prompt)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            example = self.rows[index]
            prompt = self._rendered[index]
            meta = {
                "tests": example.tests,
                "timeout_s": example.timeout_s,
                "memory_mb": example.memory_mb,
                "orig_prompt": example.prompt,
                "opponent_prompt": self._opponent_prompts[index],
            }
            func_name = infer_function_name(example.tests, example.prompt)
            if func_name:
                meta["func_name"] = func_name
            return {"prompt": prompt, "metadata": meta}

    train_dataset = PromptDataset(samples)

    def reward_fn(
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]] | None = None,
        metadata: List[Dict[str, Any]] | None = None,
        trainer_state=None,
        **_: Any,
    ) -> List[float]:
        nonlocal trainer_ref
        if completion_ids is not None:
            _ = completion_ids
        state = trainer_state
        metadata = metadata or [{} for _ in completions]

        def _normalize_step(value) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _classify_reason(
            stats: Dict[str, Any] | None,
            passed: bool,
            meta: Dict[str, Any] | None = None,
        ) -> Optional[str]:
            if passed:
                return None
            if not isinstance(stats, dict):
                stats = {}

            reason = stats.get("reason")
            if isinstance(reason, str) and reason:
                return reason

            def _reason_from_text(payload: object) -> Optional[str]:
                if not isinstance(payload, str) or not payload:
                    return None
                upper = payload.upper()
                if "TIMEOUT" in upper or "KILLED" in upper:
                    return "timeout"
                if "MEMORYERROR" in upper or "OOM" in upper:
                    return "memory_error"
                if "ASSERT" in upper:
                    return "assertion_failed"
                if "TRACEBACK" in upper or "EXCEPTION" in upper:
                    return "exec_error"
                return None

            traces = stats.get("exec_traces")
            if isinstance(traces, list) and traces:
                primary = traces[0]
                if isinstance(primary, dict):
                    text_reason = _reason_from_text(primary.get("stderr"))
                    if text_reason:
                        return text_reason
                    ret_code = primary.get("returncode")
                    if isinstance(ret_code, int) and ret_code != 0:
                        return "exec_error"

            if isinstance(meta, dict):
                meta_reason = _reason_from_text(meta.get("stderr"))
                if meta_reason:
                    return meta_reason

            passes = stats.get("passes")
            total = stats.get("total")
            if isinstance(total, int):
                if total == 0:
                    return "no_tests"
                if isinstance(passes, int) and passes < total:
                    return "tests_failed"

            base_val = stats.get("base")
            if isinstance(base_val, (int, float)) and base_val < 1.0:
                return "score_below_threshold"

            return None

        step_id = getattr(state, "global_step", None)
        base_scores: List[float] = []
        policy_pass_flags: List[bool] = []
        for idx, (prompt_text, output, meta) in enumerate(zip(prompts, completions, metadata)):
            score, stats = blended_reward(
                output,
                meta.get("tests", []),
                {
                    "timeout_s": meta.get("timeout_s", 2),
                    "memory_mb": meta.get("memory_mb", 256),
                    "stderr": meta.get("stderr", ""),
                },
                collect_exec=log_executor_output,
                exec_store=exec_store,
                exec_max_bytes=exec_truncate_bytes,
            )
            score = max(0.0, min(1.0, score))
            penalties = cfg_map.get("penalties", {}) or {}
            if stats.get("reason") == "format_error":
                fmt_pen = float(penalties.get("format_error_penalty", 0.0))
                if fmt_pen > 0.0:
                    score = max(0.0, score - fmt_pen)
            base_scores.append(score)
            if stats.get("reason") == "format_error":
                policy_pass = False
            else:
                policy_pass = float(stats.get("base", 0.0)) >= 1.0
            policy_pass_flags.append(policy_pass)
            if gen_logger is not None:
                payload = {
                    "ts": time.time(),
                    "source": "policy",
                    "step": step_id,
                    "prompt": prompt_text,
                    "completion": output,
                    "score": score,
                    "passed": policy_pass,
                    "length_tokens": len(output.strip().split()),
                    "metadata": dict(meta),
                }
                if log_executor_output and "exec_traces" in stats:
                    payload["exec_traces"] = stats["exec_traces"]
                if debug_full_logs:
                    payload["user_prompt"] = meta.get("orig_prompt", "")
                gen_logger.log(payload)

            metrics_record = {
                "source": "policy",
                "step": _normalize_step(step_id),
                "passed": bool(policy_pass),
                "score": float(score),
                "func_name": meta.get("func_name"),
                "retries": 0,
            }
            reason = _classify_reason(stats, policy_pass, meta)
            if reason:
                metrics_record["reason"] = reason
            append_jsonl(metrics_path, metrics_record)
        # --- BEGIN interstep logging ---
        if cfg_map.get("log_intersteps"):
            step = getattr(state, "global_step", None)
            print(f"[Interstep] global_step={step}")
            if prompts:
                prompt_preview = prompts[0][:120].replace("\n", " ")
                completion_preview = completions[0][:200].replace("\n", " ")
                print(f"[Prompt] {prompt_preview}")
                print(f"[Completion] {completion_preview}")
                print(f"[Score] {base_scores[0] if base_scores else None}")
        # --- END interstep logging ---
        if log_intersteps:
            step = getattr(state, "global_step", None)
            print(f"[Stage] reward_fn start | step={step} prompts={len(prompts)}")

        if sp_cfg.enabled and sp_manager is not None:
            try:
                opp_enable_thinking = bool(opponent_enable_thinking)
                opponent_prompt_strings: List[str] = []
                for prompt_text, meta in zip(prompts, metadata):
                    user_orig = meta.get("orig_prompt") or ""
                    user_prompt = meta.get("opponent_prompt", user_orig)
                    system_text = format_system_prompt(allow_thinking=opp_enable_thinking)
                    message_bundle = [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_prompt},
                    ]
                    try:
                        rendered = tokenizer.apply_chat_template(
                            message_bundle,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=opp_enable_thinking,
                        )
                    except TypeError:
                        rendered = tokenizer.apply_chat_template(
                            message_bundle,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    opponent_prompt_strings.append(rendered)

                opponent_outputs = sp_manager.generate_opponent(
                    opponent_prompt_strings,
                    opponent_completion_len,
                )
            except Exception as exc:
                print(f"[Warning] Opponent provider failure: {exc}")
                if trainer_ref is not None:
                    try:
                        trainer_ref.save_state()
                        trainer_ref.save_model(trainer_ref.args.output_dir)
                    except Exception:
                        pass
                return base_scores
            tests_per_example = [meta.get("tests", []) for meta in metadata]
            sp_scores = sp_manager.compute_scores(
                completions,
                opponent_outputs,
                tests_per_example,
            )
            weight = float(sp_cfg.weight)
            combined = [(1.0 - weight) * base + weight * sp for base, sp in zip(base_scores, sp_scores)]

            opponent_pass_flags: List[bool] = []
            opponent_trace_stats: List[Dict[str, object]] = []
            assist_cfg = getattr(trainer, "_assist_cfg", {"enabled": False})
            thinking_cfg = getattr(trainer, "_thinking", {"enabled": False})
            teacher_provider = getattr(sp_manager, "teacher_provider", None)

            for idx, (opp_out, meta) in enumerate(zip(opponent_outputs, metadata)):
                opp_base, opp_stats = score_code_tests(
                    opp_out,
                    meta.get("tests", []),
                    timeout_s=meta.get("timeout_s", 2),
                    memory_mb=meta.get("memory_mb", 256),
                    collect_exec=log_executor_output,
                    exec_store=exec_store,
                    exec_max_bytes=exec_truncate_bytes,
                )
                opponent_pass_flags.append(opp_base >= 1.0)
                opponent_trace_stats.append(opp_stats)

                if hasattr(trainer, "failure_logger"):
                    try:
                        base_prompt = prompts[idx]
                        record = {
                            "prompt": base_prompt,
                            "tests": meta.get("tests", []),
                            "timeout_s": meta.get("timeout_s", 2),
                            "memory_mb": meta.get("memory_mb", 256),
                        }
                        policy_pass = policy_pass_flags[idx]
                        opp_pass = opponent_pass_flags[-1]
                        if (not policy_pass) and (not opp_pass):
                            trainer.failure_logger.log(record)
                        elif (not policy_pass) and opp_pass:
                            record = dict(record)
                            record["kind"] = "opponent_only"
                            trainer.failure_logger.log(record)
                    except Exception:
                        pass

                if assist_cfg.get("enabled", False) and teacher_provider is not None:
                    neither = (not policy_pass_flags[idx]) and (not opponent_pass_flags[-1])
                    opp_only = (not policy_pass_flags[idx]) and opponent_pass_flags[-1]
                    trigger = False
                    if neither and random.random() < float(assist_cfg.get("sample_prob_neither", 0.15)):
                        trigger = True
                    elif opp_only and random.random() < float(assist_cfg.get("sample_prob_opp_only", 0.10)):
                        trigger = True

                    if trigger:
                        try:
                            base_prompt = prompts[idx]
                            modes = assist_cfg.get("modes", ["hint", "critique"])
                            mode = modes[0] if modes else "hint"
                            tests_list = meta.get("tests", [])

                            user_orig = meta.get("orig_prompt", base_prompt)
                            assist_user_prompt_parts: List[str] = []
                            assist_user_prompt_parts.append(f"Problem:\n{user_orig}")
                            if tests_list:
                                assist_user_prompt_parts.append("Tests (must all pass):\n" + "\n".join(tests_list))
                            if mode == "critique" and completions[idx]:
                                assist_user_prompt_parts.append("Failed attempt:\n" + completions[idx])
                            assist_user_prompt = "\n\n".join(assist_user_prompt_parts)

                            teacher_messages = build_assist_messages(assist_user_prompt)
                            hint_tokens = int(assist_cfg.get("max_hint_tokens", 256))
                            try:
                                hint_raw = sp_manager.teacher_hint([teacher_messages], hint_tokens)[0]
                            except Exception as e:
                                msg = str(e)
                                if "OpenAI Responses incomplete: max_output_tokens" in msg:
                                    # Retry once with an expanded token budget but enforce an upper bound.
                                    boosted_tokens = min(1024, max(hint_tokens * 2, hint_tokens + 128))
                                    hint_raw = sp_manager.teacher_hint([teacher_messages], boosted_tokens)[0]
                                    hint_tokens = boosted_tokens
                                else:
                                    raise
                            hint_clean = hint_raw.strip()
                            if hint_clean.lower().startswith("final answer:"):
                                hint_clean = hint_clean.split(":", 1)[1].lstrip()
                            json_match = re.search(
                                r'\{"final_answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"\}',
                                hint_clean,
                                flags=re.IGNORECASE,
                            )
                            if json_match:
                                hint_clean = json_match.group(1).encode("utf-8").decode("unicode_escape")
                            hint_clean = "".join(ch for ch in hint_clean if ord(ch) < 128)
                            hint_clean = " ".join(hint_clean.split())
                            if len(hint_clean) > 300:
                                truncated = hint_clean[:300]
                                if " " in truncated:
                                    truncated = truncated[: truncated.rfind(" ")]
                                hint_clean = truncated

                            func_name = meta.get("func_name")
                            max_gate_attempts = 2
                            last_gate_error: str | None = None
                            final_structured_retry: List[Dict[str, object]] | None = None
                            retry_text = ""
                            attempts_used = 0

                            def _flatten_messages(messages: List[Dict[str, object]]) -> List[Dict[str, str]]:
                                flat: List[Dict[str, str]] = []
                                for msg in messages:
                                    role = msg.get("role", "user")
                                    content_items = msg.get("content", [])
                                    if isinstance(content_items, list):
                                        parts: List[str] = []
                                        for item in content_items:
                                            if isinstance(item, dict):
                                                txt = item.get("text") or item.get("value")
                                                if isinstance(txt, str):
                                                    parts.append(txt)
                                            elif isinstance(item, str):
                                                parts.append(item)
                                        content_text = "\n".join(parts)
                                    else:
                                        content_text = str(content_items)
                                    flat.append({"role": str(role), "content": content_text})
                                return flat

                            for gate_attempt in range(max_gate_attempts):
                                structured_retry_messages = build_retry_messages(user_orig, hint_clean)
                                if gate_attempt > 0:
                                    structured_retry_messages.append(
                                        {
                                            "role": "system",
                                            "content": [{"type": "input_text", "text": CODE_GATE_SYSTEM_NUDGE}],
                                        }
                                    )

                                retry_messages = _flatten_messages(structured_retry_messages)
                                try:
                                    kwargs_retry = {
                                        "tokenize": False,
                                        "add_generation_prompt": True,
                                    }
                                    if policy_enable_thinking:
                                        kwargs_retry["enable_thinking"] = True
                                    retry_prompt = tokenizer.apply_chat_template(retry_messages, **kwargs_retry)
                                except TypeError:
                                    retry_prompt = tokenizer.apply_chat_template(
                                        retry_messages,
                                        tokenize=False,
                                        add_generation_prompt=True,
                                    )
                                enc = tokenizer(retry_prompt, return_tensors="pt").to(model.device)
                                retry_temp = float(rlhf_cfg.get("retry_temperature", rlhf_cfg["temperature"]))
                                gen_ids = model.generate(
                                    **enc,
                                    max_new_tokens=int(rlhf_cfg["max_completion_length"]),
                                    do_sample=True,
                                    temperature=retry_temp,
                                    top_p=float(rlhf_cfg["top_p"]),
                                )
                                retry_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                                attempts_used = gate_attempt + 1

                                try:
                                    code_block = extract_last_python_block(retry_text)
                                    if not code_block:
                                        raise CodeGateError("Missing Python code block in assist output.")
                                    if func_name and not has_function_signature(code_block, func_name):
                                        raise CodeGateError(f"Function '{func_name}' not found in code block.")
                                except CodeGateError as gate_exc:
                                    last_gate_error = str(gate_exc)
                                    final_structured_retry = structured_retry_messages
                                    if gate_attempt + 1 >= max_gate_attempts:
                                        break
                                    continue
                                else:
                                    last_gate_error = None
                                    final_structured_retry = structured_retry_messages
                                    break

                            if final_structured_retry is None:
                                final_structured_retry = structured_retry_messages

                            retries_used = attempts_used - 1 if attempts_used > 0 else 0

                            retry_score, retry_stats = blended_reward(
                                retry_text,
                                meta.get("tests", []),
                                {
                                    "timeout_s": meta.get("timeout_s", 2),
                                    "memory_mb": meta.get("memory_mb", 256),
                                    "stderr": meta.get("stderr", ""),
                                },
                                collect_exec=log_executor_output,
                                exec_store=exec_store,
                                exec_max_bytes=exec_truncate_bytes,
                            )

                            retry_score = max(0.0, min(1.0, retry_score))
                            completions[idx] = retry_text
                            base_scores[idx] = retry_score
                            policy_pass_flags[idx] = float(retry_stats.get("base", 0.0)) >= 1.0
                            combined[idx] = (1.0 - weight) * base_scores[idx] + weight * sp_scores[idx]

                            status = "pass" if policy_pass_flags[idx] else "fail"
                            print(f"[Assist] step={step_id} index={idx} mode={mode} status={status}")

                            if gen_logger is not None:
                                prompt_log_value = prompts[idx] if debug_full_logs else base_prompt[:200]
                                event = {
                                    "ts": time.time(),
                                    "source": "assist",
                                    "step": step_id,
                                    "prompt": prompt_log_value,
                                    "hint": hint_clean,
                                    "retry": retry_text,
                                    "score": retry_score,
                                    "passed": policy_pass_flags[idx],
                                }
                                if debug_full_logs:
                                    event["retry_prompt"] = retry_prompt
                                    event["assist_request_messages"] = teacher_messages
                                    event["assist_user_prompt"] = assist_user_prompt
                                    event["retry_messages"] = final_structured_retry
                                    event["hint_mode"] = mode
                                    if last_gate_error:
                                        event["codegate_error"] = last_gate_error
                                if log_executor_output and "exec_traces" in retry_stats:
                                    event["exec_traces"] = retry_stats["exec_traces"]
                                gen_logger.log(event)

                            metrics_record = {
                                "source": "assist",
                                "step": _normalize_step(step_id),
                                "passed": bool(policy_pass_flags[idx]),
                                "score": float(retry_score),
                                "func_name": meta.get("func_name"),
                                "retries": int(max(0, retries_used)),
                            }
                            assist_reason = _classify_reason(retry_stats, policy_pass_flags[idx], meta)
                            if assist_reason:
                                metrics_record["reason"] = assist_reason
                            append_jsonl(metrics_path, metrics_record)
                        except Exception as assist_exc:
                            print(f"[Assist] warning: {assist_exc}")

            for (
                prompt_text,
                opp_completion,
                meta,
                base_score,
                sp_score,
                combined_score,
                opp_pass,
                opp_stats,
            ) in zip(
                prompts,
                opponent_outputs,
                metadata,
                base_scores,
                sp_scores,
                combined,
                opponent_pass_flags,
                opponent_trace_stats,
            ):
                if gen_logger is not None:
                    payload = {
                        "ts": time.time(),
                        "source": "opponent",
                        "step": step_id,
                        "prompt": prompt_text,
                        "completion": opp_completion,
                        "score": sp_score,
                        "policy_score": base_score,
                        "combined_score": combined_score,
                        "weight": weight,
                        "passed": opp_pass,
                        "length_tokens": len(opp_completion.strip().split()),
                        "metadata": dict(meta),
                    }
                    if log_executor_output and "exec_traces" in opp_stats:
                        payload["exec_traces"] = opp_stats["exec_traces"]
                    gen_logger.log(payload)

                metrics_record = {
                    "source": "opponent",
                    "step": _normalize_step(step_id),
                    "passed": bool(opp_pass),
                    "score": float(sp_score),
                    "func_name": meta.get("func_name"),
                    "retries": 0,
                }
                opponent_reason = _classify_reason(opp_stats, opp_pass, meta)
                if opponent_reason:
                    metrics_record["reason"] = opponent_reason
                append_jsonl(metrics_path, metrics_record)

            if scoreboard is not None and policy_pass_flags:
                try:
                    expected = len(prompts)

                    if len(policy_pass_flags) != expected:
                        print(
                            "[Scoreboard] WARNING: policy flag length mismatch "
                            f"policy={len(policy_pass_flags)} expected={expected}"
                        )
                        if len(policy_pass_flags) < expected:
                            policy_pass_flags.extend([False] * (expected - len(policy_pass_flags)))
                        else:
                            del policy_pass_flags[expected:]

                    if len(opponent_pass_flags) != expected:
                        print(
                            "[Scoreboard] WARNING: opponent flag length mismatch "
                            f"opponent={len(opponent_pass_flags)} expected={expected}"
                        )
                        if len(opponent_pass_flags) < expected:
                            opponent_pass_flags.extend([False] * (expected - len(opponent_pass_flags)))
                        else:
                            del opponent_pass_flags[expected:]
                    scoreboard.update_batch(policy_pass_flags, opponent_pass_flags)
                    scoreboard.write()
                except Exception as sb_exc:
                    print(f"[Scoreboard] ERROR during update: {sb_exc}")

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
        if scoreboard is not None and policy_pass_flags and not (sp_cfg.enabled and sp_manager is not None):
            try:
                expected = len(prompts)
                if len(policy_pass_flags) != expected:
                    print(
                        "[Scoreboard] WARNING: policy flag length mismatch "
                        f"policy={len(policy_pass_flags)} expected={expected}"
                    )
                    if len(policy_pass_flags) < expected:
                        policy_pass_flags.extend([False] * (expected - len(policy_pass_flags)))
                    else:
                        del policy_pass_flags[expected:]

                opponent_flags = [False] * expected
                scoreboard.update_batch(policy_pass_flags, opponent_flags)
                scoreboard.write()
            except Exception as sb_exc:
                print(f"[Scoreboard] ERROR during solo update: {sb_exc}")

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
    ia3_cfg = cfg_map.get("ia3") or {}
    warm_steps_config = cfg_map.get("warmup_steps")
    warm_steps_value = warm_steps_config if warm_steps_config is not None else ia3_cfg.get("warmup_steps", 0)
    warm_steps = int(warm_steps_value or 0)
    if hasattr(trainer, "add_callback") and warm_steps > 0:
        trainer.add_callback(UnifiedWarmupCallback(model, warm_steps))
    if hasattr(trainer, "add_callback") and getattr(model, "_soft_prompt_wrapped", False):
        freeze_steps = getattr(model, "soft_prompt_freeze_steps", 0)
        trainer.add_callback(SoftPromptSchedulerCallback(model, int(freeze_steps or 0)))
    trainer.gen_logger = gen_logger
    trainer.scoreboard = scoreboard
    trainer.log_options = {
        "log_executor_output": log_executor_output,
        "exec_store": exec_store,
        "exec_max_bytes": exec_max_bytes,
        "exec_truncate_bytes": exec_truncate_bytes,
        "debug_full_logs": debug_full_logs,
    }
    trainer.failure_logger = failure_logger
    trainer._thinking = cfg_map.get("thinking", {"enabled": False})
    trainer._assist_cfg = cfg_map.get("teacher_assist", {"enabled": False})
    trainer_ref = trainer
    if sp_manager is not None:
        sp_manager.gen_logger = gen_logger
        sp_manager.trainer = trainer
        trainer.sp_manager = sp_manager
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(TrainingMetricsCallback(out_dir=getattr(grpo_cfg, "output_dir", None)))
        if log_intersteps and hasattr(trainer, "add_callback"):
            class _StageLoggerCallback(TrainerCallback):
                def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
                    print(f"[Stage] on_step_begin | step={state.global_step} epoch={state.epoch}")
                    return control

            def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
                last = state.log_history[-1] if state.log_history else None
                if last:
                    print(f"[Stage] on_step_end   | step={state.global_step} metrics={last}")
                else:
                    print(f"[Stage] on_step_end   | step={state.global_step} metrics=<pending>")
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
                if logs:
                    print(f"[Stage] on_log        | step={state.global_step} logs={logs}")
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
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(_EnsureLmHeadDtype())
    return trainer


def main(config_path: str, *, max_steps: int | None = None) -> None:
    config = load_config(config_path)
    trainer = build_trainer(config, max_steps=max_steps)
    try:
        trainer.train()
    finally:
        if hasattr(trainer, "gen_logger"):
            try:
                trainer.gen_logger.close()
            except Exception:
                pass
        if hasattr(trainer, "failure_logger"):
            try:
                trainer.failure_logger.close()
            except Exception:
                pass
        if hasattr(trainer, "scoreboard"):
            try:
                trainer.scoreboard.write()
            except Exception as e:
                print(f"[Scoreboard] final write failed: {e}")


    if __name__ == "__main__":  # pragma: no cover
        import sys

        path = sys.argv[1] if len(sys.argv) > 1 else "azr/config.json"
        main(path)


# Backwards compatibility for tests expecting this symbol.
GRPOTrainer = GRPOTrainerWithStop
