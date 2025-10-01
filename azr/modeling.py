from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional
import math

import torch
import torch.nn as nn

from .config import AzrModelCfg
from .utils import console
from .adapters import (
    attach_attention_logit_gates,
    attach_ffn_gates,
    attach_ia3_gates,
    attach_per_layer_head_gates,
    attach_projection_dim_gates,
    attach_residual_gates,
    attach_rope_scale,
    SoftPromptEmbedding,
    attach_lora_mlp_down,
)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        # For causal models, set pad to eos
        tok.pad_token = tok.eos_token
    return tok


def setup_model(
    cfg: AzrModelCfg,
    *,
    bf16: bool = False,
    target_modules: Optional[Iterable[str]] = None,
    ia3_cfg: Optional[Mapping[str, Any]] = None,
) -> object:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = cfg.quantization or ""
    use_4bit = quant.lower() == "4bit"

    bnb_config = None
    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            )
        except Exception as e:
            console.print(f"[yellow]bitsandbytes not available or incompatible: {e}. Falling back to full precision.[/]")
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else None,
        device_map=cfg.device_map if cfg.device_map is not None else ("auto" if bnb_config is not None else None),
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    modules_source = target_modules
    if modules_source is None and getattr(cfg, "lora_target_modules", None):
        modules_source = cfg.lora_target_modules
    modules = list(modules_source) if modules_source is not None else ["q_proj", "v_proj"]
    console.print(f"LoRA target modules: {json.dumps(modules)}")
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    # After adding LoRA adapters, cast the whole model to bfloat16 to avoid dtype mismatches (e.g., lm_head stays float32).
    if bf16:
        model = model.to(dtype=torch.bfloat16)

    trainable_markers: list[str] = ["lora"]

    bitfit_cfg = None
    prefix_cfg = None
    if ia3_cfg and ia3_cfg.get("enabled"):
        init_value = float(ia3_cfg.get("init_value", 1.0))
        attach_ia3_gates(model, init_value)
        trainable_markers.append("ia3_gate")
        bitfit_cfg = ia3_cfg.get("bitfit")
        prefix_cfg = ia3_cfg.get("prefix_prompts")

        if ia3_cfg.get("per_layer_head_gates"):
            num_heads = getattr(model.config, "num_attention_heads", None) or getattr(
                model.config, "num_heads", None
            )
            kv_heads = getattr(model.config, "num_key_value_heads", None)
            if num_heads:
                attach_per_layer_head_gates(
                    model,
                    int(num_heads),
                    init_value,
                    int(kv_heads) if kv_heads else None,
                )
                trainable_markers.append("ia3_head_gate")
            else:
                console.print(
                    "[yellow]IA³ per-layer head gates requested but attention head count missing; skipping head gates.[/]"
                )

        ffn_cfg = ia3_cfg.get("ffn_channel_gates")
        if ffn_cfg and ffn_cfg.get("enabled"):
            group_size = int(ffn_cfg.get("group_size", 16))
            target = ffn_cfg.get("target", "gate_proj")
            attach_ffn_gates(
                model,
                group_size,
                float(ffn_cfg.get("init_value", 1.0)),
                target,
            )
            trainable_markers.append("channel_gate")

        mlp_cfg = ia3_cfg.get("lora_mlp_down")
        if mlp_cfg and mlp_cfg.get("enabled"):
            bottleneck = int(mlp_cfg.get("bottleneck", 4) or 4)
            init_log = float(mlp_cfg.get("init_value", 0.0))
            attach_lora_mlp_down(model, bottleneck, init_log)
            trainable_markers.append("mlp_down_adapter")

        attn_cfg = ia3_cfg.get("attention_logit_gates")
        if attn_cfg and attn_cfg.get("enabled"):
            per_head = bool(attn_cfg.get("per_head", True))
            per_layer = bool(attn_cfg.get("per_layer", True))
            shared = bool(attn_cfg.get("shared", False))
            target = attn_cfg.get("target", "query")
            init_scale = float(attn_cfg.get("init_value", 1.0))
            if init_scale <= 0:
                init_scale = 1.0
            init_log = math.log(init_scale)
            attach_attention_logit_gates(
                model,
                per_head=per_head,
                per_layer=per_layer,
                shared=shared,
                init_value=init_log,
                target=target,
            )
            trainable_markers.append("logit_gate")

        proj_cfg = ia3_cfg.get("projection_dim_gates")
        if proj_cfg and proj_cfg.get("enabled"):
            input_enabled = bool(proj_cfg.get("input_enabled", False))
            attach_projection_dim_gates(
                model,
                float(proj_cfg.get("init_value", 1.0)),
                input_enabled=input_enabled,
            )
            trainable_markers.append("projection_gate")
            if input_enabled:
                trainable_markers.append("input_projection_gate")

        rope_cfg = ia3_cfg.get("rope_scale")
        if rope_cfg and rope_cfg.get("enabled"):
            attach_rope_scale(
                model,
                per_layer=bool(rope_cfg.get("per_layer", True)),
                init_value=float(rope_cfg.get("init_value", 1.0)),
            )
            trainable_markers.append("rope_scale")

        res_cfg = ia3_cfg.get("residual_gates")
        if res_cfg and res_cfg.get("enabled"):
            attach_residual_gates(
                model,
                float(res_cfg.get("init_value", 1.0)),
                post_ffn=bool(res_cfg.get("post_ffn", False)),
            )
            trainable_markers.append("attn_residual_gate")
            if res_cfg.get("post_ffn", False):
                trainable_markers.append("ffn_residual_gate")

        ln_cfg = ia3_cfg.get("layernorm_gamma")
        use_ln_gamma = bool(ln_cfg and ln_cfg.get("enabled"))
        if use_ln_gamma:
            enable_layernorm_scale_fit(model)
            trainable_markers.extend(["layernorm.weight", "norm.weight"])

    markers = tuple(dict.fromkeys(trainable_markers))
    for name, param in model.named_parameters():
        trainable = any(marker in name for marker in markers)
        param.requires_grad = trainable

    bitfit_params = 0
    if bitfit_cfg and bitfit_cfg.get("enabled"):
        include_lm_head_bias = bool(bitfit_cfg.get("include_lm_head_bias", False))
        bitfit_params = enable_bitfit(model, include_lm_head_bias=include_lm_head_bias)
        if bitfit_params:
            console.print(f"[cyan]BitFit biases enabled: {bitfit_params:,} parameters[/]")

    if prefix_cfg and prefix_cfg.get("enabled"):
        num_tokens = int(prefix_cfg.get("num_tokens", 0) or 0)
        if num_tokens > 0:
            freeze_steps = int(prefix_cfg.get("freeze_steps", 0) or 0)
            embedding_layer = None
            if hasattr(model, "get_input_embeddings"):
                try:
                    embedding_layer = model.get_input_embeddings()
                except Exception:
                    embedding_layer = None
            embed_dim = None
            if embedding_layer is not None:
                embed_dim = getattr(embedding_layer, "embedding_dim", None)
                if embed_dim is None and hasattr(embedding_layer, "weight"):
                    embed_dim = embedding_layer.weight.shape[-1]
            if embed_dim is None:
                console.print(
                    "[yellow]Unable to determine embedding dimension for soft prompts; skipping prefix prompts.[/]"
                )
            else:
                soft_prompts = SoftPromptEmbedding(num_roles=2, num_tokens=num_tokens, embed_dim=int(embed_dim))
                model.add_module("soft_prompt_embeddings", soft_prompts)
                setattr(model, "soft_prompt_freeze_steps", max(0, freeze_steps))
                setattr(model, "_soft_prompt_num_tokens", num_tokens)
                setattr(model, "_soft_prompt_wrapped", False)
                setattr(model, "_soft_prompts_enabled", True)
                setattr(model, "_soft_prompts_eval_disabled", False)
                setattr(model, "_active_prefix_role", 0)
                trainable_markers.append("soft_prompt_embeddings")
                _ensure_soft_prompt_wrapper(model)
                if freeze_steps <= 0:
                    soft_prompts.embeds.requires_grad = True
                else:
                    soft_prompts.embeds.requires_grad = False
        else:
            console.print("[yellow]prefix_prompts enabled but num_tokens <= 0; skipping soft prompts.[/]")

    return model


def enable_layernorm_scale_fit(model: nn.Module) -> None:
    """Unfreeze LayerNorm/RMSNorm scale parameters."""

    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or "rmsnorm" in module.__class__.__name__.lower():
            weight = getattr(module, "weight", None)
            if weight is not None:
                weight.requires_grad = True


_BITFIT_ALLOWED_SUBSTRINGS = (
    ".q_proj.bias",
    ".k_proj.bias",
    ".v_proj.bias",
    ".o_proj.bias",
    ".gate_proj.bias",
    ".up_proj.bias",
    ".down_proj.bias",
)


def enable_bitfit(
    model: nn.Module,
    *,
    include_lm_head_bias: bool = False,
    allowed_substrings: Iterable[str] = _BITFIT_ALLOWED_SUBSTRINGS,
) -> int:
    """Enable bias-only tuning (BitFit) for selected linear layers.

    Returns the number of parameters switched to trainable.
    """

    total = 0
    allowed = tuple(allowed_substrings)
    for name, param in model.named_parameters():
        if not name.endswith("bias") and "bias" not in name:
            continue

        if name == "lm_head.bias" and not include_lm_head_bias:
            continue

        if any(token in name for token in allowed) or (include_lm_head_bias and name == "lm_head.bias"):
            if not param.requires_grad:
                param.requires_grad = True
            total += param.numel()

    return total


def _ensure_soft_prompt_wrapper(model: nn.Module) -> None:
    if getattr(model, "_soft_prompt_wrapped", False):
        return

    soft_prompts: SoftPromptEmbedding | None = getattr(model, "soft_prompt_embeddings", None)
    if soft_prompts is None:
        return

    original_forward = model.forward

    def forward_with_soft_prompts(*args, **kwargs):
        if not getattr(model, "_soft_prompts_enabled", False):
            return original_forward(*args, **kwargs)

        prompts: SoftPromptEmbedding | None = getattr(model, "soft_prompt_embeddings", None)
        if prompts is None:
            return original_forward(*args, **kwargs)

        if kwargs.get("past_key_values") is not None:
            return original_forward(*args, **kwargs)

        role = kwargs.pop("prefix_role", None)
        if role is None:
            role = getattr(model, "_active_prefix_role", 0)

        try:
            role_tensor = torch.as_tensor(role, dtype=torch.long)
        except Exception:
            role_tensor = torch.tensor(0, dtype=torch.long)

        inputs_embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")

        if inputs_embeds is None and input_ids is None:
            return original_forward(*args, **kwargs)

        if inputs_embeds is None and input_ids is not None:
            embedding_layer = model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            kwargs["input_ids"] = None

        if inputs_embeds is None:
            return original_forward(*args, **kwargs)

        batch_size = inputs_embeds.shape[0]

        role_tensor = role_tensor.to(inputs_embeds.device)
        if role_tensor.dim() == 0:
            role_tensor = role_tensor.expand(batch_size)
        elif role_tensor.shape[0] != batch_size:
            if role_tensor.numel() == 1:
                role_tensor = role_tensor.expand(batch_size)
            else:
                raise ValueError(
                    f"prefix_role length {role_tensor.shape[0]} does not match batch size {batch_size}"
                )

        prefix_embeds = prompts.embeds.index_select(0, role_tensor)
        prefix_embeds = prefix_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)

        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
        kwargs["inputs_embeds"] = inputs_embeds

        num_tokens = prompts.num_tokens

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], num_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            kwargs["attention_mask"] = attention_mask

        labels = kwargs.get("labels")
        if labels is not None:
            prefix_labels = torch.full(
                (labels.shape[0], num_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
            kwargs["labels"] = labels

        return original_forward(*args, **kwargs)

    model.forward = forward_with_soft_prompts  # type: ignore[assignment]
    setattr(model, "_soft_prompt_wrapped", True)


def set_soft_prompt_role(model: nn.Module, role: int) -> None:
    if hasattr(model, "_soft_prompt_wrapped"):
        setattr(model, "_active_prefix_role", int(role))


def enable_soft_prompts(model: nn.Module, enabled: bool) -> None:
    if hasattr(model, "_soft_prompt_wrapped"):
        setattr(model, "_soft_prompts_enabled", bool(enabled))
