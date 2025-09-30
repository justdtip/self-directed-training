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

    if ia3_cfg and ia3_cfg.get("enabled"):
        init_value = float(ia3_cfg.get("init_value", 1.0))
        attach_ia3_gates(model, init_value)
        trainable_markers.append("ia3_gate")

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

    return model
def enable_layernorm_scale_fit(model: nn.Module) -> None:
    """Unfreeze LayerNorm/RMSNorm scale parameters."""

    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or "rmsnorm" in module.__class__.__name__.lower():
            weight = getattr(module, "weight", None)
            if weight is not None:
                weight.requires_grad = True
