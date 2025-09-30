from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional

import torch

from .config import AzrModelCfg
from .utils import console
from .adapters import attach_ia3_gates


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

    if ia3_cfg and ia3_cfg.get("enabled"):
        attach_ia3_gates(model, float(ia3_cfg.get("init_value", 1.0)))
        for name, param in model.named_parameters():
            trainable = ("lora" in name) or ("ia3_gate" in name)
            param.requires_grad = trainable

    return model
