from __future__ import annotations

from typing import Optional, Tuple

from .config import AzrModelCfg
from .utils import console


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        # For causal models, set pad to eos
        tok.pad_token = tok.eos_token
    return tok


def setup_model(cfg: AzrModelCfg):
    import torch
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
                bnb_4bit_compute_dtype=torch.bfloat16 if cfg.rlhf.bf16 else torch.float16,
            )
        except Exception as e:
            console.print(f"[yellow]bitsandbytes not available or incompatible: {e}. Falling back to full precision.[/]")
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.rlhf.bf16 else None,
        device_map="auto" if bnb_config is not None else None,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora = cfg.lora
    peft_cfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        target_modules=lora.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model

