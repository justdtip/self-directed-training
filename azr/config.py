from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import json


@dataclass
class AzrModelCfg:
    model_id: str
    lora_r: int
    lora_alpha: int
    quantization: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzrModelCfg":
        quant = data.get("quantization", "4bit")
        quant_str = "" if quant is None else str(quant)
        return cls(
            model_id=str(data.get("model_id", "")),
            lora_r=int(data.get("lora_r", 16)),
            lora_alpha=int(data.get("lora_alpha", 32)),
            quantization=quant_str,
        )


@dataclass
class AzrSelfPlayCfg:
    enabled: bool = False
    weight: float = 0.20
    device: str = "cuda:1"
    update_every: int = 500

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzrSelfPlayCfg":
        return cls(
            enabled=bool(data.get("enabled", False)),
            weight=float(data.get("weight", 0.20)),
            device=str(data.get("device", "cuda:1")),
            update_every=int(data.get("update_every", 500)),
        )


@dataclass
class AzrTrainingCfg:
    lr: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzrTrainingCfg":
        return cls(
            lr=float(data.get("lr", 1e-5)),
            warmup_ratio=float(data.get("warmup_ratio", 0.03)),
            weight_decay=float(data.get("weight_decay", 0.0)),
        )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)


__all__ = [
    "AzrModelCfg",
    "AzrSelfPlayCfg",
    "AzrTrainingCfg",
    "load_config",
    "save_config",
]
