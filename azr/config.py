from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import yaml


def _normalize_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    azr_section = data.get('azr')
    if not isinstance(azr_section, dict):
        return data

    model_cfg = data.setdefault('model', {})
    if not isinstance(model_cfg, dict):
        model_cfg = data['model'] = {}

    model_id = azr_section.get('model_id')
    if 'model_id' not in model_cfg and model_id:
        model_cfg['model_id'] = model_id

    quant = azr_section.get('quantization')
    if 'quantization' not in model_cfg and quant is not None:
        model_cfg['quantization'] = quant

    lora = azr_section.get('lora')
    if isinstance(lora, dict):
        if 'lora_r' not in model_cfg and lora.get('r') is not None:
            model_cfg['lora_r'] = lora['r']
        if 'lora_alpha' not in model_cfg and lora.get('alpha') is not None:
            model_cfg['lora_alpha'] = lora['alpha']

    rlhf = azr_section.get('rlhf')
    if 'rlhf' not in data and isinstance(rlhf, dict):
        data['rlhf'] = rlhf

    training = azr_section.get('training')
    if 'training' not in data and isinstance(training, dict):
        data['training'] = training

    return data


@dataclass
class AzrModelCfg:
    model_id: str
    lora_r: int
    lora_alpha: int
    quantization: str
    device_map: object | None = "auto"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzrModelCfg":
        quant = data.get("quantization", "4bit")
        quant_str = "" if quant is None else str(quant)
        return cls(
            model_id=str(data.get("model_id", "")),
            lora_r=int(data.get("lora_r", 16)),
            lora_alpha=int(data.get("lora_alpha", 32)),
            quantization=quant_str,
            device_map=data.get("device_map", "auto"),
        )


@dataclass
class AzrSelfPlayCfg:
    enabled: bool = False
    weight: float = 0.20
    device: str = "cuda:1"
    update_every: int = 500
    # Optional remote opponent configuration. Expected keys include:
    #   source, provider, model_id, endpoint, api_key_env, max_concurrency, temperature, top_p.
    opponent: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzrSelfPlayCfg":
        return cls(
            enabled=bool(data.get("enabled", False)),
            weight=float(data.get("weight", 0.20)),
            device=str(data.get("device", "cuda:1")),
            update_every=int(data.get("update_every", 500)),
            opponent=data.get("opponent"),
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
        content = handle.read()
    data = yaml.safe_load(content) if content.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError("Configuration must be a mapping")
    return _normalize_legacy(data)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)


__all__ = [
    "AzrModelCfg",
    "AzrSelfPlayCfg",
    "AzrTrainingCfg",
    "load_config",
    "save_config",
]
