from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import yaml


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class RLHFCfg:
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    importance_sampling_level: str = "sequence"
    clip_range_ratio: float = 0.1
    kl_coeff: float = 0.0
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class SandboxCfg:
    timeout_s: int = 2
    memory_mb: int = 256
    max_tool_turns: int = 6


@dataclass
class WebToolCfg:
    max_bytes: int = 800_000
    user_agent: str = "AZR-Research/1.0 (+no-bots)"
    headless: bool = True


@dataclass
class AzrModelCfg:
    image: str = "vastai/base-image:@vastai-automatic-tag"
    model_id: str = "deepcogito/cogito-v1-preview-llama-8b"
    quantization: Optional[str] = "4bit"
    lora: LoraCfg = field(default_factory=LoraCfg)
    rlhf: RLHFCfg = field(default_factory=RLHFCfg)
    sandbox: SandboxCfg = field(default_factory=SandboxCfg)
    web: WebToolCfg = field(default_factory=WebToolCfg)


@dataclass
class GPUInfo:
    name: str = ""
    tflops_fp16: float | None = None
    vram_gb: float | None = None
    mem_bw_gbps: float | None = None
    pcie: str | None = None


@dataclass
class CPUInfo:
    model: str | None = None
    cores: int | None = None


@dataclass
class HardwareCfg:
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    system_ram_gb: Optional[int] = None
    storage: Optional[str] = None


@dataclass
class TrainingCfg:
    lr: float = 5e-5
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    log_dir: Optional[str] = None


@dataclass
class SelfPlayUpdateCfg:
    every_calls: int = 0
    strategy: str = "copy_lora"


@dataclass
class SelfPlayCfg:
    enabled: bool = False
    weight: float = 0.2
    device: str = "cuda:1"
    use_tool_loop: bool = False
    tie_breakers: List[str] = field(default_factory=lambda: ["pass_rate", "code_length"])
    update: SelfPlayUpdateCfg = field(default_factory=SelfPlayUpdateCfg)


@dataclass
class RootCfg:
    azr: AzrModelCfg = field(default_factory=AzrModelCfg)
    hardware: HardwareCfg = field(default_factory=HardwareCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    self_play: SelfPlayCfg = field(default_factory=SelfPlayCfg)


def _to_dataclass(cls, data: Dict[str, Any]):
    # Recursively construct dataclasses from nested dicts, resolving annotations
    from typing import get_type_hints, get_origin, get_args

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        key = f.name
        if key not in data:
            continue
        val = data[key]
        anno = type_hints.get(key, f.type)
        origin = get_origin(anno)
        args = get_args(anno)

        if isinstance(val, dict) and dataclasses.is_dataclass(anno):
            kwargs[key] = _to_dataclass(anno, val)
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            subcls = args[0]
            kwargs[key] = [
                _to_dataclass(subcls, v) if isinstance(v, dict) else v for v in (val or [])
            ]
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str) -> RootCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RootCfg()
    if not isinstance(raw, dict):
        return cfg
    if "azr" in raw and isinstance(raw["azr"], dict):
        cfg.azr = _to_dataclass(AzrModelCfg, raw["azr"])
    if "hardware" in raw and isinstance(raw["hardware"], dict):
        cfg.hardware = _to_dataclass(HardwareCfg, raw["hardware"])
    if "training" in raw and isinstance(raw["training"], dict):
        cfg.training = _to_dataclass(TrainingCfg, raw["training"])
    if "self_play" in raw and isinstance(raw["self_play"], dict):
        cfg.self_play = _to_dataclass(SelfPlayCfg, raw["self_play"])
    return cfg
