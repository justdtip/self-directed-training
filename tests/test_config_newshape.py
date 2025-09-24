import json
from pathlib import Path

from azr.config import AzrModelCfg, AzrSelfPlayCfg, load_config


def test_load_config_accepts_additional_sections(tmp_path: Path) -> None:
    payload = {
        "model": {
            "model_id": "deepcogito/cogito-v1-preview-llama-8b",
            "quantization": "4bit",
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "self_play": {
            "enabled": True,
            "weight": 0.25,
            "device": "cuda:1",
        },
        "extras": {
            "hardware": {
                "gpus": [
                    {"name": "RTX 6000 Ada", "vram_gb": 48},
                    {"name": "RTX 6000 Ada", "vram_gb": 48},
                ],
                "system_ram_gb": 172,
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_config(path)
    model_cfg = AzrModelCfg.from_dict(cfg["model"])
    assert model_cfg.lora_alpha == 32

    sp_cfg = AzrSelfPlayCfg.from_dict(cfg.get("self_play", {}))
    assert sp_cfg.enabled is True
    assert cfg["extras"]["hardware"]["system_ram_gb"] == 172
