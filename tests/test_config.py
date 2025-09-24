import json
from pathlib import Path

from azr.config import AzrModelCfg, AzrSelfPlayCfg, AzrTrainingCfg, load_config, save_config


def test_load_config_roundtrip(tmp_path: Path) -> None:
    payload = {
        "model": {
            "model_id": "sshleifer/tiny-gpt2",
            "quantization": "4bit",
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "training": {
            "lr": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        },
        "self_play": {
            "enabled": True,
            "weight": 0.3,
            "device": "cpu",
            "update_every": 100,
        },
    }
    path = tmp_path / "config.json"
    save_config(payload, path)

    loaded = load_config(path)
    assert loaded["model"]["model_id"] == "sshleifer/tiny-gpt2"

    model_cfg = AzrModelCfg.from_dict(loaded["model"])
    assert model_cfg.lora_r == 8
    training_cfg = AzrTrainingCfg.from_dict(loaded.get("training", {}))
    assert training_cfg.lr == 2e-5
    self_play_cfg = AzrSelfPlayCfg.from_dict(loaded.get("self_play", {}))
    assert self_play_cfg.enabled is True


def test_load_config_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"model_id": "m"}}), encoding="utf-8")
    loaded = load_config(path)
    model = AzrModelCfg.from_dict(loaded["model"])
    assert model.model_id == "m"
    training = AzrTrainingCfg.from_dict(loaded.get("training", {}))
    assert training.lr == 1e-5
