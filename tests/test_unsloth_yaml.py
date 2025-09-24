import yaml


def test_unsloth_yaml_loads_and_has_sections():
    with open("azr.unsloth.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "unsloth" in data and isinstance(data["unsloth"], dict)
    assert "logging" in data and isinstance(data["logging"], dict)

    u = data["unsloth"]
    assert set(["learning_rate", "warmup_ratio", "weight_decay"]).issubset(u.keys())

    lg = data["logging"]
    assert set(["log_dir", "save_steps", "eval_steps"]).issubset(lg.keys())
