import yaml


def test_symlink_yaml_loads_and_has_steps():
    with open("azr.symlink.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "steps" in data and isinstance(data["steps"], list) and data["steps"]
    step = data["steps"][0]
    assert step.get("name", "").lower().startswith("link params")
    assert "cp azr.params.yaml" in step.get("run", "")

