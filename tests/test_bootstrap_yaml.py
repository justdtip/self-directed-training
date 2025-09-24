import yaml


def test_bootstrap_yaml_loads_and_has_steps():
    with open("azr.bootstrap.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "steps" in data and isinstance(data["steps"], list) and data["steps"], "steps list required"
    for step in data["steps"]:
        assert "name" in step and "run" in step

