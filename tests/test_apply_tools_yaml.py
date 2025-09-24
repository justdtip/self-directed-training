import yaml


def test_apply_tools_yaml_structure():
    with open("azr.apply-tools.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    steps = data.get("steps")
    assert isinstance(steps, list) and steps

    names = [s.get("name", "") for s in steps]
    required = [
        "Ensure venv + deps",
        "Materialize files",
        "Link params into /opt/azr",
        "Smoke tests",
    ]
    for req in required:
        assert any(req in n for n in names)

    # Materialize step should reference azr.tools-train.impl.yaml
    mat = next(s for s in steps if "Materialize files" in s.get("name", ""))
    assert "azr.tools-train.impl.yaml" in mat.get("run", "")


def test_tools_train_impl_yaml_exists_and_loads():
    with open("azr.tools-train.impl.yaml", "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    assert isinstance(spec, dict)
    assert "files" in spec
    assert isinstance(spec["files"], list)
