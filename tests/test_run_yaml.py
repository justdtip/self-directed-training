import yaml


def test_run_yaml_structure():
    with open("azr.run.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    steps = data.get("steps")
    assert isinstance(steps, list) and steps
    names = [s.get("name", "") for s in steps]
    # Ensure all expected stage names are present
    required = [
        "Write the YAML files you saved to disk into place",
        "Execute bootstrap",
        "Materialize files",
        "Link params file",
        "Quick smoke tests (sandbox + web)",
        "Start a demo training run",
    ]
    for r in required:
        assert any(r in n for n in names)

    # Quick sanity: runner embeds a Python heredoc and references our file specs
    mat = next(s for s in steps if s["name"].startswith("Materialize files"))
    assert "azr.files.yaml" in mat["run"]
    smoke = next(s for s in steps if s["name"].startswith("Quick smoke tests"))
    assert "sys.path" in smoke["run"] and "/opt/azr" in smoke["run"]
