import io, json, os, re, tempfile, importlib.util, sys, subprocess, textwrap
import yaml


def load_files_yaml():
    with open("azr.files.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Top-level must be a mapping"
    assert "files" in data and isinstance(data["files"], list) and data["files"], "files list required"
    return data["files"]


def test_structure_and_modes():
    files = load_files_yaml()
    for entry in files:
        assert set(entry.keys()) == {"path", "mode", "content"}
        assert entry["path"].startswith("/"), "absolute path expected"
        assert re.fullmatch(r"0[0-7]{3}", entry["mode"]) is not None
        assert isinstance(entry["content"], str) and len(entry["content"]) > 0


def test_schema_json_is_valid():
    files = load_files_yaml()
    schema = next(x for x in files if x["path"].endswith("/schema.json"))
    obj = json.loads(schema["content"])  # raises if invalid
    assert "tools" in obj and isinstance(obj["tools"], list) and len(obj["tools"]) == 3
    names = [t["function"]["name"] for t in obj["tools"]]
    assert names == ["python.run", "web.search", "web.fetch"]


def test_tool_harness_run_python_executes_code():
    files = load_files_yaml()
    harness = next(x for x in files if x["path"].endswith("/tool_harness.py"))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tool_harness.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(harness["content"])    
        spec = importlib.util.spec_from_file_location("tool_harness", p)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        res = mod.run_python("print('hello')", timeout_s=2, memory_mb=128)
        assert res["returncode"] == 0
        assert "hello" in res["stdout"]


def test_sbx_py_shell_syntax_ok():
    files = load_files_yaml()
    sbx = next(x for x in files if x["path"].endswith("/sbx_py"))
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(sbx["content"]) ; p = f.name
    try:
        # shell syntax check only
        subprocess.run(["bash", "-n", p], check=True)
    finally:
        os.remove(p)


def test_python_file_syntax_parses():
    files = load_files_yaml()
    # Compile (do not import) files that have non-stdlib deps
    py_paths = ["/opt/azr/tools/web_tool.py", "/opt/azr/train_selfplay_gspo.py", "/opt/azr/opt_azr_params.py"]
    for path in py_paths:
        src = next(x for x in files if x["path"] == path)["content"]
        compile(src, path, "exec")

