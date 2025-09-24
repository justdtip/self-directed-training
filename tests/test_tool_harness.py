import pytest

import tool_harness as harness


def test_run_python_ok():
    res = harness.run_python("print('hello')", timeout_s=2, memory_mb=128)
    assert isinstance(res, dict)
    assert res["returncode"] == 0
    assert res["stdout"].strip() == "hello"
    assert "TIMEOUT" not in res["stderr"]


def test_run_python_timeout():
    res = harness.run_python("import time; time.sleep(3)", timeout_s=1, memory_mb=128)
    assert res["returncode"] == 124
    assert "TIMEOUT" in res["stderr"]


def test_run_python_memlimit():
    code = "x = 'a' * (50*1024*1024); print(len(x))"
    res = harness.run_python(code, timeout_s=2, memory_mb=32)
    assert res["returncode"] != 0 or "MemoryError" in (res["stderr"] or "")
