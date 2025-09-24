import os
import signal


def test_run_code_success():
    from azr.tools.python_tool import run_code

    result = run_code("print('hello')")
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""


def test_run_code_exception():
    from azr.tools.python_tool import run_code

    result = run_code("import sys\nsys.exit(3)")
    assert result.returncode == 3
    assert result.stdout == ""
    assert result.stderr == ""


def test_run_code_timeout(monkeypatch):
    from azr.tools import python_tool

    calls = []

    def fake_killpg(pg, sig):
        calls.append((pg, sig))

    monkeypatch.setattr(os, "killpg", fake_killpg)
    result = python_tool.run_code("import time\ntime.sleep(5)", timeout_s=1)
    assert result.returncode == 124
    assert result.stderr == "TIMEOUT"
    assert calls and calls[0] == (0, signal.SIGKILL)

