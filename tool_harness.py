"""Execution harness for running untrusted Python with sandboxing limits.

This module provides the `_limit_resources` pre-exec hook and the `run_python`
function described in SPEC_FUNCTIONS.md. It enforces CPU and memory limits
either via nsjail (preferred) or POSIX rlimits as a fallback.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Callable, Dict


def _limit_resources(memory_mb: int, cpu_s: int) -> Callable[[], None]:
    """Return a preexec function that applies rlimits before exec.

    Parameters
    ----------
    memory_mb:
        Maximum address space in megabytes.
    cpu_s:
        Maximum CPU time in seconds.

    Returns
    -------
    Callable[[], None]
        A function suitable for use as ``preexec_fn`` in ``subprocess.Popen``.
    """

    limit_bytes = int(memory_mb) * 1024 * 1024
    cpu_limit = int(cpu_s)

    def _preexec() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        os.setsid()

    return _preexec


def _nsjail_available() -> bool:
    return shutil.which("nsjail") is not None and os.path.exists("/usr/local/bin/sbx_py")


def run_python(code: str, timeout_s: int = 2, memory_mb: int = 256) -> Dict[str, object]:
    """Execute the provided Python code with sandboxing constraints.

    Parameters
    ----------
    code:
        Python source code to execute.
    timeout_s:
        Wall-clock timeout in seconds. Must be positive.
    memory_mb:
        Memory limit in megabytes. Values below 64 are bumped to 64.

    Returns
    -------
    dict
        A dictionary containing ``stdout``, ``stderr`` and ``returncode`` keys.
    """

    timeout_s = max(1, int(timeout_s))
    memory_mb = max(64, int(memory_mb))
    code_text = textwrap.dedent(code)

    fd, path = tempfile.mkstemp(suffix=".py", dir="/tmp")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as tmp:
            tmp.write(code_text)

        if _nsjail_available():
            cmd = ["/usr/local/bin/sbx_py", path, str(memory_mb), str(timeout_s)]
            preexec = os.setsid
            run_timeout = timeout_s + 1
        else:
            cmd = [sys.executable, path]
            preexec = _limit_resources(memory_mb, timeout_s)
            run_timeout = timeout_s

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec,
        )
        try:
            stdout, stderr = proc.communicate(timeout=run_timeout)
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                stdout, stderr = proc.communicate()
            return {"stdout": "", "stderr": "TIMEOUT", "returncode": 124}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

