import os, subprocess, sys, tempfile, textwrap, resource, signal

class PythonResult:
    """
    Captures stdout, stderr and return code from running untrusted code.
    """
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _limit_resources(memory_mb: int, cpu_s: int):
    """
    Pre-exec hook for subprocess.run to set resource limits.
    """
    def preexec():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        os.setsid()
    return preexec


def run_code(code: str, timeout_s: int = 2, memory_mb: int = 256) -> PythonResult:
    """
    Execute arbitrary Python code with resource caps.
    Returns a PythonResult with stdout, stderr and returncode.
    On timeout, returncode=124 and stderr contains 'TIMEOUT'.
    """
    wrapper = textwrap.dedent(f"{code}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(wrapper)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_limit_resources(memory_mb, timeout_s),
            timeout=timeout_s
        )
        return PythonResult(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(0, signal.SIGKILL)
        except Exception:
            pass
        return PythonResult(stdout="", stderr="TIMEOUT", returncode=124)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
