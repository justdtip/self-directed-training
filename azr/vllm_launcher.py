from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional

import requests

from .config import AzrModelCfg, AzrVllmCfg


class VLLMServerError(RuntimeError):
    """Raised when the managed vLLM server fails to start or respond."""


def _build_command(model_cfg: AzrModelCfg, vllm_cfg: AzrVllmCfg) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "azr.vllm_server",
        "--model-id",
        model_cfg.model_id,
        "--host",
        vllm_cfg.host,
        "--port",
        str(vllm_cfg.port),
        "--tensor-parallel-size",
        str(max(1, vllm_cfg.tensor_parallel_size)),
        "--gpu-memory-utilization",
        f"{max(0.1, min(0.98, vllm_cfg.gpu_memory_utilization)):.4f}",
        "--dtype",
        vllm_cfg.dtype,
    ]
    if vllm_cfg.enforce_eager:
        cmd.append("--enforce-eager")
    if not vllm_cfg.trust_remote_code:
        cmd.append("--no-trust-remote-code")
    if vllm_cfg.max_model_len is not None:
        cmd.extend(["--max-model-len", str(vllm_cfg.max_model_len)])
    if vllm_cfg.visible_devices:
        cmd.extend(["--visible-devices", vllm_cfg.visible_devices])
    return cmd


def _prepare_env(base: Mapping[str, str], vllm_cfg: AzrVllmCfg) -> dict[str, str]:
    env = dict(base)
    if vllm_cfg.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = vllm_cfg.visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _wait_for_health(url: str, timeout: float, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + max(5.0, timeout)
    last_exc: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise VLLMServerError(
                f"vLLM server exited early with code {proc.returncode}"
            )
        try:
            response = requests.get(url, timeout=3)
            if response.ok:
                return
        except Exception as exc:  # pragma: no cover - connectivity issues
            last_exc = exc
        time.sleep(1.0)
    if last_exc is not None:
        raise VLLMServerError(f"Timed out waiting for vLLM health endpoint: {last_exc}")
    raise VLLMServerError("Timed out waiting for vLLM health endpoint")


@contextmanager
def launch_vllm_server(
    model_cfg: AzrModelCfg,
    vllm_cfg: AzrVllmCfg,
    *,
    log_stream: Optional[object] = None,
) -> Iterator[None]:
    """Launch a vLLM server in a subprocess and tear it down afterwards."""

    cmd = _build_command(model_cfg, vllm_cfg)
    env = _prepare_env(os.environ, vllm_cfg)
    popen_kwargs: dict[str, object] = {"env": env}
    if log_stream is not None:
        popen_kwargs.update({"stdout": log_stream, "stderr": log_stream})
    proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    health_url = f"http://{vllm_cfg.host}:{vllm_cfg.port}/health/"
    try:
        _wait_for_health(health_url, vllm_cfg.server_timeout, proc)
        yield
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


__all__ = ["launch_vllm_server", "VLLMServerError"]
