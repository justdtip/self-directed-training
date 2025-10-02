import torch
import pytest

from azr.adapters import IA3HeadGate, IA3Gate


def _get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def test_head_gate_broadcast_and_finiteness():
    gate = IA3HeadGate(num_heads=4, init_log_scale=0.0)
    x = torch.randn(2, 3, 4, 8)
    y = gate.apply_to_heads(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_head_gate_warmup_clamp_band():
    gate = IA3HeadGate(num_heads=2, init_log_scale=0.0, clamp_min=0.9, clamp_max=1.1)
    gate.set_warmup_alpha(0.0)
    scale = gate.effective_scale(torch.ones(2, dtype=torch.float32))
    assert torch.all(scale >= 0.9 - 1e-6)
    assert torch.all(scale <= 1.0 + 1e-6)
    gate.set_warmup_alpha(1.0)
    scale = gate.effective_scale(torch.ones(2, dtype=torch.float32))
    assert torch.all(scale <= 1.1 + 1e-6)


def test_scalar_gate_dtype_device_alignment():
    device, dtype = _get_device_and_dtype()
    x = torch.randn(2, 16, device=device, dtype=dtype)
    gate = IA3Gate(16).to(device=device, dtype=dtype)
    y = gate(x)
    assert y.dtype == dtype
    assert y.device.type == device.type
    assert torch.isfinite(y).all()


def test_scalar_gate_clamp_limits():
    gate = IA3Gate(8, init_log_scale=3.0, clamp_min=0.9, clamp_max=1.1)
    x = torch.ones(4, 8)
    y = gate(x)
    assert torch.all(y <= 1.1 + 1e-4)
    assert torch.all(y >= 0.9 - 1e-4)
