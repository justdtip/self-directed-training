from __future__ import annotations

import torch
from torch import nn


class IA3Gate(nn.Module):
    """Per-channel multiplicative gate used for IA³ adapters."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def attach_ia3_gates(model: nn.Module, init_value: float = 1.0) -> None:
    """Attach IA³ gates to attention projection layers in ``model``.

    The gate is applied after q/k/v/o projection layers so their outputs are
    scaled by a learned factor. Existing forwards are wrapped in-place and the
    gate module is stored on the projection for later parameter filtering.
    """

    target_tags = ("q_proj", "k_proj", "v_proj", "o_proj")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(tag in name for tag in target_tags):
            continue

        gate = IA3Gate(init_value)
        original_forward = module.forward

        def gated_forward(input: torch.Tensor, *, _orig=original_forward, _gate=gate):
            return _orig(input) * _gate.scale

        module.forward = gated_forward  # type: ignore[assignment]
        setattr(module, "ia3_gate", gate)

