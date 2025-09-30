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


class IA3HeadGate(nn.Module):
    """Per-head multiplicative gate applied within attention projections."""

    def __init__(self, num_heads: int, init_value: float = 1.0) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive for IA3HeadGate")
        self.scale = nn.Parameter(
            torch.full((num_heads,), float(init_value), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        if head_dim <= 0:
            return x
        num_heads = self.scale.shape[0]
        last_dim = x.shape[-1]
        expected = num_heads * head_dim
        if last_dim != expected:
            return x
        new_shape = (*x.shape[:-1], num_heads, head_dim)
        x = x.reshape(new_shape)
        scale_shape = (1,) * (x.dim() - 2) + (num_heads, 1)
        scale = self.scale.view(*scale_shape)
        if scale.device != x.device:
            scale = scale.to(x.device)
        x = x * scale
        return x.reshape(*x.shape[:-2], expected)


class ChannelGate(nn.Module):
    """Grouped channel gate for FFN activations."""

    def __init__(self, hidden_dim: int, group_size: int, init_value: float = 1.0) -> None:
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be positive for ChannelGate")
        if hidden_dim % group_size != 0:
            raise ValueError("hidden_dim must be divisible by group_size for ChannelGate")
        num_groups = hidden_dim // group_size
        self.group_size = group_size
        self.scale = nn.Parameter(
            torch.full((num_groups,), float(init_value), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_dim = x.shape[-1]
        group_size = self.group_size
        if hidden_dim % group_size != 0:
            return x
        num_groups = hidden_dim // group_size
        new_shape = (*x.shape[:-1], num_groups, group_size)
        x = x.reshape(new_shape)
        scale = self.scale.view((1,) * (x.dim() - 2) + (num_groups, 1))
        if scale.device != x.device:
            scale = scale.to(x.device)
        x = x * scale
        return x.reshape(*x.shape[:-2], hidden_dim)


class AttentionLogitGate(nn.Module):
    """Scales attention logits by modulating query states per head or per layer."""

    def __init__(self, num_heads: int, *, per_head: bool, init_value: float = 1.0) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive for AttentionLogitGate")
        self.per_head = per_head
        size = num_heads if per_head else 1
        self.scale = nn.Parameter(
            torch.full((size,), float(init_value), dtype=torch.float32)
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        if scale.device != scores.device:
            scale = scale.to(scores.device)
        if self.per_head and scale.numel() == scores.shape[1]:
            return scores * scale.view(1, -1, 1, 1)
        return scores * scale.view(1, 1, 1, 1)

    def apply_to_query(self, query: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        if scale.device != query.device:
            scale = scale.to(query.device)
        if self.per_head and scale.numel() == query.shape[1]:
            return query * scale.view(1, -1, 1, 1)
        return query * scale.view(1, 1, 1, 1)


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
            if input.device != gate.scale.device:
                gate.to(input.device)
            return _orig(input) * _gate.scale

        module.forward = gated_forward  # type: ignore[assignment]
        setattr(module, "ia3_gate", gate)


def attach_per_layer_head_gates(
    model: nn.Module,
    num_heads: int,
    init_value: float = 1.0,
    num_kv_heads: int | None = None,
) -> None:
    """Attach per-layer IA³ head gates to attention projection layers."""

    if num_heads <= 0:
        raise ValueError("num_heads must be positive when attaching head gates")
    kv_heads = num_kv_heads or num_heads
    target_tags = ("q_proj", "k_proj", "v_proj", "o_proj")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(tag in name for tag in target_tags):
            continue

        if "k_proj" in name or "v_proj" in name:
            heads = kv_heads
        else:
            heads = num_heads
        if heads <= 0:
            continue

        head_dim = module.out_features // heads
        if head_dim <= 0 or module.out_features % heads != 0:
            continue

        gate = IA3HeadGate(heads, init_value)
        original_forward = module.forward

        def gated_forward(
            input: torch.Tensor,
            *,
            _orig=original_forward,
            _gate=gate,
            _hd=head_dim,
        ) -> torch.Tensor:
            if input.device != gate.scale.device:
                gate.to(input.device)
            out = _orig(input)
            return _gate(out, _hd)

        module.forward = gated_forward  # type: ignore[assignment]
        setattr(module, "ia3_head_gate", gate)


def attach_ffn_gates(
    model: nn.Module,
    group_size: int,
    init_value: float = 1.0,
    target: str = "gate_proj",
) -> None:
    """Attach grouped channel gates to FFN projection layers."""

    if group_size <= 0:
        raise ValueError("group_size must be positive when attaching FFN gates")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or target not in name:
            continue
        hidden_dim = module.out_features
        if hidden_dim % group_size != 0:
            continue

        gate = ChannelGate(hidden_dim, group_size, init_value)
        original_forward = module.forward

        def gated_forward(
            input: torch.Tensor,
            *,
            _orig=original_forward,
            _gate=gate,
        ) -> torch.Tensor:
            if input.device != gate.scale.device:
                gate.to(input.device)
            out = _orig(input)
            return _gate(out)

        module.forward = gated_forward  # type: ignore[assignment]
        setattr(module, "channel_gate", gate)


_ATTENTION_WRAPPED = False


def _wrap_attention_function(fn):
    if getattr(fn, "_azr_logit_gate_wrapped", False):
        return fn

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        gate: AttentionLogitGate | None = getattr(module, "logit_gate", None)
        if gate is not None:
            query = gate.apply_to_query(query)
        return fn(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

    wrapped._azr_logit_gate_wrapped = True  # type: ignore[attr-defined]
    wrapped._azr_logit_gate_base = fn  # type: ignore[attr-defined]
    return wrapped


def _ensure_attention_wrapper() -> None:
    global _ATTENTION_WRAPPED
    if _ATTENTION_WRAPPED:
        return
    try:
        from transformers.models.llama import modeling_llama as llama_mod  # type: ignore
    except Exception:  # pragma: no cover - architecture specific
        return

    llama_mod.eager_attention_forward = _wrap_attention_function(
        llama_mod.eager_attention_forward
    )
    for key, fn in list(llama_mod.ALL_ATTENTION_FUNCTIONS.items()):
        llama_mod.ALL_ATTENTION_FUNCTIONS[key] = _wrap_attention_function(fn)

    _ATTENTION_WRAPPED = True


def attach_attention_logit_gates(
    model: nn.Module,
    *,
    per_head: bool,
    init_value: float = 1.0,
) -> None:
    """Attach attention logit gates that scale query states prior to softmax."""

    _ensure_attention_wrapper()

    for module in model.modules():
        if not hasattr(module, "q_proj") or not hasattr(module, "num_heads"):
            continue
        if hasattr(module, "logit_gate"):
            continue
        num_heads = getattr(module, "num_heads", None) or getattr(
            module, "num_attention_heads", None
        )
        if not num_heads:
            continue
        try:
            gate = AttentionLogitGate(int(num_heads), per_head=per_head, init_value=init_value)
        except ValueError:
            continue
        module.add_module("logit_gate", gate)
