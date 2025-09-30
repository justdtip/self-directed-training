from __future__ import annotations

import torch
from torch import nn


def _maybe_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor if tensor.device == device else tensor.to(device)


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
        scale = _maybe_to(scale, x.device)
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
        scale = _maybe_to(scale, x.device)
        x = x * scale
        return x.reshape(*x.shape[:-2], hidden_dim)


class ResidualStreamGate(nn.Module):
    """Per-dimension gate applied to residual streams."""

    def __init__(self, hidden_dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for ResidualStreamGate")
        self.scale = nn.Parameter(
            torch.full((hidden_dim,), float(init_value), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        scale = _maybe_to(scale, x.device)
        return x * scale


class ProjectionDimGate(nn.Module):
    """Per-dimension gate applied to projection outputs."""

    def __init__(self, hidden_dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for ProjectionDimGate")
        self.scale = nn.Parameter(
            torch.full((hidden_dim,), float(init_value), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = _maybe_to(self.scale, x.device)
        return x * scale


class InputProjectionGate(nn.Module):
    """Per-dimension gate applied to projection inputs."""

    def __init__(self, input_dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for InputProjectionGate")
        self.scale = nn.Parameter(
            torch.full((input_dim,), float(init_value), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = _maybe_to(self.scale, x.device)
        return x * scale


class AttentionLogitGate(nn.Module):
    """Flexible attention gate supporting shared/per-head scaling and warm-up."""

    def __init__(
        self,
        num_heads: int,
        *,
        per_head: bool,
        shared: bool = False,
        init_value: float = 0.0,
        target: str = "query",
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive for AttentionLogitGate")
        self.per_head = per_head
        self.shared = shared
        self.target = target
        size = 1 if shared or not per_head else num_heads
        self.log_temp = nn.Parameter(
            torch.full((size,), float(init_value), dtype=torch.float32)
        )
        self.warmup_alpha: float = 1.0

    def set_warmup_alpha(self, alpha: float) -> None:
        self.warmup_alpha = float(alpha)

    def _effective_scale(self, device: torch.device) -> torch.Tensor:
        scale = torch.exp(self.log_temp)
        scale = torch.clamp(scale, 0.5, 2.0)
        scale = _maybe_to(scale, device)
        if self.warmup_alpha != 1.0:
            scale = scale * self.warmup_alpha
        return scale

    def _reshape_scale(self, scale: torch.Tensor, num_heads: int) -> torch.Tensor:
        if scale.numel() == 1 or self.shared or not self.per_head:
            return scale.view(1, 1, 1, 1)
        if scale.numel() == num_heads:
            return scale.view(1, num_heads, 1, 1)
        return scale.mean().view(1, 1, 1, 1)

    def modulate(
        self, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scale = self._effective_scale(query.device)
        if self.target == "logits":
            scale = torch.clamp(scale, min=1e-6)
            sqrt_scale = torch.sqrt(scale)
            factor_q = self._reshape_scale(sqrt_scale, query.shape[1]).to(query.dtype)
            query = query * factor_q
            if key is not None:
                factor_k = factor_q
                if factor_q.shape[1] not in (1, key.shape[1]):
                    if factor_q.shape[1] % key.shape[1] == 0:
                        group = factor_q.shape[1] // key.shape[1]
                        factor_k = factor_q.view(1, key.shape[1], group, 1, 1).mean(dim=2)
                    else:
                        factor_k = factor_q.mean(dim=1, keepdim=True)
                factor_k = factor_k.to(key.dtype)
                key = key * factor_k
            return query, key

        factor = self._reshape_scale(scale, query.shape[1]).to(query.dtype)
        query = query * factor
        return query, key


class RoPEScale(nn.Module):
    """Scales rotary positional embeddings."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        if init_value <= 0:
            init_value = 1.0
        self.log_scale = nn.Parameter(torch.tensor(float(init_value)).log())

    @property
    def scale(self) -> torch.Tensor:
        raw = torch.exp(self.log_scale)
        return torch.clamp(raw, 0.5, 2.0)

    def forward(self, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = _maybe_to(self.scale, cos.device)
        return cos * scale, sin * scale


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


def _infer_attention_heads(module: nn.Module) -> int | None:
    num_heads = getattr(module, "num_heads", None) or getattr(
        module, "num_attention_heads", None
    )
    if num_heads:
        return int(num_heads)
    head_dim = getattr(module, "head_dim", None)
    q_proj = getattr(module, "q_proj", None)
    if head_dim and q_proj is not None and hasattr(q_proj, "out_features"):
        try:
            num = int(q_proj.out_features // head_dim)
            if num > 0:
                return num
        except Exception:
            return None
    return None


def _wrap_attention_function(fn):
    if getattr(fn, "_azr_logit_gate_wrapped", False):
        return fn

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        rope_module = getattr(module, "rope_scale", None)
        if rope_module is None:
            rope_module = getattr(module, "_shared_rope_scale", None)
        if rope_module is not None:
            scale = rope_module.scale
            scale = _maybe_to(scale, query.device)
            query = query * scale.view(1, 1, 1, 1)
            if key is not None:
                key = key * scale.view(1, 1, 1, 1)

        gate: AttentionLogitGate | None = getattr(module, "logit_gate", None)
        if gate is not None:
            query, key = gate.modulate(query, key)
        return fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

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
    per_layer: bool = True,
    shared: bool = False,
    init_value: float = 1.0,
    target: str = "query",
) -> None:
    """Attach attention logit gates with optional sharing and target selection."""

    _ensure_attention_wrapper()

    if not per_layer:
        shared = True

    if target not in {"query", "logits"}:
        target = "query"

    attention_modules: list[nn.Module] = []
    for module in model.modules():
        if not hasattr(module, "q_proj"):
            continue
        if not hasattr(module, "head_dim"):
            continue
        attention_modules.append(module)

    if not attention_modules:
        return

    if shared:
        sample = attention_modules[0]
        num_heads = _infer_attention_heads(sample)
        if not num_heads:
            return
        gate = AttentionLogitGate(
            int(num_heads), per_head=per_head, shared=True, init_value=init_value, target=target
        )
        name = "logit_gate_shared"
        index = 0
        while hasattr(model, name):
            index += 1
            name = f"logit_gate_shared_{index}"
        model.add_module(name, gate)
        for module in attention_modules:
            setattr(module, "logit_gate", gate)
        return

    for module in attention_modules:
        if hasattr(module, "logit_gate"):
            continue
        num_heads = _infer_attention_heads(module)
        if not num_heads:
            continue
        try:
            gate = AttentionLogitGate(
                int(num_heads), per_head=per_head, shared=False, init_value=init_value, target=target
            )
        except ValueError:
            continue
        module.add_module("logit_gate", gate)


def attach_projection_dim_gates(
    model: nn.Module,
    init_value: float = 1.0,
    *,
    input_enabled: bool = False,
) -> None:
    """Attach per-dimension projection gates to q/k/v/o layers."""

    target_tags = ("q_proj", "k_proj", "v_proj", "o_proj")

    for name, module in model.named_modules():
        if not any(tag in name for tag in target_tags):
            continue
        if hasattr(module, "projection_gate"):
            continue
        out_features = getattr(module, "out_features", None)
        if out_features is None:
            continue
        in_features = getattr(module, "in_features", None)
        if input_enabled and in_features is None:
            continue
        try:
            gate = ProjectionDimGate(int(out_features), init_value)
        except ValueError:
            continue
        input_gate = None
        if input_enabled:
            try:
                input_gate = InputProjectionGate(int(in_features), init_value)
            except Exception:
                input_gate = None

        original_forward = module.forward

        def gated_forward(
            input_tensor: torch.Tensor,
            *f_args,
            _orig=original_forward,
            _out_gate=gate,
            _in_gate=input_gate,
            **f_kwargs,
        ):
            if _in_gate is not None:
                if input_tensor.device != _in_gate.scale.device:
                    _in_gate.to(input_tensor.device)
                input_tensor = _in_gate(input_tensor)
            output = _orig(input_tensor, *f_args, **f_kwargs)
            if isinstance(output, tuple) and output:
                first = _out_gate(output[0])
                return (first, *output[1:])
            return _out_gate(output)

        module.forward = gated_forward  # type: ignore[assignment]
        module.add_module("projection_gate", gate)
        if input_gate is not None:
            module.add_module("input_projection_gate", input_gate)


def _iter_decoder_layers(model: nn.Module) -> list[nn.Module]:
    queue = [model]
    seen: set[int] = set()
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))

        layers = getattr(node, "layers", None)
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                return list(layers)
            if hasattr(layers, "__iter__"):
                return list(layers)

        for attr in ("model", "decoder", "base_model"):
            child = getattr(node, attr, None)
            if child is not None:
                queue.append(child)

    return []


def attach_residual_gates(
    model: nn.Module,
    init_value: float = 1.0,
    *,
    post_ffn: bool = False,
) -> None:
    """Attach per-layer residual stream gates to decoder layers."""

    layers = _iter_decoder_layers(model)
    if not layers:
        return

    hidden_dim = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_dim is None:
        sample_attn = getattr(getattr(layers[0], "self_attn", None), "o_proj", None)
        if sample_attn is not None:
            hidden_dim = getattr(sample_attn, "out_features", None)
    if hidden_dim is None:
        return

    hidden_dim = int(hidden_dim)

    for layer in layers:
        attn_mod = getattr(layer, "self_attn", None)
        mlp_mod = getattr(layer, "mlp", None)
        if attn_mod is None or mlp_mod is None:
            continue
        gate_attn = getattr(layer, "attn_residual_gate", None)
        gate_ffn = getattr(layer, "ffn_residual_gate", None)

        if gate_attn is None:
            gate_attn = ResidualStreamGate(hidden_dim, init_value)
            original_attn_forward = attn_mod.forward

            def gated_attn_forward(*args, __orig=original_attn_forward, __gate=gate_attn, **kwargs):
                output = __orig(*args, **kwargs)
                if isinstance(output, tuple):
                    if not output:
                        return output
                    attn_out = __gate(output[0])
                    return (attn_out, *output[1:])
                return __gate(output)

            attn_mod.forward = gated_attn_forward  # type: ignore[assignment]
            layer.add_module("attn_residual_gate", gate_attn)

        if post_ffn:
            if gate_ffn is None:
                gate_ffn = ResidualStreamGate(hidden_dim, init_value)
                original_mlp_forward = mlp_mod.forward

                def gated_mlp_forward(*args, __orig=original_mlp_forward, __gate=gate_ffn, **kwargs):
                    output = __orig(*args, **kwargs)
                    if isinstance(output, tuple):
                        if not output:
                            return output
                        ffn_out = __gate(output[0])
                        return (ffn_out, *output[1:])
                    return __gate(output)

                mlp_mod.forward = gated_mlp_forward  # type: ignore[assignment]
                layer.add_module("ffn_residual_gate", gate_ffn)
        elif gate_ffn is not None:
            delattr(layer, "ffn_residual_gate")

        def gated_attn_forward(*args, __orig=original_attn_forward, __gate=gate_attn, **kwargs):
            output = __orig(*args, **kwargs)
            if isinstance(output, tuple):
                if not output:
                    return output
                attn_out = __gate(output[0])
                return (attn_out, *output[1:])
            return __gate(output)

        def gated_mlp_forward(*args, __orig=original_mlp_forward, __gate=gate_ffn, **kwargs):
            output = __orig(*args, **kwargs)
            if __gate is None:
                return output
            if isinstance(output, tuple):
                if not output:
                    return output
                ffn_out = __gate(output[0])
                return (ffn_out, *output[1:])
            return __gate(output)

        attn_mod.forward = gated_attn_forward  # type: ignore[assignment]
        mlp_mod.forward = gated_mlp_forward  # type: ignore[assignment]
        layer.add_module("attn_residual_gate", gate_attn)
        if gate_ffn is not None:
            layer.add_module("ffn_residual_gate", gate_ffn)


def _wrap_rotary_embedding(rotary_module: nn.Module, init_value: float) -> None:
    if hasattr(rotary_module, "rope_scale"):
        return
    scale_module = RoPEScale(init_value)
    original_forward = rotary_module.forward

    def scaled_forward(*args, __orig=original_forward, __scale=scale_module, **kwargs):
        cos, sin = __orig(*args, **kwargs)
        return __scale(cos, sin)

    rotary_module.forward = scaled_forward  # type: ignore[assignment]
    rotary_module.add_module("rope_scale", scale_module)


def attach_rope_scale(
    model: nn.Module,
    *,
    per_layer: bool = False,
    init_value: float = 1.0,
) -> None:
    """Attach learnable RoPE scaling either globally or per layer."""

    layers = _iter_decoder_layers(model)

    attention_modules: list[nn.Module] = []
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "head_dim"):
            attention_modules.append(module)

    if per_layer and attention_modules:
        for attn in attention_modules:
            if hasattr(attn, "rope_scale"):
                continue
            attn.add_module("rope_scale", RoPEScale(init_value))
        return

    shared = RoPEScale(init_value)
    name = "rope_scale_shared"
    index = 0
    while hasattr(model, name):
        index += 1
        name = f"rope_scale_shared_{index}"
    model.add_module(name, shared)
    for attn in attention_modules:
        setattr(attn, "_shared_rope_scale", shared)
