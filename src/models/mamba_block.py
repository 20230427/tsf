"""A self-contained Mamba (selective state-space) block.

This is a pure-PyTorch implementation of the Mamba-1 selective SSM
(Gu & Dao, 2023) that runs on any backend (CPU or GPU) with no custom CUDA
kernels. If the official ``mamba-ssm`` package *is* installed, ``MambaLayer``
transparently uses its fused, hardware-aware kernels instead for speed.

Why a pure fallback matters here: the official ``mamba-ssm`` /
``causal-conv1d`` kernels are frequently hard to build against brand-new
GPU/CUDA stacks. The pure path lets the
model train today; drop in the fast kernels later with zero code changes.

Shapes throughout: (B, L, D) where D == d_model.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fast path
    from mamba_ssm import Mamba as _OfficialMamba  # type: ignore

    _HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on environment
    _OfficialMamba = None
    _HAS_MAMBA_SSM = False


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the statistics in fp32 (autocast treats nn.LayerNorm this
        # way automatically, but not custom norms like this one).
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(x.dtype)


class MambaSSM(nn.Module):
    """Pure-PyTorch selective SSM core (the Mamba mixer)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        # Projects x -> (delta, B, C) parameters of the selective SSM.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is kept in log space and negated to guarantee stability (Re(A) < 0).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _selective_scan(
        self,
        u: torch.Tensor,       # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,       # (d_inner, d_state)
        B: torch.Tensor,       # (B, L, d_state)
        C: torch.Tensor,       # (B, L, d_state)
        D: torch.Tensor,       # (d_inner,)
    ) -> torch.Tensor:
        b, l, d_in = u.shape
        n = A.shape[1]
        # Zero-order-hold discretization.
        deltaA = torch.exp(delta.unsqueeze(-1) * A)               # (B, L, d_in, n)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B,L,d_in,n)

        # unbind() instead of per-step slicing: slicing a tensor that requires
        # grad makes every loop step's backward scatter-add into a fresh
        # full-size zeros tensor (O(L^2) memory traffic); unbind's backward is
        # a single stack. Identical values and gradients, much faster on CPU.
        deltaA_t = deltaA.unbind(1)
        deltaB_u_t = deltaB_u.unbind(1)
        C_t = C.unbind(1)
        h = torch.zeros(b, d_in, n, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(l):
            h = deltaA_t[t] * h + deltaB_u_t[t]                   # (B, d_in, n)
            y = torch.einsum("bdn,bn->bd", h, C_t[t])            # (B, d_in)
            ys.append(y)
        y = torch.stack(ys, dim=1)                                # (B, L, d_in)
        return y + u * D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The recurrent scan (exp(delta*A) + L-step state accumulation) is
        # numerically fragile in fp16/bf16: under AMP it under/overflows and
        # poisons training with NaNs. Run the whole mixer in fp32 and cast
        # back, mirroring what the official fused kernels do internally.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x.float()).to(in_dtype)

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x_res = self.in_proj(x)                                   # (B, L, 2*d_inner)
        x_in, res = x_res.chunk(2, dim=-1)

        # Causal depthwise conv over the sequence.
        x_in = x_in.transpose(1, 2)                               # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :l]                         # trim right pad
        x_in = x_in.transpose(1, 2)                               # (B, L, d_inner)
        x_in = F.silu(x_in)

        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)
        x_dbl = self.x_proj(x_in)                                 # (B, L, dt_rank+2n)
        delta, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))                   # (B, L, d_inner)

        y = self._selective_scan(x_in, delta, A, B_mat, C_mat, self.D)
        y = y * F.silu(res)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba layer: x + Mamba(RMSNorm(x)).

    Uses the official ``mamba_ssm.Mamba`` mixer when available (and running on
    CUDA), otherwise the pure-PyTorch :class:`MambaSSM`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self._official = None
        if use_official and _HAS_MAMBA_SSM:
            self._official = _OfficialMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.mixer = None
        else:
            self.mixer = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
            )

    @property
    def using_official_kernels(self) -> bool:
        return self._official is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        mix = self._official(normed) if self._official is not None else self.mixer(normed)
        return x + mix


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba: forward scan + backward scan, fused per layer.

    Mamba's scan is inherently directional. Over the *time* axis that is the
    right bias (causality), but over axes with no arrow of time — e.g. the
    frequency bins of a spectrum, or the variate/channel dimension — a single
    direction is arbitrary. Each layer here runs two mixers, one on the
    sequence and one on its reverse, sums them inside the residual so every
    position sees both sides, then applies a position-wise FFN (the
    S-Mamba/Transformer block recipe: mix -> FFN -> norm).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.fwd_layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.bwd_layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(2 * d_model, d_model),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return any(l.using_official_kernels for l in self.fwd_layers) or any(
            l.using_official_kernels for l in self.bwd_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd, ffn_norm, ffn in zip(
            self.fwd_layers, self.bwd_layers, self.ffn_norms, self.ffns
        ):
            # MambaLayer returns x + mix(norm(x)); combining both directions
            # and subtracting one x keeps a single residual stream.
            x = fwd(x) + bwd(x.flip(1)).flip(1) - x
            x = x + ffn(ffn_norm(x))
        return self.norm(x)


class UniMambaEncoder(nn.Module):
    """Forward-scan-only counterpart of :class:`BiMambaEncoder`.

    Same block recipe (mix -> FFN -> norm) and same interface, but a single
    direction. Exists so the bidirectional choice can be ablated rather than
    assumed: S-Mamba (Wang et al., Neurocomputing 2025, Tab. 5) reports
    uni-Mamba as a distinct arm because a unidirectional scan sees only one
    side of a sequence that has no arrow of time, and over the variate axis
    that is an arbitrary restriction.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model, d_state=d_state, d_conv=d_conv,
                    expand=expand, dt_rank=dt_rank, use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(2 * d_model, d_model),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return any(l.using_official_kernels for l in self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, ffn_norm, ffn in zip(self.layers, self.ffn_norms, self.ffns):
            x = layer(x)
            x = x + ffn(ffn_norm(x))
        return self.norm(x)


class AttentionEncoder(nn.Module):
    """Multi-head self-attention encoder, interface-compatible with
    :class:`BiMambaEncoder`.

    This is the alternative S-Mamba benchmarks its variate-correlation block
    against (their Tab. 5 replaces bi-Mamba with Attention), and the mechanism
    iTransformer uses over the variate axis. Providing it here means the
    "selective state space beats attention for cross-variate mixing" claim can
    be measured in our own pipeline instead of inherited from theirs.

    Attention over the variate axis is quadratic in the number of channels,
    which is the cost Mamba is meant to avoid; on the 862-883 channel datasets
    that difference is the point of the comparison.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        **_ignored,
    ):
        super().__init__()
        # d_model must divide evenly into heads; fall back to fewer heads.
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads //= 2
        self.attn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout,
                                      batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(2 * d_model, d_model),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for anorm, attn, fnorm, ffn in zip(
            self.attn_norms, self.attns, self.ffn_norms, self.ffns
        ):
            h = anorm(x)
            x = x + attn(h, h, h, need_weights=False)[0]
            x = x + ffn(fnorm(x))
        return self.norm(x)


def build_variate_encoder(kind: str, **kw):
    """Factory for the cross-variate mixer, so it can be ablated by name.

    'bimamba' is the default and reproduces the previous behaviour exactly.
    """
    if kind == "bimamba":
        return BiMambaEncoder(**kw)
    if kind == "unimamba":
        return UniMambaEncoder(**kw)
    if kind == "attention":
        return AttentionEncoder(**kw)
    raise ValueError(
        f"Unknown variate encoder: {kind!r} (use bimamba, unimamba, attention)"
    )


class MambaEncoder(nn.Module):
    """A stack of :class:`MambaLayer` blocks with a final norm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return any(layer.using_official_kernels for layer in self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
