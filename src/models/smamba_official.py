"""Official S-Mamba architecture (Wang et al., Neurocomputing 2025).

Faithful port of the authors' implementation (``model/S_Mamba.py`` +
``layers/Mamba_EncDec.py`` from the official repository), adapted to this
repo's model conventions (``forward(x, stats=None, return_components=False)``
returning ``(B, H, C)``).

Architecture
------------
Non-stationary instance normalization -> inverted value embedding
``Linear(seq_len, d_model)`` -> ``n_layers`` x [ bidirectional Mamba block
(forward scan plus a scan over the time-flipped input, residual) ->
LayerNorm -> Conv1d FFN (d_model -> d_ff -> d_model, GELU) -> LayerNorm ]
-> LayerNorm -> per-variate projector ``Linear(d_model, pred_len)``
-> de-normalization.

Documented deviations from the authors' code
--------------------------------------------
1. The official ``EncoderLayer`` constructs three modules (``man``, ``man2``,
   and an ``AttentionLayer``) that are **never used in ``forward``** — they
   only inflate the parameter count. This port omits them; outputs are
   identical to the official implementation.
2. The official pipeline feeds time-stamp features (``x_mark``) alongside the
   values. This repo's unified baseline protocol provides no time features
   (``DataEmbedding_inverted``'s ``x_mark=None`` path), matching the same
   data pipeline used by every other baseline here.

The ``Mamba`` blocks come from the official ``mamba_ssm`` package
(``d_conv=2``, ``expand=1``, as in the authors' code), so this baseline
requires ``mamba_ssm`` (and its ``causal_conv1d`` dependency) to be
installed with CUDA kernels — there is intentionally no pure-PyTorch
fallback for the official variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    _MAMBA_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc


class SMambaEncoderLayer(nn.Module):
    """Official S-Mamba encoder layer: BiMamba mixing + Conv1d FFN."""

    def __init__(self, mamba: nn.Module, mamba_r: nn.Module, d_model: int,
                 d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.mamba = mamba
        self.mamba_r = mamba_r
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # Bidirectional mixing: forward scan + scan over the flipped sequence.
        new_x = self.mamba(x) + self.mamba_r(x.flip(dims=[1])).flip(dims=[1])
        x = self.norm1(x + new_x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SMambaOfficial(nn.Module):
    """Official S-Mamba forecaster over inverted (variate) tokens."""

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 d_model: int = 128, d_ff: int = 256, n_layers: int = 2,
                 d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "The official S-Mamba baseline requires the mamba_ssm package "
                "(CUDA kernels). Install it in the active environment, e.g. "
                "pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.0. "
                f"Import error was: {_MAMBA_IMPORT_ERROR}"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Inverted value embedding (no time features — see module docstring).
        self.enc_embedding = nn.Sequential(
            nn.Linear(seq_len, d_model), nn.Dropout(dropout),
        )
        self.encoder = nn.ModuleList([
            SMambaEncoderLayer(
                Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1),
                Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1),
                d_model, d_ff, dropout=dropout, activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x, stats=None, return_components=False):
        """x: (B, L, C) -> forecast (B, H, C)."""
        # Non-stationary Transformer normalization (statistics detached).
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        enc_out = self.enc_embedding(x.permute(0, 2, 1))      # (B, C, d)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out = self.encoder_norm(enc_out)
        dec_out = self.projector(enc_out).permute(0, 2, 1)    # (B, H, C)

        # De-normalization.
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        out = dec_out[:, -self.pred_len:, :]
        if return_components:
            return out, {}
        return out
