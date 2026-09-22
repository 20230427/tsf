"""Time-domain branch: decomposition, linear backbone, and a sequence encoder.

The branch decomposes the input into a slow-moving trend and a seasonal
residual (a la Autoformer/DLinear) and forecasts with two cooperating paths,
both inside the branch:

1. a *linear backbone* — per-component linear maps from the look-back window
   straight to the horizon (the DLinear recipe, near-SOTA on ETT-style data);
2. a **Mamba** (selective state-space) encoder over the time axis whose
   zero-initialized head adds nonlinear corrections for long-range dynamics.

Operating directly on the raw samples lets this branch capture local,
non-periodic and trend structure that the frequency branch (which sees global
spectral content) tends to smooth out.

The encoder is selectable via
``encoder={mamba, unimamba, bimamba, attention, mlp, none}``; causal Mamba is
the default.  The additional choices are deliberately interface-compatible so
the temporal-dependency block can be replaced or removed without changing the
DLinear anchor, forecast head, or cross-variate mixer.
Processing is channel-independent: every variable is encoded with shared
weights. The branch returns both its forecast and a per-channel feature used
by the fusion gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_block import (
    AttentionEncoder,
    BiMambaEncoder,
    MambaEncoder,
    build_variate_encoder,
)


class MovingAvg(nn.Module):
    """Moving-average smoother used to extract the trend component."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> pad the ends by replication so length is preserved.
        pad = self.kernel_size - 1
        front = pad // 2
        back = pad - front
        x_pad = torch.cat(
            [x[:, :1].repeat(1, front, 1), x, x[:, -1:].repeat(1, back, 1)], dim=1
        )
        out = self.avg(x_pad.transpose(1, 2)).transpose(1, 2)
        return out


class SeriesDecomp(nn.Module):
    """Split a series into (seasonal, trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class TimeBranch(nn.Module):
    """Produce a time-domain forecast and a per-channel feature.

    The branch owns a complete forecasting path: a *linear backbone*
    (DLinear-style per-component maps from the look-back window straight to
    the horizon — decomposition + linear projection is the canonical
    time-domain model) plus a Mamba/MLP encoder whose zero-initialized head
    adds a learned nonlinear correction. Training therefore starts from an
    exact linear forecaster, but the full prediction is computed *inside*
    this branch — there is no path around the branch.

    Input:  (B, L, C)
    Output: (feature (B, C, d_model), forecast (B, C, pred_len))
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        kernel_size: int = 25,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        encoder: str = "mamba",
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_official_mamba: bool = True,
        channel_mixer_layers: int = 0,
        use_linear_backbone: bool = True,
        zero_init: bool = True,
        zero_init_head: bool | None = None,
        mixer_kind: str = "bimamba",
    ):
        super().__init__()
        self.encoder_kind = encoder
        self.use_linear_backbone = use_linear_backbone
        self.decomp = SeriesDecomp(kernel_size)
        # Linear backbone: per-component window -> horizon maps.
        self.lin_seasonal = nn.Linear(seq_len, pred_len)
        self.lin_trend = nn.Linear(seq_len, pred_len)
        # Per-timestep embedding of the [seasonal, trend] pair -> d_model.
        self.embed = nn.Linear(2, d_model)
        # Whole-window embedding per variate (S-Mamba-style): gives the
        # feature a direct global view instead of relying on the encoder's
        # last state alone.
        self.series_embed = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # Optional cross-channel mixing: a bidirectional Mamba over the
        # variate dimension (channel order is arbitrary -> bidirectional),
        # letting each channel's forecast use the other channels' state.
        # Channel-independent models leave this accuracy on the table on
        # multivariate benchmarks (electricity, traffic, weather).
        self.channel_mixer = (
            build_variate_encoder(
                mixer_kind,
                d_model=d_model,
                n_layers=channel_mixer_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_official=use_official_mamba,
                ffn_dropout=dropout,
            )
            if channel_mixer_layers > 0
            else None
        )
        # Correction head on top of the encoder feature; zero-initialized so
        # the branch's initial forecast is exactly the linear backbone.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, pred_len),
        )
        # ``zero_init_head`` isolates the correction-head initialization from
        # the frequency backbone/head and fusion gate.  ``None`` preserves the
        # historical global ``zero_init`` behaviour bit-for-bit.
        do_zero_head = zero_init if zero_init_head is None else zero_init_head
        if do_zero_head:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        if encoder in ("mamba", "unimamba"):
            self.encoder = MambaEncoder(
                d_model=d_model,
                n_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_official=use_official_mamba,
            )
        elif encoder == "bimamba":
            self.encoder = BiMambaEncoder(
                d_model=d_model,
                n_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_official=use_official_mamba,
                ffn_dropout=dropout,
            )
        elif encoder == "attention":
            self.encoder = AttentionEncoder(
                d_model=d_model,
                n_layers=mamba_layers,
                ffn_dropout=dropout,
                attn_dropout=dropout,
            )
        elif encoder == "mlp":
            # Length-mixing MLP fallback over the flattened sequence.
            self.encoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        elif encoder == "none":
            self.encoder = nn.Identity()
        else:
            raise ValueError(
                f"Unknown time encoder: {encoder!r} "
                "(use mamba, unimamba, bimamba, attention, mlp, none)"
            )

    @property
    def using_official_mamba(self) -> bool:
        return getattr(self.encoder, "using_official_kernels", False)

    def forward(self, x: torch.Tensor):
        b, l, c = x.shape
        seasonal, trend = self.decomp(x)                 # each (B, L, C)

        # Linear backbone forecast.
        y_lin = self.lin_seasonal(seasonal.transpose(1, 2)) + self.lin_trend(
            trend.transpose(1, 2)
        )                                                # (B, C, H)

        # Stack components as 2 features per (timestep, channel).
        feats = torch.stack([seasonal, trend], dim=-1)   # (B, L, C, 2)
        # Channel-independent: fold channels into the batch dimension.
        feats = feats.permute(0, 2, 1, 3).reshape(b * c, l, 2)  # (B*C, L, 2)
        h = self.dropout(self.embed(feats))              # (B*C, L, d_model)

        if self.encoder_kind in ("mamba", "unimamba"):
            h = self.encoder(h)                          # (B*C, L, d_model)
            summary = h[:, -1, :]                        # last state (causal) -> (B*C, D)
        elif self.encoder_kind == "mlp":
            h = h + self.encoder(h)                      # residual refine
            summary = h.mean(dim=1)                      # (B*C, D)
        elif self.encoder_kind in ("bimamba", "attention"):
            h = self.encoder(h)                          # global temporal context
            summary = h.mean(dim=1)                      # no privileged direction
        else:  # none: remove TD encoder, retain embedding/head/backbone
            summary = h.mean(dim=1)

        feat = summary.reshape(b, c, -1)                 # (B, C, d_model)
        feat = feat + self.series_embed(x.transpose(1, 2))  # global window view
        if self.channel_mixer is not None:
            feat = self.channel_mixer(feat)              # mix across variates
        y = self.head(feat)                              # (B, C, H)
        if self.use_linear_backbone:
            y = y + y_lin
        return feat, y
