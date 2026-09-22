"""Frequency-domain branch: rFFT + a learnable spectral encoder.

The window is transformed to the frequency domain with a real FFT, then one of
two encoders processes the spectrum (selectable via ``encoder``):

- ``linear`` — a complex-valued linear filter (two real matrices acting on the
  real and imaginary parts) that densely reweights and mixes frequency bins.
  Fixed mixing weights, O(n_freq^2) parameters; strong cheap baseline.
- ``mamba``  — a **bidirectional Mamba** (selective SSM) scanned over the
  frequency bins. Each bin's [real, imag] pair is embedded per position, and
  forward + backward scans let every bin condition on the whole spectrum with
  *input-dependent* (selective) mixing at O(n_freq) cost. Bidirectional because
  frequency has no arrow of time — a one-way scan would be an arbitrary bias.

Either way the branch captures global, periodic structure with a full-window
receptive field — complementary to the local view of the time-domain branch.
The linear path follows FreTS/FEDformer-style frequency MLPs; the Mamba path
follows FMamba-style spectral SSMs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_block import BiMambaEncoder, build_variate_encoder


class ComplexLinear(nn.Module):
    """Apply a complex linear map y = W x over the last (frequency) dim.

    W = W_r + i W_i acts on x = x_r + i x_i giving:
        y_r = W_r x_r - W_i x_i
        y_i = W_r x_i + W_i x_r
    Implemented with real weights so it runs on any backend.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.wr = nn.Linear(in_features, out_features)
        self.wi = nn.Linear(in_features, out_features)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi


class FreqBranch(nn.Module):
    """Produce a frequency-domain forecast and a per-channel feature.

    Input:  (B, L, C)
    Output: (feature (B, C, d_model), forecast (B, C, pred_len))
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        freq_hidden: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        sparsity: float = 0.0,
        encoder: str = "linear",
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_official_mamba: bool = True,
        channel_mixer_layers: int = 0,
        backbone: str = "none",
        zero_init: bool = True,
        zero_init_backbone: bool | None = None,
        zero_init_head: str = "auto",
        mixer_kind: str = "bimamba",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_freq = seq_len // 2 + 1  # rFFT output length
        self.sparsity = float(sparsity)
        self.encoder_kind = encoder

        # Optional FITS-style spectral linear backbone: a complex linear map
        # that interpolates the (low-passed) spectrum of the length-L window
        # up to the spectrum of a length-(L+H) window, then inverts it and
        # reads off the last H samples. Linear-in-frequency forecasting of
        # this form is near-SOTA on ETTh/weather, giving this branch a strong
        # linear anchor exactly like the time branch's DLinear backbone.
        # Zero-initialized (together with the deep head) so the branch starts
        # silent and learns its contribution instead of injecting noise.
        if backbone == "fits":
            self.n_out_freq = (seq_len + pred_len) // 2 + 1
            self.spec_backbone = ComplexLinear(self.n_freq, self.n_out_freq)
            # Allow the spectral-map initialization to be changed without
            # simultaneously randomizing the temporal head and fusion gate.
            do_zero_backbone = (
                zero_init if zero_init_backbone is None else zero_init_backbone
            )
            if do_zero_backbone:
                for lin in (self.spec_backbone.wr, self.spec_backbone.wi):
                    nn.init.zeros_(lin.weight)
                    nn.init.zeros_(lin.bias)
        elif backbone == "none":
            self.spec_backbone = None
        else:
            raise ValueError(f"Unknown freq backbone: {backbone!r}")

        if encoder == "linear":
            # Complex spectral filter that densely mixes frequency bins.
            self.filter = ComplexLinear(self.n_freq, self.n_freq)
            # Map the (real|imag) spectrum to the shared feature width.
            self.proj = nn.Sequential(
                nn.Linear(2 * self.n_freq, freq_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(freq_hidden, d_model),
            )
        elif encoder == "mamba":
            # Embed each bin's [real, imag] pair, scan bins bidirectionally.
            self.embed = nn.Linear(2, d_model)
            self.dropout = nn.Dropout(dropout)
            self.encoder = BiMambaEncoder(
                d_model=d_model,
                n_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_official=use_official_mamba,
            )
        else:
            raise ValueError(f"Unknown freq encoder: {encoder!r}")

        self.norm = nn.LayerNorm(d_model)
        # Optional cross-channel mixing over the variate dimension (see
        # TimeBranch): bidirectional Mamba across channels.
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
        # Spectral forecast head: this branch's own prediction of the horizon.
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, pred_len),
        )
        # Whether to zero-initialize the forecast head:
        #   'auto'   -- only when a spectral backbone is present, so the deep
        #               head acts as a correction to the linear anchor. This is
        #               the historical behaviour and remains the default, but
        #               note it leaves the head RANDOM whenever backbone='none',
        #               so the branch injects noise into the forecast from the
        #               first step.
        #   'always' -- zero-init regardless of the backbone. This makes the
        #               branch silent at initialization *without* introducing
        #               the FITS spectral map, which is what you want when the
        #               noise injection is the problem but the map's
        #               n_freq -> n_out_freq extrapolation is not affordable
        #               (that ratio grows with the horizon: 2.0x at H=96 but
        #               8.3x at H=720 for L=96).
        #   'never'  -- always random-init.
        if zero_init_head not in ("auto", "always", "never"):
            raise ValueError(
                f"Unknown zero_init_head: {zero_init_head!r} "
                "(use auto, always, never)"
            )
        do_zero = zero_init and (
            zero_init_head == "always"
            or (zero_init_head == "auto" and self.spec_backbone is not None)
        )
        if do_zero:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    @property
    def using_official_mamba(self) -> bool:
        return getattr(getattr(self, "encoder", None), "using_official_kernels", False)

    def _low_pass_mask(self, device) -> torch.Tensor:
        """Optionally zero out the highest `sparsity` fraction of frequencies."""
        if self.sparsity <= 0.0:
            return torch.ones(self.n_freq, device=device)
        keep = int(round(self.n_freq * (1.0 - self.sparsity)))
        keep = max(1, keep)
        mask = torch.zeros(self.n_freq, device=device)
        mask[:keep] = 1.0
        return mask

    def forward(self, x: torch.Tensor):
        # x: (B, L, C) -> operate along time (dim=1).
        b, l, c = x.shape
        xc = x.transpose(1, 2)                           # (B, C, L)
        spec = torch.fft.rfft(xc, dim=-1, norm="ortho")  # (B, C, n_freq) complex
        xr, xi = spec.real, spec.imag

        mask = self._low_pass_mask(x.device)
        xr = xr * mask
        xi = xi * mask

        if self.encoder_kind == "linear":
            yr, yi = self.filter(xr, xi)                 # learned spectral mixing
            feat = torch.cat([yr, yi], dim=-1)           # (B, C, 2*n_freq)
            feat = self.norm(self.proj(feat))            # (B, C, d_model)
        else:
            # mamba: bins as a sequence, channel-independent (fold C into batch).
            bins = torch.stack([xr, xi], dim=-1)         # (B, C, n_freq, 2)
            bins = bins.reshape(b * c, self.n_freq, 2)   # (B*C, n_freq, 2)
            h = self.dropout(self.embed(bins))           # (B*C, n_freq, d_model)
            h = self.encoder(h)                          # bidirectional scan
            summary = h.mean(dim=1)                      # (B*C, d_model)
            feat = self.norm(summary.reshape(b, c, -1))  # (B, C, d_model)

        if self.channel_mixer is not None:
            feat = self.channel_mixer(feat)              # mix across variates
        y = self.head(feat)                              # (B, C, H)

        if self.spec_backbone is not None:
            # FITS-style linear forecast: upsample the spectrum to length
            # L+H, invert, and take the horizon part.
            br, bi = self.spec_backbone(xr, xi)          # (B, C, n_out_freq)
            full = torch.fft.irfft(
                torch.complex(br.float(), bi.float()),
                n=self.seq_len + self.pred_len,
                dim=-1,
                norm="ortho",
            )
            y = y + full[..., self.seq_len :].to(y.dtype)  # (B, C, H)
        return feat, y
