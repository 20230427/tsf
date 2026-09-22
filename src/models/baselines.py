"""Reference baselines implemented inside this framework.

Running a baseline through the exact same data pipeline, splits, training
loop, schedule, and evaluation script as the main model removes every
implementation confound from the comparison --- the fair-comparison
requirement reviewers rightly insist on. Select via ``model.arch`` in the
config (or ``--arch`` on scripts/run_main_results.py).

Faithfulness note
-----------------
The linear and Transformer baselines here are faithful reimplementations of
well-specified architectures (DLinear, NLinear, RLinear, PatchTST,
iTransformer). ``smamba`` is a faithful port of the authors' S-Mamba
implementation (:mod:`src.models.smamba_official`) and uses the official
``mamba_ssm`` CUDA kernels (no pure-PyTorch fallback). ms-Mamba reuses this
repo's selective-scan encoder (:mod:`src.models.mamba_block`), which falls
back to a pure-PyTorch scan when the official ``mamba-ssm`` kernels are
unavailable; it follows the published design (multi-scale parallel Mamba)
but is not the authors' code.
Any baseline can be replaced by an externally supplied official adapter via
``model.external_impl='package.module:ClassName'``. ``tf4tf`` has no faithful
in-repo implementation and therefore requires that adapter. For a camera-ready
comparison, prefer official code where its constructor/output contract can be
adapted to this pipeline; the in-repo versions are controlled same-pipeline
sanity implementations, not claims of author-code identity.
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import BiMambaEncoder, MambaEncoder
from .smamba_official import SMambaOfficial
from .time_branch import SeriesDecomp


def _wrap(out: torch.Tensor, return_components: bool):
    if return_components:
        return out, {"time": out, "freq": out}
    return out


class _InstanceNorm(nn.Module):
    """RevIN-style reversible per-window, per-variate normalization.

    ``normalize`` standardizes a look-back window by its own per-variate mean
    and std (optionally with learned affine parameters); ``denormalize``
    inverts the transform on the forecast. Statistics are cached per forward
    call, the standard RevIN usage pattern.
    """

    def __init__(self, n_channels: int | None = None, affine: bool = False,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            assert n_channels is not None, "affine RevIN needs n_channels"
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, C)
        self._mean = x.mean(1, keepdim=True).detach()
        self._std = torch.sqrt(
            x.var(1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:  # y: (B, H, C)
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * self._std + self._mean


class DLinearBaseline(nn.Module):
    """Faithful DLinear (Zeng et al., AAAI 2023), shared-weights variant.

    Moving-average decomposition into seasonal and trend components, each
    forecast by a single linear map from the look-back window to the
    horizon. No normalization, no deep components --- the canonical linear
    baseline for long-term forecasting.
    """

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.lin_seasonal = nn.Linear(seq_len, pred_len)
        self.lin_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, stats: torch.Tensor = None,
                return_components: bool = False):
        seasonal, trend = self.decomp(x)                       # (B, L, C)
        y = self.lin_seasonal(seasonal.transpose(1, 2)) + self.lin_trend(
            trend.transpose(1, 2)
        )                                                      # (B, C, H)
        return _wrap(y.transpose(1, 2), return_components)     # (B, H, C)


class NLinearBaseline(nn.Module):
    """Faithful NLinear (Zeng et al., AAAI 2023): subtract-last, linear, add.

    Subtracting the last look-back value before a single linear map is a
    minimal, effective handling of distribution shift on the standard
    benchmarks.
    """

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x, stats=None, return_components=False):
        last = x[:, -1:, :]                                    # (B, 1, C)
        z = self.linear((x - last).transpose(1, 2)).transpose(1, 2)
        return _wrap(z + last, return_components)


class RLinearBaseline(nn.Module):
    """RLinear (Li et al., 2023): RevIN + a single linear map.

    A reversible instance-normalized linear projection --- the strong linear
    map the review asks us to compare against directly.
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int):
        super().__init__()
        self.norm = _InstanceNorm(n_channels, affine=True)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)
        z = self.linear(z.transpose(1, 2)).transpose(1, 2)
        return _wrap(self.norm.denormalize(z), return_components)


class PatchTSTBaseline(nn.Module):
    """PatchTST (Nie et al., ICLR 2023): channel-independent patched Transformer.

    Each variate is normalized, split into overlapping patches, embedded, and
    processed by a shared Transformer encoder; a flatten-and-linear head maps
    the patch tokens to the horizon. Channel-independent (weights shared over
    variates).
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 patch_len: int = 16, stride: int = 8, dropout: float = 0.1,
                 d_ff: int | None = None):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.norm = _InstanceNorm(n_channels, affine=True)
        n_patches = (seq_len + stride - patch_len) // stride + 1
        self.n_patches = n_patches
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff or 2 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, x, stats=None, return_components=False):
        B, L, C = x.shape
        z = self.norm.normalize(x)                             # (B, L, C)
        z = z.permute(0, 2, 1).reshape(B * C, L)               # (B*C, L)
        z = F.pad(z, (0, self.stride), mode="replicate")       # PatchTST end-pad
        patches = z.unfold(-1, self.patch_len, self.stride)    # (B*C, n_p, p_len)
        h = self.embed(patches) + self.pos                     # (B*C, n_p, d)
        h = self.encoder(self.dropout(h))                      # (B*C, n_p, d)
        y = self.head(h.reshape(B * C, -1))                    # (B*C, H)
        y = y.reshape(B, C, -1).permute(0, 2, 1)               # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class _VariateTokenModel(nn.Module):
    """Shared skeleton for inverted (variate-token) models.

    Each variate's whole look-back series is embedded into one token; a
    sequence model mixes the ``C`` tokens; a linear head projects each token
    to the horizon. iTransformer, S-Mamba, and ms-Mamba differ only in the
    token mixer (:meth:`build_mixer`).
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 d_model: int, dropout: float):
        super().__init__()
        self.norm = _InstanceNorm(n_channels, affine=False)
        self.embed = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)
        self.mixer = self.build_mixer(d_model)

    def build_mixer(self, d_model: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)                             # (B, L, C)
        tok = self.embed(z.permute(0, 2, 1))                  # (B, C, d)
        tok = self.mixer(self.dropout(tok))                    # (B, C, d)
        y = self.head(tok).permute(0, 2, 1)                    # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class ITransformerBaseline(_VariateTokenModel):
    """iTransformer (Liu et al., ICLR 2024): attention over variate tokens."""

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 n_layers=2, dropout=0.1, d_ff=None):
        self._n_heads, self._n_layers, self._d_ff = n_heads, n_layers, d_ff
        super().__init__(seq_len, pred_len, n_channels, d_model, dropout)

    def build_mixer(self, d_model):
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=self._n_heads,
            dim_feedforward=self._d_ff or 2 * d_model, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        return nn.TransformerEncoder(enc_layer, num_layers=self._n_layers)


def _build_smamba(cfg_model: dict, seq_len: int, pred_len: int,
                  n_channels: int) -> nn.Module:
    """S-Mamba branch: official port (mamba_ssm kernels required)."""
    return SMambaOfficial(
        seq_len, pred_len, n_channels,
        d_model=cfg_model.get("baseline_d_model", 128),
        d_ff=cfg_model.get("baseline_d_ff", 256),
        n_layers=cfg_model.get("baseline_layers", 2),
        d_state=cfg_model.get("baseline_d_state", 16),
        dropout=cfg_model.get("baseline_dropout", 0.1),
    )


class MsMambaBaseline(nn.Module):
    """ms-Mamba (Karadag et al., 2025): parallel Mamba blocks at multiple scales.

    The look-back is subsampled at several rates; each scale embeds its
    variate series into a token and runs a Mamba encoder, and the per-scale
    tokens are summed before a horizon head. Captures the multi-scale
    processing of ms-Mamba with this repo's selective-scan encoder.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_layers=2,
                 d_state=16, scales=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.scales = tuple(scales)
        self.norm = _InstanceNorm(n_channels, affine=False)
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.ModuleList([
            nn.Linear(math.ceil(seq_len / s), d_model) for s in self.scales
        ])
        self.encoders = nn.ModuleList([
            MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state,
                         use_official=True)
            for _ in self.scales
        ])
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)                             # (B, L, C)
        zc = z.permute(0, 2, 1)                                # (B, C, L)
        tok = 0.0
        for s, embed, enc in zip(self.scales, self.embeds, self.encoders):
            sub = zc[:, :, ::s]                                # (B, C, ceil(L/s))
            tok = tok + enc(self.dropout(embed(sub)))          # (B, C, d)
        y = self.head(tok).permute(0, 2, 1)                    # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class _TwoStageAttention(nn.Module):
    """Crossformer's Two-Stage Attention (Zhang & Yan, ICLR 2023).

    Operates on segment embeddings of shape ``(B, C, S, d)`` (variates $C$,
    time segments $S$). Stage 1 attends across *time* segments within each
    variate; stage 2 attends across the *variate* dimension through a small
    fixed set of ``factor`` learned routers, which keeps the cross-dimension
    cost linear in $C$ (essential on the 862-channel Traffic set). The router
    set is shared across time segments (a common, faithful-enough
    simplification of the per-segment routers in the paper).
    """

    def __init__(self, d_model, n_heads, factor, dropout):
        super().__init__()
        mha = lambda: nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        ffn = lambda: nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.time_attn = mha()
        self.dim_send, self.dim_recv = mha(), mha()
        self.router = nn.Parameter(torch.randn(factor, d_model) * 0.02)
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d_model) for _ in range(4))
        self.ff1, self.ff2 = ffn(), ffn()

    def forward(self, x):                                      # (B, C, S, d)
        B, C, S, d = x.shape
        # Stage 1: cross-time attention, per variate.
        t = x.reshape(B * C, S, d)
        t = self.n1(t + self.time_attn(t, t, t)[0])
        t = self.n2(t + self.ff1(t))
        x = t.reshape(B, C, S, d)
        # Stage 2: router-based cross-dimension attention, per time segment.
        v = x.permute(0, 2, 1, 3).reshape(B * S, C, d)        # (B*S, C, d)
        r = self.router.unsqueeze(0).expand(B * S, -1, -1)    # (B*S, factor, d)
        buf = self.dim_send(r, v, v)[0]                       # routers gather
        v2 = self.dim_recv(v, buf, buf)[0]                    # variates read back
        v = self.n3(v + v2)
        v = self.n4(v + self.ff2(v))
        return v.reshape(B, S, C, d).permute(0, 2, 1, 3)      # (B, C, S, d)


class CrossformerBaseline(nn.Module):
    """Crossformer (Zhang & Yan, ICLR 2023): DSW embedding + Two-Stage Attention.

    Dimension-Segment-Wise embedding splits each variate's series into
    length-``seg_len`` segments and embeds each to ``d_model``; a stack of
    Two-Stage Attention layers models cross-time and cross-variate dependence;
    a flatten-and-linear head maps to the horizon. This is a faithful
    reproduction of Crossformer's core (DSW + router-based TSA); it omits the
    hierarchical multi-scale encoder--decoder merging, using a single-scale
    encoder with a linear prediction head (a common baseline simplification),
    so it should be read as a same-pipeline Crossformer-style baseline rather
    than the authors' exact network.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 n_layers=3, seg_len=12, factor=10, dropout=0.1):
        super().__init__()
        self.seg_len = seg_len
        self.pad = (seg_len - seq_len % seg_len) % seg_len
        seg_num = (seq_len + self.pad) // seg_len
        self.norm = _InstanceNorm(n_channels, affine=True)
        self.embed = nn.Linear(seg_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, seg_num, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _TwoStageAttention(d_model, n_heads, factor, dropout)
            for _ in range(n_layers)])
        self.head = nn.Linear(seg_num * d_model, pred_len)

    def forward(self, x, stats=None, return_components=False):
        B, L, C = x.shape
        z = self.norm.normalize(x).permute(0, 2, 1)           # (B, C, L)
        if self.pad:
            z = F.pad(z, (0, self.pad), mode="replicate")     # pad time axis
        z = z.reshape(B, C, -1, self.seg_len)                 # (B, C, S, seg_len)
        h = self.embed(z) + self.pos                          # (B, C, S, d)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        y = self.head(h.reshape(B, C, -1)).permute(0, 2, 1)   # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class _ResidualBlock(nn.Module):
    """TiDE's residual block: MLP + linear skip + LayerNorm."""

    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out), nn.Dropout(dropout),
        )
        self.skip = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


class TiDEBaseline(nn.Module):
    """TiDE (Das et al., TMLR 2023): time-series dense encoder.

    Channel-independent MLP forecaster. The normalized look-back of one
    variate is flattened, pushed through a stack of residual blocks (the dense
    encoder), decoded into ``pred_len`` per-step vectors, and reduced to a
    scalar per step by the temporal decoder; a global linear map from the
    look-back to the horizon is added as a residual, which is what keeps TiDE
    competitive with the pure linear models.

    Faithfulness note: TiDE's covariate projection is omitted because this
    pipeline feeds no dated covariates to any model --- every baseline here
    sees the raw look-back only. The rest (encoder/decoder depth, temporal
    decoder, linear residual) follows the paper.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=256,
                 n_layers=2, decoder_dim=8, temporal_hidden=64, dropout=0.1):
        super().__init__()
        self.pred_len, self.decoder_dim = pred_len, decoder_dim
        self.norm = _InstanceNorm(n_channels, affine=True)
        enc = [_ResidualBlock(seq_len, d_model, d_model, dropout)]
        enc += [_ResidualBlock(d_model, d_model, d_model, dropout)
                for _ in range(n_layers - 1)]
        self.encoder = nn.Sequential(*enc)
        dec = [_ResidualBlock(d_model, d_model, d_model, dropout)
               for _ in range(n_layers - 1)]
        dec += [_ResidualBlock(d_model, d_model, pred_len * decoder_dim, dropout)]
        self.decoder = nn.Sequential(*dec)
        self.temporal = _ResidualBlock(decoder_dim, temporal_hidden, 1, dropout)
        self.residual = nn.Linear(seq_len, pred_len)

    def forward(self, x, stats=None, return_components=False):
        B, L, C = x.shape
        z = self.norm.normalize(x).permute(0, 2, 1).reshape(B * C, L)
        h = self.decoder(self.encoder(z))                      # (B*C, H*p)
        h = h.reshape(B * C, self.pred_len, self.decoder_dim)
        y = self.temporal(h).squeeze(-1) + self.residual(z)    # (B*C, H)
        y = y.reshape(B, C, self.pred_len).permute(0, 2, 1)    # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class _AutoCorrelation(nn.Module):
    """Autoformer's Auto-Correlation (Wu et al., NeurIPS 2021).

    Replaces dot-product attention with period-based dependency discovery:
    the cross-correlation of queries and keys is obtained through the FFT,
    the top-$k$ lags are selected ($k=\\lfloor c\\log L\\rfloor$), and values
    are aggregated by rolling them to those lags with softmax weights.
    """

    def __init__(self, d_model, n_heads, factor=1, dropout=0.1):
        super().__init__()
        self.h, self.factor = n_heads, factor
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        B, L, _ = q.shape
        S = kv.shape[1]
        E = self.q.out_features // self.h
        # Autoformer truncates or zero-pads keys/values to the query length so
        # that a single rolled aggregation covers both self- and cross-blocks.
        Q = self.q(q).view(B, L, self.h, E)
        K = self.k(kv).view(B, S, self.h, E)
        V = self.v(kv).view(B, S, self.h, E)
        if S < L:
            pad = torch.zeros(B, L - S, self.h, E, device=q.device, dtype=q.dtype)
            K, V = torch.cat([K, pad], 1), torch.cat([V, pad], 1)
        else:
            K, V = K[:, :L], V[:, :L]

        qf = torch.fft.rfft(Q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        kf = torch.fft.rfft(K.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)   # (B, h, E, L)

        top_k = max(1, int(self.factor * math.log(L)))
        mean_corr = corr.mean(dim=(1, 2))                          # (B, L)
        weights, delays = torch.topk(mean_corr, top_k, dim=-1)     # (B, k)
        weights = torch.softmax(weights, dim=-1)

        Vp = V.permute(0, 2, 3, 1).contiguous()                    # (B, h, E, L)
        agg = torch.zeros_like(Vp)
        for i in range(top_k):
            # roll each batch element by its own delay via index gather
            idx = (torch.arange(L, device=q.device).view(1, 1, 1, L)
                   + delays[:, i].view(B, 1, 1, 1)) % L
            agg = agg + Vp.gather(-1, idx.expand_as(Vp)) * \
                weights[:, i].view(B, 1, 1, 1)
        agg = agg.permute(0, 3, 1, 2).reshape(B, L, -1)            # (B, L, d)
        return self.out(self.dropout(agg))


class _FrequencyEnhancedBlock(nn.Module):
    """FEDformer's Frequency Enhanced Block (Zhou et al., ICML 2022).

    Projects the sequence to the spectrum, keeps a fixed randomly selected
    subset of low-frequency modes, applies a learned complex linear map to
    those modes, and transforms back. Random mode selection is what makes the
    block linear-complexity while retaining a global view of the series.
    """

    def __init__(self, d_model, seq_len, modes=32, seed=0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        n_bins = seq_len // 2 + 1
        modes = min(modes, n_bins)
        # Fixed (buffer, not parameter) random selection, as in the paper.
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_bins, generator=g)[:modes].sort().values
        self.register_buffer("index", idx)
        scale = 1.0 / (d_model * d_model)
        self.wr = nn.Parameter(scale * torch.randn(modes, d_model, d_model))
        self.wi = nn.Parameter(scale * torch.randn(modes, d_model, d_model))

    def forward(self, x, _kv=None):                            # (B, L, d)
        B, L, d = x.shape
        z = torch.fft.rfft(self.proj(x), dim=1)                # (B, F, d)
        sel = z[:, self.index, :]                              # (B, m, d)
        real = torch.einsum("bmd,mde->bme", sel.real, self.wr) - \
            torch.einsum("bmd,mde->bme", sel.imag, self.wi)
        imag = torch.einsum("bmd,mde->bme", sel.real, self.wi) + \
            torch.einsum("bmd,mde->bme", sel.imag, self.wr)
        out = torch.zeros_like(z)
        out[:, self.index, :] = torch.complex(real, imag)
        return torch.fft.irfft(out, n=L, dim=1)


class _FrequencyEnhancedAttention(nn.Module):
    """FEDformer's Frequency Enhanced Attention: the cross block.

    Queries come from the decoder and keys/values from the encoder memory.
    Each side keeps its own randomly selected mode subset; the attention
    matrix is formed between the selected query and key modes, applied to the
    value modes, and the result is written back into the query spectrum. This
    is the block that carries encoder information into the decoder --- a
    self-block reused here would leave the encoder unconnected.
    """

    def __init__(self, d_model, q_len, kv_len, modes=32, seed=0):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(d_model, d_model) for _ in range(3))
        fq, fk = q_len // 2 + 1, kv_len // 2 + 1
        m = min(modes, fq, fk)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("idx_q", torch.randperm(fq, generator=g)[:m].sort().values)
        self.register_buffer("idx_kv", torch.randperm(fk, generator=g)[:m].sort().values)
        scale = 1.0 / (d_model * d_model)
        self.wr = nn.Parameter(scale * torch.randn(m, d_model, d_model))
        self.wi = nn.Parameter(scale * torch.randn(m, d_model, d_model))

    def forward(self, q, kv):
        L = q.shape[1]
        qf = torch.fft.rfft(self.q_proj(q), dim=1)[:, self.idx_q, :]
        kf = torch.fft.rfft(self.k_proj(kv), dim=1)[:, self.idx_kv, :]
        vf = torch.fft.rfft(self.v_proj(kv), dim=1)[:, self.idx_kv, :]
        # Attention over modes, weighted by the magnitude of q * conj(k).
        qk = torch.einsum("bmd,bnd->bmn", qf, torch.conj(kf))
        attn = torch.softmax(qk.abs(), dim=-1).to(vf.dtype)
        z = torch.einsum("bmn,bnd->bmd", attn, vf)             # (B, m, d)
        real = torch.einsum("bmd,mde->bme", z.real, self.wr) - \
            torch.einsum("bmd,mde->bme", z.imag, self.wi)
        imag = torch.einsum("bmd,mde->bme", z.real, self.wi) + \
            torch.einsum("bmd,mde->bme", z.imag, self.wr)
        out = torch.zeros(q.shape[0], L // 2 + 1, q.shape[2],
                          dtype=qf.dtype, device=q.device)
        out[:, self.idx_q, :] = torch.complex(real, imag)
        return torch.fft.irfft(out, n=L, dim=1)


class _DecompEncoderLayer(nn.Module):
    """Encoder layer of the Autoformer/FEDformer skeleton: mix -> decomp ->
    conv feed-forward -> decomp, keeping only the seasonal part."""

    def __init__(self, mixer, d_model, d_ff, kernel_size, dropout):
        super().__init__()
        self.mixer = mixer
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.dec1, self.dec2 = SeriesDecomp(kernel_size), SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(x, x))
        x, _ = self.dec1(x)
        y = self.conv2(F.gelu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x, _ = self.dec2(x + self.dropout(y))
        return x


class _DecompDecoderLayer(nn.Module):
    """Decoder layer: self-mix, cross-mix against the encoder memory, and a
    conv feed-forward, with the trend extracted at each decomposition and
    accumulated through a projection (Autoformer's progressive decomposition)."""

    def __init__(self, self_mixer, cross_mixer, d_model, c_out, d_ff,
                 kernel_size, dropout):
        super().__init__()
        self.self_mixer, self.cross_mixer = self_mixer, cross_mixer
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.dec1 = SeriesDecomp(kernel_size)
        self.dec2 = SeriesDecomp(kernel_size)
        self.dec3 = SeriesDecomp(kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.trend_proj = nn.Conv1d(d_model, c_out, 3, padding=1,
                                    padding_mode="circular", bias=False)

    def forward(self, x, memory):
        x = x + self.dropout(self.self_mixer(x, x))
        x, t1 = self.dec1(x)
        x = x + self.dropout(self.cross_mixer(x, memory))
        x, t2 = self.dec2(x)
        y = self.conv2(F.gelu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x, t3 = self.dec3(x + self.dropout(y))
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class _DecompTransformer(nn.Module):
    """Shared Autoformer/FEDformer network.

    Both models are the same decomposition encoder--decoder; they differ only
    in the block that mixes tokens (Auto-Correlation vs. the Frequency
    Enhanced Block), which is supplied by ``make_mixer``. The decoder is
    seeded with the trend mean and the seasonal part of the look-back's second
    half, and the forecast is the sum of the seasonal output and the
    accumulated trend.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model, n_heads,
                 e_layers, d_layers, d_ff, kernel_size, dropout, make_mixer):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.label_len = seq_len // 2
        self.decomp = SeriesDecomp(kernel_size)
        self.enc_embed = nn.Linear(n_channels, d_model)
        self.dec_embed = nn.Linear(n_channels, d_model)
        self.encoder = nn.ModuleList([
            _DecompEncoderLayer(make_mixer(seq_len, seq_len, False), d_model,
                                d_ff, kernel_size, dropout)
            for _ in range(e_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        dec_len = self.label_len + pred_len
        self.decoder = nn.ModuleList([
            _DecompDecoderLayer(make_mixer(dec_len, dec_len, False),
                                make_mixer(dec_len, seq_len, True),
                                d_model, n_channels, d_ff, kernel_size, dropout)
            for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, n_channels)

    def forward(self, x, stats=None, return_components=False):
        B, L, C = x.shape
        seasonal, trend = self.decomp(x)
        mean = x.mean(1, keepdim=True).expand(B, self.pred_len, C)
        trend_init = torch.cat([trend[:, -self.label_len:], mean], dim=1)
        zeros = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
        seasonal_init = torch.cat([seasonal[:, -self.label_len:], zeros], dim=1)

        memory = self.enc_embed(x)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.enc_norm(memory)

        h = self.dec_embed(seasonal_init)
        trend_out = trend_init
        for layer in self.decoder:
            h, t = layer(h, memory)
            trend_out = trend_out + t
        y = self.projection(self.dec_norm(h)) + trend_out
        return _wrap(y[:, -self.pred_len:], return_components)


class AutoformerBaseline(_DecompTransformer):
    """Autoformer (Wu et al., NeurIPS 2021): decomposition + Auto-Correlation."""

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 e_layers=2, d_layers=1, d_ff=None, kernel_size=25,
                 factor=1, dropout=0.1):
        d_ff = d_ff or 2 * d_model
        super().__init__(
            seq_len, pred_len, n_channels, d_model, n_heads, e_layers,
            d_layers, d_ff, kernel_size, dropout,
            make_mixer=lambda _q, _kv, _cross: _AutoCorrelation(
                d_model, n_heads, factor, dropout))


class FEDformerBaseline(_DecompTransformer):
    """FEDformer (Zhou et al., ICML 2022): decomposition + frequency-enhanced
    blocks with random mode selection (the Fourier, not wavelet, variant)."""

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 e_layers=2, d_layers=1, d_ff=None, kernel_size=25,
                 modes=32, dropout=0.1):
        d_ff = d_ff or 2 * d_model
        self._n = 0

        def make_mixer(q_len, kv_len, cross):
            self._n += 1
            if cross:
                return _FrequencyEnhancedAttention(d_model, q_len, kv_len,
                                                   modes, seed=self._n)
            return _FrequencyEnhancedBlock(d_model, q_len, modes, seed=self._n)

        super().__init__(seq_len, pred_len, n_channels, d_model, n_heads,
                         e_layers, d_layers, d_ff, kernel_size, dropout,
                         make_mixer=make_mixer)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Baselines with faithful / repo-native implementations, keyed by ``model.arch``.
BASELINE_ARCHS = (
    "dlinear", "nlinear", "rlinear", "patchtst", "itransformer",
    "smamba", "msmamba", "crossformer", "tide", "fedformer", "autoformer",
    "tf4tf",
)


def build_baseline(arch: str, cfg: dict, n_channels: int):
    """Return a baseline module for ``arch`` or ``None`` if it is not a baseline.

    Reads shared shape from ``cfg['data']`` and optional per-baseline
    hyperparameters from ``cfg['model']`` (all have sensible defaults). ``tf4tf``
    is an adapter: it requires ``model.external_impl='package.module:ClassName'``
    pointing at the authors' implementation (constructed with the same
    ``seq_len``/``pred_len``/``n_channels`` kwargs); there is intentionally no
    fabricated in-repo TF4TF.
    """
    d = cfg["data"]
    m = cfg.get("model", {})
    seq_len, pred_len = d["seq_len"], d["pred_len"]
    # A runner may bind an architecture name to the authors' implementation.
    # This check intentionally precedes the local registry so official code can
    # replace an in-repo reimplementation without changing table labels.  It
    # also permits newly published baselines (e.g. a close dual-domain model)
    # without pretending that this repository implements them faithfully.
    spec = m.get("external_impl")
    if spec:
        mod_name, separator, cls_name = str(spec).partition(":")
        if not separator or not mod_name or not cls_name:
            raise ValueError(
                "model.external_impl must use 'package.module:ClassName'"
            )
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(
            seq_len=seq_len,
            pred_len=pred_len,
            n_channels=n_channels,
        )
    if arch not in BASELINE_ARCHS:
        return None
    dm = m.get("baseline_d_model", 128)
    if arch == "dlinear":
        return DLinearBaseline(seq_len, pred_len, m.get("time_kernel_size", 25))
    if arch == "nlinear":
        return NLinearBaseline(seq_len, pred_len)
    if arch == "rlinear":
        return RLinearBaseline(seq_len, pred_len, n_channels)
    if arch == "patchtst":
        return PatchTSTBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 3),
            patch_len=m.get("patch_len", 16), stride=m.get("patch_stride", 8),
        )
    if arch == "itransformer":
        return ITransformerBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 2),
        )
    if arch == "smamba":
        return _build_smamba(m, seq_len, pred_len, n_channels)
    if arch == "msmamba":
        return MsMambaBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 2),
            d_state=m.get("baseline_d_state", 16),
            scales=tuple(m.get("ms_scales", (1, 2, 4))),
        )
    if arch == "crossformer":
        return CrossformerBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 3),
            seg_len=m.get("cross_seg_len", 12),
            factor=m.get("cross_factor", 10),
        )
    if arch == "tide":
        return TiDEBaseline(
            seq_len, pred_len, n_channels,
            d_model=m.get("tide_hidden", 256),
            n_layers=m.get("baseline_layers", 2),
            decoder_dim=m.get("tide_decoder_dim", 8),
        )
    if arch == "autoformer":
        return AutoformerBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            e_layers=m.get("baseline_layers", 2),
            kernel_size=m.get("time_kernel_size", 25),
        )
    if arch == "fedformer":
        return FEDformerBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            e_layers=m.get("baseline_layers", 2),
            kernel_size=m.get("time_kernel_size", 25),
            modes=m.get("fed_modes", 32),
        )
    if arch == "tf4tf":
        raise NotImplementedError(
            "TF4TF has no faithful in-repo implementation. Set "
            "model.external_impl='pkg.module:ClassName' to the authors' "
            "implementation (constructed with seq_len, pred_len, "
            "n_channels), or omit tf4tf from the baseline sweep."
        )
    return None  # pragma: no cover
