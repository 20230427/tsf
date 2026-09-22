"""Analytical GPU-memory estimate for one DD-Mamba trial configuration.

The Pamba HPO gates trials with an analytic estimate of the model's memory
footprint (``estimate_pamba_memory_mb``); reproducing that idea for DD-Mamba
is cheaper and more accurate than porting their formula, because this repo can
simply build the model on CPU and count real parameters.

The estimate is deliberately rough — it only decides whether a trial is
obviously over budget before it is started (real OOM is still caught at
runtime and pruned). Components:

* weights + gradients + AdamW states: 16 bytes per trainable parameter;
* activations for the causal time-axis Mamba scan (the pure-torch fallback
  materializes a ``(B*C, L, d_inner, N)`` discretization tensor — estimated
  at full size; with the official fused CUDA kernels ~15% of it);
* activations for the bidirectional variate mixer, if present;
* a fixed CUDA/context overhead.
"""
from __future__ import annotations


def _official_kernels_available() -> bool:
    try:
        from src.models.mamba_block import _HAS_MAMBA_SSM
        return bool(_HAS_MAMBA_SSM)
    except Exception:
        return False


def _count_parameters(cfg: dict, n_channels: int) -> int:
    from src.models.dual_domain_model import build_model

    model = build_model(cfg, int(n_channels))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    return n_params


def estimate_ddmamba_memory_mb(cfg: dict, n_channels: int | None) -> float:
    """Return an estimated peak-memory figure in MB for one training run."""
    if n_channels is None:
        return 4096.0  # conservative default when the channel count is unknown

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    try:
        n_params = _count_parameters(cfg, n_channels)
    except Exception:
        d = int(model_cfg.get("d_model", 128))
        n_params = 12 * d * d  # very rough fallback

    bytes_per_param = 4 + 4 + 8  # weights + grads + AdamW (m, v)
    param_mem_mb = n_params * bytes_per_param / 1e6

    batch = int(train_cfg.get("batch_size", 32))
    channels = int(n_channels)
    seq_len = int(data_cfg.get("seq_len", 96))
    d_model = int(model_cfg.get("d_model", 128))
    d_state = int(model_cfg.get("mamba_d_state", 16))
    expand = int(model_cfg.get("mamba_expand", 2))
    d_inner = expand * d_model

    act_mb = 0.0
    if model_cfg.get("time_encoder", "mamba") == "mamba":
        # Selective-scan discretization tensor (B*C, L, d_inner, N) plus the
        # hidden-state stack; the fused kernels never materialize it in full.
        # The kernel-path factor is calibrated against the 2026-09-06
        # electricity sweep: the anchor (d512, shared mixer x2, d_state 32,
        # 321 channels, batch 16) was estimated at 18.1 GB with the old 0.15
        # factor yet OOMed a 24.5 GB card on the first step -- the official
        # kernels still autograd-save enough intermediates to need ~2x that
        # rough figure. 0.30 gates the verified-OOM combo while keeping the
        # verified-fitting d512 + d_state 16 + single-mixer combo searchable.
        scan_mb = batch * channels * seq_len * d_inner * d_state * 4 / 1e6
        act_mb += scan_mb * (0.30 if _official_kernels_available() else 2.2)
        act_mb += batch * channels * seq_len * d_model * 4 / 1e6 * 4  # embeddings/residuals

    mixer_layers = int(model_cfg.get("channel_mixer_layers", 0))
    if mixer_layers > 0:
        # Bidirectional scan over C variate tokens, per layer.
        act_mb += batch * channels * d_inner * d_state * 4 / 1e6 * 2 * mixer_layers
        act_mb += batch * channels * d_model * 4 / 1e6 * 4 * mixer_layers

    if model_cfg.get("freq_encoder", "linear") == "mamba":
        act_mb += batch * channels * (seq_len // 2 + 1) * d_inner * d_state * 4 / 1e6 * 0.15

    overhead_mb = 512.0
    return (param_mem_mb + act_mb + overhead_mb) * 1.05
