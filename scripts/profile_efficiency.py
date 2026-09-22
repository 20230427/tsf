#!/usr/bin/env python
"""Measure the engineering profile of DD-Mamba per dataset config.

The model runs on the GPU, so the memory and throughput numbers are measured
*on the device*: peak GPU memory (``torch.cuda.max_memory_allocated``)
and wall-clock latency/throughput timed with warm-up and
``torch.cuda.synchronize()``. Model size and per-forecast MACs are
hardware-independent and reported alongside.

Columns
-------
  params      trainable parameters (millions)
  MACs        multiply-accumulates for one forecast, batch=1 (millions):
              nn.Linear + nn.Conv1d + the selective-scan recurrence
  peak mem    peak GPU memory during a forward pass (MB) at the profiling
              batch size (CUDA only; CPU falls back to an activation proxy)
  latency     mean forward-pass time per batch (ms)
  throughput  forecasts per second

Usage
-----
    python scripts/profile_efficiency.py                 # full profile on GPU
    python scripts/profile_efficiency.py --dispersion    # head overhead on GPU
    python scripts/profile_efficiency.py --device cuda --batch 32 --iters 100
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dual_domain_model import build_model  # noqa: E402
from src.models.mamba_block import MambaSSM  # noqa: E402
from src.utils import load_config  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU. "
                      "Peak-GPU-memory figures require a GPU.")
        name = "cpu"
    return torch.device(name)


def benchmark(model, x, stats, device, iters=50, warmup=10):
    """Time a forward pass on ``device`` and record peak GPU memory.

    Returns (latency_ms_per_batch, throughput_forecasts_per_s, peak_mem_mb).
    ``peak_mem_mb`` is real GPU peak memory on CUDA; on CPU it is ``nan``.
    """
    model = model.to(device).eval()
    x = x.to(device)
    stats = stats.to(device) if stats is not None else None
    is_cuda = device.type == "cuda"
    batch = x.shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            model(x, stats=stats)
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, stats=stats)
        if is_cuda:
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / iters
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float("nan")
    return dt * 1e3, batch / dt, peak_mb

# (label, config, representative channel count)
CONFIGS = [
    ("ETTh1", "configs/ETTh1.yaml", 7),
    ("Exchange", "configs/exchange_rate.yaml", 8),
    ("Weather", "configs/weather.yaml", 21),
    ("Solar", "configs/solar.yaml", 137),
    ("Electricity", "configs/electricity.yaml", 321),
    ("Traffic", "configs/traffic.yaml", 862),
]


def profile(cfg, n_channels, device=None, batch=1, iters=50):
    device = device or torch.device("cpu")
    model = build_model(cfg, n_channels).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs = {"v": 0}
    act_bytes = {"v": 0}
    handles = []

    def lin_hook(m, inp, out):
        macs["v"] += out.numel() * m.in_features
        act_bytes["v"] += out.numel() * out.element_size()

    def conv_hook(m, inp, out):
        k = m.kernel_size[0]
        macs["v"] += out.numel() * (m.in_channels // m.groups) * k
        act_bytes["v"] += out.numel() * out.element_size()

    def scan_hook(m, inp, out):
        # Recurrence: deltaA*h + deltaB_u and the C-einsum, per step.
        x = inp[0]
        b, l, _ = x.shape
        macs["v"] += 2 * b * l * m.d_inner * m.d_state
        act_bytes["v"] += out.numel() * out.element_size()

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.Conv1d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, MambaSSM):
            handles.append(mod.register_forward_hook(scan_hook))

    seq_len = cfg["data"]["seq_len"]
    # MACs/activation are measured once at batch=1 (hardware-independent).
    x1 = torch.randn(1, seq_len, n_channels)
    stats = None
    disp = cfg["model"].get("dispersion", "none")
    if disp in ("fixed", "learned"):
        res = cfg["model"].get("dispersion_resolutions", [seq_len, 144, 288, 336])
        stats = torch.randn(1, 2 * len(res), n_channels).abs()
    with torch.no_grad():
        model(x1, stats=stats)
    for h in handles:
        h.remove()

    # Timing + peak memory on the target device at the profiling batch size.
    xb = torch.randn(batch, seq_len, n_channels)
    statsb = stats.repeat(batch, 1, 1) if stats is not None else None
    lat_ms, thru, peak_mb = benchmark(model, xb, statsb, device, iters=iters)

    return {
        "params": params,
        "params_m": params / 1e6,
        "macs_m": macs["v"] / 1e6,
        "act_mb": act_bytes["v"] / 1e6,
        "lat_ms": lat_ms,
        "thru": thru,
        "peak_mb": peak_mb,
    }


def profile_dispersion_overhead(path, n_channels, device=None, batch=1, iters=50):
    """Return (base, learned) profiles so the caller can report the head's
    extra parameters / compute / GPU memory / latency --- the efficiency test
    for the dispersion experiment, measured on ``device``."""
    base_cfg = load_config(path)
    base_cfg["data"]["pred_len"] = 96
    base_cfg.setdefault("model", {})["dispersion"] = "none"
    disp_cfg = load_config(path)
    disp_cfg["data"]["pred_len"] = 96
    disp_cfg.setdefault("model", {})["dispersion"] = "learned"
    disp_cfg["model"]["dispersion_resolutions"] = [96, 144, 288, 336]
    return (profile(base_cfg, n_channels, device, batch, iters),
            profile(disp_cfg, n_channels, device, batch, iters))


#: channel counts for the standard configs (for the matched-arch mode).
CH_BY_PATH = {path: ch for _, path, ch in CONFIGS}
CH_BY_PATH.update({"configs/PEMS03.yaml": 358, "configs/PEMS04.yaml": 307,
                   "configs/PEMS07.yaml": 883, "configs/PEMS08.yaml": 170})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispersion", action="store_true",
                    help="report the dispersion head's efficiency overhead")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size for timing/peak-memory (default 32)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--archs", nargs="*", default=None,
                    help="matched-efficiency mode: profile these archs on one "
                         "--config (e.g. dual_domain smamba itransformer patchtst).")
    ap.add_argument("--config", default=None,
                    help="dataset config for --archs mode")
    ap.add_argument("--channels", type=int, default=None,
                    help="variate count for --archs mode (inferred for known configs)")
    ap.add_argument("--pred_len", type=int, default=96)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[device] {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
          + f" | batch={args.batch}")

    if args.archs:
        if not args.config:
            raise SystemExit("--archs requires --config")
        ch = args.channels or CH_BY_PATH.get(args.config)
        if ch is None:
            raise SystemExit(f"Unknown channel count for {args.config}; pass --channels")
        base = load_config(args.config)
        base["data"]["pred_len"] = args.pred_len
        print(f"\nMatched efficiency on {args.config} (C={ch}, H={args.pred_len}):")
        rows = []
        for arch in args.archs:
            cfg = load_config(args.config)
            cfg["data"]["pred_len"] = args.pred_len
            cfg["model"]["arch"] = arch
            try:
                r = profile(cfg, ch, device, args.batch, args.iters)
            except NotImplementedError as e:
                print(f"{arch:12s} skipped: {e}")
                continue
            rows.append((arch, r))
            print(f"{arch:12s} params={r['params_m']:7.3f}M  MACs={r['macs_m']:8.1f}M  "
                  f"peakmem={r['peak_mb']:7.1f}MB  lat={r['lat_ms']:7.2f}ms  "
                  f"thru={r['thru']:8.0f}/s")
        print("\n% LaTeX rows (arch & params(M) & MACs(M) & peakmem(MB) & lat(ms) & thru):")
        for arch, r in rows:
            print(f"{arch} & {r['params_m']:.2f} & {r['macs_m']:.0f} & "
                  f"{r['peak_mb']:.0f} & {r['lat_ms']:.1f} & {r['thru']:.0f} \\\\")
        return

    if args.dispersion:
        print(f"\n{'dataset':12s} {'C':>4s} {'+head params':>12s} {'overhead':>8s} "
              f"{'base ms':>9s} {'+head ms':>9s} {'base mem':>9s} {'+head mem':>10s} "
              f"{'mem +%':>7s}")
        for label, path, ch in CONFIGS:
            base, disp = profile_dispersion_overhead(path, ch, device, args.batch, args.iters)
            dp = disp["params"] - base["params"]
            memd = (disp["peak_mb"] / base["peak_mb"] - 1) if base["peak_mb"] > 0 else float("nan")
            print(f"{label:12s} {ch:4d} {dp:12,d} {dp/base['params']:7.2%} "
                  f"{base['lat_ms']:9.2f} {disp['lat_ms']:9.2f} "
                  f"{base['peak_mb']:8.1f}M {disp['peak_mb']:9.1f}M {memd:6.2%}")
        print("\nThe dispersion head adds a fixed number of parameters "
              "(independent of channel count C), and its latency/GPU-memory "
              "overhead shrinks as C grows. Run with --device cuda on the GPU.")
        return

    rows = []
    for label, path, ch in CONFIGS:
        cfg = load_config(path)
        cfg["data"]["pred_len"] = 96
        r = profile(cfg, ch, device, args.batch, args.iters)
        rows.append((label, ch, r))
        print(f"{label:12s} C={ch:4d}  params={r['params_m']:7.3f}M  "
              f"MACs={r['macs_m']:8.1f}M  peakmem={r['peak_mb']:7.1f}MB  "
              f"lat={r['lat_ms']:7.2f}ms  thru={r['thru']:8.0f}/s")

    print("\n% LaTeX rows (label & C & params(M) & MACs(M) & peakmem(MB) & lat(ms) & thru):")
    for label, ch, r in rows:
        print(f"{label} & {ch} & {r['params_m']:.2f} & {r['macs_m']:.0f} & "
              f"{r['peak_mb']:.0f} & {r['lat_ms']:.1f} & {r['thru']:.0f} \\\\")


if __name__ == "__main__":
    main()
