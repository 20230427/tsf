#!/usr/bin/env python3
"""Spectrum evidence figure (paper Fig. 3).

Four panels:
  (a) a real test window (channel 0) of the checkpoint's dataset;
  (b) its rFFT amplitude spectrum, dominant bins annotated with periods;
  (c) the *learned* complex spectral filter  W = Wr + i*Wi  of the frequency
      branch (freq_branch.filter, ComplexLinear), |W| heatmap: the diagonal is
      a per-bin band-pass gain, off-diagonal mass = learned cross-bin mixing;
  (d) diagonal gain vs mean off-diagonal magnitude per row (mixing profile).

Only needs a trained checkpoint + its data; no model forward is required.

Usage
-----
    python scripts/make_spectrum_figure.py \
        --ckpt checkpoints/ETTh1_best.pt --samples-per-day 24 \
        --out paper/neurocomputing/fig_spectrum.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Manuscript palette (keep in lockstep with scripts/make_data_figures.py).
BLUE, AMBER, VIOLET, TEAL = "#2C6FBB", "#E07B39", "#7A5AA6", "#2A9D8F"
INK, MUTED, GRID = "#22303C", "#60656C", "#C9CED4"


def load_window(ckpt_path: str, sample: int = 0):
    """Return (window (L, C), cfg) from the checkpoint's *test* split."""
    import torch

    from src.data.data_loader import get_dataloaders

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg["train"]["num_workers"] = 0
    _, _, test_loader, _, _ = get_dataloaders(cfg)
    for i, batch in enumerate(test_loader):
        if i == sample:
            x = batch[0]                      # (B, L, C), standardized space
            return x[0].numpy(), cfg
    raise RuntimeError(f"test loader has fewer than {sample + 1} batches")


def filter_weight(ckpt_path: str):
    """|W| (F, F), diag (F,), offdiag-row-mean (F,) of the complex filter."""
    import torch

    ms = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
    wr = ms["freq_branch.filter.wr.weight"].float()   # (F_out, F_in)
    wi = ms["freq_branch.filter.wi.weight"].float()
    mag = torch.sqrt(wr**2 + wi**2).numpy()
    diag = np.diag(mag)
    off = (mag.sum(axis=1) - diag) / max(mag.shape[1] - 1, 1)
    return mag, diag, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ETTh1_best.pt")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--samples-per-day", type=float, default=24.0,
                    help="ETTh1=24, ETTm1/m2=96, weather=144, electricity=24")
    ap.add_argument("--out", default="paper/neurocomputing/fig_spectrum.pdf")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    win, cfg = load_window(args.ckpt, args.sample)
    L = win.shape[0]
    y = win[:, args.channel]
    F_bins = np.arange(L // 2 + 1)
    spec = np.abs(np.fft.rfft(y, norm="ortho"))
    freq_cpd = F_bins / L * args.samples_per_day     # cycles per day

    mag, diag, off = filter_weight(args.ckpt)
    F = mag.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) window
    t_hours = np.arange(L) / args.samples_per_day * 24.0
    ax_a.plot(t_hours, y, color=INK, lw=0.9)
    ax_a.set_xlabel("time (h)")
    ax_a.set_title(f"(a) input window  L={L}, "
                   f"ch={args.channel}", loc="left", fontsize=9, fontweight="bold")

    # (b) amplitude spectrum
    ax_b.plot(freq_cpd[1:], spec[1:], color=AMBER, lw=1.1, marker="o", ms=2.2)
    top = np.argsort(spec[1:])[::-1][:3] + 1        # skip DC
    for k in top:
        per = 24.0 / freq_cpd[k] if freq_cpd[k] > 0 else np.inf
        lbl = f"{per:.1f} h" if per < 48 else f"{per/24:.1f} d"
        ax_b.annotate(lbl, (freq_cpd[k], spec[k]), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=7,
                      color=MUTED)
    ax_b.set_xlabel("frequency (cycles / day)")
    ax_b.set_ylabel("|rFFT(x)|")
    ax_b.set_title("(b) amplitude spectrum", loc="left", fontsize=9,
                   fontweight="bold")

    # (c) learned filter magnitude
    im = ax_c.imshow(mag, cmap="magma", norm=PowerNorm(0.5),
                     origin="lower", aspect="equal")
    ax_c.set_xlabel("input bin $f_{in}$")
    ax_c.set_ylabel("output bin $f_{out}$")
    ax_c.set_title(f"(c) learned filter $|W_r + i\\,W_i|$, "
                   f"{F}x{F}", loc="left", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax_c, fraction=0.045, pad=0.03)
    cb.set_label("|W|", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # (d) gain vs mixing profiles
    bin_cpd = F_bins / L * args.samples_per_day
    ax_d.plot(bin_cpd, diag, color=BLUE, lw=1.2, label="diagonal gain (band-pass)")
    ax_d.plot(bin_cpd, off, color=VIOLET, lw=1.2, ls="--",
              label="off-diagonal mean (bin mixing)")
    ax_d.set_xlabel("frequency (cycles / day)")
    ax_d.set_ylabel("magnitude")
    ax_d.legend(frameon=False, fontsize=7)
    ax_d.set_title("(d) per-bin gain vs mixing", loc="left", fontsize=9,
                   fontweight="bold")

    for ax in (ax_a, ax_b, ax_d):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.4, color=GRID, alpha=0.6)
    ax_c.grid(False)

    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"[saved] {out} and {out.with_suffix('.png')}")

    # Numbers for the caption / text.
    frac_mix = (mag.sum() - diag.sum()) / mag.sum()
    print(f"filter: F={F}, off-diagonal energy share = {frac_mix:.1%}")
    print(f"dominant bins (h): "
          + ", ".join(f"{24.0/freq_cpd[k]:.1f}" if freq_cpd[k] > 0 else "inf"
                      for k in top))


if __name__ == "__main__":
    main()
