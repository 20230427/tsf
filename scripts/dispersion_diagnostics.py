#!/usr/bin/env python
"""Data-driven dispersion diagnostics: is the time-varying scale real?

For each dataset this script (no model training required):

  * standardizes each channel, computes a rolling mean and rolling standard
    deviation over the dominant daily/weekly period;
  * quantifies time-varying dispersion by the coefficient of variation of the
    rolling std over time (CV_scale = std_t(roll_std) / mean_t(roll_std));
  * tests whether the *scale itself* is non-stationary: ADF and KPSS on the
    rolling-std series (per channel, on a sample), plus a Ljung-Box test for
    residual autocorrelation.

It writes ``paper/fig_dispersion.pdf`` (series + rolling std for each dataset,
a CV_scale bar, and the measured RevIN on/off effect from the ablations) and
prints a stationarity summary. This turns the paper's "Solar's periodic zeros"
interpretation into measured evidence.

Usage:  python scripts/dispersion_diagnostics.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data.dataset import load_raw_series  # noqa: E402

# dataset -> (loader kwargs, rolling window = dominant period, label)
DATASETS = {
    "Solar":   (dict(source="csv", csv_path="data/solar_AL.txt.gz"), 144, "10-min, daily=144"),
    "Traffic": (dict(source="csv", csv_path="data/traffic.txt.gz"),  168, "hourly, weekly=168"),
    "ETTh1":   (dict(source="csv", csv_path="data/ETTh1.csv"),         24, "hourly, daily=24"),
}
# Measured RevIN on/off effect (avg MSE, from the ablation studies): positive
# = RevIN helps (removing it hurts); negative = RevIN hurts.
REVIN_EFFECT = {"Solar": -0.029, "Traffic": +0.100, "ETTh1": +0.006}

# Palette matched to the manuscript's schematic figures.
C_SERIES = "#2C6FBB"   # blue  - standardized series
C_STD = "#E07B39"      # amber - rolling std / dispersion
C_TEAL = "#2A9D8F"     # teal  - KPSS
C_POS = "#2C6FBB"      # RevIN helps (blue)
C_NEG = "#C1443C"      # RevIN hurts (red)
C_INK = "#22303C"


def rolling(x1d, w):
    """Rolling mean and std of a 1-D array (valid, centered), length-preserving."""
    n = len(x1d)
    csum = np.concatenate([[0.0], np.cumsum(x1d)])
    csq = np.concatenate([[0.0], np.cumsum(x1d ** 2)])
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) - w // 2 + w, 0, n)
    cnt = np.maximum(hi - lo, 1)
    m = (csum[hi] - csum[lo]) / cnt
    v = (csq[hi] - csq[lo]) / cnt - m ** 2
    return m, np.sqrt(np.clip(v, 1e-8, None))


def analyze(name, kwargs, w):
    raw = load_raw_series(target_columns=None, synthetic_length=0,
                          synthetic_channels=0, seed=0, **kwargs)
    # Use the training portion (first 60%) so the diagnosis matches training.
    raw = raw[: int(0.6 * len(raw))]
    z = (raw - raw.mean(0, keepdims=True)) / (raw.std(0, keepdims=True) + 1e-8)

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss

    cvs, kpss_rej, adf_nonstat, lb_rej = [], 0, 0, 0
    ch_idx = np.linspace(0, z.shape[1] - 1, min(20, z.shape[1])).astype(int)
    roll_std_all = np.zeros((len(z), z.shape[1]), dtype=np.float32)
    for c in range(z.shape[1]):
        _, rstd = rolling(z[:, c], w)
        roll_std_all[:, c] = rstd
        cvs.append(rstd.std() / (rstd.mean() + 1e-8))
    for c in ch_idx:
        rstd = roll_std_all[:, c]
        s = rstd[::max(1, len(rstd) // 3000)]      # subsample for speed
        try:
            if kpss(s, regression="c", nlags="auto")[1] < 0.05:  # reject stationarity
                kpss_rej += 1
            if adfuller(s, autolag="AIC")[1] > 0.05:             # fail to reject unit root
                adf_nonstat += 1
        except Exception:
            pass
        # Ljung-Box on the (log) rolling-std series: is the scale autocorrelated?
        try:
            lb = acorr_ljungbox(np.log(rstd + 1e-6), lags=[w], return_df=True)
            if lb["lb_pvalue"].iloc[0] < 0.05:
                lb_rej += 1
        except Exception:
            pass
    n = len(ch_idx)
    return {
        "z": z, "roll_std": roll_std_all, "cv": float(np.mean(cvs)),
        "kpss_frac": kpss_rej / n, "adf_frac": adf_nonstat / n, "lb_frac": lb_rej / n,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": C_INK,
        "text.color": C_INK, "axes.labelcolor": C_INK, "xtick.color": C_INK,
        "ytick.color": C_INK, "axes.titlesize": 9.5, "pdf.fonttype": 42,
        "svg.fonttype": "none"})

    res = {}
    for name, (kw, w, _) in DATASETS.items():
        print(f"[{name}] analyzing...", flush=True)
        res[name] = analyze(name, kw, w)

    names = list(DATASETS)
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.10, wspace=0.06)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0])

    def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.0):
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        ha="center", va=va, fontsize=7,
                        xytext=(0, 2 if h >= 0 else -2), textcoords="offset points")

    # Top row: representative channel series + rolling std band.
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        z = res[name]["z"]; rstd = res[name]["roll_std"]
        c = z.shape[1] // 2
        t = np.arange(min(len(z), 1400))
        ax.plot(t, z[t, c], color=C_SERIES, lw=0.5, alpha=0.9)
        ax2 = ax.twinx()
        ax2.plot(t, rstd[t, c], color=C_STD, lw=1.3)
        ax2.set_ylim(bottom=0)
        ax.set_title(name, fontsize=9.5, loc="left", color=C_INK, fontweight="bold")
        ax.set_xlabel("time", fontsize=8)
        ax.tick_params(labelsize=6.8)
        ax2.tick_params(labelsize=6.8, colors=C_STD)
        ax2.spines["right"].set_color(C_STD)
        if j == 0:
            ax.set_ylabel("standardized value", fontsize=8, color=C_SERIES)
        if j == 2:
            ax2.set_ylabel("rolling std", color=C_STD, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    # Shared styling for the three bottom bar panels.
    def _barpanel(cell, values, colors, ylabel, title, ylim=None, fmt="{:.2f}"):
        ax = fig.add_subplot(cell)
        bars = ax.bar(names, values, color=colors, width=0.62,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, loc="left", color=C_INK, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", color=C_INK, alpha=0.10, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
        _bar_labels(ax, bars, fmt=fmt)
        return ax

    _barpanel(gs[1, 0], [res[n]["cv"] for n in names], C_STD,
              "CV of rolling std", "(a) Time-varying scale")
    axr = _barpanel(gs[1, 1], [REVIN_EFFECT[n] for n in names],
                    [C_POS if REVIN_EFFECT[n] > 0 else C_NEG for n in names],
                    r"$\Delta$MSE (off $-$ on)", "(b) RevIN effect",
                    ylim=(-0.05, 0.118), fmt="{:+.3f}")
    axr.axhline(0, lw=0.8, color=C_INK, zorder=2)
    _barpanel(gs[1, 2], [res[n]["kpss_frac"] for n in names], C_TEAL,
              "frac. non-stationary", "(c) KPSS on rolling std", ylim=(0, 1.12))

    fig.savefig("paper/fig_dispersion.pdf")
    fig.savefig("paper/fig_dispersion.png", dpi=200)
    print("[saved] paper/fig_dispersion.pdf")

    print("\n=== Dispersion diagnostics (train split) ===")
    print(f"{'dataset':8s} {'CV_scale':>9s} {'KPSS_nonstat':>13s} "
          f"{'ADF_nonstat':>12s} {'LjungBox_ac':>12s} {'RevIN_eff':>10s}")
    for n in names:
        r = res[n]
        print(f"{n:8s} {r['cv']:9.3f} {r['kpss_frac']:13.2f} {r['adf_frac']:12.2f} "
              f"{r['lb_frac']:12.2f} {REVIN_EFFECT[n]:+10.3f}")


if __name__ == "__main__":
    main()
