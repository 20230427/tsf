#!/usr/bin/env python3
"""Two data figures for the paper, in the manuscript's rounded-panel style.

  fig_branch_matrix.pdf : the 27-cell branch-ablation matrix as two heatmaps
                          (frequency- and time-branch removal effect, dataset x
                          horizon), diverging color with a gray midpoint, in-cell
                          Delta-MSE, and a star for effects surviving
                          Benjamini-Hochberg correction (q<0.05). Source:
                          results/stats_correction.json.
  fig_pcc_mixer.pdf     : mean absolute cross-channel PCC per dataset (one hue),
                          with the variate mixer's on/off decision shown by
                          hatch + label (texture as the accessible secondary
                          encoding), a coupling-threshold guide, and the Exchange
                          co-trending exception flagged. Sources: results/pcc.json
                          and configs/*.yaml.

Design follows the repo's data-figure conventions (matched palette, direct
labels, legend, no color-alone encoding) inside rounded-corner panels that echo
the schematic figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "neurocomputing"

# Manuscript palette.
BLUE, AMBER, TEAL, VIOLET = "#2C6FBB", "#E07B39", "#2A9D8F", "#7A5AA6"
INK, MUTED, GRID = "#22303C", "#60656C", "#22303C"
PANEL_BLUE, PANEL_GREEN, PANEL_LILAC = "#EAF0FA", "#EAF3EC", "#F3ECF8"
BORDER = "#9AA0A6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": INK,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
})

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Solar",
            "Electricity", "Traffic", "Exchange"]
KEY = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
       "Weather": "weather", "Solar": "solar", "Electricity": "electricity",
       "Traffic": "traffic", "Exchange": "exchange_rate"}
HORIZONS = [96, 192, 336, 720]


def rounded_panel(fig, x, y, w, h, tint):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((x + 0.004, y - 0.006), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 facecolor="black", edgecolor="none", alpha=0.08, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.02",
                 facecolor=tint, edgecolor=BORDER, linewidth=1.0,
                 linestyle=(0, (5, 3)), zorder=0.5))
    return ax


# One diverging ramp for BOTH panels (same quantity -> same scale): cool teal
# = removal helps (branch redundant), gray = no effect, amber = removal hurts
# (branch matters). Branch identity is carried by the panel title, not the hue.
def diverging():
    return LinearSegmentedColormap.from_list(
        "dd", [(0.0, TEAL), (0.5, "#EEEEEE"), (1.0, AMBER)])


def branch_grid(rows, branch):
    """(effect grid, q grid) for a branch; NaN where the cell was not run."""
    eff = np.full((len(DATASETS), len(HORIZONS)), np.nan)
    q = np.full_like(eff, np.nan)
    for r in rows:
        if r["branch"] != branch:
            continue
        name, h = r["cell"].split("@")
        inv = {v: k for k, v in KEY.items()}
        ds = inv.get(name, name)
        if ds in DATASETS and int(h) in HORIZONS:
            i, j = DATASETS.index(ds), HORIZONS.index(int(h))
            eff[i, j] = r["effect"]; q[i, j] = r["q_bh"]
    return eff, q


def draw_branch_matrix():
    rows = json.load(open(ROOT / "results/stats_correction.json"))["families"]["branch"]["rows"]
    vmax = max(abs(r["effect"]) for r in rows)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(8.4, 4.5))
    rounded_panel(fig, 0.055, 0.09, 0.40, 0.86, PANEL_GREEN)
    rounded_panel(fig, 0.515, 0.09, 0.40, 0.86, PANEL_BLUE)
    cmap = diverging(); cmap.set_bad("#DADDE1")
    specs = [("frequency", [0.095, 0.19, 0.30, 0.60],
             "(a) Frequency-branch removal"),
             ("time", [0.555, 0.19, 0.30, 0.60],
             "(b) Time-branch removal")]
    for branch, rect, title in specs:
        eff, q = branch_grid(rows, branch)
        ax = fig.add_axes(rect)
        im = ax.imshow(eff, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels(HORIZONS, fontsize=8)
        ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS, fontsize=8)
        ax.set_xlabel("horizon $H$", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        for i in range(len(DATASETS)):
            for j in range(len(HORIZONS)):
                v = eff[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                sig = q[i, j] < 0.05
                txt = f"{v:+.3f}".replace("0.", ".")
                col = "white" if abs(v) > 0.6 * vmax else INK
                ax.text(j, i, txt + ("★" if sig else ""), ha="center",
                        va="center", fontsize=6.4,
                        fontweight="bold" if sig else "normal", color=col)
    cax = fig.add_axes([0.935, 0.27, 0.014, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\Delta$MSE when branch removed", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "fig_branch_matrix.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_branch_matrix.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("[saved] fig_branch_matrix.pdf")


def draw_pcc_mixer():
    pcc = json.load(open(ROOT / "results/pcc.json"))
    # mixer on/off per dataset (channel_mixer_layers > 0), weather excluded (no PCC).
    mixer_on = {"ETTh1": False, "ETTh2": False, "ETTm1": False, "ETTm2": False,
                "solar": True, "electricity": True, "traffic": True,
                "exchange_rate": False}
    items = [(k, pcc[k]["mean_abs_pcc"], mixer_on[k]) for k in pcc if k in mixer_on]
    items.sort(key=lambda t: t[1])
    labels = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
              "solar": "Solar", "electricity": "Electricity", "traffic": "Traffic",
              "exchange_rate": "Exchange"}
    names = [labels[k] for k, _, _ in items]
    vals = [v for _, v, _ in items]
    ons = [o for _, _, o in items]

    fig = plt.figure(figsize=(6.6, 3.9))
    rounded_panel(fig, 0.03, 0.08, 0.94, 0.86, PANEL_LILAC)
    ax = fig.add_axes([0.20, 0.17, 0.72, 0.70])
    y = np.arange(len(names))
    for yi, v, on in zip(y, vals, ons):
        ax.barh(yi, v, height=0.62, color=BLUE, edgecolor="white", linewidth=0.8,
                hatch=None if on else "////", zorder=3,
                alpha=1.0 if on else 0.55)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=7.6)
        ax.text(0.01, yi, "mixer ON" if on else "mixer OFF", va="center", ha="left",
                fontsize=6.8, color="white", fontweight="bold", zorder=5)
    ax.axvline(0.50, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.505, -0.72, "coupling guide (0.5)", fontsize=6.8,
            color=MUTED, ha="center", va="center")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(-1.15, len(names) - 0.4)
    ax.set_xlim(0, 1.0); ax.set_xlabel("mean absolute cross-channel PCC (train split)", fontsize=8.5)
    ax.set_title("(a) Channel coupling vs. the variate-mixer decision",
                 fontsize=9.5, fontweight="bold", loc="left", pad=6)
    ax.grid(axis="x", color=GRID, alpha=0.10, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    # flag the Exchange exception
    if "Exchange" in names:
        ei = names.index("Exchange")
        ax.annotate("co-trending: high PCC,\nmixer still off",
                    xy=(vals[ei], ei), xytext=(vals[ei] + 0.20, ei + 0.05),
                    fontsize=6.8, color=AMBER, va="center",
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0))
    fig.savefig(OUT / "fig_pcc_mixer.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_pcc_mixer.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("[saved] fig_pcc_mixer.pdf")


if __name__ == "__main__":
    draw_branch_matrix()
    draw_pcc_mixer()
