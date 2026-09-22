#!/usr/bin/env python3
"""Branch-decomposition case study (paper Fig. 5).

Shows, for one test window of the case-study dump, how the gated fusion
combines the two branch forecasts:

  (a) history + horizon: ground truth, fused forecast, and both branch
      forecasts (time = blue dashed, freq = amber dashed); 'now' line at the
      forecast origin; the channel's gate g annotated;
  (b) the same fused forecast as the *weighted sum*: stacked areas of
      g*y_time and (1-g)*y_freq -- the weight-combination view.

Source (produced by scripts/case_study/run_case_study.py, standardized space):
    output/case_study/record_ETTm1/components_96/{fused,time,freq,true,gate}.npy
    history window is re-read from the checkpoint's test split via meta.json.

Usage
-----
    python scripts/make_case_figure.py \
        --components output/case_study/record_ETTm1/components_96 \
        --channel 0 --sample 0 --out paper/neurocomputing/fig_case.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLUE, AMBER, VIOLET = "#2C6FBB", "#E07B39", "#7A5AA6"
INK, MUTED, GRID = "#22303C", "#60656C", "#C9CED4"
ETT_CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_history(ckpt_path: str, sample: int, channel: int):
    """History window (L,) of that test sample, standardized space."""
    import torch

    from src.data.data_loader import get_dataloaders

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg["train"]["num_workers"] = 0
    _, _, test_loader, _, _ = get_dataloaders(cfg)
    for i, batch in enumerate(test_loader):
        if i == sample:
            return batch[0][0, :, channel].numpy()      # (L,)
    raise RuntimeError("sample not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components",
                    default="output/case_study/record_ETTm1/components_96")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--out", default="paper/neurocomputing/fig_case.pdf")
    args = ap.parse_args()
    d = Path(args.components)

    meta = json.load(open(d / "meta.json"))
    s, c = args.sample, args.channel
    time_y = np.load(d / "time.npy")[s, :, c]        # (H,)
    freq_y = np.load(d / "freq.npy")[s, :, c]
    fused_y = np.load(d / "fused.npy")[s, :, c]
    true_y = np.load(d / "true.npy")[s, :, c]
    g = float(np.load(d / "gate.npy").reshape(-1, meta["channels"])[s, c])

    hist = load_history(meta["checkpoint"], s, c)
    L, H = len(hist), len(true_y)
    t_all = np.arange(L + H)
    t_cut = L

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # (a) trajectory view
    ax_a.plot(t_all[:t_cut], hist, color=MUTED, lw=0.9, label="history")
    ax_a.plot(t_all[t_cut - 1:], np.concatenate([[hist[-1]], true_y]),
              color=INK, lw=1.2, label="ground truth")
    ax_a.plot(t_all[t_cut - 1:], np.concatenate([[hist[-1]], time_y]),
              color=BLUE, lw=1.1, ls="--", label="time branch")
    ax_a.plot(t_all[t_cut - 1:], np.concatenate([[hist[-1]], freq_y]),
              color=AMBER, lw=1.1, ls="--", label="freq branch")
    ax_a.plot(t_all[t_cut - 1:], np.concatenate([[hist[-1]], fused_y]),
              color=VIOLET, lw=1.6, label="fused (ours)")
    ax_a.axvline(t_cut - 1, lw=0.8, color=GRID)
    ax_a.text(t_cut - 1, ax_a.get_ylim()[1], " now", fontsize=7,
              color=MUTED, va="top")
    ch_name = ETT_CHANNELS[c] if c < len(ETT_CHANNELS) else f"ch {c}"
    ax_a.set_title(f"(a) {meta['dataset']} {ch_name}, "
                   f"$g={g:.2f}$", loc="left", fontsize=9, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2,
                columnspacing=0.9, handlelength=1.6)
    ax_a.set_xlabel("time step")

    # (b) weighted-combination view (same y-limits as the fused signal)
    w_t, w_f = g * time_y, (1.0 - g) * freq_y
    lo = min(w_t.min(), w_f.min(), 0)
    hi = max(w_t.max(), w_f.max(), 0)
    t_h = np.arange(H)
    ax_b.axhline(0, lw=0.6, color=GRID)
    ax_b.fill_between(t_h, lo, w_t, color=BLUE, alpha=0.30, lw=0)
    ax_b.fill_between(t_h, lo, w_f, color=AMBER, alpha=0.30, lw=0)
    ax_b.plot(t_h, w_t, color=BLUE, lw=1.1,
              label=f"$g\\,y_{{time}}$  ($g={g:.2f}$)")
    ax_b.plot(t_h, w_f, color=AMBER, lw=1.1,
              label=f"$(1-g)\\,y_{{freq}}$  ($1-g={1-g:.2f}$)")
    ax_b.plot(t_h, fused_y, color=VIOLET, lw=1.6, label="$\\hat y$ = sum")
    ax_b.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax_b.set_title("(b) gated combination", loc="left", fontsize=9,
                   fontweight="bold")
    ax_b.legend(frameon=False, fontsize=7, loc="upper left")
    ax_b.set_xlabel("horizon step")

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.4, color=GRID, alpha=0.6)

    fig.tight_layout(w_pad=1.6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"[saved] {out} and {out.with_suffix('.png')}")
    print(f"sample={s} channel={c} ({ch_name}) gate={g:.3f} "
          f"MSE_fused={np.mean((fused_y - true_y) ** 2):.4f} "
          f"MSE_time={np.mean((time_y - true_y) ** 2):.4f} "
          f"MSE_freq={np.mean((freq_y - true_y) ** 2):.4f}")


if __name__ == "__main__":
    main()
