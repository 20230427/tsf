#!/usr/bin/env python3
"""Fusion-gate / weight-combination figure (paper Fig. 4).

The fusion gate g in (0,1) is the convex weight on the *time* branch:
    y = g * y_time + (1 - g) * y_freq      (fusion='gated', per channel)

Two panels:
  (a) per-channel mean gate over the whole test set (sorted), with the
      equal-mix line 0.5 and the init line sigmoid(2.2)~0.90;
  (b) distribution of g over all (window, channel) pairs.

Preferred source is the case-study dump (full test set, no model reload):
    output/case_study/record_ETTm1/components_96/gate.npy   (N, C, 1)
Alternative: extract with a forward hook from any 'gated' checkpoint
(--ckpt; needs the data files of that config; GPU recommended when the
model was trained with use_official_mamba=true).

Usage
-----
    python scripts/make_gate_figure.py \
        --gate-npy output/case_study/record_ETTm1/components_96/gate.npy \
        --name ETTm1 --out paper/neurocomputing/fig_gate.pdf
    python scripts/make_gate_figure.py --ckpt checkpoints/ETTh1_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLUE, AMBER, VIOLET = "#2C6FBB", "#E07B39", "#7A5AA6"
INK, MUTED, GRID = "#22303C", "#60656C", "#C9CED4"
GATE_INIT = 0.90  # sigmoid(2.2), see src/models/fusion.py GATE_BIAS_INIT

ETT_CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def gate_from_npy(path: str) -> np.ndarray:
    return np.load(path).astype(float).squeeze()      # (N, C)


def gate_from_ckpt(ckpt_path: str) -> tuple[np.ndarray, str]:
    """Forward-hook extraction over the test split; returns ((N, C), name)."""
    import torch

    from src.data.data_loader import get_dataloaders
    from src.models.dual_domain_model import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg["train"]["num_workers"] = 0
    n_channels = state["n_channels"]
    name = cfg["data"]["csv_path"]
    _, _, test_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg, n_channels)
    model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    assert cfg["model"].get("fusion", "gated") == "gated", \
        "hook extraction assumes fusion='gated'"

    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out.detach().float().cpu())

    h = model.fusion.gate.register_forward_hook(hook)
    with torch.no_grad():
        for batch in test_loader:
            model(batch[0].to(device))
    h.remove()
    return torch.cat(captured).squeeze(-1).numpy(), name     # (N, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-npy", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default="ETTm1")
    ap.add_argument("--out", default="paper/neurocomputing/fig_gate.pdf")
    args = ap.parse_args()

    if args.gate_npy:
        g = gate_from_npy(args.gate_npy)
    elif args.ckpt:
        g, args.name = gate_from_ckpt(args.ckpt)
    else:
        ap.error("one of --gate-npy / --ckpt is required")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 0.7, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    per_ch = g.mean(axis=0)                    # (C,)
    order = np.argsort(per_ch)[::-1]
    labels = (ETT_CHANNELS + [f"ch {i}" for i in range(len(per_ch))])
    labels = [labels[i] for i in order]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.5),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) per-channel mean gate, sorted
    xs = np.arange(len(order))
    vals = per_ch[order]
    colors = [BLUE if v >= 0.5 else AMBER for v in vals]
    ax_a.bar(xs, vals, 0.62, color=colors)
    ax_a.axhline(0.5, lw=0.9, color=INK, ls=":")
    ax_a.axhline(GATE_INIT, lw=0.9, color=MUTED, ls="--")
    ax_a.text(len(xs) - 0.4, 0.505, "equal mix 0.5", fontsize=7,
              color=INK, ha="right", va="bottom")
    ax_a.text(len(xs) - 0.4, GATE_INIT - 0.012, f"init {GATE_INIT:.2f}",
              fontsize=7, color=MUTED, ha="right", va="top")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels, rotation=30, ha="right")
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("mean gate  $g$  (weight on time branch)")
    ax_a.set_title(f"(a) per-channel mean gate, {args.name} (test set)",
                   loc="left", fontsize=9, fontweight="bold")

    # (b) distribution over all (window, channel)
    ax_b.hist(g.ravel(), bins=40, color=VIOLET, alpha=0.85)
    ax_b.axvline(g.mean(), lw=1.0, color=INK)
    ax_b.text(g.mean(), ax_b.get_ylim()[1] * 0.92,
              f"mean {g.mean():.2f}", fontsize=7.5, ha="left",
              va="top", color=INK)
    ax_b.set_xlabel("gate $g$")
    ax_b.set_ylabel("# (window, channel)")
    ax_b.set_title("(b) gate distribution", loc="left", fontsize=9,
                   fontweight="bold")

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
    print(f"{args.name}: mean g = {g.mean():.3f}; per-channel = "
          + np.array2string(per_ch, precision=3, floatmode="fixed"))


if __name__ == "__main__":
    main()
