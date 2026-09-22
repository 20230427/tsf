"""Evaluate a trained checkpoint and optionally save a forecast plot.

Usage
-----
    python -m src.evaluate --checkpoint checkpoints/dual_domain_default_best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import get_dataloaders
from .models import build_model
from .utils import all_metrics, describe_device, get_device, save_prediction_plot, set_seed


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, stats in loader:
        x = x.to(device)
        out = model(x, stats=stats).cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a dual-domain checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", default=None, help="Optional path to save a forecast PNG.")
    parser.add_argument("--channel", type=int, default=0, help="Channel to plot.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[device] {describe_device(device)}")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    n_channels = state["n_channels"]
    set_seed(cfg["experiment"]["seed"])

    _, _, test_loader, _, loaded_channels = get_dataloaders(cfg)
    if loaded_channels != n_channels:
        print(
            f"[warn] channel mismatch: checkpoint={n_channels}, data={loaded_channels}."
        )

    model = build_model(cfg, n_channels).to(device)
    model.load_state_dict(state["model_state"])

    preds, trues = collect_predictions(model, test_loader, device)
    metrics = all_metrics(preds, trues)
    print(f"[test] mse={metrics['mse']:.4f} mae={metrics['mae']:.4f}")

    if args.plot:
        save_prediction_plot(preds, trues, args.plot, channel=args.channel)
        print(f"[plot] saved forecast to {args.plot}")


if __name__ == "__main__":
    main()
