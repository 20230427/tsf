"""Miscellaneous training utilities: config loading, seeding, device, plotting."""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _coerce(value: str):
    """Best-effort string -> {bool, int, float, None, str} for CLI overrides."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply `--key value` CLI overrides onto the nested config.

    Keys may be dotted (``train.lr``) or bare (``lr``); a bare key is matched
    against the first section that already contains it.
    """
    sections = ["experiment", "data", "model", "train"]
    for key, raw in overrides:
        val = _coerce(raw)
        if "." in key:
            section, field = key.split(".", 1)
            cfg.setdefault(section, {})[field] = val
            continue
        placed = False
        for sec in sections:
            if key in cfg.get(sec, {}):
                cfg[sec][key] = val
                placed = True
                break
        if not placed:
            # Unknown bare key -> stash under a generic namespace.
            cfg.setdefault("cli", {})[key] = val
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config."
    )


def parse_overrides(unknown: list) -> list:
    """Turn a flat ['--lr', '0.001', '--epochs', '5'] list into pairs."""
    pairs = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                pairs.append((key, unknown[i + 1]))
                i += 2
            else:
                pairs.append((key, "true"))  # bare flag
                i += 1
        else:
            i += 1
    return pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx} ({name}, sm_{cap[0]}{cap[1]})"
    return "cpu"


def save_prediction_plot(
    pred: np.ndarray,
    true: np.ndarray,
    path: str,
    channel: int = 0,
    max_points: Optional[int] = 500,
) -> None:
    """Save a quick prediction-vs-truth line plot for one sample/channel."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # plotting is optional

    p = pred[0, :, channel]
    t = true[0, :, channel]
    if max_points:
        p = p[:max_points]
        t = t[:max_points]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(t, label="ground truth", linewidth=1.5)
    plt.plot(p, label="prediction", linewidth=1.5)
    plt.title(f"Forecast (channel {channel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
