"""Factory helpers that turn a config dict into ready-to-use DataLoaders."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from .dataset import ETT_BORDERS, Scaler, build_splits, load_raw_series


def apply_channel_permutation(
    data: np.ndarray, seed: Optional[int]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply one deterministic channel order to the complete series.

    Inputs and targets receive the same permutation before splitting, so this
    is an architecture-order stress test rather than a label corruption.  A
    local NumPy generator keeps the permutation independent of model seeding.
    """
    if seed is None:
        return data, None
    order = np.random.default_rng(int(seed)).permutation(data.shape[1])
    return np.ascontiguousarray(data[:, order]), order


def get_dataloaders(
    cfg: dict,
    *,
    include_test: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Scaler, int]:
    """Build train/val/test DataLoaders and the fitted Scaler.

    Returns
    -------
    train_loader, val_loader, test_loader-or-None, scaler, n_channels.
    ``include_test=False`` is the leakage-safe mode for candidate selection.
    """
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    data = load_raw_series(
        source=dcfg["source"],
        csv_path=dcfg.get("csv_path"),
        target_columns=dcfg.get("target_columns"),
        synthetic_length=dcfg.get("synthetic_length", 8000),
        synthetic_channels=dcfg.get("synthetic_channels", 7),
        seed=cfg["experiment"]["seed"],
        npz_key=dcfg.get("npz_key", "data"),
        npz_feature=dcfg.get("npz_feature", 0),
    )
    data, channel_order = apply_channel_permutation(
        data, dcfg.get("channel_permutation_seed")
    )
    if channel_order is not None:
        preview = ",".join(str(int(i)) for i in channel_order[:12])
        suffix = ",..." if len(channel_order) > 12 else ""
        print(
            f"[data] channel permutation seed="
            f"{dcfg['channel_permutation_seed']} | order=[{preview}{suffix}]"
        )
    n_channels = data.shape[1]

    protocol = dcfg.get("split_protocol", "ratio")
    if protocol == "ratio":
        borders = None
    elif protocol in ETT_BORDERS:
        borders = ETT_BORDERS[protocol]
    else:
        raise ValueError(
            f"Unknown data.split_protocol: {protocol!r} (use ratio, ETTh, ETTm)"
        )

    # Multi-resolution statistics are only materialized when the model uses a
    # dispersion head (keeps the default pipeline unchanged / cheap).
    mcfg = cfg.get("model", {})
    stats_resolutions = None
    if mcfg.get("dispersion", "none") in ("fixed", "learned"):
        stats_resolutions = mcfg.get(
            "dispersion_resolutions", [dcfg["seq_len"], 144, 288, 336]
        )

    train_ds, val_ds, test_ds, scaler = build_splits(
        data=data,
        seq_len=dcfg["seq_len"],
        pred_len=dcfg["pred_len"],
        train_ratio=dcfg["train_ratio"],
        val_ratio=dcfg["val_ratio"],
        scale=dcfg.get("scale", True),
        borders=borders,
        stats_resolutions=stats_resolutions,
        include_test=include_test,
    )

    common = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = (DataLoader(test_ds, shuffle=False, **common)
                   if test_ds is not None else None)
    return train_loader, val_loader, test_loader, scaler, n_channels
