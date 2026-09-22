"""Sliding-window datasets for multivariate time series forecasting."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Scaler:
    """Standardize channels using statistics fitted on the training split."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# Subdirectories (under the data root) that may hold co-located .pt caches
# for the raw csv/txt/npz benchmark files.
PT_CACHE_SUBDIRS = (
    "ETT-small", "weather", "electricity", "traffic",
    "exchange_rate", "Solar", "PEMS", "illness",
)


def _resolve_pt_path(path: Optional[str]) -> Optional[str]:
    """Find a co-located .pt cache for a csv/txt/npz data path.

    Follows the pamba data_provider convention: same basename with a ``.pt``
    suffix, either next to the original file or in one of the known dataset
    subdirectories beside it. Returns None when no cache exists so callers
    fall back to the original file format.
    """
    if not path:
        return None
    root = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    candidates = [os.path.join(root, stem + ".pt")]
    candidates += [os.path.join(root, sub, stem + ".pt") for sub in PT_CACHE_SUBDIRS]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def _load_pt_values(pt_path: str, feature_index: int = 0) -> np.ndarray:
    """Read the (time, channels) array from a pamba-style .pt cache.

    The cache stores ``{'values': (T, C) ndarray, ...}`` (``dates`` /
    ``columns`` are optional and ignored). A ``(T, N, F)`` stack is reduced
    over the feature axis like the npz path, honoring ``feature_index``.
    """
    cached = torch.load(pt_path, weights_only=False, map_location="cpu")
    arr = np.asarray(
        cached["values"] if isinstance(cached, dict) else cached, dtype=np.float32
    )
    if arr.ndim == 3:            # (T, N, F) -> select one feature -> (T, N)
        arr = arr[:, :, feature_index]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected .pt array shape {arr.shape} in {pt_path!r}")
    return arr


def generate_synthetic(length: int, channels: int, seed: int = 0) -> np.ndarray:
    """Create a multi-seasonal synthetic signal with trend and noise.

    Each channel mixes several sinusoids (so the frequency branch has real
    structure to exploit), a slow trend (for the time branch), and Gaussian
    noise. Returns an array of shape (length, channels).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    series = np.zeros((length, channels), dtype=np.float32)
    for c in range(channels):
        n_components = rng.integers(2, 5)
        signal = np.zeros(length, dtype=np.float64)
        for _ in range(n_components):
            period = rng.uniform(12, 336)
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * t / period + phase)
        trend = rng.uniform(-1e-3, 1e-3) * t
        noise = rng.normal(0, 0.3, size=length)
        series[:, c] = (signal + trend + noise).astype(np.float32)
    return series


class SlidingWindowDataset(Dataset):
    """Yields ``(input_window, target_window, stats)`` from a contiguous series.

    ``stats`` carries multi-resolution historical statistics for the dispersion
    head: for each resolution ``r`` in ``stats_resolutions`` it holds the mean
    and standard deviation of the ``r`` samples ending at the forecast origin
    (strictly before the target), shape ``(2*len(res), C)``. When no
    resolutions are given ``stats`` is an empty ``(0, C)`` tensor. All statistics
    use only information available before the forecast origin (no leakage).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (time, channels) already scaled if desired.
    seq_len, pred_len : int
        Look-back and horizon lengths.
    stats_resolutions : list[int] | None
        Past-context lengths for the dispersion statistics (e.g. 96/144/288/336).
    start_offset : int
        Number of leading rows reserved as history-only context; valid windows
        are indexed from ``start_offset`` so that boundary windows can look back
        into the reserved context for their statistics.
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 stats_resolutions: Optional[list] = None, start_offset: int = 0):
        if data.ndim != 2:
            raise ValueError(f"data must be 2D (time, channels), got {data.shape}")
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stats_resolutions = list(stats_resolutions) if stats_resolutions else []
        self.start_offset = start_offset
        n = len(self.data) - start_offset - seq_len - pred_len + 1
        if n <= 0:
            raise ValueError(
                f"Series too short ({len(self.data)}) for start_offset="
                f"{start_offset} + seq_len={seq_len} + pred_len={pred_len}."
            )
        self.n_samples = n

    def __len__(self) -> int:
        return self.n_samples

    def _stats(self, origin: int) -> np.ndarray:
        """Multi-resolution (mean, std) of history ending at ``origin``."""
        rows = []
        for r in self.stats_resolutions:
            hist = self.data[max(0, origin - r):origin]      # (<=r, C)
            rows.append(hist.mean(axis=0))
            rows.append(hist.std(axis=0))
        return np.stack(rows, axis=0).astype(np.float32)     # (2*len(res), C)

    def __getitem__(self, idx: int):
        s = self.start_offset + idx
        e = s + self.seq_len
        x = self.data[s:e]
        y = self.data[e : e + self.pred_len]
        if self.stats_resolutions:
            stats = self._stats(e)
        else:
            stats = np.zeros((0, x.shape[1]), dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(stats)


def load_raw_series(
    source: str,
    csv_path: Optional[str],
    target_columns: Optional[list],
    synthetic_length: int,
    synthetic_channels: int,
    seed: int,
    npz_key: str = "data",
    npz_feature: int = 0,
) -> np.ndarray:
    """Load the full (time, channels) array from synthetic, CSV, or NPZ source.

    ``npz`` covers the PEMS traffic-flow benchmark files (PEMS03/04/07/08),
    which store a ``(T, N, F)`` array under ``npz_key`` (default ``"data"``);
    feature ``npz_feature`` (default 0, traffic flow) is taken across the
    ``N`` sensor channels, giving a ``(T, N)`` multivariate series.
    """
    if source == "synthetic":
        return generate_synthetic(synthetic_length, synthetic_channels, seed=seed)
    if source in ("csv", "npz"):
        # Prefer a co-located .pt cache over the raw csv/txt/npz file.
        pt_path = _resolve_pt_path(csv_path)
        if pt_path:
            print(f"[data] using pt cache: {pt_path}")
            return _load_pt_values(pt_path, npz_feature)
    if source == "npz":
        if not csv_path:
            raise ValueError("data.csv_path must point at the .npz when source == 'npz'.")
        with np.load(csv_path) as npz:
            arr = npz[npz_key] if npz_key in npz else npz[list(npz.files)[0]]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:            # (T, N, F) -> select one feature -> (T, N)
            arr = arr[:, :, npz_feature]
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected NPZ array shape {arr.shape} in {csv_path!r}")
        return arr
    if source == "csv":
        if not csv_path:
            raise ValueError("data.csv_path must be set when source == 'csv'.")
        if csv_path.endswith((".txt", ".txt.gz")):
            # LSTNet-style matrix (e.g. Solar-Energy's solar_AL.txt):
            # headerless, comma-separated, no timestamp column.
            df = pd.read_csv(csv_path, header=None)
        else:
            df = pd.read_csv(csv_path)
            # Drop a leading timestamp/date column if present.
            first = df.columns[0]
            if (
                df[first].dtype == object
                or "date" in str(first).lower()
                or "time" in str(first).lower()
            ):
                df = df.drop(columns=[first])
        if target_columns:
            df = df[target_columns]
        df = df.select_dtypes(include=[np.number])
        return df.to_numpy(dtype=np.float32)
    raise ValueError(f"Unknown data source: {source!r}")


# Canonical ETT benchmark borders (Informer/Autoformer/TSLib protocol):
# 12 months train / 4 months val / 4 months test; rows beyond month 20 are
# NOT used. Hourly: 30*24 rows per month; 15-min: 4x that.
ETT_BORDERS = {
    "ETTh": (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24),        # 8640/11520/14400
    "ETTm": (12 * 30 * 96, 16 * 30 * 96, 20 * 30 * 96),        # 34560/46080/57600
}


def build_splits(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    scale: bool,
    borders: Optional[Tuple[int, int, int]] = None,
    stats_resolutions: Optional[list] = None,
    include_test: bool = True,
) -> Tuple[
    SlidingWindowDataset,
    SlidingWindowDataset,
    Optional[SlidingWindowDataset],
    Scaler,
]:
    """Chronologically split, scale (train-fit), and window the series.

    Validation and test windows are extended backwards by `seq_len` so that
    their first prediction target still has a full look-back window, without
    leaking future data across the split boundary's targets. Set
    ``include_test=False`` during hyperparameter selection so no test dataset
    is constructed before the selected configuration is frozen.

    If ``borders`` is given as absolute row indices (train_end, val_end,
    test_end), it overrides the ratio split — required to reproduce the
    canonical ETT protocol, whose published numbers use fixed month borders
    and discard the tail of the file. Without it, results on ETT are NOT
    comparable to the literature.
    """
    n = len(data)
    if borders is not None:
        b0, b1, b2 = borders
        if b0 >= n:
            raise ValueError(
                f"Split border train_end={b0} exceeds series length {n}; "
                "check data.split_protocol vs the loaded file."
            )
        n_train = b0
        n_val = min(b1, n) - b0
        data = data[: min(b2, n)]
        n = len(data)
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

    train_raw = data[:n_train]
    scaler = Scaler.fit(train_raw)
    proc = scaler.transform(data) if scale else data

    # Reserve enough backward context that boundary windows of val/test can
    # look back far enough for the longest dispersion resolution (no leakage:
    # the context is real past data, just before the split).
    ctx = max(seq_len, max(stats_resolutions) if stats_resolutions else 0)
    off = ctx - seq_len

    train_slice = proc[:n_train]
    val_slice = proc[max(0, n_train - ctx) : n_train + n_val]
    val_off = off if n_train - ctx >= 0 else n_train - seq_len

    kw = {"stats_resolutions": stats_resolutions}
    train_ds = SlidingWindowDataset(train_slice, seq_len, pred_len, **kw)
    val_ds = SlidingWindowDataset(val_slice, seq_len, pred_len,
                                  start_offset=val_off, **kw)
    test_ds = None
    if include_test:
        test_slice = proc[max(0, n_train + n_val - ctx) :]
        test_off = (off if n_train + n_val - ctx >= 0
                    else (n_train + n_val) - seq_len)
        test_ds = SlidingWindowDataset(test_slice, seq_len, pred_len,
                                       start_offset=test_off, **kw)
    return train_ds, val_ds, test_ds, scaler
