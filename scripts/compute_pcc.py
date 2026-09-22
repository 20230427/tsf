#!/usr/bin/env python
"""Cross-channel Pearson correlation (PCC) statistics per dataset.

Quantifies how strongly a dataset's variates are linearly related on the
TRAINING split (canonical ETT borders; 70% elsewhere), reporting the mean
and median absolute off-diagonal PCC and the fraction of channel pairs with
|r| > 0.6. These statistics ground the per-dataset variate-mixer decision in
a measurable property of the data: cross-channel capacity should pay where
channels are strongly correlated and overfit where they are not.

Usage:  python scripts/compute_pcc.py       # writes results/pcc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ETT_BORDERS, load_raw_series  # noqa: E402

DATASETS = {
    "ETTh1": ("data/ETTh1.csv", "ETTh"),
    "ETTh2": ("data/ETTh2.csv", "ETTh"),
    "ETTm1": ("data/ETTm1.csv", "ETTm"),
    "ETTm2": ("data/ETTm2.csv", "ETTm"),
    "weather": ("data/weather.csv", None),
    "solar": ("data/solar_AL.txt.gz", None),
    "electricity": ("data/electricity.txt.gz", None),
    "traffic": ("data/traffic.txt.gz", None),
    "exchange_rate": ("data/exchange_rate.txt.gz", None),
}


def pcc_stats(x: np.ndarray) -> dict:
    # Drop constant channels (zero variance) to keep corrcoef defined.
    keep = x.std(axis=0) > 1e-8
    x = x[:, keep]
    c = np.corrcoef(x, rowvar=False)
    n = c.shape[0]
    off = np.abs(c[np.triu_indices(n, k=1)])
    return {
        "channels": int(n),
        "mean_abs_pcc": float(off.mean()),
        "median_abs_pcc": float(np.median(off)),
        "frac_gt_0.6": float((off > 0.6).mean()),
    }


def main():
    out = {}
    for name, (path, protocol) in DATASETS.items():
        if not Path(path).exists():
            print(f"{name:14s} SKIP (file not available)")
            continue
        data = load_raw_series(source="csv", csv_path=path, target_columns=None,
                               synthetic_length=0, synthetic_channels=0, seed=0)
        n_train = ETT_BORDERS[protocol][0] if protocol else int(0.7 * len(data))
        stats = pcc_stats(data[:n_train].astype(np.float64))
        out[name] = stats
        print(f"{name:14s} C={stats['channels']:4d}  mean|r|={stats['mean_abs_pcc']:.3f}  "
              f"median|r|={stats['median_abs_pcc']:.3f}  frac|r|>0.6={stats['frac_gt_0.6']:.2f}")
    Path("results").mkdir(exist_ok=True)
    with open("results/pcc.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[saved] results/pcc.json")


if __name__ == "__main__":
    main()
