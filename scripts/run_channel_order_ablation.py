#!/usr/bin/env python
"""Channel-order stress test for the cross-variate encoder.

Each condition permutes every input and target channel consistently before the
chronological split.  The same model seeds are paired across the original and
permuted orders.  This exposes order sensitivity without corrupting labels.

Example
-------
python scripts/run_channel_order_ablation.py --config configs/weather.yaml \
    --permutation-seeds 11 22 33 44 55 --model-seeds 6
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import apply_channel_permutation  # noqa: E402
from src.data.dataset import load_raw_series  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import (  # noqa: E402
    apply_overrides,
    atomic_write_json,
    config_sha256,
    load_config,
    parse_overrides,
    provenance_fields,
)


MIXER_VARIANTS = {
    "bimamba": {"mixer_kind": "bimamba"},
    "attention": {"mixer_kind": "attention"},
    "none": {"channel_mixer_layers": 0},
}


def _raw_channel_count(cfg: dict) -> int:
    d = cfg["data"]
    data = load_raw_series(
        source=d["source"],
        csv_path=d.get("csv_path"),
        target_columns=d.get("target_columns"),
        synthetic_length=d.get("synthetic_length", 8000),
        synthetic_channels=d.get("synthetic_channels", 7),
        seed=cfg["experiment"]["seed"],
        npz_key=d.get("npz_key", "data"),
        npz_feature=d.get("npz_feature", 0),
    )
    return int(data.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--permutation-seeds", nargs="+", type=int,
                        default=[11, 22, 33, 44, 55])
    parser.add_argument("--model-seeds", type=int, default=6,
                        help="matched training seeds per order and mixer")
    parser.add_argument("--variants", nargs="+", choices=list(MIXER_VARIANTS),
                        default=list(MIXER_VARIANTS))
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--validation-only", action="store_true",
        help="do not construct/evaluate the test split",
    )
    args, unknown = parser.parse_known_args()

    base = load_config(args.config)
    base = apply_overrides(base, parse_overrides(unknown))
    base_seed = int(base["experiment"]["seed"])
    base_name = base["experiment"]["name"]
    n_channels = _raw_channel_count(base)
    orders = [("original", None)] + [
        (f"perm_{seed}", seed) for seed in args.permutation_seeds
    ]

    records = []
    grouped: dict[str, dict[str, list[dict]]] = {}
    for mixer_name in args.variants:
        grouped[mixer_name] = {}
        for order_name, permutation_seed in orders:
            runs = []
            for seed_offset in range(args.model_seeds):
                cfg = copy.deepcopy(base)
                cfg["model"].update(MIXER_VARIANTS[mixer_name])
                if mixer_name != "none":
                    cfg["model"]["channel_mixer_layers"] = max(
                        1, int(base["model"].get("channel_mixer_layers", 1))
                    )
                cfg["data"]["channel_permutation_seed"] = permutation_seed
                cfg["experiment"]["seed"] = base_seed + seed_offset
                cfg["experiment"]["name"] = (
                    f"{base_name}_order_{mixer_name}_{order_name}_s{seed_offset}"
                )
                print(
                    f"\n===== mixer={mixer_name} order={order_name} "
                    f"model_seed={base_seed + seed_offset} ====="
                )
                out = train(cfg, evaluate_test=not args.validation_only)
                metrics = (
                    out["best_val_metrics"] if args.validation_only
                    else out["test_metrics"]
                )
                _, permutation = apply_channel_permutation(
                    np.empty((1, n_channels), dtype=np.float32), permutation_seed
                )
                run = {
                    "mixer": mixer_name,
                    "order": order_name,
                    "channel_permutation_seed": permutation_seed,
                    "channel_order": (
                        list(range(n_channels)) if permutation is None
                        else permutation.astype(int).tolist()
                    ),
                    "seed": base_seed + seed_offset,
                    "horizon": cfg["data"]["pred_len"],
                    "phase": out["evaluation_scope"],
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "param_count": out["param_count"],
                    "active_param_count": out["active_param_count"],
                    "best_val_loss": out["best_val_loss"],
                    "best_epoch": out["best_epoch"],
                    "checkpoint": out["checkpoint"],
                    "run_config_sha256": config_sha256(cfg),
                }
                runs.append(run)
                records.append(run)
            grouped[mixer_name][order_name] = runs

    summary = {}
    for mixer_name, by_order in grouped.items():
        reference = by_order["original"]
        ref_by_seed = {r["seed"]: r for r in reference}
        summary[mixer_name] = {}
        for order_name, runs in by_order.items():
            deltas = [r["mse"] - ref_by_seed[r["seed"]]["mse"] for r in runs]
            summary[mixer_name][order_name] = {
                "mse_mean": float(np.mean([r["mse"] for r in runs])),
                "mse_std": float(np.std([r["mse"] for r in runs])),
                "mae_mean": float(np.mean([r["mae"] for r in runs])),
                "paired_delta_mse_mean": float(np.mean(deltas)),
                "paired_delta_mse_std": float(np.std(deltas)),
                "runs": runs,
            }

    output = provenance_fields(
        base, config_path=args.config,
        seed_values=[base_seed + i for i in range(args.model_seeds)],
        cwd=Path(__file__).resolve().parents[1],
    )
    output.update({
        "analysis": "channel_order_ablation",
        "evaluation_scope": (
            "validation_only" if args.validation_only else "validation_and_test"
        ),
        "permutation_seeds": args.permutation_seeds,
        "model_seeds": args.model_seeds,
        "variants": args.variants,
        "horizons": [base["data"]["pred_len"]],
        "run_records": records,
        "summary": summary,
    })
    out_path = args.out or os.path.join(
        base["experiment"]["checkpoint_dir"],
        f"channel_order_ablation_{base_name}.json",
    )
    atomic_write_json(out_path, output)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
