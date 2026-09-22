#!/usr/bin/env python
"""Multi-seed main-results runner: horizons x seeds -> mean +/- std.

Produces the statistically reliable version of the main results table for
one dataset config, and supports in-framework baselines through the same
pipeline for fair comparison.

Usage
-----
    # 3-seed DD-Mamba results across the standard horizons:
    python scripts/run_main_results.py --config configs/ETTh1.yaml --seeds 3

    # Same protocol/pipeline, DLinear baseline:
    python scripts/run_main_results.py --config configs/ETTh1.yaml \
        --seeds 3 --arch dlinear

    # Extra overrides are forwarded to every run:
    python scripts/run_main_results.py --config configs/solar.yaml \
        --seeds 3 --use_revin false

Writes checkpoints/main_<name>[_<arch>].json and prints a markdown table
(mean +/- std per horizon, plus the average row).
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import (  # noqa: E402
    apply_overrides,
    atomic_write_json,
    config_sha256,
    load_config,
    parse_overrides,
    provenance_fields,
)


def mean_std(values):
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5
    return m, s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--horizons", type=int, nargs="*",
                        default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arch", default=None,
                        help="Override model.arch (e.g. dlinear).")
    args, unknown = parser.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    if args.arch:
        base_cfg["model"]["arch"] = args.arch
    base_name = base_cfg["experiment"]["name"]
    tag = f"{base_name}_{args.arch}" if args.arch else base_name
    base_seed = base_cfg["experiment"]["seed"]

    results, run_records = {}, []
    for h in args.horizons:
        runs = []
        for s in range(args.seeds):
            cfg = copy.deepcopy(base_cfg)
            cfg["data"]["pred_len"] = h
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{tag}_h{h}_s{s}"
            print(f"\n===== {tag} | H={h} | seed {base_seed + s} =====")
            out = train(cfg)
            run = {
                **out["test_metrics"],
                "seed": base_seed + s,
                "horizon": h,
                "arch": base_cfg["model"].get("arch", "dual_domain"),
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out["best_val_metrics"],
                "best_epoch": out["best_epoch"],
                "param_count": out["param_count"],
                "checkpoint": out["checkpoint"],
            }
            runs.append(run)
            run_records.append(run)
        results[h] = {
            "runs": runs,
            "mse": mean_std([r["mse"] for r in runs]),
            "mae": mean_std([r["mae"] for r in runs]),
        }

    lines = [
        f"\n## {tag} — {args.seeds} seeds (mean ± std)\n",
        "| H | MSE | MAE |", "|---|---|---|",
    ]
    for h in args.horizons:
        (mm, ms), (am, asd) = results[h]["mse"], results[h]["mae"]
        lines.append(f"| {h} | {mm:.3f} ± {ms:.3f} | {am:.3f} ± {asd:.3f} |")
    avg_mse = sum(results[h]["mse"][0] for h in args.horizons) / len(args.horizons)
    avg_mae = sum(results[h]["mae"][0] for h in args.horizons) / len(args.horizons)
    lines.append(f"| avg | {avg_mse:.3f} | {avg_mae:.3f} |")
    print("\n".join(lines))

    ckpt_dir = base_cfg["experiment"]["checkpoint_dir"]
    out_path = os.path.join(ckpt_dir, f"main_{tag}.json")
    record = provenance_fields(
        base_cfg, config_path=args.config,
        seed_values=[base_seed + s for s in range(args.seeds)],
        cwd=Path(__file__).resolve().parents[1],
    )
    record.update({
        "arch": base_cfg["model"].get("arch", "dual_domain"),
        "horizons": list(args.horizons),
        "seeds": args.seeds,
        "run_records": run_records,
        "results": {
            str(h): {
                "runs": results[h]["runs"],
                "mse_mean": results[h]["mse"][0],
                "mse_std": results[h]["mse"][1],
                "mae_mean": results[h]["mae"][0],
                "mae_std": results[h]["mae"][1],
            }
            for h in args.horizons
        },
    })
    atomic_write_json(out_path, record)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
