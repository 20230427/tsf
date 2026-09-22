#!/usr/bin/env python
"""Branch-ablation matrix: full / time-only / freq-only on every dataset.

Motivation
----------
The paper's frequency-branch (dual-domain) evidence currently rests on a
single cell: Solar, H=96, time-only, +0.0114 MSE with a 95% CI barely
excluding zero. One marginal result on one dataset at one horizon cannot
carry the conclusion. This script replaces "Solar as the single test" with
the full matrix: for each dataset and horizon it trains the full model and
both single-branch variants under shared seeds, then reports paired per-seed
deltas with t-based 95% CIs for every cell.

Run on the training GPU (hours of compute; not a CPU job):

    python scripts/run_branch_matrix.py                    # all 9 datasets, H=96, 3 seeds
    python scripts/run_branch_matrix.py --horizons 96 192 336 720
    python scripts/run_branch_matrix.py --datasets ETTh1 solar weather --seeds 3

Each (dataset, horizon) cell reuses the ablation harness with variants
full / time_only / freq_only, so results are directly comparable to the
existing tables. Output: checkpoints/branch_matrix.json plus a rendered
markdown/LaTeX summary with one row per cell and a star when the paired CI
excludes zero.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import (  # noqa: E402
    atomic_write_json,
    config_sha256,
    load_config,
    provenance_fields,
    validate_provenance_record,
)
from scripts.statistical_audit import student_t_critical_975  # noqa: E402

DATASETS = {
    "ETTh1": "configs/ETTh1.yaml",
    "ETTh2": "configs/ETTh2.yaml",
    "ETTm1": "configs/ETTm1.yaml",
    "ETTm2": "configs/ETTm2.yaml",
    "weather": "configs/weather.yaml",
    "solar": "configs/solar.yaml",
    "electricity": "configs/electricity.yaml",
    "traffic": "configs/traffic.yaml",
    "exchange_rate": "configs/exchange_rate.yaml",
}
VARIANTS = {"full": None, "time_only": "time_only", "freq_only": "freq_only"}
def paired_ci(full_runs, var_runs):
    d = [v - f for f, v in zip(full_runs, var_runs)]
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = student_t_critical_975(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def run_cell(cfg_path, horizon, seeds, dataset=None):
    base = load_config(cfg_path)
    base["data"]["pred_len"] = horizon
    name = base["experiment"]["name"]
    seed0 = base["experiment"]["seed"]
    cell, run_records = {}, []
    for variant, fusion in VARIANTS.items():
        runs = []
        for s in range(seeds):
            cfg = copy.deepcopy(base)
            if fusion is not None:
                cfg["model"]["fusion"] = fusion
            cfg["experiment"]["seed"] = seed0 + s
            cfg["experiment"]["name"] = f"{name}_bm_{variant}_H{horizon}_s{s}"
            print(f"\n===== {name} H={horizon} {variant} seed={seed0 + s} =====",
                  flush=True)
            out = train(cfg)
            metrics = dict(out["test_metrics"])
            runs.append(metrics["mse"])
            run_records.append({
                **metrics,
                "seed": seed0 + s,
                "horizon": horizon,
                "arch": cfg["model"].get("arch", "dual_domain"),
                "dataset": dataset or name,
                "variant": variant,
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out["best_val_metrics"],
                "best_epoch": out["best_epoch"],
                "param_count": out["param_count"],
                "checkpoint": out["checkpoint"],
            })
        cell[variant] = runs
    return cell, run_records


def summarize(results):
    md = ["| dataset | H | freq removed Δ [95% CI] | time removed Δ [95% CI] |",
          "|---|---|---|---|"]
    tex = []
    for key, cell in sorted(results.items()):
        ds, h = key.rsplit("@", 1)
        rows = {}
        for variant in ("time_only", "freq_only"):
            m, lo, hi = paired_ci(cell["full"], cell[variant])
            star = "*" if (lo > 0 or hi < 0) else ""
            rows[variant] = f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
        md.append(f"| {ds} | {h} | {rows['time_only']} | {rows['freq_only']} |")
        tex.append(f"{ds} & {h} & {rows['time_only']} & {rows['freq_only']} \\\\")
    return "\n".join(md), "\n".join(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--horizons", nargs="*", type=int, default=[96])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="checkpoints/branch_matrix.json")
    args = ap.parse_args()

    resolved_configs = {ds: load_config(DATASETS[ds]) for ds in args.datasets}
    seed_values = sorted({
        cfg["experiment"]["seed"] + offset
        for cfg in resolved_configs.values()
        for offset in range(args.seeds)
    })
    resolved_bundle = {
        "runner": {
            "datasets": list(args.datasets),
            "horizons": list(args.horizons),
            "seeds_per_cell": args.seeds,
            "variants": list(VARIANTS),
        },
        "dataset_configs": resolved_configs,
    }

    results, run_records, resume_history = {}, [], []
    if os.path.exists(args.out):  # resume support
        with open(args.out, encoding="utf-8") as handle:
            existing = json.load(handle)
        validation = validate_provenance_record(existing)
        if validation["status"] != "verified":
            details = "; ".join(validation["reasons"])
            raise SystemExit(
                f"Refusing to resume {validation['status']} result file "
                f"{args.out}: {details}. Use a new --out path; the legacy "
                "file will not be rewritten."
            )
        if existing.get("seeds") != args.seeds:
            raise SystemExit(
                f"Refusing to mix seeds={args.seeds} with existing "
                f"seeds={existing.get('seeds')} in {args.out}."
            )
        existing_resolved = existing.get("resolved_config", {})
        existing_configs = existing_resolved.get("dataset_configs", {})
        requested_cells = {
            f"{dataset}@{horizon}"
            for dataset in args.datasets
            for horizon in args.horizons
        }
        unexpected_cells = sorted(set(existing.get("results", {})) - requested_cells)
        if unexpected_cells:
            raise SystemExit(
                "Refusing to mix cells outside the requested design: "
                + ", ".join(unexpected_cells)
            )
        for dataset in {
            cell.rsplit("@", 1)[0] for cell in existing.get("results", {})
        }:
            if dataset not in existing_configs:
                raise SystemExit(
                    f"Refusing to resume: prior resolved config for {dataset} is missing."
                )
            if config_sha256(existing_configs[dataset]) != config_sha256(
                resolved_configs[dataset]
            ):
                raise SystemExit(
                    f"Refusing to resume: resolved config changed for {dataset}."
                )
        results = existing.get("results", {})
        run_records = existing.get("run_records", [])
        resume_history = list(existing.get("resume_history", []))
        resume_history.append({
            "created_at_utc": existing.get("created_at_utc"),
            "config_sha256": existing.get("config_sha256"),
            "git": existing.get("git"),
            "environment": existing.get("environment"),
            "cli": existing.get("cli"),
            "completed_cells": sorted(results),
            "run_record_count": len(run_records),
        })
        print(f"[resume] loaded {len(results)} finished cells from {args.out}")

    for ds in args.datasets:
        for h in args.horizons:
            key = f"{ds}@{h}"
            if key in results:
                print(f"[skip] {key} already done")
                continue
            results[key], cell_records = run_cell(
                DATASETS[ds], h, args.seeds, dataset=ds
            )
            run_records.extend(cell_records)
            record = provenance_fields(
                resolved_bundle, config_path=None, seed_values=seed_values,
                cwd=Path(__file__).resolve().parents[1],
            )
            record.update({
                "dataset_config_paths": {ds_name: DATASETS[ds_name]
                                         for ds_name in args.datasets},
                "datasets": list(args.datasets),
                "horizons": list(args.horizons),
                "architectures": sorted({r["arch"] for r in run_records}),
                "seeds": args.seeds,
                "resume_history": resume_history,
                "run_records": run_records,
                "results": results,
            })
            atomic_write_json(args.out, record)
            print(f"[saved] {args.out} ({len(results)} cells)")

    md, tex = summarize(results)
    print("\n## Branch-ablation matrix (paired ΔMSE vs full; * = CI excludes 0)\n")
    print(md)
    print("\n% LaTeX rows:\n" + tex)


if __name__ == "__main__":
    main()
