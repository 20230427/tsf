#!/usr/bin/env python3
"""Inspect Optuna HPO studies: best trial, parameters, and overview table.

Usage
-----
    python scripts/hpo/hpo_dump.py --study DDMamba_ETTh1_pl96
    python scripts/hpo/hpo_dump.py --list
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.hpo.study_manager import parse_study_name  # noqa: E402


def _load(name: str):
    import optuna

    from src.hpo.study_manager import load_study

    return load_study(name)


def dump_study(name: str) -> int:
    study = _load(name)
    finished = [t for t in study.trials if t.state.is_finished()]
    complete = [t for t in study.trials
                if t.state.name == "COMPLETE" and t.value is not None]
    print(f"study: {name}")
    print(f"  trials: {len(study.trials)} ({len(complete)} complete, "
          f"{len(finished) - len(complete)} pruned/failed)")
    if not complete:
        print("  no completed trials")
        return 0
    best = study.best_trial
    print(f"  best: trial #{best.number} value={study.best_value:.6f}")
    for key, value in sorted(best.params.items()):
        print(f"    {key}: {value}")
    attrs = {k: v for k, v in best.user_attrs.items() if k != "error_trace"}
    for key in ("mse", "mae", "val_loss", "param_count", "best_epoch",
                "config_sha256", "status"):
        if key in attrs:
            print(f"    @{key}: {attrs[key]}")
    return 0


def list_studies() -> int:
    pattern = os.path.join("output", "hpo", "optuna", "DDMamba_*.db")
    dbs = sorted(glob.glob(pattern))
    if not dbs:
        print(f"no studies found under {pattern}")
        return 0
    print(f"{'study':40} {'trials':>7} {'best':>12}  best params (abbrev.)")
    for db in dbs:
        name = os.path.splitext(os.path.basename(db))[0]
        try:
            study = _load(name)
        except Exception as e:
            print(f"{name:40} unreadable ({e})")
            continue
        try:
            best = study.best_trial
            best_str = f"{study.best_value:.6f}"
            abbrev = " ".join(f"{k.split('.')[-1]}={v}"
                              for k, v in sorted(best.params.items())[:4])
            row = f"{name:40} {len(study.trials):>7} {best_str:>12}  {abbrev}"
        except ValueError:
            row = f"{name:40} {len(study.trials):>7} {'--':>12}  (no completed trials)"
        print(row)
        dataset, pred_len = parse_study_name(name)
        json_path = os.path.join("output", "hpo", f"{name}_best_params.json")
        if os.path.exists(json_path):
            print(f"{'':40} best-params JSON: {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--study", help="study name, e.g. DDMamba_ETTh1_pl96")
    group.add_argument("--list", action="store_true", help="summarize all studies")
    args = parser.parse_args()
    if args.list:
        return list_studies()
    return dump_study(args.study)


if __name__ == "__main__":
    raise SystemExit(main())
