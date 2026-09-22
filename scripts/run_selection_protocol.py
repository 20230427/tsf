#!/usr/bin/env python
"""Leakage-safe validation-only switch selection.

Candidate runs construct only train/validation loaders and call
``train(..., evaluate_test=False)``. After the validation metric freezes one
candidate, a separate final-confirmation phase retrains only that candidate
with test evaluation enabled. The output records the declared search space,
selection metric, run budget, and complete provenance.

Usage
-----
    python scripts/run_selection_protocol.py --config configs/solar.yaml \
        --horizons 96 --seeds 3 \
        --switch model.use_revin=true,false \
        --switch model.channel_mixer_layers=0,1 \
        --fixed model.use_revin=true,model.channel_mixer_layers=0
"""
from __future__ import annotations

import argparse
import copy
import itertools
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


SELECTION_METRIC = "mean_best_validation_loss_across_horizons_and_seeds"


def parse_value(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_assignments(text: str) -> dict:
    """Convert comma-separated dotted assignments to a dictionary."""
    out = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, separator, val = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid assignment: {item!r}")
        out[key.strip()] = parse_value(val.strip())
    return out


def set_by_path(cfg: dict, dotted: str, value) -> None:
    section, _, key = dotted.partition(".")
    if not key:
        section, key = "model", section
    cfg.setdefault(section, {})[key] = value


def apply_candidate(cfg: dict, candidate: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for path, value in candidate.items():
        set_by_path(cfg, path, value)
    return cfg


def candidate_label(candidate: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in candidate.items()) or "default"


def _mean(values):
    return sum(values) / len(values)


def eval_candidate(base_cfg, candidate, horizons, seeds, base_seed, tag,
                   train_fn=None):
    """Evaluate one candidate using validation only; never expose test data."""
    train_fn = train if train_fn is None else train_fn
    per_horizon, all_val_losses, run_records = {}, [], []
    for horizon in horizons:
        horizon_losses, horizon_runs = [], []
        for offset in range(seeds):
            cfg = apply_candidate(base_cfg, candidate)
            seed = base_seed + offset
            cfg["data"]["pred_len"] = horizon
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["name"] = f"{tag}_h{horizon}_s{offset}"
            out = train_fn(cfg, evaluate_test=False)
            if "test_metrics" in out:
                raise RuntimeError(
                    "validation-only train call returned test_metrics; refusing "
                    "to continue a potentially test-informed selection"
                )
            run = {
                "phase": "candidate_selection",
                "seed": seed,
                "horizon": horizon,
                "arch": cfg["model"].get("arch", "dual_domain"),
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out.get("best_val_metrics"),
                "best_epoch": out.get("best_epoch"),
                "epochs_ran": out.get("epochs_ran"),
                "param_count": out.get("param_count"),
                "checkpoint": out.get("checkpoint"),
            }
            horizon_runs.append(run)
            run_records.append(run)
            horizon_losses.append(out["best_val_loss"])
            all_val_losses.append(out["best_val_loss"])
        per_horizon[str(horizon)] = {
            "val_loss": _mean(horizon_losses),
            "runs": horizon_runs,
        }
    return {
        "candidate": copy.deepcopy(candidate),
        "label": candidate_label(candidate),
        "per_horizon": per_horizon,
        "val_loss": _mean(all_val_losses),
        "run_records": run_records,
    }


def select_by_validation(candidate_results):
    """Freeze the lowest validation-loss candidate with deterministic ties."""
    if not candidate_results:
        raise ValueError("candidate_results must not be empty")
    index, selected = min(
        enumerate(candidate_results), key=lambda item: (item[1]["val_loss"], item[0])
    )
    return index, selected


def confirm_frozen_candidate(base_cfg, candidate, horizons, seeds, base_seed,
                             tag, train_fn=None):
    """Run test evaluation only after ``candidate`` has been frozen."""
    train_fn = train if train_fn is None else train_fn
    per_horizon, run_records = {}, []
    for horizon in horizons:
        horizon_runs = []
        for offset in range(seeds):
            cfg = apply_candidate(base_cfg, candidate)
            seed = base_seed + offset
            cfg["data"]["pred_len"] = horizon
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["name"] = f"{tag}_h{horizon}_s{offset}"
            out = train_fn(cfg, evaluate_test=True)
            metrics = dict(out["test_metrics"])
            run = {
                **metrics,
                "phase": "frozen_final_confirmation",
                "seed": seed,
                "horizon": horizon,
                "arch": cfg["model"].get("arch", "dual_domain"),
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out.get("best_val_metrics"),
                "best_epoch": out.get("best_epoch"),
                "epochs_ran": out.get("epochs_ran"),
                "param_count": out.get("param_count"),
                "checkpoint": out.get("checkpoint"),
            }
            horizon_runs.append(run)
            run_records.append(run)
        per_horizon[str(horizon)] = {
            "mse": _mean([run["mse"] for run in horizon_runs]),
            "mae": _mean([run["mae"] for run in horizon_runs]),
            "runs": horizon_runs,
        }
    return {"per_horizon": per_horizon, "run_records": run_records}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--horizons", nargs="*", type=int, default=[96])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--switch", action="append", default=[],
        help="dotted.path=v1,v2[,v3]; repeatable; Cartesian product searched",
    )
    parser.add_argument(
        "--fixed", default=None,
        help="predeclared fixed assignment recorded for comparison; it does not "
             "influence validation selection",
    )
    parser.add_argument("--out", default=None)
    args, unknown = parser.parse_known_args()

    if args.seeds < 1 or not args.horizons:
        raise SystemExit("--seeds must be >=1 and --horizons must not be empty")
    base_cfg = apply_overrides(load_config(args.config), parse_overrides(unknown))
    base_name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]

    switch_axes = {}
    for spec in args.switch:
        key, separator, vals = spec.partition("=")
        values = [parse_value(v.strip()) for v in vals.split(",") if v.strip()]
        if not separator or not key.strip() or not values:
            raise SystemExit(f"Invalid --switch specification: {spec!r}")
        switch_axes[key.strip()] = values
    if not switch_axes:
        raise SystemExit("Provide at least one --switch path=v1,v2")

    paths = list(switch_axes)
    grid = [dict(zip(paths, combo))
            for combo in itertools.product(*(switch_axes[path] for path in paths))]
    selection_runs = len(grid) * len(args.horizons) * args.seeds
    final_runs = len(args.horizons) * args.seeds
    print(f"[selection] {len(grid)} candidates x {len(args.horizons)} horizons "
          f"x {args.seeds} seeds = {selection_runs} validation-only runs")

    candidate_results = []
    for index, candidate in enumerate(grid):
        print(f"\n===== candidate {index + 1}/{len(grid)}: "
              f"{candidate_label(candidate)} =====")
        result = eval_candidate(
            base_cfg, candidate, args.horizons, args.seeds, base_seed,
            f"{base_name}_sel{index}",
        )
        result["candidate_index"] = index
        candidate_results.append(result)

    selected_index, selected = select_by_validation(candidate_results)
    frozen_candidate = copy.deepcopy(selected["candidate"])
    print(f"\n[freeze] candidate {selected_index}: {selected['label']} "
          f"({SELECTION_METRIC}={selected['val_loss']:.6f})")
    final_confirmation = confirm_frozen_candidate(
        base_cfg, frozen_candidate, args.horizons, args.seeds, base_seed,
        f"{base_name}_final_confirm",
    )

    candidate_run_records = [
        run for result in candidate_results for run in result["run_records"]
    ]
    for result in candidate_results:
        # Avoid duplicating the standardized records in two large locations.
        result.pop("run_records")
    run_records = candidate_run_records + final_confirmation["run_records"]
    final_confirmation = dict(final_confirmation)
    final_confirmation.pop("run_records")

    fixed_candidate = parse_assignments(args.fixed) if args.fixed else {}
    record = provenance_fields(
        base_cfg, config_path=args.config,
        seed_values=[base_seed + offset for offset in range(args.seeds)],
        cwd=Path(__file__).resolve().parents[1],
    )
    record.update({
        "horizons": list(args.horizons),
        "seeds": args.seeds,
        "switch_space": copy.deepcopy(switch_axes),
        "search_space": {
            "axes": copy.deepcopy(switch_axes),
            "candidates": copy.deepcopy(grid),
            "candidate_count": len(grid),
        },
        "budget": {
            "horizons": len(args.horizons),
            "seeds_per_horizon": args.seeds,
            "candidate_count": len(grid),
            "selection_training_runs": selection_runs,
            "final_confirmation_runs": final_runs,
            "total_training_runs": selection_runs + final_runs,
        },
        "selection": {
            "metric": SELECTION_METRIC,
            "direction": "minimize",
            "test_metrics_used": False,
            "selected_candidate_index": selected_index,
            "selected_candidate": frozen_candidate,
            "selected_value": selected["val_loss"],
        },
        "protocol": {
            "candidate_phase": "validation_only",
            "candidate_test_loader_constructed": False,
            "candidate_test_metrics_emitted": False,
            "final_test_phase": "after_configuration_freeze",
        },
        "candidates": candidate_results,
        "models": {
            "fixed": {
                "label": candidate_label(fixed_candidate),
                "candidate": fixed_candidate,
                "role": "predeclared_reference_not_used_for_selection",
            },
            "validation_selected": {
                "label": selected["label"],
                "candidate": frozen_candidate,
                "val_loss": selected["val_loss"],
                "per_horizon_validation": selected["per_horizon"],
                "final_confirmation": final_confirmation,
            },
        },
        "run_records": run_records,
    })

    out_path = args.out or os.path.join(
        base_cfg["experiment"]["checkpoint_dir"], f"selection_{base_name}.json"
    )
    atomic_write_json(out_path, record)

    print("\n" + "=" * 70)
    print(f"selected: {selected['label']}")
    print(f"validation score: {selected['val_loss']:.6f}")
    for horizon, values in final_confirmation["per_horizon"].items():
        print(f"final H={horizon}: test MSE={values['mse']:.4f} "
              f"MAE={values['mae']:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
