#!/usr/bin/env python
"""Unified, same-pipeline baseline comparison.

Runs DD-Mamba and every baseline through the IDENTICAL data pipeline, splits,
training loop, schedule, and evaluation for one dataset --- the fair
comparison the review requires (baselines are otherwise quoted from S-Mamba,
which mixes implementations). Each architecture is trained over ``--seeds``
seeds and ``--horizons`` horizons; results are collected into one JSON and a
markdown table.

Usage
-----
    # DD-Mamba vs the standard baselines on ETTh1, 5 seeds:
    python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
        --seeds 5 --archs dual_domain dlinear nlinear rlinear patchtst \
        itransformer smamba msmamba

    # Include TF4TF by pointing at the authors' implementation:
    python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
        --archs tf4tf --external_impl tf4tf_official.model:TF4TF

Writes checkpoints/unified_<name>.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
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
from scripts.run_selection_protocol import apply_candidate  # noqa: E402

#: The comparison set reported in the paper's main tables, ordered as they
#: appear there: our model, the Mamba forecasters, the attention forecasters,
#: then the MLP/linear ones. ``nlinear`` and ``msmamba`` are extras kept from
#: the earlier sweep and are not in the paper's baseline columns.
DEFAULT_ARCHS = ["dual_domain", "smamba", "itransformer", "rlinear", "patchtst",
                 "crossformer", "tide", "dlinear", "fedformer", "autoformer",
                 "nlinear", "msmamba"]


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5
    return m, s


def run_arch(base_cfg, arch, horizons, seeds, base_seed, external_impl,
             selected_overrides=None):
    cfg0 = copy.deepcopy(base_cfg)
    cfg0["model"]["arch"] = arch
    if selected_overrides:
        cfg0 = apply_candidate(cfg0, selected_overrides)
    if external_impl:
        cfg0["model"]["external_impl"] = external_impl
    per_h, run_records = {}, []
    for h in horizons:
        mses, maes, runs = [], [], []
        for s in range(seeds):
            cfg = copy.deepcopy(cfg0)
            cfg["data"]["pred_len"] = h
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_{arch}_h{h}_s{s}"
            print(f"\n===== {arch} | H={h} | seed {base_seed + s} =====")
            out = train(cfg)
            metrics = dict(out["test_metrics"])
            run = {
                **metrics,
                "seed": base_seed + s,
                "horizon": h,
                "arch": arch,
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out["best_val_metrics"],
                "best_epoch": out["best_epoch"],
                "param_count": out["param_count"],
                "checkpoint": out["checkpoint"],
            }
            runs.append(run)
            run_records.append(run)
            mses.append(metrics["mse"])
            maes.append(metrics["mae"])
        per_h[str(h)] = {"runs": runs, "mse": mean_std(mses), "mae": mean_std(maes)}
    return per_h, run_records


def parse_external_implementations(values):
    """Parse repeatable ARCH=module:Class adapters.

    A value without ``ARCH=`` remains a backward-compatible TF4TF adapter.
    """
    mapping = {}
    for value in values or []:
        if "=" in value:
            arch, implementation = value.split("=", 1)
        else:
            arch, implementation = "tf4tf", value
        arch, implementation = arch.strip(), implementation.strip()
        if not arch or not implementation or ":" not in implementation:
            raise ValueError(f"invalid external implementation: {value!r}")
        if arch in mapping:
            raise ValueError(f"duplicate external implementation for {arch!r}")
        mapping[arch] = implementation
    return mapping


def load_selection_report(path, base_cfg, requested_archs, require_selected):
    """Load frozen validation selections and reject stale/mismatched reports."""
    if not path:
        if require_selected:
            raise ValueError("--require-selected-baselines needs --selection-report")
        return {}, None
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("report_schema") != "ddmamba-baseline-selection-v1":
        raise ValueError("unsupported baseline selection report")
    if payload.get("selection_scope") != "validation_only_no_test_loader":
        raise ValueError("selection report is not certified validation-only")
    if payload.get("config_sha256") != config_sha256(base_cfg):
        raise ValueError(
            "dataset config differs from the frozen selection report; rerun "
            "baseline selection instead of applying stale hyperparameters"
        )
    frozen = payload.get("frozen_architectures")
    if not isinstance(frozen, dict):
        raise ValueError("selection report lacks frozen_architectures")
    selected = {}
    missing = []
    for architecture in requested_archs:
        if architecture == "dual_domain":
            continue
        entry = frozen.get(architecture)
        if not isinstance(entry, dict) or not isinstance(entry.get("overrides"), dict):
            missing.append(architecture)
        else:
            selected[architecture] = copy.deepcopy(entry["overrides"])
    if missing and require_selected:
        raise ValueError(
            "requested baselines have no frozen validation selection: "
            + ", ".join(missing)
        )
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return selected, {"path": str(report_path), "sha256": digest}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--archs", nargs="*", default=DEFAULT_ARCHS)
    ap.add_argument("--horizons", nargs="*", type=int, default=[96, 192, 336, 720])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--external_impl", action="append", default=[],
        help="repeatable ARCH=module:Class official adapter; a bare module:Class "
             "is treated as TF4TF for backward compatibility",
    )
    ap.add_argument(
        "--selection-report", default=None,
        help="validation-only report from run_baseline_selection.py",
    )
    ap.add_argument(
        "--require-selected-baselines", action="store_true",
        help="refuse any non-DD architecture missing from the frozen report",
    )
    ap.add_argument("--out", default=None)
    args, unknown = ap.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]

    try:
        external_implementations = parse_external_implementations(args.external_impl)
        selected_overrides, selection_source = load_selection_report(
            args.selection_report, base_cfg, args.archs,
            args.require_selected_baselines,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    results, failures, run_records = {}, {}, []
    for arch in args.archs:
        try:
            results[arch], arch_records = run_arch(
                base_cfg, arch, args.horizons, args.seeds, base_seed,
                external_implementations.get(arch),
                selected_overrides.get(arch),
            )
            run_records.extend(arch_records)
        except NotImplementedError as e:
            print(f"[skip] {arch}: {e}")
            failures[arch] = str(e)
        except Exception as e:  # keep the sweep going; record the failure
            print(f"[error] {arch}: {e}")
            traceback.print_exc()
            failures[arch] = str(e)

    out_path = args.out or os.path.join(
        base_cfg["experiment"]["checkpoint_dir"], f"unified_{name}.json")
    record = provenance_fields(
        base_cfg, config_path=args.config,
        seed_values=[base_seed + s for s in range(args.seeds)],
        cwd=Path(__file__).resolve().parents[1],
    )
    record.update({
        "architectures": list(args.archs),
        "architecture_overrides": {
            architecture: selected_overrides.get(architecture, {})
            for architecture in args.archs
        },
        "baseline_selection_source": selection_source,
        "external_implementations": external_implementations,
        "external_impl": args.external_impl,
        "horizons": list(args.horizons),
        "seeds": args.seeds,
        "run_records": run_records,
        "results": results,
        "failures": failures,
    })
    atomic_write_json(out_path, record)

    # Markdown table: avg MSE/MAE over horizons per arch.
    print(f"\n## Unified same-pipeline comparison — {name} ({args.seeds} seeds)\n")
    print("| Arch | " + " | ".join(f"H{h} MSE" for h in args.horizons) + " | Avg MSE | Avg MAE |")
    print("|---" * (len(args.horizons) + 3) + "|")
    for arch, per_h in results.items():
        cells = [f"{per_h[str(h)]['mse'][0]:.3f}" for h in args.horizons]
        avg_mse = sum(per_h[str(h)]["mse"][0] for h in args.horizons) / len(args.horizons)
        avg_mae = sum(per_h[str(h)]["mae"][0] for h in args.horizons) / len(args.horizons)
        print(f"| {arch} | " + " | ".join(cells) + f" | {avg_mse:.3f} | {avg_mae:.3f} |")
    if failures:
        print("\nSkipped/failed:", ", ".join(failures))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
