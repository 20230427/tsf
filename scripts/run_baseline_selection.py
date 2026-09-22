#!/usr/bin/env python3
"""Select baseline hyperparameters under an equal validation-only budget.

This is the tuning stage for a fair same-pipeline comparison.  It never builds
or evaluates a test loader.  The output contains one frozen override set per
architecture; pass that report to ``run_unified_baselines.py`` for the separate
final evaluation stage.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_selection_protocol import eval_candidate, select_by_validation  # noqa: E402
from src.utils import (  # noqa: E402
    apply_overrides,
    atomic_write_json,
    config_sha256,
    load_config,
    parse_overrides,
    provenance_fields,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "ddmamba-baseline-selection-v1"


def candidate_grid(axes: dict) -> list[dict]:
    """Expand a deterministic dotted-key Cartesian product."""
    if not isinstance(axes, dict) or not axes:
        raise ValueError("each architecture requires a non-empty axes mapping")
    keys = list(axes)
    values = []
    for key in keys:
        axis = axes[key]
        if not isinstance(axis, list) or not axis:
            raise ValueError(f"search axis {key!r} must be a non-empty list")
        values.append(axis)
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def build_search_plan(search_spec: dict, architectures: list[str], *, allow_unequal: bool) -> dict[str, list[dict]]:
    declared = search_spec.get("architectures")
    if not isinstance(declared, dict):
        raise ValueError("search specification lacks architectures")
    plan = {}
    for architecture in architectures:
        if architecture not in declared:
            raise ValueError(f"no predeclared search space for {architecture!r}")
        entry = declared[architecture]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid search entry for {architecture!r}")
        plan[architecture] = candidate_grid(entry.get("axes"))

    expected = int(search_spec.get("trial_budget_per_architecture", 0))
    counts = {architecture: len(candidates) for architecture, candidates in plan.items()}
    if expected <= 0:
        raise ValueError("trial_budget_per_architecture must be positive")
    unequal = {architecture: count for architecture, count in counts.items() if count != expected}
    if unequal and not allow_unequal:
        raise ValueError(
            "search budget mismatch; refusing an unfair sweep: "
            + ", ".join(f"{name}={count}" for name, count in unequal.items())
            + f" (expected {expected})"
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="dataset configuration")
    parser.add_argument(
        "--search-space", default="configs/baseline_search_spaces.yaml",
        help="predeclared equal-budget search specification",
    )
    parser.add_argument(
        "--archs", nargs="+",
        default=["dlinear", "rlinear", "patchtst", "itransformer", "smamba", "msmamba"],
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-unequal-budget", action="store_true",
        help="explicit exploratory override; the report remains labeled unequal",
    )
    args, unknown = parser.parse_known_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    if len(set(args.archs)) != len(args.archs):
        raise SystemExit("--archs contains duplicates")

    base_cfg = apply_overrides(load_config(args.config), parse_overrides(unknown))
    search_spec = load_config(args.search_space)
    if search_spec.get("schema_version") != "ddmamba-baseline-search-v1":
        raise SystemExit("unsupported baseline search-space schema")
    try:
        plan = build_search_plan(
            search_spec, list(args.archs), allow_unequal=args.allow_unequal_budget
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    base_seed = int(base_cfg["experiment"]["seed"])
    dataset_name = str(base_cfg["experiment"]["name"])
    all_run_records = []
    frozen = {}
    candidate_audit = {}
    for architecture in args.archs:
        results = []
        for candidate_index, overrides in enumerate(plan[architecture]):
            candidate = {"model.arch": architecture, **copy.deepcopy(overrides)}
            tag = f"{dataset_name}_baseline_select_{architecture}_c{candidate_index}"
            print(
                f"\n===== {architecture} candidate {candidate_index + 1}/"
                f"{len(plan[architecture])}: {overrides} ====="
            )
            result = eval_candidate(
                base_cfg, candidate, args.horizons, args.seeds, base_seed, tag
            )
            result["candidate_index"] = candidate_index
            all_run_records.extend(result.pop("run_records"))
            results.append(result)
        selected_index, selected = select_by_validation(results)
        selected_overrides = {
            key: value for key, value in selected["candidate"].items()
            if key != "model.arch"
        }
        frozen[architecture] = {
            "candidate_index": selected_index,
            "overrides": selected_overrides,
            "mean_validation_loss": selected["val_loss"],
            "selection_metric": search_spec["selection_metric"],
            "trial_budget": len(results),
        }
        candidate_audit[architecture] = results
        print(
            f"[freeze] {architecture}: candidate {selected_index}, "
            f"validation={selected['val_loss']:.6g}"
        )

    report = provenance_fields(
        base_cfg,
        config_path=args.config,
        seed_values=[base_seed + offset for offset in range(args.seeds)],
        cwd=ROOT,
    )
    report.update(
        {
            "report_schema": REPORT_SCHEMA,
            "search_space_path": args.search_space,
            "search_space_sha256": config_sha256(search_spec),
            "selection_scope": "validation_only_no_test_loader",
            "selection_metric": search_spec["selection_metric"],
            "trial_budget_per_architecture": search_spec["trial_budget_per_architecture"],
            "equal_budget_enforced": not args.allow_unequal_budget,
            "architectures": list(args.archs),
            "horizons": list(args.horizons),
            "run_records": all_run_records,
            "frozen_architectures": frozen,
            "candidate_audit": candidate_audit,
        }
    )
    output = Path(args.out) if args.out else Path(
        base_cfg["experiment"]["checkpoint_dir"]
    ) / f"baseline_selection_{dataset_name}.json"
    atomic_write_json(output, report)
    print(f"\n[saved] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
