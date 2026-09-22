#!/usr/bin/env python3
"""Aggregate matrix ablations into paired horizon-average and per-H audits.

Primary inference averages the requested horizons *within each matched seed*
before testing.  This avoids treating four models from the same seed as four
independent observations.  BH correction is applied separately by the
predeclared mechanism family and by analysis scope (primary vs per-horizon).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.statistical_audit import (  # noqa: E402
    SCHEMA_VERSION,
    apply_bh_family,
    format_number,
    pair_run_records,
    statistics_from_observations,
    write_json,
    write_text,
)


REMOVAL_VARIANTS = {
    "no_channel_mixer", "no_time_encoder", "time_only", "freq_only",
    "no_linear_backbone", "no_revin", "with_revin",
}
PLACEMENT_VARIANTS = {
    "both_mixer", "shared_mixer", "time_mixer_only", "freq_mixer_only",
}
INITIALIZATION_VARIANTS = {
    "rand_init", "time_head_random_init", "time_head_zero_init",
    "freq_backbone_random_init", "freq_backbone_zero_init",
    "freq_head_random_init", "freq_head_zero_init", "fusion_random_init",
    "fusion_zero_init",
}
NORMALIZATION_VARIANTS = {
    "revin_alpha_learned", "revin_alpha_channel", "disp_base", "disp_revin",
    "disp_fixed", "disp_learned",
}
SPECTRAL_VARIANTS = {
    "no_fits", "with_fits", "freq_mamba", "freq_linear",
    "freq_sparsity_0", "freq_sparsity_20", "freq_sparsity_40",
    "freq_sparsity_60",
}


def variant_family(variant: str) -> str:
    if variant in REMOVAL_VARIANTS:
        return "component_removal"
    if variant in PLACEMENT_VARIANTS:
        return "mixer_placement"
    if variant in INITIALIZATION_VARIANTS:
        return "initialization"
    if variant in NORMALIZATION_VARIANTS:
        return "normalization_reconstruction"
    if variant in SPECTRAL_VARIANTS:
        return "spectral_strategy"
    if variant.startswith("fusion_"):
        return "fusion_strategy"
    if variant.startswith(("time_", "vc_")):
        return "encoder_replacement"
    return "other_component"


def _load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    sources = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_records = payload.get("run_records")
        if not isinstance(source_records, list):
            continue
        if not any(run.get("variant") for run in source_records):
            continue
        dataset = payload.get("dataset")
        if dataset is None:
            config = payload.get("config")
            dataset = Path(config).stem if config else path.stem.split("_H", 1)[0]
        for run in source_records:
            if not run.get("variant"):
                continue
            item = dict(run)
            item.setdefault("dataset", dataset)
            item["source_file"] = str(path)
            records.append(item)
        sources.append(str(path))
    if not records:
        raise ValueError("no ablation run_records found in the selected JSON files")
    return records, sources


def _index_records(records: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    index = {}
    phases = {run.get("phase") for run in records}
    if len(phases) != 1:
        raise ValueError(f"refusing to mix evaluation scopes: {sorted(phases, key=str)}")
    for run in records:
        key = (
            str(run["dataset"]), int(run["horizon"]),
            str(run["variant"]), int(run["seed"]),
        )
        if key in index:
            raise ValueError(f"duplicate ablation cell: {key}")
        index[key] = run
    return index


def _row(
    dataset: str,
    variant: str,
    scope: str,
    horizon: int | None,
    reference_runs: list[dict[str, Any]],
    comparator_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    observations, pairing = pair_run_records(reference_runs, comparator_runs, "mse")
    horizon_label = "avg" if horizon is None else f"h{horizon}"
    return {
        "id": f"{scope}|{variant_family(variant)}|{dataset}|{horizon_label}|{variant}",
        "scope": scope,
        "family": variant_family(variant),
        "dataset": dataset,
        "horizon": horizon,
        "metric": "mse",
        "reference_variant": "full",
        "comparator_variant": variant,
        "pairing": pairing,
        "observations": observations,
        "statistics": statistics_from_observations(observations),
        "multiplicity": {},
    }


def build_audit(records: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    index = _index_records(records)
    datasets = sorted({key[0] for key in index})
    rows = []

    for dataset in datasets:
        variants = sorted({key[2] for key in index if key[0] == dataset} - {"full"})
        for variant in variants:
            variant_keys = [key for key in index if key[0] == dataset and key[2] == variant]
            variant_horizons = {key[1] for key in variant_keys}
            missing = set(horizons) - variant_horizons
            if missing:
                raise ValueError(
                    f"{dataset}/{variant} lacks horizons {sorted(missing)}; "
                    "do not compute a selective horizon average"
                )
            seed_sets = []
            for horizon in horizons:
                ref_seeds = {
                    key[3] for key in index
                    if key[:3] == (dataset, horizon, "full")
                }
                cmp_seeds = {
                    key[3] for key in index
                    if key[:3] == (dataset, horizon, variant)
                }
                if ref_seeds != cmp_seeds:
                    raise ValueError(
                        f"unmatched seeds for {dataset}/{variant}/H{horizon}: "
                        f"full={sorted(ref_seeds)}, variant={sorted(cmp_seeds)}"
                    )
                seed_sets.append(ref_seeds)
                ref_runs = [index[(dataset, horizon, "full", seed)]
                            for seed in sorted(ref_seeds)]
                cmp_runs = [index[(dataset, horizon, variant, seed)]
                            for seed in sorted(cmp_seeds)]
                rows.append(_row(
                    dataset, variant, "per_horizon", horizon, ref_runs, cmp_runs
                ))
            if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
                raise ValueError(
                    f"seed set changes across horizons for {dataset}/{variant}"
                )
            aggregate_ref, aggregate_cmp = [], []
            for seed in sorted(seed_sets[0]):
                aggregate_ref.append({
                    "seed": seed,
                    "mse": sum(index[(dataset, h, "full", seed)]["mse"]
                               for h in horizons) / len(horizons),
                })
                aggregate_cmp.append({
                    "seed": seed,
                    "mse": sum(index[(dataset, h, variant, seed)]["mse"]
                               for h in horizons) / len(horizons),
                })
            rows.append(_row(
                dataset, variant, "horizon_average_primary", None,
                aggregate_ref, aggregate_cmp,
            ))

    families = []
    family_keys = sorted({(row["scope"], row["family"]) for row in rows})
    for scope, family in family_keys:
        members = [row["id"] for row in rows
                   if row["scope"] == scope and row["family"] == family]
        family_id = f"{scope}.{family}"
        families.append(apply_bh_family(
            rows, family_id, members,
            f"{family} contrasts in the {scope} analysis",
        ))

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "ablation_matrix_paired_audit",
        "difference_definition": "variant_minus_full; positive favors the full model",
        "primary_endpoint": "per-seed arithmetic mean MSE across requested horizons",
        "horizons": horizons,
        "families": families,
        "rows": rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Ablation Matrix Statistical Audit", "",
        f"Primary endpoint: {audit['primary_endpoint']}.", "",
        "## Horizon-average primary comparisons", "",
        "| Dataset | Variant | Family | n | Full | Variant | ΔMSE | 95% CI | raw p | BH q | exact p |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in audit["rows"]:
        if row["scope"] != "horizon_average_primary":
            continue
        stats = row["statistics"]
        family_id = f"{row['scope']}.{row['family']}"
        correction = row["multiplicity"][family_id]
        ci = stats["ci95"]
        lines.append(
            f"| {row['dataset']} | `{row['comparator_variant']}` | "
            f"{row['family']} | {stats['n']} | "
            f"{format_number(stats['mean_reference'])} | "
            f"{format_number(stats['mean_comparator'])} | "
            f"{format_number(stats['mean_paired_difference'])} | "
            f"[{format_number(ci['low'])}, {format_number(ci['high'])}] | "
            f"{format_number(stats['p_value_raw'])} | "
            f"{format_number(correction['q_bh'])} | "
            f"{format_number(stats['exact_sign_flip_p_two_sided'])} |"
        )
    lines.extend(["", "Per-horizon rows and exact BH values are retained in the JSON."])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("checkpoints/ablation_matrix"))
    parser.add_argument("--pattern", default="*_core.json",
                        help="analyze one predeclared chain at a time")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[96, 192, 336, 720])
    parser.add_argument("--out", type=Path,
                        default=Path("generated/ablation_matrix_audit.json"))
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No JSON files match {args.input_dir / args.pattern}")
    records, sources = _load_records(paths)
    audit = build_audit(records, args.horizons)
    audit["sources"] = sources
    serialized = write_json(args.out, audit)
    markdown_path = args.out.with_suffix(".md")
    write_text(markdown_path, render_markdown(serialized))
    print(f"[saved] {args.out}")
    print(f"[saved] {markdown_path}")


if __name__ == "__main__":
    main()
