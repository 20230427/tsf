#!/usr/bin/env python3
"""Generate a complete paired audit for same-pipeline unified baselines.

The script reads legacy or seed-identified ``results/unified_<dataset>.json``
files and writes a machine-readable JSON audit plus Markdown and LaTeX
appendices under ``generated/statistical_audit``.  Human tables are rendered
only after the JSON has been written and read back.

Two multiplicity scopes are reported side by side:

* ``displayed``: the five baseline architectures predeclared for the paper's
  matched comparison (S-Mamba, iTransformer, RLinear, PatchTST, DLinear);
* ``full``: every non-DD architecture found in the unified result files.

Both average-over-horizon and per-horizon families are explicit.  Legacy run
positions remain anonymous pair indices; they are never relabeled as seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.statistical_audit import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_observations,
    apply_bh_family,
    benjamini_hochberg,
    format_number,
    latex_escape,
    pair_run_records,
    paired_statistics,
    statistics_from_observations,
    write_json,
    write_text,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "generated" / "statistical_audit"

REFERENCE_MODEL = "dual_domain"
DISPLAYED_BASELINES = ["smamba", "itransformer", "rlinear", "patchtst", "dlinear"]
DATASET_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
NICE = {
    "configs/ETTh1.yaml": "ETTh1",
    "configs/ETTh2.yaml": "ETTh2",
    "configs/ETTm1.yaml": "ETTm1",
    "configs/ETTm2.yaml": "ETTm2",
    "configs/weather.yaml": "Weather",
    "configs/electricity.yaml": "Electricity",
    "configs/solar.yaml": "Solar",
    "configs/traffic.yaml": "Traffic",
    "configs/exchange_rate.yaml": "Exchange",
}


def _dataset_sort_key(name: str) -> tuple[int, str]:
    return (
        DATASET_ORDER.index(name) if name in DATASET_ORDER else len(DATASET_ORDER),
        name,
    )


def _baseline_sort_key(name: str) -> tuple[int, str]:
    return (
        DISPLAYED_BASELINES.index(name)
        if name in DISPLAYED_BASELINES
        else len(DISPLAYED_BASELINES),
        name,
    )


def per_seed_avg_mse(cell_by_h: dict[str, Any], horizons: Sequence[int]) -> list[float]:
    """Backward-compatible helper: positional per-run average over horizons."""
    first_runs = cell_by_h[str(horizons[0])]["runs"]
    count = len(first_runs)
    if any(len(cell_by_h[str(horizon)]["runs"]) != count for horizon in horizons):
        raise ValueError("horizon run counts differ")
    return [
        sum(cell_by_h[str(horizon)]["runs"][index]["mse"] for horizon in horizons)
        / len(horizons)
        for index in range(count)
    ]


def load_unified(results_dir: Path = RESULTS) -> dict[str, dict[str, Any]]:
    """Load source files without treating prior analysis output as raw data."""
    paths = sorted(results_dir.glob("unified_*.json"))
    paths = [path for path in paths if path.name != "unified_analysis.json"]
    if not paths and results_dir == RESULTS:
        paths = [Path(path) for path in sorted(glob.glob(str(ROOT / "JSON*.json")))]
    datasets: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in payload or REFERENCE_MODEL not in payload["results"]:
            continue
        dataset = NICE.get(payload.get("config"), payload.get("config", path.stem))
        payload["_source_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        datasets[dataset] = payload
    return datasets


def _make_row(
    dataset: str,
    baseline: str,
    horizon: int | None,
    observations: list[dict[str, Any]],
    pairing: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    level = "average_over_horizons" if horizon is None else "per_horizon"
    horizon_label = "average" if horizon is None else f"h{horizon}"
    return {
        "id": f"unified|{dataset}|{horizon_label}|{baseline}",
        "analysis": "unified_baseline",
        "level": level,
        "dataset": dataset,
        "horizon": horizon,
        "metric": "mse",
        "reference_model": REFERENCE_MODEL,
        "comparator_model": baseline,
        "source_file": source_path,
        "pairing": pairing,
        "observations": observations,
        "statistics": statistics_from_observations(observations),
        "multiplicity": {},
    }


def build_unified_audit(
    results_dir: Path = RESULTS,
    displayed_baselines: Sequence[str] = DISPLAYED_BASELINES,
) -> dict[str, Any]:
    datasets = load_unified(results_dir)
    if not datasets:
        raise FileNotFoundError(f"No unified raw result files found under {results_dir}")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    full_baselines: set[str] = set()
    missing_displayed: dict[str, list[str]] = {}
    legacy_sources: list[str] = []

    for dataset in sorted(datasets, key=_dataset_sort_key):
        payload = datasets[dataset]
        results = payload["results"]
        horizons = [int(horizon) for horizon in payload["horizons"]]
        baselines = sorted(
            (architecture for architecture in results if architecture != REFERENCE_MODEL),
            key=_baseline_sort_key,
        )
        full_baselines.update(baselines)
        missing = [baseline for baseline in displayed_baselines if baseline not in baselines]
        if missing:
            missing_displayed[dataset] = missing
        source_has_legacy = False

        for baseline in baselines:
            by_horizon: dict[int, list[dict[str, Any]]] = {}
            pairing_by_horizon: dict[int, dict[str, Any]] = {}
            for horizon in horizons:
                reference_runs = results[REFERENCE_MODEL][str(horizon)]["runs"]
                comparator_runs = results[baseline][str(horizon)]["runs"]
                observations, pairing = pair_run_records(reference_runs, comparator_runs, "mse")
                by_horizon[horizon] = observations
                pairing_by_horizon[horizon] = pairing
                source_has_legacy = source_has_legacy or pairing["legacy"]
                rows.append(
                    _make_row(
                        dataset,
                        baseline,
                        horizon,
                        observations,
                        pairing,
                        payload["_source_path"],
                    )
                )

            average_observations, average_pairing = aggregate_observations(by_horizon)
            rows.append(
                _make_row(
                    dataset,
                    baseline,
                    None,
                    average_observations,
                    average_pairing,
                    payload["_source_path"],
                )
            )

        if source_has_legacy:
            legacy_sources.append(payload["_source_path"])
        sources.append(
            {
                "dataset": dataset,
                "path": payload["_source_path"],
                "declared_run_count": payload.get("seeds"),
                "horizons": horizons,
                "architectures": [REFERENCE_MODEL, *baselines],
                "seed_identity_status": (
                    "not_recorded_legacy_positional_pairing"
                    if source_has_legacy
                    else "explicit_seed_ids"
                ),
            }
        )

    families: list[dict[str, Any]] = []
    displayed_set = set(displayed_baselines)
    for level, label in (
        ("average_over_horizons", "average"),
        ("per_horizon", "per_horizon"),
    ):
        level_rows = [row for row in rows if row["level"] == level]
        displayed_members = [
            row["id"] for row in level_rows if row["comparator_model"] in displayed_set
        ]
        full_members = [row["id"] for row in level_rows]
        families.append(
            apply_bh_family(
                rows,
                f"unified.{label}.displayed",
                displayed_members,
                (
                    f"Predeclared displayed-baseline {label} family: "
                    f"{', '.join(displayed_baselines)} across all available datasets"
                ),
            )
        )
        families.append(
            apply_bh_family(
                rows,
                f"unified.{label}.full",
                full_members,
                f"Full observed {label} family: every non-{REFERENCE_MODEL} architecture",
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "unified_baseline_statistical_audit",
        "metric": "mse",
        "difference_definition": "comparator_minus_reference; positive means DD-Mamba is better",
        "reference_model": REFERENCE_MODEL,
        "family_policy": {
            "displayed_baselines_predeclared": list(displayed_baselines),
            "displayed_family_rule": "all available dataset comparisons against the predeclared list",
            "full_family_rule": "all available dataset comparisons against every observed non-reference architecture",
            "full_baselines_observed": sorted(full_baselines, key=_baseline_sort_key),
            "missing_displayed_baselines_by_dataset": missing_displayed,
        },
        "seed_identity_policy": {
            "rule": "Use explicit recorded seed IDs only; never infer IDs from run positions or counts.",
            "legacy_sources": legacy_sources,
            "legacy_pair_label": "pair_index",
        },
        "sources": sources,
        "families": families,
        "rows": rows,
    }


def _family_q(row: dict[str, Any], family_id: str, field: str = "q_bh") -> Any:
    return row.get("multiplicity", {}).get(family_id, {}).get(field)


def render_unified_markdown(audit: dict[str, Any], include_title: bool = True) -> str:
    """Render Markdown exclusively from a serialized audit payload."""
    lines: list[str] = []
    if include_title:
        lines.extend(["# Unified Baseline Statistical Audit", ""])
    lines.extend(
        [
            "## Audit conventions",
            "",
            f"- Difference: `{audit['difference_definition']}`.",
            "- Primary inference: paired two-sided Student t test with a 95% t interval.",
            "- Small-n sensitivity: exact two-sided sign-flip enumeration; its minimum attainable p-value is shown without asymptotic substitution.",
            "- Multiplicity: BH q-values are reported for both the predeclared displayed family and the full observed family.",
            "- Legacy provenance: a `pair_index` is an anonymous positional pairing key, not a seed ID.",
            "",
            "## Family registry",
            "",
            "| Family ID | Size | Description |",
            "|---|---:|---|",
        ]
    )
    for family in audit["families"]:
        lines.append(f"| `{family['id']}` | {family['size']} | {family['description']} |")

    lines.extend(
        [
            "",
            "## Average-over-horizon paired results",
            "",
            "| Dataset | Baseline | n | DD | Baseline | Diff | 95% CI | t(df) | raw p | BH q displayed | BH q full | exact sign-flip p | Relative % | dz |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    average_rows = [row for row in audit["rows"] if row["level"] == "average_over_horizons"]
    for row in average_rows:
        stats = row["statistics"]
        ci = stats["ci95"]
        t_df = f"{format_number(stats['t_statistic'])} ({stats['df']})"
        lines.append(
            "| {dataset} | {baseline} | {n} | {ref} | {comp} | {effect} | [{lo}, {hi}] | "
            "{tdf} | {p} | {qd} | {qf} | {perm} | {relative} | {dz} |".format(
                dataset=row["dataset"],
                baseline=row["comparator_model"],
                n=stats["n"],
                ref=format_number(stats["mean_reference"]),
                comp=format_number(stats["mean_comparator"]),
                effect=format_number(stats["mean_paired_difference"]),
                lo=format_number(ci["low"]),
                hi=format_number(ci["high"]),
                tdf=t_df,
                p=format_number(stats["p_value_raw"]),
                qd=format_number(_family_q(row, "unified.average.displayed")),
                qf=format_number(_family_q(row, "unified.average.full")),
                perm=format_number(stats["exact_sign_flip_p_two_sided"]),
                relative=format_number(stats["relative_effect_percent"]),
                dz=format_number(stats["standardized_effect_dz"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-horizon paired results",
            "",
            "| Dataset | H | Baseline | n | Diff | 95% CI | t | df | raw p | BH q displayed | BH q full | exact p | Relative % | dz |",
            "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    per_horizon_rows = [row for row in audit["rows"] if row["level"] == "per_horizon"]
    for row in per_horizon_rows:
        stats = row["statistics"]
        ci = stats["ci95"]
        lines.append(
            "| {dataset} | {horizon} | {baseline} | {n} | {effect} | [{lo}, {hi}] | {t} | {df} | "
            "{p} | {qd} | {qf} | {perm} | {relative} | {dz} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                baseline=row["comparator_model"],
                n=stats["n"],
                effect=format_number(stats["mean_paired_difference"]),
                lo=format_number(ci["low"]),
                hi=format_number(ci["high"]),
                t=format_number(stats["t_statistic"]),
                df=stats["df"],
                p=format_number(stats["p_value_raw"]),
                qd=format_number(_family_q(row, "unified.per_horizon.displayed")),
                qf=format_number(_family_q(row, "unified.per_horizon.full")),
                perm=format_number(stats["exact_sign_flip_p_two_sided"]),
                relative=format_number(stats["relative_effect_percent"]),
                dz=format_number(stats["standardized_effect_dz"]),
            )
        )

    lines.extend(
        [
            "",
            "## Seed-level / legacy-pair observations",
            "",
            "`seed_id` is `not recorded` for legacy inputs; the pair index must not be cited as a seed.",
            "",
            "| Comparison ID | Pair index | Seed ID | Reference MSE | Comparator MSE | Paired diff |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in audit["rows"]:
        for observation in row["observations"]:
            seed_id = observation["seed_id"] if observation["seed_id"] is not None else "not recorded"
            lines.append(
                f"| `{row['id']}` | {observation['pair_index']} | {seed_id} | "
                f"{format_number(observation['reference'])} | "
                f"{format_number(observation['comparator'])} | "
                f"{format_number(observation['paired_difference'])} |"
            )

    lines.extend(["", "## Explicit family members", ""])
    for family in audit["families"]:
        lines.append(f"### `{family['id']}` ({family['size']})")
        lines.append("")
        lines.extend(f"- `{member}`" for member in family["members"])
        lines.append("")
    return "\n".join(lines)


def render_unified_latex(audit: dict[str, Any], include_heading: bool = True) -> str:
    """Render a standalone appendix fragment from the serialized JSON."""
    lines: list[str] = []
    if include_heading:
        lines.extend(
            [
                r"\section{Unified baseline statistical audit}",
                r"\label{app:unified-statistical-audit}",
            ]
        )
    lines.extend(
        [
            r"The paired difference is comparator MSE minus DD-Mamba MSE; positive values favor DD-Mamba. "
            r"Paired two-sided Student $t$ tests and 95\% $t$ intervals are primary. Exact two-sided sign-flip tests are reported as a small-sample sensitivity analysis. Legacy pair indices are not seed identifiers.",
            "",
            r"\begin{longtable}{llrrrrrrrr}",
            r"\caption{Average-over-horizon paired baseline audit. $q_d$ and $q_f$ denote BH correction in the displayed and full families.}\\",
            r"\toprule",
            r"Dataset & Baseline & $n$ & Diff & CI low & CI high & $p$ & $q_d$ & $q_f$ & $p_{SF}$ \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Dataset & Baseline & $n$ & Diff & CI low & CI high & $p$ & $q_d$ & $q_f$ & $p_{SF}$ \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in [item for item in audit["rows"] if item["level"] == "average_over_horizons"]:
        stats = row["statistics"]
        ci = stats["ci95"]
        values = [
            latex_escape(row["dataset"]),
            latex_escape(row["comparator_model"]),
            str(stats["n"]),
            format_number(stats["mean_paired_difference"]),
            format_number(ci["low"]),
            format_number(ci["high"]),
            format_number(stats["p_value_raw"]),
            format_number(_family_q(row, "unified.average.displayed")),
            format_number(_family_q(row, "unified.average.full")),
            format_number(stats["exact_sign_flip_p_two_sided"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    lines.extend(
        [
            r"\begin{longtable}{lrllrrrrrr}",
            r"\caption{Per-horizon paired baseline audit.}\\",
            r"\toprule",
            r"Dataset & $H$ & Baseline & $n$ & Diff & CI low & CI high & $p$ & $q_d$ & $q_f$ \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Dataset & $H$ & Baseline & $n$ & Diff & CI low & CI high & $p$ & $q_d$ & $q_f$ \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in [item for item in audit["rows"] if item["level"] == "per_horizon"]:
        stats = row["statistics"]
        ci = stats["ci95"]
        values = [
            latex_escape(row["dataset"]),
            str(row["horizon"]),
            latex_escape(row["comparator_model"]),
            str(stats["n"]),
            format_number(stats["mean_paired_difference"]),
            format_number(ci["low"]),
            format_number(ci["high"]),
            format_number(stats["p_value_raw"]),
            format_number(_family_q(row, "unified.per_horizon.displayed")),
            format_number(_family_q(row, "unified.per_horizon.full")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    lines.extend(
        [
            r"\begin{longtable}{llrrrr}",
            r"\caption{Seed-level or anonymous legacy-pair observations. A dash in Seed denotes an unrecorded identifier.}\\",
            r"\toprule",
            r"Comparison & Seed & Pair & Reference & Comparator & Difference \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Comparison & Seed & Pair & Reference & Comparator & Difference \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in audit["rows"]:
        for observation in row["observations"]:
            values = [
                latex_escape(row["id"]),
                latex_escape(observation["seed_id"] if observation["seed_id"] is not None else "--"),
                str(observation["pair_index"]),
                format_number(observation["reference"]),
                format_number(observation["comparator"]),
                format_number(observation["paired_difference"]),
            ]
            lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def write_unified_bundle(audit: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    json_path = output_dir / f"{stem}.json"
    serialized = write_json(json_path, audit)
    markdown_path = output_dir / f"{stem}.md"
    latex_path = output_dir / f"{stem}.tex"
    write_text(markdown_path, render_unified_markdown(serialized))
    write_text(latex_path, render_unified_latex(serialized))
    return {"json": json_path, "markdown": markdown_path, "latex": latex_path}


# Compatibility names used by earlier tests and scripts.
def paired_test(full: Sequence[float], variant: Sequence[float]) -> dict[str, Any]:
    stats = paired_statistics(full, variant)
    return {
        "effect": stats["mean_paired_difference"],
        "p": stats["p_value_raw"],
        "lo": stats["ci95"]["low"],
        "hi": stats["ci95"]["high"],
        "n": stats["n"],
        "t": stats["t_statistic"],
        "df": stats["df"],
        "p_sign_flip": stats["exact_sign_flip_p_two_sided"],
        "relative_effect_percent": stats["relative_effect_percent"],
        "standardized_effect_dz": stats["standardized_effect_dz"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default="unified_baseline_audit")
    args = parser.parse_args(argv)
    audit = build_unified_audit(args.results_dir)
    paths = write_unified_bundle(audit, args.output_dir, args.stem)
    print(
        f"Unified audit: {len(audit['rows'])} tests across "
        f"{len(audit['sources'])} datasets; outputs: "
        + ", ".join(str(path) for path in paths.values())
    )


if __name__ == "__main__":
    main()
