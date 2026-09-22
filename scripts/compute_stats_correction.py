#!/usr/bin/env python3
"""Audit branch-removal and component-ablation paired comparisons.

The correction families are explicit in each generated audit:

* every observed branch-removal comparison forms one family (54 in the
  current legacy manuscript; 72 after the planned complete 9x4 rerun);
* all 29 component-placement/replacement comparisons form one family.

Each output row contains anonymous legacy pair observations (or explicit seed
IDs when present), paired differences, effect/CI/t/df/raw p/BH q, an exact
sign-flip sensitivity test, relative effect, and paired standardized effect.
JSON is the source of truth for the generated Markdown and LaTeX appendices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.statistical_audit import (  # noqa: E402
    SCHEMA_VERSION,
    apply_bh_family,
    benjamini_hochberg,
    format_number,
    latex_escape,
    pair_numeric_series,
    pair_run_records,
    paired_statistics,
    statistics_from_observations,
    student_t_two_sided_p,
    write_json,
    write_text,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "generated" / "statistical_audit"
DEFAULT_COMPONENT_SOURCES = [
    ("Ablation_ETTh1.json", "ETTh1"),
    ("Ablation_solar_final.json", "Solar"),
    ("Ablation_weather.json", "Weather"),
]


def t_sf_two_sided(t_statistic: float, df: int) -> float:
    """Backward-compatible name retained for existing imports."""
    return student_t_two_sided_p(t_statistic, df)


def paired_test(full: Sequence[float], variant: Sequence[float]) -> dict[str, Any]:
    """Backward-compatible flat view over the expanded paired audit."""
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


def _relative_source_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _make_row(
    row_id: str,
    analysis: str,
    dataset: str,
    horizon: int | None,
    reference_variant: str,
    comparator_variant: str,
    observations: list[dict[str, Any]],
    pairing: dict[str, Any],
    source_file: str,
    branch_removed: str | None = None,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "analysis": analysis,
        "dataset": dataset,
        "horizon": horizon,
        "metric": "mse",
        "reference_variant": reference_variant,
        "comparator_variant": comparator_variant,
        "branch_removed": branch_removed,
        "source_file": source_file,
        "pairing": pairing,
        "observations": observations,
        "statistics": statistics_from_observations(observations),
        "multiplicity": {},
    }


def build_component_audit(
    results_dir: Path = RESULTS,
    branch_filename: str = "Branch_matrix.json",
    component_sources: Sequence[tuple[str, str]] = DEFAULT_COMPONENT_SOURCES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    legacy_sources: list[str] = []

    branch_path = results_dir / branch_filename
    branch_payload = json.loads(branch_path.read_text(encoding="utf-8"))
    branch_results = branch_payload["results"]
    branch_run_records = branch_payload.get("run_records")
    if branch_run_records is not None and not isinstance(branch_run_records, list):
        raise ValueError(f"{branch_path} run_records must be a list when present")
    branch_source = _relative_source_path(branch_path)
    branch_legacy = False
    for cell in sorted(branch_results):
        values = branch_results[cell]
        dataset, horizon_text = cell.rsplit("@", 1)
        horizon = int(horizon_text)
        for comparator_variant, branch_removed in (
            ("time_only", "frequency"),
            ("freq_only", "time"),
        ):
            if branch_run_records is not None:
                reference_runs = [
                    run for run in branch_run_records
                    if run.get("dataset") == dataset
                    and int(run.get("horizon", -1)) == horizon
                    and run.get("variant") == "full"
                ]
                comparator_runs = [
                    run for run in branch_run_records
                    if run.get("dataset") == dataset
                    and int(run.get("horizon", -1)) == horizon
                    and run.get("variant") == comparator_variant
                ]
                if not reference_runs or not comparator_runs:
                    raise ValueError(
                        f"{branch_path} has run_records but lacks explicit records "
                        f"for {cell} full/{comparator_variant}"
                    )
                observations, pairing = pair_run_records(
                    reference_runs, comparator_runs, "mse"
                )
            else:
                observations, pairing = pair_numeric_series(
                    values["full"], values[comparator_variant]
                )
            branch_legacy = branch_legacy or pairing["legacy"]
            rows.append(
                _make_row(
                    f"branch|{dataset}|h{horizon}|remove_{branch_removed}",
                    "branch_removal",
                    dataset,
                    horizon,
                    "full",
                    comparator_variant,
                    observations,
                    pairing,
                    branch_source,
                    branch_removed,
                )
            )
    if branch_legacy:
        legacy_sources.append(branch_source)
    sources.append(
        {
            "path": branch_source,
            "kind": "branch_removal_matrix",
            "declared_run_count": branch_payload.get("seeds"),
            "seed_identity_status": (
                "not_recorded_legacy_positional_pairing" if branch_legacy else "explicit_seed_ids"
            ),
        }
    )

    component_source_paths: list[str] = []
    for filename, dataset in component_sources:
        path = results_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload["results"]
        source_file = _relative_source_path(path)
        component_source_paths.append(source_file)
        source_legacy = False
        reference_runs = entries["full"]["runs"]
        for variant, entry in entries.items():
            if variant == "full":
                continue
            observations, pairing = pair_run_records(reference_runs, entry["runs"], "mse")
            source_legacy = source_legacy or pairing["legacy"]
            rows.append(
                _make_row(
                    f"component|{dataset}|{variant}",
                    "component_ablation",
                    dataset,
                    None,
                    "full",
                    variant,
                    observations,
                    pairing,
                    source_file,
                )
            )
        if source_legacy:
            legacy_sources.append(source_file)
        sources.append(
            {
                "path": source_file,
                "kind": "component_ablation",
                "dataset": dataset,
                "declared_run_count": payload.get("seeds"),
                "seed_identity_status": (
                    "not_recorded_legacy_positional_pairing"
                    if source_legacy
                    else "explicit_seed_ids"
                ),
            }
        )

    branch_members = [row["id"] for row in rows if row["analysis"] == "branch_removal"]
    component_members = [
        row["id"] for row in rows if row["analysis"] == "component_ablation"
    ]
    families = [
        apply_bh_family(
            rows,
            "branch_removal.all",
            branch_members,
            "All frequency- and time-branch removals across every observed dataset-horizon cell",
        ),
        apply_bh_family(
            rows,
            "component_placement.all",
            component_members,
            "All placement, replacement, normalization, initialization, and fusion comparisons",
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "component_and_branch_statistical_audit",
        "metric": "mse",
        "difference_definition": "comparator_minus_full; positive means removing/replacing the component hurts",
        "family_policy": {
            "branch_family_predeclared": "all branch removals in Branch_matrix.json",
            "component_family_predeclared": component_source_paths,
            "expected_branch_family_size": 2 * len(branch_results),
            "expected_component_family_size": 29,
            "observed_branch_family_size": len(branch_members),
            "observed_component_family_size": len(component_members),
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


def _family_id(row: dict[str, Any]) -> str:
    return (
        "branch_removal.all"
        if row["analysis"] == "branch_removal"
        else "component_placement.all"
    )


def render_component_markdown(audit: dict[str, Any], include_title: bool = True) -> str:
    lines: list[str] = []
    if include_title:
        lines.extend(["# Component and Branch Statistical Audit", ""])
    lines.extend(
        [
            "## Audit conventions",
            "",
            f"- Difference: `{audit['difference_definition']}`.",
            "- Paired two-sided t tests and 95% t intervals are primary; exact sign-flip tests are small-n sensitivity analyses.",
            "- BH correction is performed independently for the branch-removal and component-placement families.",
            "- Legacy pair indices remain anonymous and are not seed IDs.",
            "",
            "## Family registry",
            "",
            "| Family ID | Size | Description |",
            "|---|---:|---|",
        ]
    )
    for family in audit["families"]:
        lines.append(f"| `{family['id']}` | {family['size']} | {family['description']} |")

    for analysis, heading in (
        ("branch_removal", "Branch-removal comparisons"),
        ("component_ablation", "Component-placement and replacement comparisons"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| ID | Dataset | H | Variant | n | Full | Comparator | Diff | 95% CI | t | df | raw p | BH q | exact p | exact BH q | Relative % | dz |",
                "|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in audit["rows"] if item["analysis"] == analysis]:
            stats = row["statistics"]
            ci = stats["ci95"]
            correction = row["multiplicity"][_family_id(row)]
            lines.append(
                "| `{id}` | {dataset} | {horizon} | {variant} | {n} | {ref} | {comp} | {effect} | "
                "[{lo}, {hi}] | {t} | {df} | {p} | {q} | {exact} | {exactq} | {relative} | {dz} |".format(
                    id=row["id"],
                    dataset=row["dataset"],
                    horizon=row["horizon"] if row["horizon"] is not None else "--",
                    variant=row["comparator_variant"],
                    n=stats["n"],
                    ref=format_number(stats["mean_reference"]),
                    comp=format_number(stats["mean_comparator"]),
                    effect=format_number(stats["mean_paired_difference"]),
                    lo=format_number(ci["low"]),
                    hi=format_number(ci["high"]),
                    t=format_number(stats["t_statistic"]),
                    df=stats["df"],
                    p=format_number(stats["p_value_raw"]),
                    q=format_number(correction["q_bh"]),
                    exact=format_number(stats["exact_sign_flip_p_two_sided"]),
                    exactq=format_number(correction["q_bh_exact_sign_flip"]),
                    relative=format_number(stats["relative_effect_percent"]),
                    dz=format_number(stats["standardized_effect_dz"]),
                )
            )

    lines.extend(
        [
            "",
            "## Seed-level / legacy-pair observations",
            "",
            "| Comparison ID | Pair index | Seed ID | Full MSE | Comparator MSE | Paired diff |",
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


def render_component_latex(audit: dict[str, Any], include_heading: bool = True) -> str:
    lines: list[str] = []
    if include_heading:
        lines.extend(
            [
                r"\section{Component and branch statistical audit}",
                r"\label{app:component-statistical-audit}",
            ]
        )
    lines.extend(
        [
            r"Differences are comparator MSE minus full-model MSE. Positive values mean that removal or replacement hurts. Paired two-sided Student $t$ tests and 95\% $t$ intervals are primary; exact sign-flip tests are sensitivity analyses. Legacy pair indices are not seed identifiers.",
            "",
            r"\begin{longtable}{lllrrrrrrrr}",
            r"\caption{Complete component and branch statistical audit.}\\",
            r"\toprule",
            r"Dataset & $H$ & Variant & $n$ & Diff & CI low & CI high & $t$ & df & $p$ & $q$ \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Dataset & $H$ & Variant & $n$ & Diff & CI low & CI high & $t$ & df & $p$ & $q$ \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in audit["rows"]:
        stats = row["statistics"]
        ci = stats["ci95"]
        correction = row["multiplicity"][_family_id(row)]
        values = [
            latex_escape(row["dataset"]),
            str(row["horizon"] if row["horizon"] is not None else "--"),
            latex_escape(row["comparator_variant"]),
            str(stats["n"]),
            format_number(stats["mean_paired_difference"]),
            format_number(ci["low"]),
            format_number(ci["high"]),
            format_number(stats["t_statistic"]),
            str(stats["df"]),
            format_number(stats["p_value_raw"]),
            format_number(correction["q_bh"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])

    lines.extend(
        [
            r"\begin{longtable}{llrrrr}",
            r"\caption{Seed-level or anonymous legacy-pair component observations.}\\",
            r"\toprule",
            r"Comparison & Seed & Pair & Full & Comparator & Difference \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Comparison & Seed & Pair & Full & Comparator & Difference \\",
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


def write_component_bundle(audit: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    json_path = output_dir / f"{stem}.json"
    serialized = write_json(json_path, audit)
    markdown_path = output_dir / f"{stem}.md"
    latex_path = output_dir / f"{stem}.tex"
    write_text(markdown_path, render_component_markdown(serialized))
    write_text(latex_path, render_component_latex(serialized))
    return {"json": json_path, "markdown": markdown_path, "latex": latex_path}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument(
        "--branch-file", default="Branch_matrix.json",
        help="branch-matrix JSON filename inside --results-dir; use a new "
             "provenance-enabled file instead of resuming a legacy artifact",
    )
    parser.add_argument(
        "--component-source", action="append", default=[], metavar="FILE=DATASET",
        help="repeatable component-ablation source inside --results-dir; "
             "defaults to the three legacy manuscript files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default="component_branch_audit")
    args = parser.parse_args(argv)
    component_sources = DEFAULT_COMPONENT_SOURCES
    if args.component_source:
        parsed = []
        for specification in args.component_source:
            filename, separator, dataset = specification.partition("=")
            if not separator or not filename.strip() or not dataset.strip():
                parser.error(f"invalid --component-source: {specification!r}")
            parsed.append((filename.strip(), dataset.strip()))
        component_sources = parsed
    audit = build_component_audit(
        args.results_dir,
        branch_filename=args.branch_file,
        component_sources=component_sources,
    )
    paths = write_component_bundle(audit, args.output_dir, args.stem)
    print(
        f"Component audit: {len(audit['rows'])} tests; outputs: "
        + ", ".join(str(path) for path in paths.values())
    )


if __name__ == "__main__":
    main()
