#!/usr/bin/env python3
"""Generate the complete JSON-sourced statistical audit report bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_unified_baselines import (
    DEFAULT_OUTPUT,
    RESULTS,
    build_unified_audit,
    render_unified_latex,
    render_unified_markdown,
)
from scripts.compute_stats_correction import (
    DEFAULT_COMPONENT_SOURCES,
    build_component_audit,
    render_component_latex,
    render_component_markdown,
)
from scripts.statistical_audit import SCHEMA_VERSION, write_json, write_text


def render_combined_markdown(payload: dict) -> str:
    return "\n\n".join(
        [
            "# Complete Statistical Audit\n\n"
            "This appendix is generated from the adjacent machine-readable JSON. "
            "No table performs a second statistical calculation.",
            render_unified_markdown(payload["unified_baseline_audit"], include_title=False),
            render_component_markdown(payload["component_branch_audit"], include_title=False),
        ]
    )


def render_combined_latex(payload: dict) -> str:
    return "\n\n".join(
        [
            r"\section{Complete statistical audit}" + "\n" +
            r"\label{app:complete-statistical-audit}" + "\n" +
            r"All tables in this appendix are generated from the adjacent machine-readable JSON audit.",
            render_unified_latex(payload["unified_baseline_audit"], include_heading=False),
            render_component_latex(payload["component_branch_audit"], include_heading=False),
        ]
    )


def write_combined_bundle(
    payload: dict, output_dir: Path, stem: str = "complete_statistical_audit"
) -> dict[str, Path]:
    json_path = output_dir / f"{stem}.json"
    # Reports are deliberately derived from the serialized/reloaded JSON.
    serialized = write_json(json_path, payload)
    markdown_path = output_dir / f"{stem}.md"
    latex_path = output_dir / f"{stem}.tex"
    write_text(markdown_path, render_combined_markdown(serialized))
    write_text(latex_path, render_combined_latex(serialized))
    return {"json": json_path, "markdown": markdown_path, "latex": latex_path}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument(
        "--branch-file", default="Branch_matrix.json",
        help="branch-matrix filename inside --results-dir",
    )
    parser.add_argument(
        "--component-source", action="append", default=[], metavar="FILE=DATASET",
        help="repeatable component-ablation source inside --results-dir",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default="complete_statistical_audit")
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

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "complete_statistical_audit",
        "report_generation_rule": "Markdown and LaTeX tables are derived from this serialized JSON.",
        "unified_baseline_audit": build_unified_audit(args.results_dir),
        "component_branch_audit": build_component_audit(
            args.results_dir,
            branch_filename=args.branch_file,
            component_sources=component_sources,
        ),
    }
    paths = write_combined_bundle(payload, args.output_dir, args.stem)
    print("Generated: " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
