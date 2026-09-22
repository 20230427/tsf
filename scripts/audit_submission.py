#!/usr/bin/env python3
"""Fail-closed audit for the Neurocomputing submission package.

The audit is deliberately independent of PyTorch and LaTeX so it can run in a
clean CI job before a manuscript is built.  It does not certify scientific
validity; it prevents known draft-only artefacts (placeholders, legacy results
without configuration provenance, incomplete statistical reports, and broken
PDFs) from silently entering a submission archive.

Examples
--------
    python scripts/audit_submission.py
    python scripts/audit_submission.py --json submission_audit.json
    python scripts/audit_submission.py --allow-legacy-results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "neurocomputing" / "DD-Mamba_Neurocomputing_condensed_revision(1)"
RESULTS = ROOT / "results"
PROVENANCE_RESULTS = RESULTS / "provenance"


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str | None = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _nested(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur: Any = mapping
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                break
            cur = cur[key]
        else:
            return cur
    return None


def _provenance(record: dict[str, Any]) -> dict[str, Any] | None:
    # Current runners store provenance fields at the top level so legacy
    # consumers can still read ``config``/``results`` without unwrapping a
    # metadata object.
    if (
        isinstance(record.get("schema_version"), str)
        and isinstance(record.get("resolved_config"), dict)
    ):
        return record
    value = _nested(
        record,
        ("provenance",),
        ("run_provenance",),
        ("metadata", "provenance"),
        ("_meta", "provenance"),
    )
    return value if isinstance(value, dict) else None


def _resolved_config(provenance: dict[str, Any]) -> Any:
    return _nested(
        provenance,
        ("resolved_config",),
        ("config", "resolved"),
        ("configuration", "resolved"),
    )


def _config_hash(provenance: dict[str, Any]) -> Any:
    return _nested(
        provenance,
        ("config_sha256",),
        ("config", "sha256"),
        ("configuration", "sha256"),
    )


def _seed_values(record: dict[str, Any], provenance: dict[str, Any]) -> Any:
    return _nested(
        record,
        ("seed_values",),
        ("metadata", "seed_values"),
    ) or _nested(provenance, ("seed_values",), ("seeds", "values"))


def check_required_files() -> list[Finding]:
    findings: list[Finding] = []
    required = (
        PAPER / "main.tex",
        PAPER / "supplementary_material.tex",
        PAPER / "refs.bib",
        PAPER / "highlights.tex",
        PAPER / "main.pdf",
        PAPER / "supplementary_material.pdf",
    )
    for path in required:
        if not path.exists():
            findings.append(Finding("error", "missing-file", "Required submission file is missing.", str(path)))
    return findings


def check_manuscript() -> list[Finding]:
    path = PAPER / "main.tex"
    if not path.exists():
        return []
    text = _read_text(path)
    findings: list[Finding] = []
    placeholder_patterns = {
        "placeholder-author": r"First Author|Author Name|example\.edu|TODO \(submission\).*author",
        "placeholder-repository": r"repository URL to be inserted|\[repository URL",
        "placeholder-credit": r"TODO \(submission\).*CRediT|Author 1:|Author 2:|replace this prompt with verified\s+per-author CRediT",
        "placeholder-conflict": r"To be completed and approved by all authors before submission",
        "placeholder-funding": r"TODO \(submission\).*funding|To be completed before submission: list every funder",
        "placeholder-acknowledgements": r"To be completed before submission, or removed if there are no",
    }
    for code, pattern in placeholder_patterns.items():
        if re.search(pattern, text, flags=re.I):
            findings.append(Finding("error", code, "Submission placeholder remains in main.tex.", str(path)))

    required_sections = {
        "missing-credit": ("CRediT authorship contribution statement",),
        "missing-competing-interest": ("Declaration of competing interest", "Conflict of Interest"),
        "missing-data-availability": ("Data availability",),
        "missing-ai-declaration": ("Declaration of generative AI",),
        "missing-funding": ("Funding",),
        "missing-ethics": ("Ethics",),
    }
    for code, markers in required_sections.items():
        if not any(marker.lower() in text.lower() for marker in markers):
            findings.append(Finding("error", code, f"Required section is absent: {markers[0]}.", str(path)))

    supplement = PAPER / "supplementary_material.tex"
    if supplement.exists() and re.search(r"search ranges? (?:are|is) (?:provided|reported)", text, flags=re.I):
        supp_text = _read_text(supplement)
        if not re.search(r"search (?:space|range)|trial budget|selection metric", supp_text, flags=re.I):
            findings.append(Finding(
                "error",
                "unsupported-search-range-claim",
                "main.tex promises search ranges, but the supplement contains no search-space specification.",
                str(supplement),
            ))
    return findings


def _plain_highlight(item: str) -> str:
    item = re.sub(r"%.*", "", item)
    item = re.sub(r"\\(?:textbf|emph|textit)\{([^{}]*)\}", r"\1", item)
    item = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", "", item)
    item = item.replace("{", "").replace("}", "")
    return " ".join(item.split())


def check_highlights() -> list[Finding]:
    path = PAPER / "highlights.tex"
    if not path.exists():
        return []
    text = _read_text(path)
    items = [_plain_highlight(x) for x in re.findall(r"\\item\s+([^\r\n]+)", text)]
    findings: list[Finding] = []
    if not 3 <= len(items) <= 5:
        findings.append(Finding("error", "highlight-count", f"Expected 3-5 highlights, found {len(items)}.", str(path)))
    for index, item in enumerate(items, 1):
        if len(item) > 85:
            findings.append(Finding("error", "highlight-length", f"Highlight {index} has {len(item)} characters (limit 85).", str(path)))
        if re.search(r"\bPCC\b", item):
            findings.append(Finding("warning", "highlight-abbreviation", f"Highlight {index} uses unexplained abbreviation PCC.", str(path)))
    return findings


def check_pdf(path: Path) -> list[Finding]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if len(data) < 1024 or not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        return [Finding("error", "invalid-pdf", "PDF is truncated or structurally invalid.", str(path))]
    return []


def check_artifact_freshness(target: Path, dependencies: Iterable[Path]) -> list[Finding]:
    """Reject a built artifact older than any existing declared source."""
    if not target.exists():
        return []
    stale_dependencies = [
        dependency for dependency in dependencies
        if dependency.exists() and dependency.stat().st_mtime_ns > target.stat().st_mtime_ns
    ]
    if not stale_dependencies:
        return []
    names = ", ".join(dependency.name for dependency in stale_dependencies)
    return [Finding(
        "error",
        "stale-built-artifact",
        f"Built artifact predates source/dependency: {names}.",
        str(target),
    )]


def _iter_submission_results() -> Iterable[Path]:
    """Prefer isolated provenance-enabled outputs over legacy root files."""
    provenance_files = sorted(PROVENANCE_RESULTS.glob("*.json"))
    if provenance_files:
        yield from provenance_files
        return
    for path in sorted(RESULTS.glob("unified_*.json")):
        if path.name not in {"unified_analysis.json", "unified_audit.json"}:
            yield path


def check_result_provenance(allow_legacy: bool) -> list[Finding]:
    findings: list[Finding] = []
    paths = list(_iter_submission_results())
    if not paths:
        return [Finding("error", "missing-results", "No result artifacts were found.", str(RESULTS))]
    for path in paths:
        try:
            record = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding("error", "invalid-result-json", str(exc), str(path)))
            continue
        provenance = _provenance(record)
        severity = "warning" if allow_legacy else "error"
        if provenance is None:
            findings.append(Finding(severity, "legacy-result", "Result has no immutable run provenance; it must not drive a submission table.", str(path)))
            continue
        if _resolved_config(provenance) is None:
            findings.append(Finding("error", "missing-resolved-config", "Provenance does not embed the resolved runtime configuration.", str(path)))
        config_hash = _config_hash(provenance)
        if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", config_hash):
            findings.append(Finding("error", "missing-config-hash", "Provenance requires a SHA-256 hash of the resolved configuration.", str(path)))
        seeds = _seed_values(record, provenance)
        if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
            findings.append(Finding("error", "missing-seed-values", "Explicit integer seed_values are required; a seed count is insufficient.", str(path)))
            continue
        if len(set(seeds)) != len(seeds):
            findings.append(Finding("error", "duplicate-seed-values", "seed_values contains duplicates.", str(path)))

        architectures = record.get("architectures")
        results = record.get("results")
        horizons = record.get("horizons")
        if isinstance(architectures, list) and isinstance(results, dict) and isinstance(horizons, list):
            failures = record.get("failures", {})
            if failures:
                findings.append(Finding("error", "failed-unified-runs", f"Unified sweep records failures: {sorted(failures)}.", str(path)))
            missing_architectures = [name for name in architectures if name not in results]
            if missing_architectures:
                findings.append(Finding("error", "missing-unified-architectures", "Unified results omit requested architectures: " + ", ".join(missing_architectures), str(path)))
            for architecture in architectures:
                per_horizon = results.get(architecture)
                if not isinstance(per_horizon, dict):
                    continue
                for horizon in horizons:
                    cell = per_horizon.get(str(horizon))
                    runs = cell.get("runs") if isinstance(cell, dict) else None
                    if not isinstance(runs, list):
                        findings.append(Finding("error", "missing-unified-cell", f"Missing {architecture} H={horizon} run records.", str(path)))
                        continue
                    run_seeds = [run.get("seed") for run in runs if isinstance(run, dict)]
                    if (
                        len(runs) != len(seeds)
                        or not all(isinstance(seed, int) for seed in run_seeds)
                        or sorted(run_seeds) != sorted(seeds)
                    ):
                        findings.append(Finding("error", "incomplete-unified-seeds", f"{architecture} H={horizon} does not contain exactly the declared seed set.", str(path)))
    return findings


def check_baseline_selection_sources() -> list[Finding]:
    """Require frozen validation selections for provenance-enabled baselines."""
    findings: list[Finding] = []
    if not PROVENANCE_RESULTS.exists():
        return findings
    for path in sorted(PROVENANCE_RESULTS.glob("unified_*.json")):
        try:
            record = json.loads(_read_text(path))
        except json.JSONDecodeError:
            continue
        architectures = record.get("architectures", [])
        baselines = [name for name in architectures if name != "dual_domain"] if isinstance(architectures, list) else []
        if not baselines:
            continue
        source = record.get("baseline_selection_source")
        if not isinstance(source, dict):
            findings.append(Finding("error", "missing-baseline-selection", "Matched baselines lack a frozen validation-selection source.", str(path)))
            continue
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.exists():
            findings.append(Finding("error", "missing-baseline-selection-file", "Frozen baseline-selection report cannot be found.", str(source_path)))
            continue
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != source.get("sha256"):
            findings.append(Finding("error", "changed-baseline-selection", "Baseline-selection report changed after final evaluation.", str(source_path)))
        try:
            selection = json.loads(_read_text(source_path))
        except json.JSONDecodeError as exc:
            findings.append(Finding("error", "invalid-baseline-selection", str(exc), str(source_path)))
            continue
        if selection.get("selection_scope") != "validation_only_no_test_loader":
            findings.append(Finding("error", "test-informed-baseline-selection", "Baseline search was not validation-only.", str(source_path)))
        if not selection.get("equal_budget_enforced"):
            findings.append(Finding("error", "unequal-baseline-budget", "Equal validation trial budgets were not enforced.", str(source_path)))
        record_provenance = _provenance(record)
        if record_provenance is not None and selection.get("config_sha256") != _config_hash(record_provenance):
            findings.append(Finding("error", "mismatched-baseline-config", "Frozen baseline search used a different resolved dataset configuration.", str(source_path)))
        frozen = selection.get("frozen_architectures")
        candidates = selection.get("candidate_audit")
        declared_budget = selection.get("trial_budget_per_architecture")
        if not isinstance(frozen, dict) or not isinstance(candidates, dict) or not isinstance(declared_budget, int):
            findings.append(Finding("error", "incomplete-baseline-selection", "Frozen selection lacks candidates, winners, or a declared trial budget.", str(source_path)))
            continue
        for architecture in baselines:
            winner = frozen.get(architecture)
            trials = candidates.get(architecture)
            if not isinstance(winner, dict) or not isinstance(trials, list):
                findings.append(Finding("error", "missing-baseline-candidates", f"No complete search record for {architecture}.", str(source_path)))
                continue
            if winner.get("trial_budget") != declared_budget or len(trials) != declared_budget:
                findings.append(Finding("error", "baseline-budget-mismatch", f"{architecture} does not have exactly {declared_budget} candidate trials.", str(source_path)))
            for trial in trials:
                if not isinstance(trial, dict) or "test_metrics" in trial or "test_mse" in trial or "test_mae" in trial:
                    findings.append(Finding("error", "test-metric-in-baseline-search", f"{architecture} candidate audit contains test output.", str(source_path)))
                    break
    return findings


def check_statistical_outputs() -> list[Finding]:
    findings: list[Finding] = []
    family_registry: dict[str, int] = {}
    preferred = ROOT / "generated" / "provenance_audit" / "complete_statistical_audit.json"
    legacy = ROOT / "generated" / "statistical_audit" / "complete_statistical_audit.json"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        return [Finding(
            "error", "missing-statistics",
            "Missing complete statistical audit; run scripts/generate_statistical_audit.py.",
            str(path),
        )]
    try:
        record = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        return [Finding("error", "invalid-statistics-json", str(exc), str(path))]

    for section_name in ("unified_baseline_audit", "component_branch_audit"):
        section = record.get(section_name)
        if not isinstance(section, dict):
            findings.append(Finding("error", "missing-statistics-section", f"Missing audit section: {section_name}.", str(path)))
            continue
        families = section.get("families")
        rows = section.get("rows")
        if not isinstance(families, list) or not families:
            findings.append(Finding("error", "missing-family-definition", f"{section_name} has no family registry.", str(path)))
            continue
        if not isinstance(rows, list) or not rows:
            findings.append(Finding("error", "missing-statistics-rows", f"{section_name} has no comparison rows.", str(path)))
            continue
        policy = section.get("family_policy", {})
        if section_name == "component_branch_audit" and isinstance(policy, dict):
            for label in ("branch", "component"):
                expected = policy.get(f"expected_{label}_family_size")
                observed = policy.get(f"observed_{label}_family_size")
                if not isinstance(expected, int) or not isinstance(observed, int) or expected != observed:
                    findings.append(Finding(
                        "error",
                        "incomplete-statistical-family",
                        f"{label} family is incomplete: observed={observed!r}, expected={expected!r}.",
                        str(path),
                    ))
        known_members: set[str] = set()
        for family in families:
            members = family.get("members") if isinstance(family, dict) else None
            size = family.get("size") if isinstance(family, dict) else None
            if not isinstance(members, list) or size != len(members):
                findings.append(Finding("error", "invalid-family-size", f"Family size/member mismatch in {section_name}.", str(path)))
            else:
                known_members.update(str(member) for member in members)
                if isinstance(family.get("id"), str):
                    family_registry[family["id"]] = size
        for row in rows:
            if not isinstance(row, dict):
                findings.append(Finding("error", "invalid-statistics-row", f"Non-object row in {section_name}.", str(path)))
                continue
            statistics = row.get("statistics")
            required = ("n", "mean_paired_difference", "ci95", "t_statistic", "df", "p_value_raw", "exact_sign_flip_p_two_sided")
            if not isinstance(statistics, dict) or any(field not in statistics for field in required):
                findings.append(Finding("error", "incomplete-statistics", f"Comparison {row.get('id')} lacks required paired statistics.", str(path)))
            if not isinstance(row.get("observations"), list):
                findings.append(Finding("error", "missing-observations", f"Comparison {row.get('id')} lacks run-level observations.", str(path)))
            if row.get("id") not in known_members:
                findings.append(Finding("error", "unregistered-comparison", f"Comparison {row.get('id')} is absent from every declared family.", str(path)))

    manuscript_path = PAPER / "main.tex"
    if manuscript_path.exists():
        manuscript = _read_text(manuscript_path)
        declared_patterns = {
            "unified.average.full": r"BH family of\s+(\d+)\s+average-MSE contrasts",
            "component_placement.all": r"(\d+)\s+placement tests",
            "branch_removal.all": r"(\d+)\s+branch-removal tests",
        }
        for family_id, pattern in declared_patterns.items():
            match = re.search(pattern, manuscript, flags=re.I)
            expected = family_registry.get(family_id)
            if match and isinstance(expected, int) and int(match.group(1)) != expected:
                findings.append(Finding(
                    "error",
                    "manuscript-family-size-mismatch",
                    f"main.tex declares {match.group(1)} tests for {family_id}, but the serialized audit has {expected}.",
                    str(manuscript_path),
                ))
    return findings


def run_audit(allow_legacy: bool) -> list[Finding]:
    findings = check_required_files()
    findings.extend(check_manuscript())
    findings.extend(check_highlights())
    for name in ("main.pdf", "supplementary_material.pdf", "highlights.pdf"):
        findings.extend(check_pdf(PAPER / name))
    statistical_json = ROOT / "generated" / "provenance_audit" / "complete_statistical_audit.json"
    if not statistical_json.exists():
        statistical_json = ROOT / "generated" / "statistical_audit" / "complete_statistical_audit.json"
    findings.extend(check_artifact_freshness(
        PAPER / "fig_branch_matrix.pdf",
        (PAPER / "draw_smamba_style_branch_matrix.py", statistical_json),
    ))
    findings.extend(check_artifact_freshness(
        PAPER / "main.pdf",
        (
            PAPER / "main.tex",
            PAPER / "refs.bib",
            PAPER / "fig_architecture.pdf",
            PAPER / "fig_branch_matrix.pdf",
        ),
    ))
    findings.extend(check_artifact_freshness(
        PAPER / "supplementary_material.pdf",
        (
            PAPER / "supplementary_material.tex",
            PAPER / "fig_selective_blocks.pdf",
            PAPER / "fig_pcc_mixer.pdf",
            PAPER / "fig_dispersion.pdf",
        ),
    ))
    findings.extend(check_artifact_freshness(
        PAPER / "highlights.pdf", (PAPER / "highlights.tex",)
    ))
    findings.extend(check_result_provenance(allow_legacy=allow_legacy))
    findings.extend(check_baseline_selection_sources())
    findings.extend(check_statistical_outputs())
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-legacy-results", action="store_true", help="Downgrade missing result provenance to a warning for draft-only builds.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the audit report as JSON.")
    args = parser.parse_args()
    findings = run_audit(allow_legacy=args.allow_legacy_results)
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    report = {
        "schema_version": 1,
        "ready_for_submission": not errors,
        "counts": {"errors": len(errors), "warnings": len(warnings)},
        "findings": [asdict(finding) for finding in findings],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for finding in findings:
        where = f" [{finding.path}]" if finding.path else ""
        print(f"{finding.severity.upper():7s} {finding.code}: {finding.message}{where}")
    print(f"\nsubmission-ready={str(not errors).lower()} errors={len(errors)} warnings={len(warnings)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
