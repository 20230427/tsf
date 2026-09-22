"""Standard-library-only tests for the statistical audit generators."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_unified_baselines import (
    build_unified_audit,
    render_unified_markdown,
    write_unified_bundle,
)
from scripts.compute_stats_correction import build_component_audit
from scripts.statistical_audit import (
    benjamini_hochberg,
    exact_sign_flip_p,
    pair_run_records,
    paired_statistics,
    student_t_critical_975,
)


def _run(mse, seed_marker="missing", seed_value=None):
    run = {"mse": mse, "mae": mse}
    if seed_marker != "missing":
        run[seed_marker] = seed_value
    return run


def _write_toy_unified(results_dir: Path) -> None:
    payload = {
        "config": "configs/toy.yaml",
        "seeds": 2,
        "horizons": [1, 2],
        "results": {
            "dual_domain": {
                "1": {"runs": [_run(1.0), _run(1.1)]},
                "2": {"runs": [_run(2.0), _run(2.2)]},
            },
            "smamba": {
                "1": {"runs": [_run(1.2), _run(1.3)]},
                "2": {"runs": [_run(2.3), _run(2.5)]},
            },
            "extra": {
                "1": {"runs": [_run(0.9), _run(1.0)]},
                "2": {"runs": [_run(1.9), _run(2.0)]},
            },
        },
    }
    (results_dir / "unified_toy.json").write_text(json.dumps(payload), encoding="utf-8")


class StatisticalAuditTests(unittest.TestCase):
    def assertClose(self, actual, expected, tolerance=1e-10):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            msg=f"{actual!r} != {expected!r}",
        )

    def test_benjamini_hochberg_is_monotone_in_original_order(self):
        q_values = benjamini_hochberg([0.01, 0.04, 0.03])
        for actual, expected in zip(q_values, [0.03, 0.04, 0.04]):
            self.assertClose(actual, expected)

    def test_complete_paired_statistics_and_exact_sign_flip(self):
        stats = paired_statistics([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertEqual(stats["n"], 3)
        self.assertClose(stats["mean_paired_difference"], 2.0)
        self.assertClose(stats["sd_paired_difference"], 1.0)
        self.assertClose(stats["t_statistic"], 2.0 * math.sqrt(3.0))
        self.assertEqual(stats["df"], 2)
        self.assertClose(stats["exact_sign_flip_p_two_sided"], 0.25)
        self.assertClose(stats["relative_effect_percent"], 100.0)
        self.assertClose(stats["standardized_effect_dz"], 2.0)
        self.assertClose(student_t_critical_975(2), 4.30265273, tolerance=1e-7)
        self.assertClose(exact_sign_flip_p([1.0, 2.0, 3.0]), 0.25)

    def test_legacy_pairing_never_fabricates_seed_ids(self):
        observations, pairing = pair_run_records(
            [_run(1.0), _run(2.0)], [_run(1.1), _run(2.2)]
        )
        self.assertEqual(pairing["basis"], "legacy_positional_order")
        self.assertFalse(pairing["seed_ids_available"])
        self.assertEqual([item["pair_index"] for item in observations], [1, 2])
        self.assertEqual([item["seed_id"] for item in observations], [None, None])

    def test_explicit_seed_pairing_matches_ids_not_positions(self):
        reference = [_run(1.0, "seed", 11), _run(2.0, "seed", 12)]
        comparator = [_run(2.4, "seed", 12), _run(1.3, "seed", 11)]
        observations, pairing = pair_run_records(reference, comparator)
        self.assertEqual(pairing["basis"], "explicit_seed_id")
        self.assertEqual([item["seed_id"] for item in observations], [11, 12])
        for actual, expected in zip(
            [item["comparator"] for item in observations], [1.3, 2.4]
        ):
            self.assertClose(actual, expected)

    def test_partial_seed_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "partially recorded"):
            pair_run_records(
                [_run(1.0, "seed", 1), _run(2.0)], [_run(1.1), _run(2.2)]
            )

    def test_unified_audit_declares_displayed_and_full_families(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            _write_toy_unified(results_dir)
            audit = build_unified_audit(results_dir, displayed_baselines=["smamba"])
        families = {family["id"]: family for family in audit["families"]}
        self.assertEqual(families["unified.average.displayed"]["size"], 1)
        self.assertEqual(families["unified.average.full"]["size"], 2)
        self.assertEqual(families["unified.per_horizon.displayed"]["size"], 2)
        self.assertEqual(families["unified.per_horizon.full"]["size"], 4)
        self.assertEqual(
            audit["family_policy"]["displayed_baselines_predeclared"], ["smamba"]
        )
        self.assertTrue(
            all(
                observation["seed_id"] is None
                for row in audit["rows"]
                for observation in row["observations"]
            )
        )
        average_smamba = next(
            row
            for row in audit["rows"]
            if row["level"] == "average_over_horizons"
            and row["comparator_model"] == "smamba"
        )
        self.assertClose(average_smamba["statistics"]["mean_paired_difference"], 0.25)
        self.assertIn("unified.average.displayed", average_smamba["multiplicity"])
        self.assertIn("unified.average.full", average_smamba["multiplicity"])

    def test_human_tables_are_rendered_from_serialized_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_toy_unified(root)
            audit = build_unified_audit(root, displayed_baselines=["smamba"])
            paths = write_unified_bundle(audit, root / "generated", "toy")
            serialized = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown = paths["markdown"].read_text(encoding="utf-8")
            latex = paths["latex"].read_text(encoding="utf-8")
        self.assertEqual(markdown, render_unified_markdown(serialized).rstrip() + "\n")
        self.assertIn("not recorded", markdown)
        self.assertIn("\\begin{longtable}", latex)

    def test_component_audit_records_explicit_family_members(self):
        branch = {
            "seeds": 3,
            "results": {
                "Toy@1": {
                    "full": [1.0, 1.1, 0.9],
                    "time_only": [1.2, 1.3, 1.1],
                    "freq_only": [1.0, 1.0, 1.0],
                }
            },
        }
        component = {
            "seeds": 3,
            "results": {
                "full": {"runs": [_run(1.0), _run(1.1), _run(0.9)]},
                "variant": {"runs": [_run(1.2), _run(1.3), _run(1.1)]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            (results_dir / "branch.json").write_text(json.dumps(branch), encoding="utf-8")
            (results_dir / "component.json").write_text(
                json.dumps(component), encoding="utf-8"
            )
            audit = build_component_audit(
                results_dir,
                branch_filename="branch.json",
                component_sources=[("component.json", "Toy")],
            )
        families = {family["id"]: family for family in audit["families"]}
        self.assertEqual(families["branch_removal.all"]["size"], 2)
        self.assertEqual(families["component_placement.all"]["size"], 1)
        self.assertEqual(len(families["branch_removal.all"]["members"]), 2)
        self.assertTrue(
            all(
                observation["seed_id"] is None
                for row in audit["rows"]
                for observation in row["observations"]
            )
        )

    def test_branch_audit_uses_top_level_seed_records_when_available(self):
        branch = {
            "seeds": 2,
            "results": {
                "Toy@1": {
                    "full": [1.0, 2.0],
                    "time_only": [1.3, 2.4],
                    "freq_only": [1.1, 2.2],
                }
            },
            # Deliberately reverse comparator order: pairing must follow seed,
            # not the positional arrays retained for backward compatibility.
            "run_records": [
                {"dataset": "Toy", "horizon": 1, "variant": "full", "seed": 11, "mse": 1.0},
                {"dataset": "Toy", "horizon": 1, "variant": "full", "seed": 12, "mse": 2.0},
                {"dataset": "Toy", "horizon": 1, "variant": "time_only", "seed": 12, "mse": 2.4},
                {"dataset": "Toy", "horizon": 1, "variant": "time_only", "seed": 11, "mse": 1.3},
                {"dataset": "Toy", "horizon": 1, "variant": "freq_only", "seed": 12, "mse": 2.2},
                {"dataset": "Toy", "horizon": 1, "variant": "freq_only", "seed": 11, "mse": 1.1},
            ],
        }
        component = {
            "seeds": 2,
            "results": {
                "full": {"runs": [_run(1.0, "seed", 11), _run(2.0, "seed", 12)]},
                "variant": {"runs": [_run(1.2, "seed", 11), _run(2.3, "seed", 12)]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            (results_dir / "branch.json").write_text(json.dumps(branch), encoding="utf-8")
            (results_dir / "component.json").write_text(json.dumps(component), encoding="utf-8")
            audit = build_component_audit(
                results_dir,
                branch_filename="branch.json",
                component_sources=[("component.json", "Toy")],
            )
        branch_rows = [row for row in audit["rows"] if row["analysis"] == "branch_removal"]
        self.assertEqual(len(branch_rows), 2)
        self.assertTrue(all(row["pairing"]["basis"] == "explicit_seed_id" for row in branch_rows))
        self.assertEqual(
            [item["seed_id"] for item in branch_rows[0]["observations"]],
            [11, 12],
        )
        self.assertEqual(audit["family_policy"]["expected_branch_family_size"], 2)


if __name__ == "__main__":
    unittest.main()
