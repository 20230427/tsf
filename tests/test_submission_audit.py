"""Standard-library tests for the fail-closed submission gate."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_submission


class SubmissionAuditTests(unittest.TestCase):
    def test_stale_built_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "main.pdf"
            source = root / "main.tex"
            target.write_bytes(b"%PDF-1.4\n%%EOF\n")
            source.write_text("revised", encoding="utf-8")
            os.utime(target, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))
            findings = audit_submission.check_artifact_freshness(target, [source])
        self.assertEqual([item.code for item in findings], ["stale-built-artifact"])

    def test_conflict_of_interest_title_satisfies_required_section(self):
        manuscript = r"""
        \section*{CRediT authorship contribution statement}
        Verified contribution roles.
        \section*{Conflict of Interest}
        Verified declaration.
        \section*{Ethics declaration}
        Not applicable.
        \section*{Funding}
        Verified declaration.
        \section*{Data availability}
        Archive details.
        \section*{Declaration of generative AI}
        Verified declaration.
        """
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            (paper / "main.tex").write_text(manuscript, encoding="utf-8")
            (paper / "supplementary_material.tex").write_text("trial budget", encoding="utf-8")
            with patch.object(audit_submission, "PAPER", paper):
                findings = audit_submission.check_manuscript()
        self.assertFalse(
            any(item.code == "missing-competing-interest" for item in findings)
        )
        self.assertEqual(findings, [])

    def test_frozen_selection_hash_is_verified_and_tampering_is_detected(self):
        selection = {
            "selection_scope": "validation_only_no_test_loader",
            "equal_budget_enforced": True,
            "trial_budget_per_architecture": 1,
            "frozen_architectures": {
                "smamba": {"trial_budget": 1, "overrides": {"train.lr": 0.001}}
            },
            "candidate_audit": {
                "smamba": [{"candidate": {"train.lr": 0.001}, "val_loss": 0.2}]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            selection_path = result_dir / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            digest = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            unified = {
                "architectures": ["dual_domain", "smamba"],
                "baseline_selection_source": {
                    "path": str(selection_path),
                    "sha256": digest,
                },
            }
            (result_dir / "unified_toy.json").write_text(
                json.dumps(unified), encoding="utf-8"
            )
            with patch.object(audit_submission, "PROVENANCE_RESULTS", result_dir):
                self.assertEqual(
                    audit_submission.check_baseline_selection_sources(), []
                )
                selection_path.write_text(
                    json.dumps({**selection, "changed": True}), encoding="utf-8"
                )
                findings = audit_submission.check_baseline_selection_sources()
        self.assertTrue(
            any(item.code == "changed-baseline-selection" for item in findings)
        )

    def test_incomplete_unified_seed_grid_is_rejected(self):
        resolved = {"experiment": {"name": "toy"}}
        config_hash = hashlib.sha256(
            json.dumps(
                resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        record = {
            "schema_version": "ddmamba-experiment-v1",
            "resolved_config": resolved,
            "config_sha256": config_hash,
            "seed_values": [11, 12],
            "architectures": ["dual_domain", "smamba"],
            "horizons": [96],
            "failures": {"smamba": "out of memory"},
            "results": {
                "dual_domain": {"96": {"runs": [{"seed": 11}, {"seed": 12}]}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            (result_dir / "unified_toy.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with patch.object(audit_submission, "PROVENANCE_RESULTS", result_dir):
                findings = audit_submission.check_result_provenance(False)
        codes = {item.code for item in findings}
        self.assertIn("failed-unified-runs", codes)
        self.assertIn("missing-unified-architectures", codes)


if __name__ == "__main__":
    unittest.main()
