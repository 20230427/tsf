"""Tests for validation-only config application."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import apply_selection


def _report(*, include_test_oracle: bool = False) -> dict:
    models = {
        "validation_selected": {
            "candidate": {"model.use_revin": False},
        }
    }
    if include_test_oracle:
        models["test_oracle"] = {"candidate": {"model.use_revin": True}}
    return {
        "config": "configs/toy.yaml",
        "selection": {
            "test_metrics_used": False,
            "selected_candidate": {"model.use_revin": False},
        },
        "protocol": {
            "candidate_test_loader_constructed": False,
            "candidate_test_metrics_emitted": False,
            "final_test_phase": "after_configuration_freeze",
        },
        "models": models,
    }


class ApplySelectionTests(unittest.TestCase):
    def test_valid_report_updates_only_the_frozen_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs" / "toy.yaml"
            config.parent.mkdir()
            config.write_text("model:\n  use_revin: true  # old\n", encoding="utf-8")
            report = root / "selection.json"
            report.write_text(json.dumps(_report()), encoding="utf-8")
            with patch.object(apply_selection, "ROOT", root):
                apply_selection.apply_one(report, dry_run=False)
            updated = config.read_text(encoding="utf-8")
        self.assertIn("use_revin: false", updated)
        self.assertIn("# old", updated)

    def test_test_oracle_report_is_rejected_without_modifying_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "configs" / "toy.yaml"
            config.parent.mkdir()
            original = "model:\n  use_revin: true\n"
            config.write_text(original, encoding="utf-8")
            report = root / "selection.json"
            report.write_text(
                json.dumps(_report(include_test_oracle=True)), encoding="utf-8"
            )
            with patch.object(apply_selection, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "test-oracle"):
                    apply_selection.apply_one(report, dry_run=False)
            self.assertEqual(config.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
