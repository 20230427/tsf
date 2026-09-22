"""CPU/standard-library tests for ZIP result auditing, not model training."""
import copy
import unittest
from unittest.mock import MagicMock

from scripts.analyze_lookback_results import (
    DEFAULT_LENGTHS, audit_records, canonical_hash, describe, style_axis,
)


class LookbackAuditTests(unittest.TestCase):
    def make_record(self, length=24):
        cfg = {
            "experiment": {"name": f"electricity_L{length}", "seed": 2024,
                           "checkpoint_dir": "checkpoints/test"},
            "data": {"seq_len": length, "pred_len": 96,
                     "csv_path": "data/electricity.csv"},
            "model": {"d_model": 128}, "train": {"epochs": 10},
        }
        payload = {
            "schema_version": "ddmamba-experiment-v1", "resolved_config": cfg,
            "config_sha256": canonical_hash(cfg), "seed_values": [2024],
            "seeds": 1, "horizons": [96], "architectures": ["dual_domain"],
            "architecture_overrides": {}, "results": {}, "failures": {},
            "run_records": [], "environment": {}, "git": {"commit": None},
        }
        run_cfg = copy.deepcopy(cfg)
        run_cfg["model"]["arch"] = "dual_domain"
        run_cfg["experiment"]["name"] += "_dual_domain_h96_s0"
        run = {"seed": 2024, "arch": "dual_domain", "horizon": 96,
               "mse": 0.15, "mae": 0.25, "best_epoch": 2,
               "param_count": 100, "run_config_sha256": canonical_hash(run_cfg)}
        # Deliberately wrong cached mean: audit recomputes from the raw run.
        payload["results"]["dual_domain"] = {"96": {"runs": [run], "mse": [999, 0]}}
        payload["run_records"] = [copy.deepcopy(run)]
        return (f"electricity_L{length}.json", payload, "fixture_digest")

    def audit(self, records, lengths=(24,)):
        return audit_records(records, list(lengths), ["dual_domain"], [2024], 96, "electricity")

    def test_complete_recomputes_metrics_and_single_seed_has_no_std(self):
        summary = self.audit([self.make_record()])
        self.assertTrue(summary["complete"], summary["issues"])
        cell = summary["cells"]["24"]["dual_domain"]["mse"]
        self.assertEqual(cell["mean"], 0.15)
        self.assertIsNone(cell["sample_std"])

    def test_missing_length_fails_closed(self):
        self.assertFalse(self.audit([self.make_record()], (24, 48))["complete"])

    def test_duplicate_runs_are_rejected(self):
        record = self.make_record()
        self.assertFalse(self.audit([record, copy.deepcopy(record)])["complete"])

    def test_config_drift_is_rejected(self):
        other = self.make_record(48)
        other[1]["resolved_config"]["train"]["epochs"] = 20
        other[1]["config_sha256"] = canonical_hash(other[1]["resolved_config"])
        summary = self.audit([self.make_record(), other], (24, 48))
        self.assertEqual(summary["fixed_config_variants"], 2)
        self.assertFalse(summary["complete"])

    def test_bad_config_hash_is_rejected(self):
        record = self.make_record()
        record[1]["config_sha256"] = "invalid"
        self.assertFalse(self.audit([record])["complete"])

    def test_nonfinite_metric_is_rejected(self):
        record = self.make_record()
        run = record[1]["results"]["dual_domain"]["96"]["runs"][0]
        run["mse"] = float("nan")
        record[1]["run_records"] = [copy.deepcopy(run)]
        self.assertFalse(self.audit([record])["complete"])

    def test_sample_std_uses_n_minus_one(self):
        self.assertAlmostEqual(describe([1.0, 2.0, 3.0])["sample_std"], 1.0)

    def test_negative_metric_is_rejected(self):
        record = self.make_record()
        run = record[1]["results"]["dual_domain"]["96"]["runs"][0]
        run["mse"] = -0.1
        record[1]["run_records"] = [copy.deepcopy(run)]
        self.assertFalse(self.audit([record])["complete"])

    def test_bad_run_config_hash_is_rejected(self):
        record = self.make_record()
        run = record[1]["results"]["dual_domain"]["96"]["runs"][0]
        run["run_config_sha256"] = "invalid"
        record[1]["run_records"] = [copy.deepcopy(run)]
        self.assertFalse(self.audit([record])["complete"])

    def test_axis_displays_every_lookback_tick(self):
        ax = MagicMock()
        style_axis(ax, DEFAULT_LENGTHS)
        ax.set_xticks.assert_called_once_with(
            DEFAULT_LENGTHS, labels=[str(length) for length in DEFAULT_LENGTHS])
        ax.tick_params.assert_called_once_with(
            axis="x", labelrotation=90, labelsize=8.5)

    def test_detail_axis_displays_every_tick_in_its_range(self):
        ax = MagicMock()
        lengths = [length for length in DEFAULT_LENGTHS if length >= 144]
        style_axis(ax, lengths)
        self.assertEqual(len(lengths), 9)
        ax.set_xticks.assert_called_once_with(
            lengths, labels=[str(length) for length in lengths])


if __name__ == "__main__":
    unittest.main()
