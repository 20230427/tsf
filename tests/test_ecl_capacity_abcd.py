"""Measurement-integrity tests; fixtures are not experimental results."""
import copy
import unittest

from scripts.plot_ecl_capacity_abcd import distribution, from_capacity


class ECLCapacityFigureTests(unittest.TestCase):
    def payload(self):
        p = {"dataset": "electricity", "channels": 321, "prediction_length": 96,
             "evaluation_scope": "validation_only", "anchor_lookback": 96,
             "base_d_model": 512, "base_learning_rate": 0.0005,
             "lookbacks": [24, 96], "d_models": [64, 128, 256, 512],
             "learning_rate_multipliers": [0.5, 1, 1.5], "matched_seed_count": 1}
        keys = {(v, 512, 0.0005) for v in p["lookbacks"]}
        keys.update((96, v, 0.0005) for v in p["d_models"])
        keys.update((96, 512, float(f"{0.0005 * v:.15g}"))
                    for v in p["learning_rate_multipliers"])
        records = [{"dataset": "electricity", "prediction_length": 96,
                    "phase": "validation_only", "seed": 2024,
                    "lookback": l, "d_model": width, "learning_rate": lr,
                    "freq_hidden": 512, "param_count": width * 100 + l,
                    "run_config_sha256": "fixture_hash", "mse": 0.15, "mae": 0.25,
                    "training_ms_per_iteration": 10.0,
                    "peak_allocated_gpu_memory_mb": 1024.0}
                   for l, width, lr in sorted(keys)]
        return {"schema": "ddmamba-capacity-staircase-v1", "protocol": p,
                "failures": [], "run_records": records}

    def test_complete_single_seed_has_no_error_bars(self):
        data = from_capacity(self.payload())
        self.assertTrue(data["complete"])
        self.assertEqual(data["missing_panels"], [])
        self.assertEqual(data["lookback"][0]["memory_gib"]["mean"], 1.0)
        self.assertIsNone(data["width"][0]["mse"]["sample_std"])

    def test_weather_is_rejected(self):
        p = self.payload()
        p["protocol"]["dataset"] = "weather"
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_test_metrics_are_not_used_as_validation_metrics(self):
        p = self.payload()
        p["protocol"]["evaluation_scope"] = "validation_and_test"
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_incomplete_width_sweep_is_rejected(self):
        p = self.payload()
        p["run_records"] = [r for r in p["run_records"] if r["d_model"] != 64]
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_duplicate_is_rejected(self):
        p = self.payload()
        p["run_records"].append(copy.deepcopy(p["run_records"][0]))
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_unmatched_seeds_are_rejected(self):
        p = self.payload()
        p["run_records"][0]["seed"] = 2025
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_freq_hidden_drift_is_not_a_one_factor_sweep(self):
        p = self.payload()
        p["run_records"][0]["freq_hidden"] = 64
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_missing_resource_measurement_is_rejected(self):
        p = self.payload()
        for r in p["run_records"]:
            if r["d_model"] == 512 and r["learning_rate"] == 0.0005:
                r["training_ms_per_iteration"] = None
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_sample_sd_is_not_cached_population_sd(self):
        self.assertEqual(distribution([1.0, 2.0, 3.0])["sample_std"], 1.0)

    def test_nonfinite_metric_is_rejected(self):
        p = self.payload()
        p["run_records"][0]["mse"] = float("nan")
        with self.assertRaises(ValueError):
            from_capacity(p)

    def test_matched_multiple_seeds_recompute_sample_sd(self):
        p = self.payload()
        originals = p["run_records"]
        p["protocol"]["matched_seed_count"] = 3
        p["run_records"] = []
        for offset in range(3):
            for original in originals:
                r = copy.deepcopy(original)
                r["seed"] = 2024 + offset
                r["mse"] = 0.1 + 0.1 * offset
                p["run_records"].append(r)
        data = from_capacity(p)
        self.assertEqual(data["seeds"], [2024, 2025, 2026])
        self.assertAlmostEqual(data["width"][0]["mse"]["sample_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
