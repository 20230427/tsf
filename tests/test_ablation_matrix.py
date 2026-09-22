from __future__ import annotations

import pytest

from scripts.analyze_ablation_matrix import build_audit


def _records(variants=("time_only", "time_mlp"), horizons=(96, 192)):
    rows = []
    for horizon in horizons:
        for seed in range(6):
            full = 1.0 + 0.01 * seed + 0.0001 * horizon
            rows.append({
                "dataset": "Toy", "horizon": horizon, "variant": "full",
                "seed": seed, "mse": full, "phase": "validation_and_test",
            })
            for variant in variants:
                penalty = 0.1 if variant == "time_only" else 0.05
                rows.append({
                    "dataset": "Toy", "horizon": horizon, "variant": variant,
                    "seed": seed, "mse": full + penalty,
                    "phase": "validation_and_test",
                })
    return rows


def test_horizon_average_is_paired_within_seed_and_family_corrected():
    audit = build_audit(_records(), [96, 192])
    primary = [row for row in audit["rows"]
               if row["scope"] == "horizon_average_primary"]
    assert len(primary) == 2
    by_variant = {row["comparator_variant"]: row for row in primary}
    assert by_variant["time_only"]["statistics"]["n"] == 6
    assert by_variant["time_only"]["statistics"]["mean_paired_difference"] \
        == pytest.approx(0.1)
    assert by_variant["time_mlp"]["statistics"]["mean_paired_difference"] \
        == pytest.approx(0.05)
    family_ids = {family["id"] for family in audit["families"]}
    assert "horizon_average_primary.component_removal" in family_ids
    assert "horizon_average_primary.encoder_replacement" in family_ids


def test_horizon_average_rejects_incomplete_variant_cells():
    rows = _records(variants=("time_only",), horizons=(96, 192))
    rows = [row for row in rows
            if not (row["variant"] == "time_only" and row["horizon"] == 192)]
    with pytest.raises(ValueError, match="lacks horizons"):
        build_audit(rows, [96, 192])


def test_audit_rejects_mixed_validation_and_test_scopes():
    rows = _records(variants=("time_only",), horizons=(96, 192))
    rows[-1]["phase"] = "validation_only"
    with pytest.raises(ValueError, match="evaluation scopes"):
        build_audit(rows, [96, 192])
