from __future__ import annotations

import json

from scripts.run_selection_protocol import (
    confirm_frozen_candidate,
    eval_candidate,
    select_by_validation,
)
from src.data.dataset import build_splits, generate_synthetic
from src.utils import (
    atomic_write_json,
    config_sha256,
    provenance_fields,
    validate_provenance_record,
)


def _config(tmp_path):
    return {
        "data": {
            "seq_len": 24,
            "pred_len": 12,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
        },
        "model": {"arch": "dual_domain", "use_revin": True},
        "train": {"device": "cpu"},
        "experiment": {
            "name": "audit_test",
            "seed": 7,
            "checkpoint_dir": str(tmp_path),
        },
    }


def _run_record():
    return {"seed": 7, "horizon": 12, "arch": "dual_domain", "mse": 0.1}


def test_provenance_round_trip_and_atomic_write(tmp_path):
    cfg = _config(tmp_path)
    record = provenance_fields(
        cfg, config_path="configs/test.yaml", seed_values=[7],
        cli_argv=["runner.py", "--seeds", "1"], cwd=tmp_path,
    )
    record["run_records"] = [_run_record()]
    assert record["config_sha256"] == config_sha256(record["resolved_config"])
    assert record["git"] == {"commit": None, "dirty": None}
    assert validate_provenance_record(record)["status"] == "verified"

    output = tmp_path / "nested" / "result.json"
    atomic_write_json(output, record)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == record["schema_version"]
    assert not list(output.parent.glob(".result.json.*.tmp"))


def test_legacy_is_unverifiable_and_tampering_is_invalid(tmp_path):
    legacy = {"config": "configs/old.yaml", "results": {}}
    verdict = validate_provenance_record(legacy)
    assert verdict["status"] == "legacy_unverifiable"

    cfg = _config(tmp_path)
    record = provenance_fields(cfg, config_path=None, seed_values=[7], cwd=tmp_path)
    record["run_records"] = [_run_record()]
    record["resolved_config"]["model"]["use_revin"] = False
    verdict = validate_provenance_record(record)
    assert verdict["status"] == "invalid"
    assert any("config_sha256" in reason for reason in verdict["reasons"])


def test_validation_only_selection_never_requests_or_returns_test(tmp_path):
    calls = []

    def fake_train(cfg, *, evaluate_test=True):
        calls.append(evaluate_test)
        base = {
            "best_val_loss": 0.2 if cfg["model"]["use_revin"] else 0.1,
            "best_val_metrics": {"loss": 0.1, "mse": 0.1, "mae": 0.2},
            "best_epoch": 2,
            "epochs_ran": 3,
            "param_count": 10,
            "checkpoint": "fake.pt",
        }
        if evaluate_test:
            base["test_metrics"] = {"mse": 0.3, "mae": 0.4, "loss": 0.3}
        return base

    cfg = _config(tmp_path)
    yes = eval_candidate(cfg, {"model.use_revin": True}, [12], 2, 7,
                         "yes", train_fn=fake_train)
    no = eval_candidate(cfg, {"model.use_revin": False}, [12], 2, 7,
                        "no", train_fn=fake_train)
    assert calls == [False, False, False, False]
    assert "test_mse" not in yes and "test_mae" not in yes
    assert all("mse" not in run for run in yes["run_records"])

    index, selected = select_by_validation([yes, no])
    assert index == 1 and selected["candidate"] == {"model.use_revin": False}
    confirm = confirm_frozen_candidate(
        cfg, selected["candidate"], [12], 2, 7, "confirm", train_fn=fake_train
    )
    assert calls[-2:] == [True, True]
    assert all(run["phase"] == "frozen_final_confirmation"
               for run in confirm["run_records"])


def test_include_test_false_does_not_construct_test_dataset():
    data = generate_synthetic(length=400, channels=3, seed=1)
    train_ds, val_ds, test_ds, _ = build_splits(
        data, seq_len=24, pred_len=12, train_ratio=0.7, val_ratio=0.1,
        scale=True, include_test=False,
    )
    assert len(train_ds) > 0 and len(val_ds) > 0
    assert test_ds is None
