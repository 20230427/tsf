"""Tests for the Optuna HPO stack (src/hpo, scripts/hpo).

CPU-only: the objective is exercised against a fake ``train`` so no dataset,
GPU, or mamba kernel is required.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import optuna
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hpo import gpu_budget  # noqa: E402
from src.hpo.cleanup import (  # noqa: E402
    classify_checkpoint,
    run_cleanup,
)
from src.hpo.objective import HpoLeakageError, create_objective  # noqa: E402
from src.hpo.search_space import (  # noqa: E402
    anchor_params,
    build_trial_cfg,
    expand_forwarded_params,
    load_spec,
    normalize_conditions,
    resolve_space,
    snap_to_space,
    suggest_params,
)
from src.hpo.study_manager import create_study, export_results, study_name  # noqa: E402


SPEC_PATH = ROOT / "configs" / "hpo_search_spaces.yaml"


@pytest.fixture()
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture()
def base_cfg():
    return {
        "experiment": {"name": "ETTh1", "seed": 2024, "checkpoint_dir": "checkpoints"},
        "data": {"seq_len": 96, "pred_len": 96, "batch_size": 32},
        "model": {
            "d_model": 128, "time_encoder": "mamba", "mamba_layers": 1,
            "mamba_d_state": 16, "freq_hidden": 128, "time_kernel_size": 25,
            "freq_sparsity": 0.4, "freq_backbone": "fits",
            "channel_mixer_layers": 0, "mixer_placement": "both",
            "time_dropout": 0.2, "freq_dropout": 0.1, "head_dropout": 0.1,
            "use_revin": True, "arch": "dual_domain",
        },
        "train": {"lr": 5e-4, "epochs": 1, "batch_size": 32, "amp": False},
    }


# ---------------------------------------------------------------- search space

def test_spec_schema_valid(spec):
    space, fixed = resolve_space(spec, "ETTh1")
    assert "train.lr" in space
    assert space["model.time_encoder"]["choices"][0] == "mamba"
    assert True in space["model.use_revin"]["choices"]
    assert fixed == {}
    pems_space, pems_fixed = resolve_space(spec, "PEMS04")
    assert "train.amp" not in pems_space
    assert pems_fixed["train.amp"] is False


def test_forced_defaults_prefer_mamba_and_revin(spec):
    forced = spec["forced_defaults"]
    assert forced["model.time_encoder"] == "mamba"
    assert forced["model.use_revin"] is True


def test_pinned_profile(spec):
    """The tuned profile pins single-option switches and narrows two grids."""
    space, _ = resolve_space(spec, "ETTh1")
    # Single-option pins (marked inline in the YAML).
    assert space["model.time_encoder"]["choices"] == ["mamba"]
    assert space["model.freq_backbone"]["choices"] == ["fits"]
    assert space["model.use_revin"]["choices"] == [True]
    # Narrowed grids.
    assert space["model.channel_mixer_layers"]["choices"] == [1, 2]
    assert space["model.mamba_layers"]["choices"] == [1, 2, 3]


def test_search_fixed_overlap_rejected(spec):
    broken = copy.deepcopy(spec)
    broken.setdefault("datasets", {})["ETTh1"] = {
        "fixed": {"train.lr": 1e-4},  # also in base.search
    }
    with pytest.raises(ValueError, match="both searched and fixed"):
        resolve_space(broken, "ETTh1")


def test_param_groups_cover_and_order_the_space(spec):
    """The 3 multivariate groups cover every searched key in a fixed order."""
    from src.hpo.search_space import PARAM_GROUPS

    space, _ = resolve_space(spec, "ETTh1")
    grouped = [key for _, group in PARAM_GROUPS for key in group]
    assert set(grouped) >= set(space)  # every searched key belongs to a group
    # Architecture switches first, then model sizing/regularization, training last.
    order = [name for name, _ in PARAM_GROUPS]
    assert order == ["architecture", "model", "training"]
    flat = [key for _, group in PARAM_GROUPS for key in group]
    assert flat.index("model.time_encoder") < flat.index("model.d_model")
    assert flat.index("model.d_model") < flat.index("train.lr")
    # Architecture group holds exactly the component switches.
    arch = dict(PARAM_GROUPS)["architecture"]
    assert set(arch) == {"model.time_encoder", "model.channel_mixer_layers",
                         "model.mixer_placement", "model.freq_backbone",
                         "model.use_revin"}
    # The three dropouts are searched as ONE shared parameter with forwarding.
    assert "model.dropout" in dict(PARAM_GROUPS)["model"]
    assert not any(key in space for key in
                   ("model.time_dropout", "model.freq_dropout",
                    "model.head_dropout"))


def test_shared_dropout_forwarding(spec):
    """model.dropout fans out to the three real dropout keys everywhere."""
    space, _ = resolve_space(spec, "ETTh1")
    params = {"model.dropout": 0.2}
    notes = normalize_conditions(params, space)
    assert params["model.time_dropout"] == 0.2
    assert params["model.freq_dropout"] == 0.2
    assert params["model.head_dropout"] == 0.2
    assert notes and "model.dropout=0.2" in notes[0]
    # Pure helper does not mutate its input.
    source = {"model.dropout": 0.3}
    expanded = expand_forwarded_params(source, space)
    assert expanded["model.head_dropout"] == 0.3
    assert source == {"model.dropout": 0.3}
    # A forward target may not itself be searched.
    broken = copy.deepcopy(spec)
    broken["base"]["search"]["model.head_dropout"] = {
        "type": "categorical", "choices": [0.1, 0.2]}
    with pytest.raises(ValueError, match="forward_to targets"):
        resolve_space(broken, "ETTh1")


def test_anchor_dropout_read_from_tuned_config(spec, base_cfg):
    """The anchor derives model.dropout from the tuned config's dropouts."""
    anchor = anchor_params(base_cfg, spec, "ETTh1")
    assert anchor["model.dropout"] == 0.2  # from model.time_dropout: 0.2
    assert anchor["model.dropout"] in \
        resolve_space(spec, "ETTh1")[0]["model.dropout"]["choices"]


def test_normalize_conditions_pins_placement(base_cfg):
    params = {"model.channel_mixer_layers": 0, "model.mixer_placement": "shared"}
    notes = normalize_conditions(params)
    assert params["model.mixer_placement"] == "both"
    assert notes
    params = {"model.channel_mixer_layers": 2, "model.mixer_placement": "shared"}
    assert normalize_conditions(params) == []
    assert params["model.mixer_placement"] == "shared"


def test_snap_to_space_nearest():
    spec = {"type": "categorical", "choices": [0.0, 0.1, 0.2, 0.3]}
    assert snap_to_space(0.4, spec) == 0.3
    assert snap_to_space(0.1, spec) == 0.1
    spec = {"type": "categorical", "choices": ["mamba", "mlp"]}
    assert snap_to_space("mamba", spec) == "mamba"
    assert snap_to_space("attention", spec) == "mamba"  # off-grid -> first choice
    spec = {"type": "loguniform", "low": 1e-4, "high": 2e-3}
    assert snap_to_space(5e-3, spec) == 2e-3


def test_anchor_params_snap_tuned_config(spec, base_cfg):
    anchor = anchor_params(base_cfg, spec, "ETTh1")
    space, _ = resolve_space(spec, "ETTh1")
    assert set(anchor) == set(space)
    assert anchor["model.time_encoder"] == "mamba"  # forced default / pinned
    assert anchor["model.use_revin"] is True
    assert anchor["model.freq_sparsity"] == 0.3  # tuned 0.4 snapped to grid
    assert anchor["model.channel_mixer_layers"] == 1  # tuned 0 snapped into [1, 2]
    assert anchor["model.freq_backbone"] == "fits"  # pinned
    for key, value in anchor.items():
        param_spec = space[key]
        if param_spec["type"] == "categorical":
            assert value in param_spec["choices"]


def test_electricity_d_model_capped_for_24gb(spec):
    """Electricity (321 channels): the d512 tier OOMs a 24GB card on the
    first training step at every horizon, so the space is narrowed
    Pamba-style (their heaviest datasets cap d_model the same way) and the
    tuned d512 recipe snaps to d256 for the anchor trial."""
    space, _ = resolve_space(spec, "electricity")
    assert space["model.d_model"]["choices"] == [128, 256]
    cfg = {"experiment": {"name": "electricity"}, "model": {"d_model": 512}}
    anchor = anchor_params(cfg, spec, "electricity")
    assert anchor["model.d_model"] == 256


def test_build_trial_cfg_precedence(base_cfg, spec):
    _, fixed = resolve_space(spec, "PEMS04")
    forced = dict(spec["forced_defaults"])
    cfg = build_trial_cfg(
        base_cfg, fixed, forced,
        {"model.time_encoder": "mlp", "model.d_model": 256},
        name="unit_t0_s0", seed=7, epochs=3,
    )
    assert cfg["model"]["time_encoder"] == "mlp"  # suggestion beats forced default
    assert cfg["model"]["use_revin"] is True  # forced default applies
    assert cfg["train"]["amp"] is False  # dataset fixed wins last
    assert cfg["model"]["d_model"] == 256
    assert cfg["experiment"]["name"] == "unit_t0_s0"
    assert cfg["experiment"]["seed"] == 7
    assert cfg["train"]["epochs"] == 3


# --------------------------------------------------------------------- objective


class _FakeTrain:
    def __init__(self):
        self.calls = []

    def __call__(self, cfg, *, evaluate_test=True):
        self.calls.append((copy.deepcopy(cfg), evaluate_test))
        out = {
            "best_val_loss": 0.5,
            "best_val_metrics": {"mse": 0.5, "mae": 0.4, "loss": 0.5},
            "best_epoch": 1, "epochs_ran": 1, "param_count": 1234,
            "checkpoint": None,
        }
        if evaluate_test:
            out["test_metrics"] = {"mse": 0.42, "mae": 0.39, "loss": 0.42}
        return out


@pytest.fixture()
def fake_train(monkeypatch):
    fake = _FakeTrain()
    import src.hpo.objective as objective_module

    monkeypatch.setattr(objective_module, "train", fake)
    return fake


def _tiny_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["checkpoint_dir"] = "checkpoints/hpo_test"
    return cfg


def _run_study(objective, n_trials=2):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study


def _patch_estimate(monkeypatch, mb=100.0):
    import src.hpo.objective as objective_module

    monkeypatch.setattr(objective_module, "estimate_ddmamba_memory_mb",
                        lambda cfg, n: mb)


def test_objective_runs_and_records_attrs(fake_train, spec, base_cfg, monkeypatch):
    _patch_estimate(monkeypatch)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    study = _run_study(objective, n_trials=2)
    assert len(fake_train.calls) == 2
    for (call_cfg, evaluate_test), trial in zip(fake_train.calls, study.trials):
        assert evaluate_test is True
        assert call_cfg["experiment"]["name"].startswith("ETTh1_pl96_hpo_t")
        assert trial.user_attrs["status"] == "ok"
        assert trial.user_attrs["mse"] == pytest.approx(0.42)
        assert trial.user_attrs["val_loss"] == pytest.approx(0.5)
        assert trial.user_attrs["param_count"] == 1234
    assert study.best_value == pytest.approx(0.42)


def test_objective_uses_pamba_log_style(fake_train, spec, base_cfg, monkeypatch):
    """HPO trials must train with the compact Pamba-style epoch logs.

    The study logs are redirected to files, where a tqdm bar would emit one
    line per batch; the objective therefore forces ``train.log_style='pamba'``
    (three plain lines per epoch, no bar) without mutating the caller's cfg.
    """
    _patch_estimate(monkeypatch)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    _run_study(objective, n_trials=1)
    assert fake_train.calls, "fake train was never invoked"
    trial_cfg = fake_train.calls[0][0]
    assert trial_cfg["train"]["log_style"] == "pamba"
    assert "log_style" not in cfg["train"]


def test_objective_trial_names_include_pred_len(fake_train, spec, base_cfg,
                                                monkeypatch):
    """Parallel per-horizon studies must never share checkpoint paths.

    Regression test for the 2026-09-06 electricity sweep: trial names
    lacking pred_len let four concurrent studies clobber each other's
    ``checkpoints/<name>_best.pt`` (state_dict size-mismatch crashes that
    surfaced as five consecutive pruned trials and a dead study).
    """
    _patch_estimate(monkeypatch)
    names = []
    for pred_len in (96, 192):
        cfg = _tiny_cfg(base_cfg)
        cfg["data"]["pred_len"] = pred_len
        objective = create_objective(cfg, spec, metric="mse",
                                     dedicated_gpu=True, n_channels=7)
        _run_study(objective, n_trials=1)
        names.append(fake_train.calls[-1][0]["experiment"]["name"])
    assert names[0] != names[1]
    assert "pl96" in names[0]
    assert "pl192" in names[1]


def test_objective_applies_params_to_cfg(fake_train, spec, base_cfg, monkeypatch):
    _patch_estimate(monkeypatch)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    study = _run_study(objective, n_trials=1)
    call_cfg, _ = fake_train.calls[0]
    params = study.trials[0].params
    assert call_cfg["model"]["d_model"] == params["model.d_model"]
    assert call_cfg["train"]["lr"] == params["train.lr"]
    assert call_cfg["model"]["time_encoder"] == params["model.time_encoder"]
    # Shared dropout is transparently forwarded to all three real keys.
    assert call_cfg["model"]["time_dropout"] == params["model.dropout"]
    assert call_cfg["model"]["freq_dropout"] == params["model.dropout"]
    assert call_cfg["model"]["head_dropout"] == params["model.dropout"]
    if params["model.channel_mixer_layers"] == 0:
        assert call_cfg["model"]["mixer_placement"] == "both"


def test_objective_val_mode_never_sees_test(fake_train, spec, base_cfg, monkeypatch):
    _patch_estimate(monkeypatch)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="val_loss", dedicated_gpu=True,
                                 n_channels=7)
    study = _run_study(objective, n_trials=1)
    _, evaluate_test = fake_train.calls[0]
    assert evaluate_test is False
    assert study.best_value == pytest.approx(0.5)


def test_objective_leak_guard_fails_trial(fake_train, spec, base_cfg, monkeypatch):
    _patch_estimate(monkeypatch)

    def leaking_train(cfg, *, evaluate_test=True):
        out = _FakeTrain()(cfg, evaluate_test=evaluate_test)
        out["test_metrics"] = {"mse": 0.1}  # leak even in val mode
        return out

    import src.hpo.objective as objective_module

    monkeypatch.setattr(objective_module, "train", leaking_train)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="val_loss", dedicated_gpu=True,
                                 n_channels=7)
    study = optuna.create_study(direction="minimize")
    with pytest.raises(HpoLeakageError):
        study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_objective_cleans_trial_artifacts(fake_train, spec, base_cfg, monkeypatch,
                                          tmp_path):
    _patch_estimate(monkeypatch)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    results_json = ckpt_dir / "ETTh1_pl96_hpo_t0_s0_results.json"
    results_json.write_text("{}")
    checkpoint_pt = ckpt_dir / "ETTh1_pl96_hpo_t0_s0_best.pt"
    checkpoint_pt.write_bytes(b"x")

    import src.hpo.objective as objective_module

    original_train = objective_module.train

    def cleanup_train(cfg, *, evaluate_test=True):
        out = original_train(cfg, evaluate_test=evaluate_test)
        out["checkpoint"] = str(checkpoint_pt)
        return out

    monkeypatch.setattr(objective_module, "train", cleanup_train)
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["checkpoint_dir"] = str(ckpt_dir)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    _run_study(objective, n_trials=1)
    assert not checkpoint_pt.exists()
    assert not results_json.exists()


def test_objective_skips_over_budget(fake_train, spec, base_cfg, monkeypatch):
    import src.hpo.objective as objective_module

    monkeypatch.setattr(objective_module, "TOTAL_GPU_MEMORY_MB", 1024)
    _patch_estimate(monkeypatch, mb=999999.0)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    study = _run_study(objective, n_trials=1)
    assert fake_train.calls == []  # never trained
    assert study.trials[0].value == float("inf")
    assert study.trials[0].user_attrs["status"] == "skipped_over_budget"


def _artifacts_writing_train(ckpt_dir, exc):
    """Fake train that writes the deterministic trial artifacts, then fails."""

    def failing_train(cfg, *, evaluate_test=True):
        name = cfg["experiment"]["name"]
        (ckpt_dir / f"{name}_best.pt").write_bytes(b"x")
        (ckpt_dir / f"{name}_results.json").write_text("{}")
        raise exc

    return failing_train


@pytest.mark.parametrize("exc,expected_state", [
    (RuntimeError("boom"), optuna.trial.TrialState.PRUNED),
    # dedicated-GPU runtime OOM completes with inf instead of pruning
    (RuntimeError("CUDA out of memory"), optuna.trial.TrialState.COMPLETE),
])
def test_objective_failed_trial_cleans_artifacts(exc, expected_state, spec,
                                                 base_cfg, monkeypatch,
                                                 tmp_path):
    """Error / OOM paths must not leak the trial's half-written artifacts."""
    import src.hpo.objective as objective_module

    _patch_estimate(monkeypatch)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    monkeypatch.setattr(objective_module, "train",
                        _artifacts_writing_train(ckpt_dir, exc))
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["checkpoint_dir"] = str(ckpt_dir)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    study = _run_study(objective, n_trials=1)
    assert study.trials[0].state == expected_state
    assert list(ckpt_dir.iterdir()) == []  # checkpoint + results JSON gone


def test_objective_keep_checkpoint_retains_failed_trial_artifacts(
        spec, base_cfg, monkeypatch, tmp_path):
    """--keep-checkpoint opts out of failed-trial cleanup too."""
    import src.hpo.objective as objective_module

    _patch_estimate(monkeypatch)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    monkeypatch.setattr(objective_module, "train",
                        _artifacts_writing_train(ckpt_dir, RuntimeError("boom")))
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["checkpoint_dir"] = str(ckpt_dir)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7, keep_checkpoint=True)
    _run_study(objective, n_trials=1)
    names = sorted(p.name for p in ckpt_dir.iterdir())
    assert names == ["ETTh1_pl96_hpo_t0_s0_best.pt",
                     "ETTh1_pl96_hpo_t0_s0_results.json"]


# ------------------------------------------------------------------- gpu budget


@pytest.fixture()
def budget_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_budget, "HPO_DIR", str(tmp_path))
    monkeypatch.setattr(gpu_budget, "BUDGET_LOCK_FILE", str(tmp_path / "lock"))
    monkeypatch.setattr(gpu_budget, "BUDGET_JSON_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setattr(gpu_budget, "MULTI_BUDGET_FILE", str(tmp_path / "multi.json"))
    return tmp_path


def test_gpu_budget_acquire_release(budget_paths):
    budget = gpu_budget.GPUBudget(total_mb=1000)
    assert budget.acquire("t1", 600, timeout=1)
    assert budget.acquire("t2", 600, timeout=0.1) is False  # would overcommit
    budget.release("t1")
    assert budget.acquire("t2", 600, timeout=1)
    budget.release("t2")


def test_gpu_budget_stale_pid_reaped(budget_paths, monkeypatch):
    budget = gpu_budget.GPUBudget(total_mb=1000)
    with open(gpu_budget.BUDGET_JSON_FILE, "w") as f:
        json.dump({"used_mb": 500.0,
                   "trials": {"dead": {"pid": 999999, "mb": 500.0}}}, f)
    assert budget.acquire("fresh", 800, timeout=1)  # stale entry reaped first


def test_multi_gpu_budget_reserves_freest(budget_paths):
    budget = gpu_budget.MultiGPUBudget([0, 1], total_mb=1000)
    assert budget.reserve("s1", 900, timeout=1) == 0
    assert budget.reserve("s2", 900, timeout=1) == 1
    assert budget.reserve("s3", 900, timeout=0.1) is None  # both cards full
    budget.unreserve("s1")
    assert budget.reserve("s3", 900, timeout=1) == 0


# --------------------------------------------------------------- study manager


def test_study_roundtrip_and_export(tmp_path, monkeypatch, fake_train, spec,
                                    base_cfg):
    from src.hpo import study_manager

    monkeypatch.setattr(study_manager, "STORAGE_DIR", str(tmp_path / "optuna"))
    monkeypatch.setattr(study_manager, "CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(study_manager, "OUTPUT_HPO_DIR", str(tmp_path))
    monkeypatch.setattr(study_manager, "STORAGE_TEMPLATE",
                        "sqlite:///" + str(tmp_path / "optuna" / "{}.db"))
    _patch_estimate(monkeypatch)

    name = study_name("UnitTest", 96)
    study = create_study(name, n_startup_trials=1)
    study.enqueue_trial(anchor_params(base_cfg, spec, "ETTh1"))
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mse", dedicated_gpu=True,
                                 n_channels=7)
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    space, _ = resolve_space(spec, "ETTh1")
    export_results(name, fill={"train.amp": False, "model.time_encoder": "mamba"},
                   space=space)
    best_json = tmp_path / f"{name}_best_params.json"
    assert best_json.exists()
    payload = json.loads(best_json.read_text())
    # Pamba format: a single best-record document, no non-SOTA trial records.
    assert payload["version"] == "ddmamba-hpo-best-v2"
    assert "run_records" not in payload
    assert payload["metric"] == "mse"
    assert payload["best_value"] == pytest.approx(0.42)
    # Complete parameter set: fill merges under suggested params, and the
    # shared dropout is expanded to the three real keys (directly applicable).
    assert payload["best_params"]["train.amp"] is False
    assert payload["best_params"]["experiment.seed"] == 2024
    assert "model.d_model" in payload["best_params"]
    shared = payload["best_params"]["model.dropout"]
    assert payload["best_params"]["model.time_dropout"] == shared
    assert payload["best_params"]["model.freq_dropout"] == shared
    assert payload["best_params"]["model.head_dropout"] == shared
    ordered_attrs = list(payload["best_trial_user_attrs"])
    assert ordered_attrs[:2] == ["mse", "mae"]
    assert (tmp_path / "csv" / f"{name}_results.csv").exists()


def test_export_selects_best_test_mse_not_optimized_metric(tmp_path, monkeypatch,
                                                            fake_train, spec,
                                                            base_cfg):
    """Even when the study optimizes MAE, the export reports the best-MSE trial."""
    from src.hpo import study_manager

    monkeypatch.setattr(study_manager, "STORAGE_DIR", str(tmp_path / "optuna"))
    monkeypatch.setattr(study_manager, "CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(study_manager, "OUTPUT_HPO_DIR", str(tmp_path))
    monkeypatch.setattr(study_manager, "STORAGE_TEMPLATE",
                        "sqlite:///" + str(tmp_path / "optuna" / "{}.db"))
    _patch_estimate(monkeypatch)

    # MSE 0.42 / MAE 0.39 vs MSE 0.50 / MAE 0.30: optimizing MAE picks the
    # second trial, but the export must always report the best test MSE.
    outcomes = [{"mse": 0.42, "mae": 0.39}, {"mse": 0.50, "mae": 0.30}]

    def varying_train(cfg, *, evaluate_test=True):
        out = {
            "best_val_loss": 0.5, "best_val_metrics": {}, "best_epoch": 1,
            "epochs_ran": 1, "param_count": 1, "checkpoint": None,
        }
        if evaluate_test:
            idx = len(varying_train.calls)
            metrics = outcomes[min(idx, len(outcomes) - 1)]
            out["test_metrics"] = {**metrics, "loss": metrics["mse"]}
        varying_train.calls.append(cfg)
        return out

    varying_train.calls = []

    import src.hpo.objective as objective_module

    monkeypatch.setattr(objective_module, "train", varying_train)

    name = study_name("UnitTestMae", 96)
    study = create_study(name, n_startup_trials=1)
    cfg = _tiny_cfg(base_cfg)
    objective = create_objective(cfg, spec, metric="mae", dedicated_gpu=True,
                                 n_channels=7)
    study.optimize(objective, n_trials=2)
    # study optimized MAE; best-by-MAE is trial #1
    assert study.best_trial.number == 1
    export_results(name)
    payload = json.loads((tmp_path / f"{name}_best_params.json").read_text())
    # export still reports the best test-MSE record
    assert payload["metric"] == "mse"
    assert payload["best_trial_number"] == 0
    assert payload["best_value"] == pytest.approx(0.42)


def test_export_inf_only_study_writes_csv_not_best_params(tmp_path, monkeypatch):
    """A study whose completed trials are all inf (e.g. every run OOMed)
    must export the CSV without raising instead of crashing on
    json.dump(allow_nan=False) -- the failure mode that aborted the
    2026-09-06 electricity_pl192/pl720 exports."""
    from src.hpo import study_manager

    monkeypatch.setattr(study_manager, "STORAGE_DIR", str(tmp_path / "optuna"))
    monkeypatch.setattr(study_manager, "CSV_DIR", str(tmp_path / "csv"))
    monkeypatch.setattr(study_manager, "OUTPUT_HPO_DIR", str(tmp_path))
    monkeypatch.setattr(study_manager, "STORAGE_TEMPLATE",
                        "sqlite:///" + str(tmp_path / "optuna" / "{}.db"))

    name = study_name("UnitTestInf", 96)
    study = create_study(name, n_startup_trials=1)
    study.optimize(lambda trial: float("inf"), n_trials=1)
    assert study.trials[0].value == float("inf")

    export_results(name)  # must not raise
    assert (tmp_path / "csv" / f"{name}_results.csv").exists()
    assert not (tmp_path / f"{name}_best_params.json").exists()


# -------------------------------------------------------------------- cleanup


def test_cleanup_classify_checkpoint():
    new = classify_checkpoint("electricity_pl192_hpo_t3_s0_best.pt")
    assert new == {"dataset": "electricity", "pred_len": 192, "trial": 3,
                   "seed": 0, "legacy": False}
    legacy = classify_checkpoint("illness_hpo_t7_s1_best.pt")
    assert legacy == {"dataset": "illness", "pred_len": None, "trial": 7,
                      "seed": 1, "legacy": True}
    # new naming wins over the legacy pattern for the same filename
    assert classify_checkpoint("electricity_pl96_hpo_t9_s0_best.pt")["pred_len"] == 96
    # regular experiment checkpoints and JSONs are never matched
    assert classify_checkpoint("ETTh1_best.pt") is None
    assert classify_checkpoint("electricity_pl192_hpo_t3_s0_results.json") is None


def test_cleanup_plan_and_run(tmp_path):
    ckpt = tmp_path / "ck"
    ckpt.mkdir()
    pids = tmp_path / "pids"
    pids.mkdir()

    def touch(path, data=b"x"):
        path.write_bytes(data)

    new = ckpt / "electricity_pl192_hpo_t3_s0_best.pt"
    touch(new, b"x" * 1000)
    legacy = ckpt / "illness_hpo_t7_s1_best.pt"
    touch(legacy, b"y")
    other_ds = ckpt / "weather_pl96_hpo_t1_s0_best.pt"
    touch(other_ds, b"z")
    regular = ckpt / "ETTh1_best.pt"
    touch(regular, b"r")  # regular experiment checkpoint: never matched
    results_json = ckpt / "electricity_pl192_hpo_t3_s0_results.json"
    results_json.write_text("{}")  # JSON: preserve by policy

    dead_pid = pids / "gone_pl96.pid"
    dead_pid.write_text("1999999")
    live_pid = pids / "live_pl96.pid"
    live_pid.write_text(str(os.getpid()))

    # no liveness protection by design: every HPO trial checkpoint is listed
    report = run_cleanup(confirm=False, checkpoint_dir=ckpt, pid_dir=pids)
    listed = {Path(p).name for p, _ in report["checkpoints"]}
    assert listed == {new.name, legacy.name, other_ds.name}
    assert str(regular.name) not in listed
    assert report["pid_files"] == [str(dead_pid)]
    # dry run: nothing deleted
    assert all(p.exists() for p in (new, legacy, other_ds, regular, dead_pid))

    # dataset / pred_len filters restrict the scope
    filtered = run_cleanup(confirm=False, checkpoint_dir=ckpt, pid_dir=pids,
                           dataset="electricity", pred_len=192)
    assert [Path(p).name for p, _ in filtered["checkpoints"]] == [new.name]

    run_cleanup(confirm=True, checkpoint_dir=ckpt, pid_dir=pids)
    assert not new.exists()
    assert not legacy.exists()
    assert not other_ds.exists()
    assert not dead_pid.exists()
    # out-of-scope artifacts survive
    assert regular.exists()
    assert results_json.exists()
    assert live_pid.exists()
