"""CPU-only tests for the hyperparameter-sensitivity harness.

Covers the pure helpers of scripts/hyperparam/run_sensitivity.py (value
parsing, the single-override contract, TSV append, per-cell checkpoint
cleanup) and the wiring between the runner's sweep registry, the launcher's
bash registry, and the plotting notebook. No GPU or dataset is touched.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml

from scripts.hyperparam import run_sensitivity
from scripts.hyperparam.run_sensitivity import (
    CONFIG_KEY,
    SWEEP_RANGES,
    TXT_COLUMNS,
    append_txt_row,
    apply_sweep_override,
    build_experiment_name,
    cell_artifact_paths,
    cleanup_cell_artifacts,
    fmt_value,
    parse_sweep_value,
    purge_done_checkpoints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_cfg() -> dict:
    return {
        "experiment": {"name": "x", "seed": 1, "checkpoint_dir": "checkpoints"},
        "data": {"pred_len": 96},
        "model": {"d_model": 128, "freq_hidden": 128, "mamba_d_state": 16},
        "train": {"lr": 0.0001},
    }


def test_parse_sweep_value_int_vs_float():
    assert parse_sweep_value("128") == 128
    assert isinstance(parse_sweep_value("128"), int)
    assert parse_sweep_value("1e-5") == 1e-5
    assert isinstance(parse_sweep_value("1e-5"), float)
    assert parse_sweep_value("0.0002") == 2e-4
    # Unparseable values pass through untouched (runner's grid check rejects).
    assert parse_sweep_value("banana") == "banana"


def test_fmt_value_canonical_strings():
    assert fmt_value(128) == "128"
    assert fmt_value(1e-5) == "1e-05"
    assert fmt_value(0.0002) == "0.0002"


def test_apply_sweep_override_changes_exactly_one_key():
    for param, (section, key) in CONFIG_KEY.items():
        cfg = _base_cfg()
        before = copy.deepcopy(cfg)
        apply_sweep_override(cfg, param, 64)
        changed = [
            (s, k)
            for s in cfg
            for k in cfg[s]
            if cfg[s][k] != before.get(s, {}).get(k)
        ]
        assert changed == [(section, key)], f"{param} must touch only {section}.{key}"
        assert cfg[section][key] == 64


def test_build_experiment_name_is_unique_and_filesystem_safe():
    a = build_experiment_name("ETTm1", "d_model", 128, 2023, 96)
    b = build_experiment_name("ETTm1", "d_model", 256, 2023, 96)
    c = build_experiment_name("ETTm1", "d_model", 128, 2024, 96)
    d = build_experiment_name("weather", "lr", 1e-5, 2022, 96)
    assert len({a, b, c, d}) == 4
    # float values are mangled ('.'/'-' replaced) so no path surprises
    assert "." not in d and "-" not in d
    assert d.startswith("weather_hparam_lr_")


def test_append_txt_row_writes_header_once_and_keeps_columns(tmp_path):
    row = {
        "dataset": "ETTm1", "pred_len": 96, "sweep_param": "d_model",
        "sweep_value": 128, "seed": 2023, "mse": "0.31", "mae": "0.36",
        "status": "ok", "elapsed_s": "1.0",
        "timestamp": "2026-08-18 00:00:00",
    }
    append_txt_row(row, "ETTm1", output_dir=tmp_path)
    row2 = dict(row, seed=2024, status="oom", mse="", mae="")
    append_txt_row(row2, "ETTm1", output_dir=tmp_path)

    path = tmp_path / "sensitivity_results_ETTm1.txt"
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].split("\t") == TXT_COLUMNS  # header exactly once
    assert lines[1].split("\t") == [str(row[c]) for c in TXT_COLUMNS]
    assert len(lines[2].split("\t")) == len(TXT_COLUMNS)


def test_sweep_registry_maps_onto_real_config_keys():
    default = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())
    for param, (section, key) in CONFIG_KEY.items():
        assert key in default[section], f"{param} -> {section}.{key} missing"
        assert param in SWEEP_RANGES


def test_bash_registry_matches_python_registry():
    """scripts/hyperparam/sensitivity_common.sh `sweep` lines must declare the
    same param grids as SWEEP_RANGES (the idempotency check compares against
    these strings, and the plot assumes the shared grid)."""
    sh = (PROJECT_ROOT / "scripts" / "hyperparam" / "sensitivity_common.sh").read_text()
    declared = dict(re.findall(r"^sweep\s+(\S+)\s+\"([^\"]+)\"", sh, re.M))
    assert set(declared) == set(SWEEP_RANGES)
    for param, values in declared.items():
        bash_grid = [float(v) for v in values.split()]
        assert bash_grid == [float(v) for v in SWEEP_RANGES[param]], param


def test_plot_notebook_exists_and_covers_the_same_params():
    nb_path = PROJECT_ROOT / "output" / "hyperparam" / "sensitivity_plot.ipynb"
    nb = json.loads(nb_path.read_text())
    src = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    for param in SWEEP_RANGES:
        assert f'"{param}"' in src, f"notebook missing sweep param {param}"


# ---------------------------------------------------------------------------
# Checkpoint hygiene: every possible cell name must be unique (a collision
# would make one cell's cleanup delete another's live checkpoint), and the
# purge must only touch artifacts of cells already recorded "ok".
# ---------------------------------------------------------------------------


def test_all_possible_cell_names_are_unique():
    """The full dispatchable grid: datasets x params x values x seeds x
    pred_lens. Experiment names embed all five components (with the sweep
    value mangled), so no two cells can share a name -- cleanup can never
    mistake one cell's checkpoint for another's."""
    names = []
    for ds in ("ETTm1", "weather"):
        for param, values in SWEEP_RANGES.items():
            for v in values:
                for seed in range(2022, 2027):
                    for pl in (96, 192, 336, 720):
                        names.append(build_experiment_name(ds, param, v, seed, pl))
    assert len(names) == len(set(names)) == 800
    # lr is the tricky one: floats whose mangled strings must stay distinct
    lr_names = [build_experiment_name("ETTm1", "lr", v, 2023, 96)
                for v in SWEEP_RANGES["lr"]]
    assert len(set(lr_names)) == 5
    # names are filesystem-safe (no path metacharacters)
    for n in names:
        assert not set(n) & {".", "/", "-", " "}


def test_cell_artifact_paths_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sensitivity, "CHECKPOINT_DIR", tmp_path)
    name = build_experiment_name("ETTm1", "d_model", 128, 2023, 96)
    pt, js = cell_artifact_paths(name)
    assert pt.parent == tmp_path
    assert pt.name == f"{name}_best.pt"
    assert js.name == f"{name}_results.json"

    pt.write_bytes(b"x")
    js.write_bytes(b"x")
    (tmp_path / "unrelated_best.pt").write_bytes(b"x")  # not a purged cell
    cleanup_cell_artifacts(name)
    assert not pt.exists() and not js.exists()
    assert (tmp_path / "unrelated_best.pt").exists()
    # idempotent: missing_ok must not raise on a second call
    cleanup_cell_artifacts(name)


def test_purge_done_checkpoints_only_removes_ok_cells(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpt"
    tsv_dir = tmp_path / "tsv"
    ckpt_dir.mkdir()
    tsv_dir.mkdir()
    monkeypatch.setattr(run_sensitivity, "CHECKPOINT_DIR", ckpt_dir)

    def touch(ds, param, value, seed, pl=96, status="ok"):
        row = {
            "dataset": ds, "pred_len": pl, "sweep_param": param,
            "sweep_value": value, "seed": seed, "mse": "0.1", "mae": "0.2",
            "status": status, "elapsed_s": "1.0", "timestamp": "t",
        }
        append_txt_row(row, ds, output_dir=tsv_dir)
        for p in cell_artifact_paths(build_experiment_name(ds, param, value, seed, pl)):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        return row

    # three DISTINCT cells: a previously-ok one (purge target), a failed one
    # (kept), and "self" -- an ok cell currently being retried (kept via
    # exclude; a retry of a recorded cell must not delete its own artifacts
    # mid-run)
    touch("ETTm1", "d_model", 128, 2022)                                    # ok
    touch("ETTm1", "d_model", 256, 2023, status="error:RuntimeError:x")     # failed
    touch("ETTm1", "d_model", 512, 2023)                                    # ok = self
    self_name = build_experiment_name("ETTm1", "d_model", 512, 2023, 96)

    removed = purge_done_checkpoints("ETTm1", exclude=self_name, tsv_dir=tsv_dir)
    assert removed == 2  # only the seed-2022 ok cell's .pt + .json
    gone_pt, gone_js = cell_artifact_paths(
        build_experiment_name("ETTm1", "d_model", 128, 2022, 96))
    assert not gone_pt.exists() and not gone_js.exists()
    self_pt, _ = cell_artifact_paths(self_name)
    failed_pt, _ = cell_artifact_paths(
        build_experiment_name("ETTm1", "d_model", 256, 2023, 96))
    assert self_pt.exists() and failed_pt.exists()
    # no TSV -> no-op, no crash
    assert purge_done_checkpoints("weather", exclude="x", tsv_dir=tsv_dir) == 0
    # float spellings from the TSV round-trip through the name builder
    touch("ETTm1", "lr", 1e-05, 2024)
    removed = purge_done_checkpoints(
        "ETTm1", exclude=self_name, tsv_dir=tsv_dir)
    # lr cell (2) + self was excluded again; already-purged ok cell counts 0
    assert removed == 2
