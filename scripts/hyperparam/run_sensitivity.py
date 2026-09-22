#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis for DD-Mamba.

Single-experiment runner: each invocation trains and tests exactly one
(dataset, sweep_param, pred_len, sweep_value, seed) cell. All non-swept
hyperparameters come from the dataset's tuned config
``configs/<dataset>.yaml`` (the single source of truth); exactly one dotted
key is overridden per run.

Adapted from the Pamba sensitivity harness (references/hyperparam_scripts);
differences: configs come from configs/*.yaml instead of a profile resolver,
the freq-branch width (freq_hidden) replaces geo_d_model, and there is no
separate freq-branch state dim (mamba_d_state is shared), so the fourth
swept dimension is the learning rate.

Results are appended as one TSV row to
``output/hyperparam/sensitivity_results_<dataset>.txt`` (same 10-column
schema as Pamba, so the launcher's awk idempotency check and the plotting
notebook carry over). The TSV row is the experiment's only durable record:
each cell's checkpoint and per-run results.json under
``checkpoints/hyperparam/`` are deleted after evaluation (see
``cleanup_cell_artifacts``; ``--keep-checkpoint`` opts out), and artifacts
leaked by abnormally-exited cells are reaped on the next run
(``purge_done_checkpoints``).

Usage:
    python scripts/hyperparam/run_sensitivity.py \
        --dataset ETTm1 --sweep_param d_model --sweep_value 128 \
        --pred_len 96 --seed 2023
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hyperparam"

# Checkpoint dir for sensitivity runs, as an ABSOLUTE path: isolated from
# curated checkpoints, and absolute so train()'s writes and this runner's
# cleanup always resolve the same files regardless of the caller's CWD.
# Artifacts ({name}_best.pt, {name}_results.json) are deleted after each
# cell's evaluation (see cleanup_cell_artifacts); the TSV row is the
# experiment's only durable record, so 200+ cells cannot fill the disk.
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "hyperparam"

#: sweep_param (TSV name) -> value grid. Kept in lockstep with
#: scripts/hyperparam/sensitivity_common.sh and output/hyperparam/
#: sensitivity_plot.ipynb -- edit all three together.
SWEEP_RANGES = {
    "d_model": [64, 128, 256, 512, 1024],
    "freq_hidden": [16, 32, 64, 128, 256],
    "mamba_d_state": [2, 4, 8, 16, 32],
    "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
}

#: sweep_param (TSV name) -> dotted config key in configs/<dataset>.yaml.
CONFIG_KEY = {
    "d_model": ("model", "d_model"),
    "freq_hidden": ("model", "freq_hidden"),
    "mamba_d_state": ("model", "mamba_d_state"),
    "lr": ("train", "lr"),
}

TXT_COLUMNS = [
    "dataset",
    "pred_len",
    "sweep_param",
    "sweep_value",
    "seed",
    "mse",
    "mae",
    "status",
    "elapsed_s",
    "timestamp",
]

TXT_HEADER = "\t".join(TXT_COLUMNS)


def parse_sweep_value(raw: str):
    """'128' -> 128 (int), '1e-5' / '0.0002' -> float."""
    try:
        return int(raw) if "." not in raw and "e" not in raw.lower() else float(raw)
    except ValueError:
        return raw


def fmt_value(value) -> str:
    """Canonical string for experiment names / tick labels: 1e-05 -> '1e-05'."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def apply_sweep_override(cfg: dict, param: str, value) -> dict:
    """Set exactly one (section, key) cell of the config in place."""
    section, key = CONFIG_KEY[param]
    cfg[section][key] = value
    return cfg


def build_experiment_name(dataset: str, param: str, value, seed: int, pred_len: int) -> str:
    v = fmt_value(value).replace(".", "p").replace("-", "m")
    return f"{dataset}_hparam_{param}_{v}_s{seed}_H{pred_len}"


def append_txt_row(row: dict, dataset: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Append one TSV row; create the file with header on first use.

    The launcher normally pre-creates the header, but standalone runs (and
    tests) may hit a missing file, so header creation is idempotent here.
    Appends are single short lines (< PIPE_BUF), so concurrent workers on
    one host stay atomic without a lock.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sensitivity_results_{dataset}.txt"
    if not path.exists():
        path.write_text(TXT_HEADER + "\n", encoding="utf-8")
    line = "\t".join(str(row[h]) for h in TXT_COLUMNS) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def cell_artifact_paths(name: str) -> tuple:
    """(best.pt, results.json) paths a cell owns under CHECKPOINT_DIR.

    The caller owns the name: experiment names embed dataset, sweep_param,
    the mangled sweep value, seed AND pred_len (build_experiment_name), so
    two cells can never share a name -- verified across the full 800-cell
    grid (2 datasets x 4 params x 5 values x 5 seeds x 4 pred_lens) in
    tests/test_hyperparam.py.
    """
    return (
        CHECKPOINT_DIR / f"{name}_best.pt",
        CHECKPOINT_DIR / f"{name}_results.json",
    )


def cleanup_cell_artifacts(name: str) -> None:
    """Delete one cell's checkpoint + per-run JSON after evaluation.

    Best-effort: never raises (cleanup must not turn a finished run into an
    error row). The best checkpoint has already been reloaded for the final
    evaluation inside train(), so it has served its purpose.
    """
    for p in cell_artifact_paths(name):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def purge_done_checkpoints(dataset: str, exclude: str,
                           tsv_dir: Path = OUTPUT_DIR) -> int:
    """Delete artifacts left by cells already recorded as "ok" in the TSV.

    This covers leaks from abnormal exits (SIGKILL, power loss) where the
    finally-block cleanup never ran. RACE-SAFETY RULE: only cells whose row
    already has status "ok" are purged -- the launcher never re-dispatches
    those, so a concurrent worker (--para N slots/GPU) cannot be training
    one. `exclude` is additionally kept (this very process may be re-running
    a failed cell). Returns the number of artifacts removed.
    """
    tsv = tsv_dir / f"sensitivity_results_{dataset}.txt"
    if not tsv.exists():
        return 0
    removed = 0
    with open(tsv, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(TXT_COLUMNS) or not parts[7].startswith("ok"):
                continue
            try:
                name = build_experiment_name(
                    parts[0], parts[2], parse_sweep_value(parts[3]),
                    int(parts[4]), int(parts[1]),
                )
            except (ValueError, TypeError):
                continue  # malformed row: skip, never block a new run
            if name == exclude:
                continue
            for p in cell_artifact_paths(name):
                try:
                    if p.exists():
                        p.unlink()
                        removed += 1
                except OSError:
                    pass
    return removed


def main():
    cli = argparse.ArgumentParser(description="Single-experiment hyperparameter sweep")
    cli.add_argument("--dataset", type=str, required=True,
                     help="config stem, e.g. ETTm1 or weather")
    cli.add_argument("--sweep_param", type=str, required=True,
                     choices=list(SWEEP_RANGES.keys()))
    cli.add_argument("--sweep_value", type=str, required=True)
    cli.add_argument("--pred_len", type=int, default=96)
    cli.add_argument("--seed", type=int, default=2023)
    cli.add_argument("--validation-only", action="store_true",
                     help="evaluate on the validation split only; never "
                          "construct the test split (for cheap screening)")
    cli.add_argument("--keep-checkpoint", action="store_true",
                     help="keep this cell's checkpoint + results.json after "
                          "evaluation (default: both are deleted; the TSV "
                          "row is the durable record) -- for debugging a "
                          "single cell")
    opts = cli.parse_args()

    dataset = opts.dataset
    config_path = CONFIG_DIR / f"{dataset}.yaml"
    if not config_path.exists():
        print(f"ERROR: no config at {config_path} (dataset must be a config "
              f"stem, e.g. ETTm1, weather)")
        sys.exit(1)

    if opts.pred_len not in [96, 192, 336, 720]:
        print(f"ERROR: Unsupported pred_len={opts.pred_len}. "
              f"Available: [96, 192, 336, 720]")
        sys.exit(1)

    param = opts.sweep_param
    value = parse_sweep_value(opts.sweep_value)
    grid = SWEEP_RANGES[param]
    if not any(float(value) == float(v) for v in grid):
        print(f"ERROR: {value!r} is not on the {param} grid {grid}")
        sys.exit(1)

    import torch

    cfg = load_config(str(config_path))
    cfg["data"]["pred_len"] = opts.pred_len
    cfg["experiment"]["seed"] = opts.seed
    cfg["experiment"]["checkpoint_dir"] = str(CHECKPOINT_DIR)
    cfg["experiment"]["name"] = build_experiment_name(
        dataset, param, value, opts.seed, opts.pred_len
    )
    apply_sweep_override(cfg, param, value)

    from src.train import train

    tag = f"[{dataset} pl={opts.pred_len} {param}={fmt_value(value)} seed={opts.seed}]"

    # Reap artifacts leaked by earlier abnormally-exited cells (SIGKILL,
    # power loss). Only cells already recorded "ok" are touched, so this
    # can never race a concurrently training worker; our own name is
    # excluded because this run may be a retry of a failed cell.
    purged = purge_done_checkpoints(dataset, exclude=cfg["experiment"]["name"])
    if purged:
        print(f"{tag} purged {purged} stale artifact(s) from previously-ok cells")

    row = {
        "dataset": dataset,
        "pred_len": opts.pred_len,
        "sweep_param": param,
        "sweep_value": value,
        "seed": opts.seed,
        "mse": "",
        "mae": "",
        "status": "running",
        "elapsed_s": "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"{tag} Starting training ...")
    t0 = time.time()
    try:
        out = train(cfg, evaluate_test=not opts.validation_only)
        metrics = (
            out["best_val_metrics"] if opts.validation_only
            else out["test_metrics"]
        )
        mse_val = metrics.get("mse") if metrics else None
        mae_val = metrics.get("mae") if metrics else None
        if (
            mse_val is not None
            and mae_val is not None
            and np.isfinite(mse_val)
            and np.isfinite(mae_val)
        ):
            row["mse"] = f"{float(mse_val):.8f}"
            row["mae"] = f"{float(mae_val):.8f}"
            row["status"] = "ok"
        else:
            # Non-finite metrics mean training diverged (see the AMP note in
            # AGENTS.md); record it as a result, do not retry blindly.
            row["status"] = "no_metrics"

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            row["status"] = "oom"
            print(f"{tag} OOM")
        else:
            msg = str(e).replace("\n", " ")[:160]
            row["status"] = f"error:RuntimeError:{msg}"
            print(f"{tag} RuntimeError: {e}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        row["status"] = f"error:{type(e).__name__}:{msg}"
        print(f"{tag} Exception: {e}")
        traceback.print_exc()
    finally:
        row["elapsed_s"] = f"{time.time() - t0:.1f}"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clean the sweep's disk footprint by default: both the best
        # checkpoint (already reloaded for the final evaluation inside
        # train()) and the per-run results.json are deleted; the TSV row is
        # the experiment's durable record. --keep-checkpoint opts out for
        # debugging a single cell.
        if not opts.keep_checkpoint:
            cleanup_cell_artifacts(cfg["experiment"]["name"])

    print(
        f"{tag} Done: status={row['status']}, mse={row.get('mse') or 'N/A'}, "
        f"mae={row.get('mae') or 'N/A'}, elapsed={row['elapsed_s']}s"
    )

    append_txt_row(row, dataset)


if __name__ == "__main__":
    main()
