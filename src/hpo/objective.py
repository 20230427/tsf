"""Optuna objective: one trial = one (or more) full DD-Mamba training run(s).

Ported from ``references/pamba/src/hpo/objective.py``. Key adaptations:

* the Pamba ``Exp_Long_Term_Forecast(args)`` experiment class is replaced by
  this repo's programmatic entry point ``src.train.train(cfg, evaluate_test=...)``
  returning a metrics dict directly (no metrics-file fallback chain needed);
* the argparse ``Namespace`` assembly becomes ``build_trial_cfg`` — dotted
  overrides layered onto the resolved dataset config;
* per Pamba protocol (and the user's explicit choice) the default metric is
  the **test-set MSE** of the trial's best-validation checkpoint.
  ``--metric val_loss`` is the leakage-safe escape hatch: it trains with
  ``evaluate_test=False`` and optimizes ``best_val_loss`` instead, reusing the
  guard ``run_selection_protocol.eval_candidate`` relies on.

Disk hygiene mirrors ``scripts/hyperparam/run_sensitivity.py``: the Optuna DB
is each trial's durable record, so a trial's checkpoint and per-run results
JSON are deleted once its metrics are captured (``--keep-checkpoint``
retains them). Failed trials (error, runtime OOM, leak-guard) are cleaned
the same way in the objective's ``finally`` block, so an abandoned trial
never leaks its half-written artifacts; leftovers from hard-killed
processes are reclaimed by ``scripts/hpo/hpo_clean.py``.
"""
from __future__ import annotations

import copy
import gc
import os
import traceback

import numpy as np
import optuna
import torch

from src.train import train
from src.utils import config_sha256
from src.hpo.gpu_budget import (
    GPUBudget,
    TOTAL_GPU_MEMORY_MB,
    MultiGPUBudget,
    NoopBudget,
)
from src.hpo.memory_estimate import estimate_ddmamba_memory_mb
from src.hpo.search_space import (
    build_trial_cfg,
    normalize_conditions,
    resolve_space,
    suggest_params,
)


class HpoLeakageError(RuntimeError):
    """A validation-only trial unexpectedly produced test metrics."""


def _remove_paths(paths) -> None:
    """Best-effort unlink; missing/unreadable files are silently ignored."""
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _cleanup_trial_artifacts(cfg: dict, out: dict, keep_checkpoint: bool) -> None:
    """Delete a finished trial's checkpoint and results JSON (disk hygiene)."""
    if keep_checkpoint:
        return
    paths = []
    checkpoint = out.get("checkpoint")
    if checkpoint:
        paths.append(checkpoint)
    name = cfg.get("experiment", {}).get("name", "")
    checkpoint_dir = cfg.get("experiment", {}).get("checkpoint_dir", "checkpoints")
    if name:
        paths.append(os.path.join(checkpoint_dir, f"{name}_results.json"))
    _remove_paths(paths)


def _trial_artifact_paths(base_cfg: dict, dataset: str, pred_len: int,
                          trial_number: int, seeds: int) -> list[str]:
    """Every path a trial may have created, across all seed offsets.

    Trial names are deterministic (``{ds}_pl{pl}_hpo_t{n}_s{i}``), so the
    cleanup in the objective's ``finally`` block does not need the configs
    that were actually built before a failure.
    """
    checkpoint_dir = base_cfg.get("experiment", {}).get(
        "checkpoint_dir", "checkpoints")
    paths = []
    for offset in range(max(1, int(seeds))):
        name = f"{dataset}_pl{pred_len}_hpo_t{trial_number}_s{offset}"
        paths.append(os.path.join(checkpoint_dir, f"{name}_best.pt"))
        paths.append(os.path.join(checkpoint_dir, f"{name}_results.json"))
    return paths


def create_objective(
    base_cfg: dict,
    spec: dict,
    *,
    metric: str = "mse",
    gpu_ids: list[int] | None = None,
    dedicated_gpu: bool = False,
    stop_check=None,
    seeds: int = 1,
    keep_checkpoint: bool = False,
    n_channels: int | None = None,
    epochs: int | None = None,
):
    """Build the Optuna objective for one (dataset, pred_len) study.

    ``base_cfg`` must already carry the target ``data.pred_len``. ``spec`` is
    the loaded ``configs/hpo_search_spaces.yaml`` document.

    Trial training always runs with ``train.log_style='pamba'`` (Pamba's
    compact per-epoch lines, no tqdm bar): study logs are redirected to
    files, where a progress bar would emit one line per batch. The caller's
    config dict is deep-copied first, never mutated.
    """
    dataset = str(base_cfg["experiment"]["name"])
    pred_len = int(base_cfg["data"]["pred_len"])
    base_seed = int(base_cfg["experiment"]["seed"])
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg.setdefault("train", {})["log_style"] = "pamba"
    space, fixed = resolve_space(spec, dataset)
    forced_defaults = dict(spec.get("forced_defaults") or {})
    seeds = max(1, int(seeds))
    if metric not in ("mse", "mae", "val_loss"):
        raise ValueError(f"unknown metric {metric!r} (use mse, mae, val_loss)")
    evaluate_test = metric != "val_loss"

    if dedicated_gpu:
        budget = NoopBudget()
    elif gpu_ids and len(gpu_ids) > 1:
        budget = MultiGPUBudget(gpu_ids)
    else:
        budget = GPUBudget()

    def objective(trial) -> float:
        trial_id = f"trial_{trial.number}"
        if stop_check and stop_check():
            raise optuna.TrialPruned("stop requested")

        try:
            params = suggest_params(trial, space)
            notes = normalize_conditions(params, space)
            if notes:
                trial.set_user_attr("normalized_conditions", notes)

            first_cfg = build_trial_cfg(
                base_cfg, fixed, forced_defaults, params,
                # The name must carry pred_len: parallel per-horizon studies
                # share checkpoints/<name>_best.pt, and trial numbers restart
                # from 0 in every study, so without it concurrent studies
                # clobber each other's checkpoints (observed as state_dict
                # size-mismatch crashes in the 2026-09-06 electricity sweep).
                name=f"{dataset}_pl{pred_len}_hpo_t{trial.number}_s0",
                seed=base_seed, epochs=epochs,
            )
            est_mb = estimate_ddmamba_memory_mb(first_cfg, n_channels)
            trial.set_user_attr("est_mb", round(float(est_mb), 1))
            if est_mb > TOTAL_GPU_MEMORY_MB:
                print(f"[OOM-skip] trial {trial.number}: estimated {est_mb:.0f} MB "
                      f"> limit {TOTAL_GPU_MEMORY_MB} MB")
                trial.set_user_attr("status", "skipped_over_budget")
                return float("inf")

            acquired = budget.acquire(trial_id, est_mb, timeout=1800)
            if not acquired:
                trial.set_user_attr("status", "timeout_waiting_gpu")
                raise optuna.TrialPruned("timeout_waiting_gpu")

            try:
                values, last_out, last_cfg = [], None, None
                for offset in range(seeds):
                    cfg = build_trial_cfg(
                        base_cfg, fixed, forced_defaults, params,
                        name=f"{dataset}_pl{pred_len}_hpo_t{trial.number}_s{offset}",
                        seed=base_seed + offset, epochs=epochs,
                    )
                    out = train(cfg, evaluate_test=evaluate_test)
                    if not evaluate_test and "test_metrics" in out:
                        raise HpoLeakageError(
                            "validation-only HPO trial returned test_metrics; "
                            "refusing to continue a potentially test-informed "
                            "selection"
                        )
                    if evaluate_test:
                        values.append(float(out["test_metrics"][metric]))
                    else:
                        values.append(float(out["best_val_loss"]))
                    _cleanup_trial_artifacts(cfg, out, keep_checkpoint)
                    last_out, last_cfg = out, cfg

                value = float(np.mean(values))
                trial.set_user_attr("status", "ok")
                trial.set_user_attr("seed", base_seed)
                trial.set_user_attr("seeds_used", seeds)
                if evaluate_test:
                    trial.set_user_attr(
                        "mse", float(last_out["test_metrics"]["mse"]))
                    trial.set_user_attr(
                        "mae", float(last_out["test_metrics"]["mae"]))
                trial.set_user_attr("val_loss", float(last_out["best_val_loss"]))
                trial.set_user_attr("param_count", last_out.get("param_count"))
                trial.set_user_attr("best_epoch", last_out.get("best_epoch"))
                trial.set_user_attr("config_sha256", config_sha256(last_cfg))
                return value

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] trial {trial.number}: CUDA out of memory")
                    trial.set_user_attr("status", "OOM")
                    if dedicated_gpu:
                        return float("inf")
                    raise optuna.TrialPruned("OOM_contention")
                raise

        except (HpoLeakageError, optuna.TrialPruned):
            raise
        except Exception as e:
            trial.set_user_attr("status", "error")
            trial.set_user_attr("error_trace", traceback.format_exc())
            print(f"[ERROR] trial {trial.number}: {type(e).__name__}: {e}")
            raise optuna.TrialPruned(f"error: {type(e).__name__}: {e}")
        finally:
            # Disk hygiene covers failed trials too: an error / runtime OOM /
            # leak-guard path abandons the trial, so whatever artifacts it
            # already wrote are removed here (a successful trial has already
            # cleaned up per seed offset; the re-remove is a no-op).
            if not keep_checkpoint:
                _remove_paths(_trial_artifact_paths(
                    base_cfg, dataset, pred_len, trial.number, seeds))
            budget.release(trial_id)
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # Expose for tests without reaching into the closure.
    objective.space = space  # type: ignore[attr-defined]
    objective.fixed = fixed  # type: ignore[attr-defined]
    return objective
