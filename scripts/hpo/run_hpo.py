#!/usr/bin/env python3
"""Optuna HPO runner for DD-Mamba (ported from references/pamba scripts/hpo/hpo_tune.py).

Single-phase design: one TPE study per (dataset, pred_len) jointly searches
every key of ``configs/hpo_search_spaces.yaml``; the dataset's tuned config
(snapped into the grid) is enqueued as the anchor trial. Single-GPU mode runs
every study on ``--gpu`` with shared MB accounting; multi-GPU mode
(``--nodes > 1``) lets each (dataset, pred_len) study reserve a whole GPU from
a file-locked pool, so studies run in parallel across cards. Studies resume
automatically from their SQLite database under ``output/hpo/optuna``. The
best-params export follows the Pamba format: a single record for the best
test-MSE trial and its complete parameter set.

Examples
--------
    # Single GPU, one study, 50 trials:
    python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len 96 --n_trials 50

    # All standard horizons, detached with logs in log/hpo/:
    python scripts/hpo/run_hpo.py --config configs/ETTh1.yaml --pred_len all --bg

    # Four GPUs, studies spread across cards:
    python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len all --nodes 4 --gpu 0

    # Print the resolved space / anchor / memory estimate without training:
    python scripts/hpo/run_hpo.py --config configs/ETTh1.yaml --pred_len 96 --dry-run
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_PRED_LENS = [96, 192, 336, 720]

_stop_requested = False
_current_study = None


def _handle_sigterm(signum, frame):
    global _stop_requested, _current_study
    _stop_requested = True
    if _current_study is not None:
        _current_study.stop()


signal.signal(signal.SIGTERM, _handle_sigterm)


class ConsecutivePrunedStopper:
    """Stop a study after a run of consecutive pruned/inf trials (ported from Pamba)."""

    def __init__(self, max_consecutive: int = 5, max_inf: int = 10):
        self.max_consecutive = max_consecutive
        self.max_inf = max_inf

    def __call__(self, study, trial):
        import numpy as np
        import optuna

        pruned = 0
        infs = 0
        for t in reversed(study.trials):
            if t.state == optuna.trial.TrialState.PRUNED:
                pruned += 1
                infs = 0
            elif (t.state == optuna.trial.TrialState.COMPLETE
                  and t.value is not None and np.isinf(t.value)):
                infs += 1
            else:
                break
            if pruned >= self.max_consecutive:
                print(f"[STOP] {pruned} consecutive pruned trials -- stopping study")
                study.stop()
                break
            if infs >= self.max_inf:
                print(f"[STOP] {infs} consecutive Complete-inf trials -- stopping study")
                study.stop()
                break


def _probe_n_channels(cfg: dict) -> int:
    """Build train/val loaders once to learn the channel count (cached .pt)."""
    from src.data.data_loader import get_dataloaders

    probe = copy.deepcopy(cfg)
    probe.setdefault("experiment", {})
    probe["experiment"]["name"] = probe["experiment"].get("name", "hpo") + "_hpo_probe"
    probe.setdefault("train", {})
    probe["train"]["epochs"] = 1
    _, _, _, _, n_channels = get_dataloaders(probe, include_test=False)
    return int(n_channels)


def _resolve_pred_lens(arg: str) -> list[int]:
    if arg == "all":
        return list(DEFAULT_PRED_LENS)
    values = [int(v) for v in arg.replace(",", " ").split()]
    if not values:
        raise SystemExit("--pred_len produced no horizons")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="dataset configuration (configs/*.yaml)")
    parser.add_argument("--pred_len", default="all",
                        help="'all' (=96/192/336/720) or a space/comma list")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--metric", default="mse", choices=["mse", "mae", "val_loss"],
                        help="optimization target (default test MSE, Pamba protocol; "
                             "val_loss is the leakage-safe variant). The export always "
                             "reports the best test-MSE trial regardless of this choice")
    parser.add_argument("--gpu", type=int, default=0, help="first physical GPU index")
    parser.add_argument("--nodes", type=int, default=1,
                        help="number of GPUs (1=single, >1=multi-GPU mode)")
    parser.add_argument("--seeds", type=int, default=1, help="seeds per trial (averaged)")
    parser.add_argument("--epochs", type=int, default=0,
                        help="override train.epochs (0 = use the dataset config)")
    parser.add_argument("--timeout", type=int, default=None, help="per-study timeout (s)")
    parser.add_argument("--bg", action="store_true",
                        help="append output to log/hpo/<dataset>_<pl>_study.log")
    parser.add_argument("--search-space", default="configs/hpo_search_spaces.yaml")
    parser.add_argument("--n-startup", type=int, default=10,
                        help="TPE random startup trials")
    parser.add_argument("--keep-checkpoint", action="store_true",
                        help="retain per-trial checkpoints (default: delete after recording)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved space/anchor/estimate and exit")
    args, unknown = parser.parse_known_args()

    # --- GPU environment: set CUDA_VISIBLE_DEVICES BEFORE importing torch ----
    gpu_manager = None
    if args.nodes > 1:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        system_gpus = len(result.stdout.strip().split("\n"))
        physical_ids = list(range(args.gpu, args.gpu + args.nodes))
        if max(physical_ids) >= system_gpus:
            print(f"ERROR: requested GPUs {physical_ids} but system has {system_gpus}")
            return 1
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical_ids)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # --- heavy imports (torch et al.) after the env is pinned ----------------
    from src.utils import load_config, parse_overrides, apply_overrides, config_sha256
    from src.hpo.search_space import anchor_params, build_trial_cfg, load_spec, \
        resolve_space
    from src.hpo.memory_estimate import estimate_ddmamba_memory_mb
    from src.hpo.gpu_budget import MultiGPUBudget, TOTAL_GPU_MEMORY_MB
    from src.hpo.objective import create_objective
    from src.hpo.study_manager import (
        create_study,
        export_results,
        make_best_callback,
        study_name,
    )

    base_cfg = apply_overrides(load_config(args.config), parse_overrides(unknown))
    dataset = str(base_cfg["experiment"]["name"])
    spec = load_spec(args.search_space)
    epochs_override = args.epochs or None
    pred_lens = _resolve_pred_lens(args.pred_len)

    mode = (f"multi-GPU (physical {physical_ids}, runtime 0..{args.nodes - 1})"
            if args.nodes > 1
            else f"single GPU (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    print(f"[hpo] dataset={dataset} metric={args.metric} trials={args.n_trials} | {mode}")
    print(f"[hpo] search space: {args.search_space} "
          f"(sha256 {config_sha256(spec)[:12]})")

    n_channels = None

    for pred_len in pred_lens:
        if _stop_requested:
            break

        log_file = None
        if args.bg:
            os.makedirs("log/hpo", exist_ok=True)
            log_path = os.path.join("log", "hpo", f"{dataset}_{pred_len}_study.log")
            log_file = open(log_path, "a", buffering=1)
            sys.stdout = log_file
            sys.stderr = log_file

        reserved_gpu = None
        study_tag = f"{dataset}_pl{pred_len}"
        try:
            cfg = copy.deepcopy(base_cfg)
            cfg["data"]["pred_len"] = pred_len

            if n_channels is None:
                n_channels = _probe_n_channels(cfg)
                print(f"[hpo] n_channels={n_channels}")

            name = study_name(dataset, pred_len)
            space, fixed = resolve_space(spec, dataset)
            forced = dict(spec.get("forced_defaults") or {})
            anchor = anchor_params(cfg, spec, dataset)
            # Pamba-style complete-parameter fill for the best-params export:
            # protocol defaults + dataset-locked assignments. Suggested trial
            # parameters always win over ``fill`` for overlapping keys.
            fill = {**forced, **fixed}

            anchor_cfg = build_trial_cfg(
                cfg, fixed, forced, anchor,
                name=f"{dataset}_hpo_anchor_est", seed=int(cfg["experiment"]["seed"]),
                epochs=epochs_override,
            )
            est_anchor = estimate_ddmamba_memory_mb(anchor_cfg, n_channels)
            print(f"\n{'=' * 60}\n[hpo] study {name} | anchor est. {est_anchor:.0f} MB "
                  f"(GPU limit {TOTAL_GPU_MEMORY_MB} MB)")
            if args.dry_run:
                print("[hpo] resolved search space:")
                for key in sorted(space):
                    print(f"    {key}: {space[key]}")
                if forced:
                    print(f"[hpo] forced_defaults: {forced}")
                if fixed:
                    print(f"[hpo] dataset fixed: {fixed}")
                print("[hpo] anchor trial params (tuned config snapped to grid):")
                for key in sorted(anchor):
                    print(f"    {key}: {anchor[key]}")
                continue

            if args.nodes > 1:
                gpu_manager = MultiGPUBudget(list(range(args.nodes)))
                reserved_gpu = gpu_manager.reserve(study_tag, min(est_anchor * 1.2,
                                                                  TOTAL_GPU_MEMORY_MB),
                                                   timeout=7200)
                if reserved_gpu is None:
                    print(f"[hpo] SKIP {study_tag}: GPU reservation timeout")
                    continue
                print(f"[hpo] study reserved GPU {reserved_gpu}")

            study = create_study(name, n_startup_trials=args.n_startup)
            global _current_study
            _current_study = study

            if len(study.trials) == 0:
                study.enqueue_trial(anchor)
                print(f"[hpo] enqueued anchor trial: {anchor}")

            objective = create_objective(
                cfg, spec,
                metric=args.metric,
                gpu_ids=None,
                dedicated_gpu=True,  # GPU allocation is done at study level
                stop_check=lambda: _stop_requested,
                seeds=args.seeds,
                keep_checkpoint=args.keep_checkpoint,
                n_channels=n_channels,
                epochs=epochs_override,
            )

            finished = len([t for t in study.trials if t.state.is_finished()])
            remaining = max(0, args.n_trials - finished)
            print(f"[hpo] {finished} finished trials in DB; running {remaining} more")
            if remaining > 0:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    timeout=args.timeout,
                    callbacks=[make_best_callback(name, fill=fill, space=space),
                               ConsecutivePrunedStopper()],
                )

            try:
                best = study.best_trial
                print(f"\n[hpo] best trial #{best.number}: {study.best_value:.6f} "
                      f"({args.metric})")
                for key, value in sorted(best.params.items()):
                    print(f"    {key}: {value}")
            except ValueError:
                print("\n[hpo] no completed trials")

            export_results(name, fill=fill, space=space)
        except Exception as e:
            print(f"[ERROR] study {study_tag}: {e}")
            traceback.print_exc()
        finally:
            _current_study = None
            if reserved_gpu is not None and gpu_manager is not None:
                gpu_manager.unreserve(study_tag)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if log_file is not None:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                log_file.close()

    print("\n[hpo] all studies completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
