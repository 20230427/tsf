"""Optuna study management: SQLite storage, TPE sampler, result export.

Ported from ``references/pamba/src/hpo/study_manager.py``. Studies live under
``output/hpo/optuna`` (one SQLite file per dataset x pred_len). The best-params
JSON follows the Pamba export format exactly: a single record for the **best
test-MSE trial** (selected by the ``mse`` user attribute regardless of the
optimization metric) with a complete, ready-to-apply parameter set — no
non-SOTA trial records. Full per-trial history remains available in the CSV
export and the SQLite database.
"""
from __future__ import annotations

import math
import os

import optuna
from optuna.samplers import TPESampler

from src.utils import atomic_write_json

OUTPUT_HPO_DIR = os.path.join("output", "hpo")
STORAGE_DIR = os.path.join(OUTPUT_HPO_DIR, "optuna")
CSV_DIR = os.path.join(OUTPUT_HPO_DIR, "csv")
STORAGE_TEMPLATE = "sqlite:///" + os.path.join(STORAGE_DIR, "{}.db")

BEST_PARAMS_SCHEMA = "ddmamba-hpo-best-v2"

# Pamba-style deterministic ordering of the exported user attributes: the
# headline test metrics first, then reproducibility metadata.
_USER_ATTRS_ORDER = ("mse", "mae", "val_loss", "seed", "param_count",
                     "best_epoch", "est_mb", "config_sha256", "status",
                     "seeds_used")
_USER_ATTRS_EXCLUDE = {"error_trace", "normalized_conditions"}


def study_name(dataset: str, pred_len: int) -> str:
    return f"DDMamba_{dataset}_pl{pred_len}"


def parse_study_name(name: str) -> tuple[str, int]:
    rest = name[len("DDMamba_"):] if name.startswith("DDMamba_") else name
    idx = rest.rfind("_pl")
    if idx <= 0:
        return rest, 0
    try:
        return rest[:idx], int(rest[idx + 3:])
    except ValueError:
        return rest, 0


def _storage_url(name: str) -> str:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    return STORAGE_TEMPLATE.format(name)


def create_study(name: str, *, direction: str = "minimize",
                 n_startup_trials: int = 10) -> optuna.Study:
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        seed=2023,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    return optuna.create_study(
        study_name=name,
        storage=_storage_url(name),
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )


def load_study(name: str) -> optuna.Study:
    return optuna.load_study(study_name=name, storage=_storage_url(name))


def _ordered(src: dict, order: tuple) -> dict:
    result = {}
    for key in order:
        if key in src:
            result[key] = src[key]
    for key, value in src.items():
        if key not in result:
            result[key] = value
    return result


def _select_best_trial(study) -> tuple:
    """Return (trial, metric, value) for the best **test-MSE** trial.

    Pamba protocol: the export always reports the best test-MSE result, even
    when the study was optimized for mae or val_loss. Trials without a finite
    ``mse`` user attribute (validation-only studies) fall back to Optuna's
    ``study.best_trial``.
    """
    best_mse, best = None, None
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        mse = trial.user_attrs.get("mse")
        if mse is None or not math.isfinite(mse):
            continue
        if best_mse is None or mse < best_mse:
            best_mse, best = mse, trial
    if best is not None:
        return best, "mse", float(best_mse)
    return study.best_trial, "val_loss", float(study.best_value)


def export_results(name: str, fill: dict | None = None, space: dict | None = None):
    """Write the trials CSV and the Pamba-format best-params JSON.

    ``fill`` carries the spec's ``forced_defaults`` + dataset ``fixed``
    assignments (and any other non-searched defaults) so ``best_params`` is a
    *complete* parameter set, mirroring Pamba's ``_DEFAULT_FILL`` +
    ``override_fixed`` merge. ``space`` (the resolved search space) expands
    ``forward_to`` declarations — e.g. the shared ``model.dropout`` fans out to
    ``time_dropout``/``freq_dropout``/``head_dropout`` — so the exported set is
    directly applicable. Suggested trial parameters always win over ``fill``
    for overlapping keys.
    """
    import pandas as pd

    study = load_study(name)
    os.makedirs(CSV_DIR, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(CSV_DIR, f"{name}_results.csv"), index=False)

    try:
        best, metric, value = _select_best_trial(study)
    except ValueError:
        print(f"[export] no completed trials in {name}; CSV only")
        return df

    if not math.isfinite(value):
        # Every completed trial is inf (e.g. all-OOM anchor runs). A
        # best-params record would be meaningless, and json.dump(allow_nan
        # =False) refuses inf outright -- export the CSV and stop there.
        print(f"[export] no finite completed trials in {name} "
              f"(best {metric}={value}); CSV only")
        return df

    from src.hpo.search_space import expand_forwarded_params

    best_params = expand_forwarded_params({**(fill or {}), **dict(best.params)},
                                          space)
    seed = best.user_attrs.get("seed")
    if seed is not None:
        best_params["experiment.seed"] = int(seed)

    user_attrs = _ordered(
        {key: (None if isinstance(value, float) and not math.isfinite(value)
               else value)
         for key, value in best.user_attrs.items()
         if key not in _USER_ATTRS_EXCLUDE},
        _USER_ATTRS_ORDER,
    )

    payload = {
        "study_name": name,
        "version": BEST_PARAMS_SCHEMA,
        "metric": metric,
        "best_value": value,
        "best_params": best_params,  # dotted overrides, ready to apply
        "best_trial_number": int(best.number),
        "best_trial_user_attrs": user_attrs,
    }
    os.makedirs(OUTPUT_HPO_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_HPO_DIR, f"{name}_best_params.json")
    atomic_write_json(out_path, payload)
    print(f"[export] {out_path}")
    return df


def make_best_callback(name: str, fill: dict | None = None,
                       space: dict | None = None):
    """Re-export (replace) the best test-MSE record on every new best."""

    def _callback(study, trial):
        try:
            best, _, _ = _select_best_trial(study)
            if best.number == trial.number:
                print(f" * new best test MSE record (trial #{trial.number}) "
                      f"-- results exported")
                export_results(name, fill=fill, space=space)
        except ValueError:
            pass

    return _callback
