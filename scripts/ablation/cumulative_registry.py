#!/usr/bin/env python
"""Single source of truth for the cumulative (progressive step-down) ablation.

Design mirrors the Pamba progressive step-down table
(``references/pamba/src/ablation/registry.py``): a fixed, ordered list of
component-removal steps whose overrides accumulate -- step k applies its own
overrides on top of steps 1..k-1, starting from the per-dataset tuned config
(``configs/<dataset>.yaml``, the "Full" baseline). The chain ends at a bare
time-domain Mamba forecaster, the counterpart of Pamba's "Mamba-Patch"
terminal step.

A step is a structural NO-OP for a dataset when every key it touches is
already at its target value in the (k-1)-configuration (per-dataset recipes
often disable a component up front, e.g. channel_mixer_layers: 0 on ETT).
No-op cells are skipped at runtime (``[skip-noop]``, no GPU time) and are
rendered in the xlsx as "value copied from the previous chain variant +
yellow fill" (Pamba's two-layer no-op treatment, adapted: the copy source is
the previous variant, which for a cumulative chain is bit-identical by
construction).

Consumers (keep in lockstep -- this module is the only place the chain is
defined):

  - ``scripts/ablation/gen_cells.py``              cell matrix + no-op flags
  - ``scripts/ablation/ablation_common.sh``        runtime skip of no-op cells
  - ``scripts/ablation/collect_cumulative.py``     per-cell config verification
  - ``scripts/ablation/update_cumulative_table.py``  xlsx no-op blocks

No model/training code is modified: every step expands to plain dotted
config overrides understood by ``python -m src.train`` (``--model.fusion
time_only`` ...), so the official training entry point stays the sole runner.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# The cumulative chain (ordered). Each entry:
#   name      -- run-name token and result key
#   display   -- human label for tables
#   overrides -- dotted config overrides; step k inherits 1..k-1
# The removal order peels regularizers/encapsulation first and core structure
# last: fusion gating -> frequency-branch enhancements -> cross-variate
# mixing -> time-branch linear residual -> input normalization -> the whole
# frequency branch. Reordering the list changes the published table; do so
# only with a fresh sweep (cells are keyed by variant name, so stale results
# for a renamed variant are ignored rather than mixed in).
# ---------------------------------------------------------------------------
CUMULATIVE_STEPS = [
    {
        "name": "full",
        "display": "Full DD-Mamba",
        "overrides": {},
    },
    {
        "name": "no_gated_fusion",
        "display": "w/o gated fusion",
        "overrides": {"model.fusion": "sum"},
    },
    {
        "name": "no_freq_enh",
        "display": "w/o freq enhancements",
        "overrides": {"model.freq_backbone": "none", "model.freq_sparsity": 0.0},
    },
    {
        "name": "no_mixer",
        "display": "w/o variate mixer",
        "overrides": {"model.channel_mixer_layers": 0},
    },
    {
        "name": "no_lin_backbone",
        "display": "w/o linear backbone",
        "overrides": {"model.time_linear_backbone": False},
    },
    {
        "name": "no_revin",
        "display": "w/o RevIN",
        "overrides": {"model.use_revin": False},
    },
    {
        "name": "time_only_mamba",
        "display": "Time-only Mamba",
        "overrides": {"model.fusion": "time_only"},
    },
]

VARIANT_NAMES = [step["name"] for step in CUMULATIVE_STEPS]
VARIANT_DISPLAYS = {step["name"]: step["display"] for step in CUMULATIVE_STEPS}

# build_model (src/models/dual_domain_model.py) defaults for every key the
# chain touches. Configs may omit a key; the effective value then comes from
# here, so no-op detection must consult these too.
MODEL_DEFAULTS = {
    "model.fusion": "gated",
    "model.freq_backbone": "none",
    "model.freq_sparsity": 0.0,
    "model.channel_mixer_layers": 1,
    "model.time_linear_backbone": True,
    "model.use_revin": True,
}

# ---------------------------------------------------------------------------
# Dataset matrix. The sweep runs in two waves:
#   wave 1 (DONE, 168 cells, 2026-09-04): ETTh1, ETTm2, weather, electricity,
#     solar, exchange_rate, PEMS04 -- commented below, kept for traceability.
#   wave 2 (current default): ETTh2, ETTm1, PEMS08, illness -- the remaining
#     datasets of the 11-dataset ladder table.
# DEFAULT_DATASETS is what a bare `sbatch scripts/batch/batch_cumulative_ablation.sh`
# runs; completed wave-1 stems stay in OPTIONAL_DATASETS so a full rerun via
# `--datasets ETTh1 ...` still validates. Horizon convention follows the repo's
# HPO/robustness runners: standard 96/192/336/720, PEMS 12/24/48/96, Illness
# 24/36/48/60 (weekly data, community convention).
# ---------------------------------------------------------------------------
STANDARD_PRED_LENS = [96, 192, 336, 720]
PEMS_PRED_LENS = [12, 24, 48, 96]
ILLNESS_PRED_LENS = [24, 36, 48, 60]

# Wave-1 datasets (already collected into output/cumulative_ablation/):
# DEFAULT_DATASETS = [
#     "ETTh1", "ETTm2", "weather", "electricity", "solar",
#     "exchange_rate", "PEMS04",
# ]
DEFAULT_DATASETS = [
    "ETTh2", "ETTm1", "PEMS08", "illness",
]
OPTIONAL_DATASETS = [
    "ETTh1", "ETTm2", "weather", "electricity", "solar",
    "exchange_rate", "PEMS04", "traffic", "PEMS03", "PEMS07",
]


def config_path_for(dataset: str) -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / f"{dataset}.yaml"


def pred_lens_for(dataset: str) -> list[int]:
    if dataset.startswith("PEMS"):
        return list(PEMS_PRED_LENS)
    if dataset == "illness":
        return list(ILLNESS_PRED_LENS)
    return list(STANDARD_PRED_LENS)


def variant_index(name: str) -> int:
    try:
        return VARIANT_NAMES.index(name)
    except ValueError:
        raise ValueError(
            f"unknown cumulative variant {name!r}; valid: {', '.join(VARIANT_NAMES)}"
        ) from None


def resolve_overrides(step_index: int) -> dict[str, object]:
    """Merged dotted overrides for steps 0..step_index inclusive."""
    if not 0 <= step_index < len(CUMULATIVE_STEPS):
        raise IndexError(f"step index {step_index} out of range")
    merged: dict[str, object] = {}
    for step in CUMULATIVE_STEPS[: step_index + 1]:
        merged.update(step["overrides"])
    return merged


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def cumulative_flags(step_index: int) -> list[str]:
    """Flat CLI override list for `python -m src.train`, cumulative to step."""
    flags: list[str] = []
    for key, value in resolve_overrides(step_index).items():
        flags.extend([f"--{key}", _format_value(value)])
    return flags


def effective_value(cfg: dict, dotted_key: str):
    """Typed value of dotted_key in cfg, falling back to MODEL_DEFAULTS.

    Mirrors src.utils.apply_overrides semantics for dotted keys (section
    created on demand); the fallback matters only for no-op detection since
    configs may omit keys that build_model defaults.
    """
    section, _, field = dotted_key.partition(".")
    sub = cfg.get(section, {})
    if isinstance(sub, dict) and field in sub:
        return sub[field]
    return MODEL_DEFAULTS.get(dotted_key)


def effective_config(config_path, step_index: int, seed: int | None = None) -> dict:
    """Load configs/<ds>.yaml and apply the cumulative overrides in-memory."""
    cfg = copy.deepcopy(load_config(str(config_path)))
    for key, value in resolve_overrides(step_index).items():
        section, _, field = key.partition(".")
        cfg.setdefault(section, {})[field] = value
    if seed is not None:
        cfg.setdefault("experiment", {})["seed"] = int(seed)
    return cfg


def _values_equal(a, b) -> bool:
    """Strict equality that never conflates bool with int (False == 0 trap)."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def is_noop(config_path, step_index: int) -> bool:
    """True when step ``step_index`` changes nothing vs the previous step.

    The comparison base is the (step_index-1)-configuration (base config for
    step 0), i.e. the previous link of the chain; only the step's own marginal
    keys matter. Keys already set to the target by an earlier step or by the
    dataset recipe make the step a no-op iff ALL its marginal keys match.
    Step 0 ("full", the baseline) is never a no-op: it has no overrides but
    must always be trained -- it is the reference every delta is measured
    against.
    """
    if step_index <= 0:
        return False
    base = effective_config(config_path, step_index - 1)
    for key, target in CUMULATIVE_STEPS[step_index]["overrides"].items():
        if not _values_equal(effective_value(base, key), target):
            return False
    return True


def noop_variants_for_dataset(config_path) -> dict[str, str]:
    """Map variant name -> previous non-noop variant name for the xlsx filler.

    The copy source skips over no-op variants: if both ``no_mixer`` and
    ``no_lin_backbone`` are no-ops, ``no_revin`` copies from ``no_freq_enh``
    (the latest variant that actually differed from the base config).
    """
    result: dict[str, str] = {}
    last_effective = VARIANT_NAMES[0]
    for idx in range(1, len(CUMULATIVE_STEPS)):
        name = VARIANT_NAMES[idx]
        if is_noop(config_path, idx):
            result[name] = last_effective
        else:
            last_effective = name
    return result


def dataset_seed(config_path) -> int:
    cfg = load_config(str(config_path))
    return int(cfg.get("experiment", {}).get("seed", 2024))


def run_name(dataset: str, variant: str, pred_len: int, seed: int) -> str:
    return f"{dataset}_cum_{variant}_h{pred_len}_s{seed}"
