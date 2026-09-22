"""Declarative Optuna search space for DD-Mamba HPO.

Ported from ``references/pamba/src/hpo/search_space.py``. The space lives in
``configs/hpo_search_spaces.yaml`` (schema ``ddmamba-hpo-search-v1``) so it can
be edited without touching code. Semantics follow Pamba, expressed with this
repo's dotted config keys:

* ``base.search``  -- parameters searched for every dataset.
* ``base.fixed``   -- dotted assignments locked for every dataset (a fixed key
  is removed from the search space).
* ``datasets.<name>.search`` -- per-dataset distribution overrides.
* ``datasets.<name>.fixed``  -- per-dataset locked assignments.
* ``forced_defaults`` -- dotted assignments applied to every trial *before*
  the sampled parameters (protocol defaults, e.g. ``model.time_encoder:
  mamba``) and to the anchor trial; searched keys still override them.

This is a single-phase design: every key in the resolved space is sampled
jointly by TPE (the Pamba Phase-1/Phase-2 split was deliberately dropped).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

SCHEMA_VERSION = "ddmamba-hpo-search-v1"
DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "hpo_search_spaces.yaml"

_SPEC_TYPES = {"categorical", "int", "float", "loguniform", "uniform"}
_VALID_SECTIONS = {"experiment", "data", "model", "train"}

# Multivariate parameter groups. Suggestion always happens in this fixed,
# deterministic order (architecture switches -> model sizing/regularization ->
# training) so sampled parameter dicts and Optuna logs are reproducible and
# readable, mirroring the grouped-suggestion format of the reference HPO
# design. ``TPESampler(multivariate=True, group=True)`` models the joint
# dependencies of the whole space.
ARCHITECTURE_GROUP = (
    # Component switches: which architecture blocks exist at all.
    "model.time_encoder",        # causal Mamba vs MLP over the time axis
    "model.channel_mixer_layers",  # variate BiMamba depth (0 = channel-independent)
    "model.mixer_placement",     # one mixer per branch vs weight-tied across branches
    "model.freq_backbone",       # FITS spectral linear anchor on/off
    "model.use_revin",           # instance normalization on/off
)

MODEL_GROUP = (
    # Backbone sizing and regularization of the surviving components.
    "model.d_model",             # shared hidden width of both branches
    "model.mamba_layers",        # stacked Mamba blocks per encoder
    "model.mamba_d_state",       # SSM state dimension N
    "model.freq_hidden",         # frequency-encoder width
    "model.time_kernel_size",    # moving-average window for trend extraction
    "model.freq_sparsity",       # low-pass cutoff (fraction of highest bins dropped)
    "model.dropout",             # shared dropout, forwarded to time/freq/head
)

TRAIN_GROUP = (
    # Training-side hyperparameters.
    "train.lr",
)

PARAM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("architecture", ARCHITECTURE_GROUP),
    ("model", MODEL_GROUP),
    ("training", TRAIN_GROUP),
)


def _suggest_order(space: dict) -> list[str]:
    ordered = [key for _, group in PARAM_GROUPS for key in group if key in space]
    ordered += [key for key in space if key not in ordered]
    return ordered


def suggest_params(trial, space: dict) -> dict:
    """Sample every key of the space, grouped and in the fixed order above."""
    params: dict = {}
    for name in _suggest_order(space):
        params[name] = suggest_one(trial, name, space[name])
    return params


def load_spec(path: str | Path | None = None) -> dict:
    """Load and validate the YAML search-space specification."""
    path = Path(path) if path is not None else DEFAULT_SPEC_PATH
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"HPO search space {path} is not a mapping")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported HPO search-space schema: {spec.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION!r})"
        )
    validate_spec(spec)
    return spec


def _validate_param(key: str, spec: dict) -> None:
    if "." not in key or key.split(".", 1)[0] not in _VALID_SECTIONS:
        raise ValueError(f"search key {key!r} must be dotted with a known section")
    if not isinstance(spec, dict) or spec.get("type") not in _SPEC_TYPES:
        raise ValueError(f"search entry {key!r} lacks a valid type: {spec!r}")
    t = spec["type"]
    if t == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"categorical {key!r} requires a non-empty choices list")
    else:
        low, high = spec.get("low"), spec.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ValueError(f"{t} {key!r} requires numeric low/high")
        if not low < high:
            raise ValueError(f"{t} {key!r} requires low < high (got {low} >= {high})")
        if t in ("loguniform",) and low <= 0:
            raise ValueError(f"loguniform {key!r} requires low > 0")
    forward = spec.get("forward_to")
    if forward is not None:
        if (not isinstance(forward, list) or not forward
                or not all(isinstance(target, str) and "." in target
                           for target in forward)):
            raise ValueError(
                f"forward_to of {key!r} must be a non-empty list of dotted keys"
            )


def validate_spec(spec: dict) -> None:
    base = spec.get("base") or {}
    for key, entry in (base.get("search") or {}).items():
        _validate_param(key, entry)
    for key in base.get("fixed") or {}:
        if "." not in key:
            raise ValueError(f"fixed key {key!r} must be dotted")
    for name, entry in (spec.get("datasets") or {}).items():
        for key, param in (entry.get("search") or {}).items():
            _validate_param(key, param)
        for key in entry.get("fixed") or {}:
            if "." not in key:
                raise ValueError(f"fixed key {key!r} in dataset {name!r} must be dotted")
    for key in spec.get("forced_defaults") or {}:
        if "." not in key:
            raise ValueError(f"forced_defaults key {key!r} must be dotted")


def _dataset_section(spec: dict, dataset: str) -> dict:
    return (spec.get("datasets") or {}).get(dataset) or {}


def resolve_space(spec: dict, dataset: str) -> tuple[dict, dict]:
    """Return (search_space, fixed_assignments) for one dataset.

    The dataset section's ``search`` entries replace the base distributions
    key-by-key; dataset ``fixed`` keys are removed from the space (a key may
    never be both searched and fixed).
    """
    base = spec.get("base") or {}
    space = dict(base.get("search") or {})
    fixed = dict(base.get("fixed") or {})
    section = _dataset_section(spec, dataset)
    space.update(section.get("search") or {})
    fixed.update(section.get("fixed") or {})
    overlap = sorted(set(space) & set(fixed))
    if overlap:
        raise ValueError(
            f"keys are both searched and fixed for dataset {dataset!r}: {overlap}"
        )
    for key in fixed:
        space.pop(key, None)
    forwarded = [target for entry in space.values()
                 if isinstance(entry, dict)
                 for target in entry.get("forward_to") or []]
    clash = sorted(set(forwarded) & set(space))
    if clash:
        raise ValueError(
            f"forward_to targets are themselves searched for dataset "
            f"{dataset!r}: {clash}"
        )
    return space, fixed


def get_dotted(cfg: dict, key: str, default=None):
    section, _, field = key.partition(".")
    return cfg.get(section, {}).get(field, default)


def set_dotted(cfg: dict, key: str, value) -> None:
    section, _, field = key.partition(".")
    cfg.setdefault(section, {})[field] = value


def suggest_one(trial, name: str, spec: dict):
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if t == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if t == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    raise ValueError(f"Unknown param type {t!r} for {name!r}")


def expand_forwarded_params(params: dict, space: dict | None) -> dict:
    """Return a copy of ``params`` with every ``forward_to`` target filled in.

    A searched key may declare ``forward_to: [<dotted key>, ...]``: its sampled
    value is transparently written to each target (e.g. one shared
    ``model.dropout`` fans out to ``time_dropout``/``freq_dropout``/
    ``head_dropout``). Pure function — the input dict is not modified.
    """
    expanded = dict(params)
    for key, entry in (space or {}).items():
        targets = entry.get("forward_to") if isinstance(entry, dict) else None
        if targets and key in expanded:
            for target in targets:
                expanded[target] = expanded[key]
    return expanded


def normalize_conditions(params: dict, space: dict | None = None) -> list[str]:
    """Apply post-suggest conditional constraints and forwards, in place.

    * ``forward_to`` expansion: every searched key declaring ``forward_to``
      writes its sampled value to each target key (recorded as a note).
    * ``model.mixer_placement`` is meaningless when
      ``model.channel_mixer_layers == 0`` (no mixer exists); pin it to ``both``
      so TPE does not waste probability mass on a dead axis.

    Returns the list of normalizations applied (recorded as a trial user
    attribute).
    """
    notes = []
    for key, entry in (space or {}).items():
        targets = entry.get("forward_to") if isinstance(entry, dict) else None
        if targets and key in params:
            for target in targets:
                params[target] = params[key]
            notes.append(f"{key}={params[key]} -> {', '.join(targets)}")
    if params.get("model.channel_mixer_layers") == 0 and "model.mixer_placement" in params:
        params["model.mixer_placement"] = "both"
        notes.append("mixer_placement=both (channel_mixer_layers=0)")
    return notes


def _nearest(value, choices):
    numeric = all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices)
    if numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(choices, key=lambda c: abs(c - value))
    return choices[0]


def snap_to_space(value, spec: dict):
    """Snap an arbitrary config value into the spec's support.

    Used when enqueuing the anchor trial from the dataset's tuned config: a
    tuned value outside the declared grid (e.g. ``freq_sparsity: 0.4`` against
    choices ``[0.0, 0.1, 0.2, 0.3]``) moves to the nearest grid point.
    """
    t = spec["type"]
    if t == "categorical":
        choices = spec["choices"]
        if value in choices:
            return value
        return _nearest(value, choices)
    low, high = spec["low"], spec["high"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return low
    value = min(max(value, low), high)
    if t == "int":
        return int(round(value))
    return float(value)


def anchor_params(cfg: dict, spec: dict, dataset: str | None = None) -> dict:
    """Anchor trial parameters: the tuned config (plus protocol defaults).

    Reads the current value of every searched key from ``cfg`` after applying
    ``forced_defaults``, then snaps each value into the declared support so
    ``study.enqueue_trial`` receives in-grid parameters. For keys declaring
    ``forward_to`` (e.g. the shared ``model.dropout``), the anchor value is
    derived from the first target that exists in the tuned config.
    """
    space, _ = resolve_space(spec, dataset or str(cfg.get("experiment", {}).get("name", "")))
    forced = dict(spec.get("forced_defaults") or {})
    anchor = {}
    for key, param_spec in space.items():
        value = forced.get(key, get_dotted(cfg, key))
        if value is None and param_spec.get("forward_to"):
            for target in param_spec["forward_to"]:
                candidate = get_dotted(cfg, target)
                if candidate is not None:
                    value = candidate
                    break
        anchor[key] = snap_to_space(value, param_spec)
    return anchor


def build_trial_cfg(base_cfg: dict, fixed: dict, forced_defaults: dict, params: dict,
                    *, name: str, seed: int, epochs: int | None = None) -> dict:
    """Assemble one trial's resolved config.

    Precedence: base config < forced_defaults < sampled params < fixed
    (dataset-locked decisions always win).
    """
    cfg = copy.deepcopy(base_cfg)
    for key, value in {**forced_defaults, **params}.items():
        set_dotted(cfg, key, value)
    for key, value in fixed.items():
        set_dotted(cfg, key, value)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = name
    cfg["experiment"]["seed"] = int(seed)
    if epochs:
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = int(epochs)
    return cfg
