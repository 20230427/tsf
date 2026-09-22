#!/usr/bin/env python3
"""Pure-Python primitives for reproducible paired statistical audits.

The project has two generations of result files.  Newer files may record an
explicit seed on every run; legacy files only preserve run order.  This module
never turns a positional index into a seed identifier.  Legacy pairs therefore
carry ``seed_id=None`` and an explicit ``legacy_positional_order`` provenance
label in every machine-readable and human-readable report.

The inferential convention is always ``comparator - reference`` for MSE.  A
positive effect therefore means that the reference model has lower error.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "1.0"
SEED_KEYS = ("seed_id", "seed")


def _betacf(a: float, b: float, x: float) -> float:
    max_iterations, epsilon, floor = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c

        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    factor = math.exp(log_beta_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _betacf(a, b, x) / a
    return 1.0 - factor * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_statistic: float, df: int) -> float:
    """Return ``P(|T| >= |t|)`` without SciPy."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(t_statistic):
        return 0.0
    x = df / (df + t_statistic * t_statistic)
    return min(1.0, max(0.0, _betai(df / 2.0, 0.5, x)))


def student_t_critical_975(df: int) -> float:
    """Compute the two-sided 95% critical value by deterministic bisection."""
    if df <= 0:
        raise ValueError("df must be positive")
    low, high = 0.0, 1.0
    while student_t_two_sided_p(high, df) > 0.05:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if student_t_two_sided_p(middle, df) > 0.05:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted q-values."""
    values = [float(value) for value in p_values]
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value: {value!r}")
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [1.0] * count
    previous = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        candidate = values[index] * count / (reverse_rank + 1)
        previous = min(previous, candidate, 1.0)
        adjusted[index] = previous
    return adjusted


def exact_sign_flip_p(differences: Sequence[float], max_exact_n: int = 20) -> float:
    """Exact two-sided paired randomization p-value by enumerating sign flips."""
    diffs = [float(value) for value in differences]
    if not diffs:
        raise ValueError("at least one paired difference is required")
    if len(diffs) > max_exact_n:
        raise ValueError(
            f"exact sign-flip enumeration is limited to n <= {max_exact_n}; got {len(diffs)}"
        )
    observed = abs(sum(diffs) / len(diffs))
    tolerance = 1e-15 * max(1.0, observed)
    extreme = 0
    total = 1 << len(diffs)
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        permuted = abs(sum(sign * value for sign, value in zip(signs, diffs)) / len(diffs))
        if permuted + tolerance >= observed:
            extreme += 1
    return extreme / total


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def paired_statistics(reference: Sequence[float], comparator: Sequence[float]) -> dict[str, Any]:
    """Compute the complete paired audit requested by the manuscript.

    ``reference`` is DD-Mamba/full and ``comparator`` is a baseline/variant.
    The reported difference is comparator minus reference.
    """
    reference_values = [float(value) for value in reference]
    comparator_values = [float(value) for value in comparator]
    if len(reference_values) != len(comparator_values):
        raise ValueError("paired samples must have equal lengths")
    if not reference_values:
        raise ValueError("at least one paired observation is required")
    if not all(math.isfinite(value) for value in reference_values + comparator_values):
        raise ValueError("paired samples must be finite")

    differences = [comp - ref for ref, comp in zip(reference_values, comparator_values)]
    sample_size = len(differences)
    effect = _mean(differences)
    reference_mean = _mean(reference_values)
    comparator_mean = _mean(comparator_values)
    exact_p = exact_sign_flip_p(differences)

    if sample_size < 2:
        return {
            "n": sample_size,
            "mean_reference": reference_mean,
            "mean_comparator": comparator_mean,
            "mean_paired_difference": effect,
            "difference_definition": "comparator_minus_reference",
            "sd_paired_difference": None,
            "standard_error": None,
            "ci95": {"low": None, "high": None, "method": "t"},
            "t_statistic": None,
            "df": 0,
            "p_value_raw": None,
            "exact_sign_flip_p_two_sided": exact_p,
            "relative_effect_percent": (
                100.0 * effect / reference_mean if reference_mean != 0.0 else None
            ),
            "standardized_effect_dz": None,
            "degenerate_zero_variance": None,
        }

    variance = sum((value - effect) ** 2 for value in differences) / (sample_size - 1)
    difference_sd = math.sqrt(variance)
    standard_error = difference_sd / math.sqrt(sample_size)
    df = sample_size - 1
    degenerate = standard_error == 0.0
    if degenerate:
        t_statistic = None
        raw_p = 0.0 if effect != 0.0 else 1.0
    else:
        t_statistic = effect / standard_error
        raw_p = student_t_two_sided_p(t_statistic, df)
    critical = student_t_critical_975(df)
    half_width = critical * standard_error

    return {
        "n": sample_size,
        "mean_reference": reference_mean,
        "mean_comparator": comparator_mean,
        "mean_paired_difference": effect,
        "difference_definition": "comparator_minus_reference",
        "sd_paired_difference": difference_sd,
        "standard_error": standard_error,
        "ci95": {"low": effect - half_width, "high": effect + half_width, "method": "t"},
        "t_statistic": t_statistic,
        "df": df,
        "p_value_raw": raw_p,
        "exact_sign_flip_p_two_sided": exact_p,
        "relative_effect_percent": (
            100.0 * effect / reference_mean if reference_mean != 0.0 else None
        ),
        "standardized_effect_dz": effect / difference_sd if difference_sd != 0.0 else None,
        "degenerate_zero_variance": degenerate,
    }


def _run_seed_id(run: dict[str, Any]) -> Any | None:
    for key in SEED_KEYS:
        if key in run:
            return run[key]
    return None


def pair_run_records(
    reference_runs: Sequence[dict[str, Any]],
    comparator_runs: Sequence[dict[str, Any]],
    metric: str = "mse",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair run records by explicit seed, or preserve legacy positional order.

    A top-level seed count or a run position is not a seed ID.  Mixed explicit
    and missing identifiers are rejected instead of silently guessed.
    """
    reference_runs = list(reference_runs)
    comparator_runs = list(comparator_runs)
    reference_ids = [_run_seed_id(run) for run in reference_runs]
    comparator_ids = [_run_seed_id(run) for run in comparator_runs]
    explicit_flags = [value is not None for value in reference_ids + comparator_ids]

    if any(explicit_flags) and not all(explicit_flags):
        raise ValueError("seed identifiers are only partially recorded; refusing implicit pairing")

    observations: list[dict[str, Any]] = []
    if explicit_flags and all(explicit_flags):
        if len(set(map(str, reference_ids))) != len(reference_ids):
            raise ValueError("duplicate reference seed identifiers")
        if len(set(map(str, comparator_ids))) != len(comparator_ids):
            raise ValueError("duplicate comparator seed identifiers")
        comparator_by_seed = {
            str(seed_id): run for seed_id, run in zip(comparator_ids, comparator_runs)
        }
        if set(map(str, reference_ids)) != set(comparator_by_seed):
            raise ValueError("reference and comparator seed identifiers do not match")
        for pair_index, (seed_id, reference_run) in enumerate(
            zip(reference_ids, reference_runs), start=1
        ):
            comparator_run = comparator_by_seed[str(seed_id)]
            reference_value = float(reference_run[metric])
            comparator_value = float(comparator_run[metric])
            observations.append(
                {
                    "pair_index": pair_index,
                    "seed_id": seed_id,
                    "reference": reference_value,
                    "comparator": comparator_value,
                    "paired_difference": comparator_value - reference_value,
                }
            )
        pairing = {
            "basis": "explicit_seed_id",
            "seed_ids_available": True,
            "legacy": False,
            "warning": None,
        }
        return observations, pairing

    if len(reference_runs) != len(comparator_runs):
        raise ValueError("legacy positional pairing requires equal run counts")
    for pair_index, (reference_run, comparator_run) in enumerate(
        zip(reference_runs, comparator_runs), start=1
    ):
        reference_value = float(reference_run[metric])
        comparator_value = float(comparator_run[metric])
        observations.append(
            {
                "pair_index": pair_index,
                "seed_id": None,
                "reference": reference_value,
                "comparator": comparator_value,
                "paired_difference": comparator_value - reference_value,
            }
        )
    pairing = {
        "basis": "legacy_positional_order",
        "seed_ids_available": False,
        "legacy": True,
        "warning": (
            "The source records run order but no seed identifiers. Pair indices are retained "
            "for auditing and must not be interpreted as seed IDs."
        ),
    }
    return observations, pairing


def pair_numeric_series(
    reference: Sequence[float], comparator: Sequence[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair legacy numeric arrays without inventing seed identifiers."""
    reference_runs = [{"mse": value} for value in reference]
    comparator_runs = [{"mse": value} for value in comparator]
    return pair_run_records(reference_runs, comparator_runs, metric="mse")


def statistics_from_observations(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return paired_statistics(
        [observation["reference"] for observation in observations],
        [observation["comparator"] for observation in observations],
    )


def aggregate_observations(
    observations_by_horizon: dict[int, Sequence[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Average matched observations over horizons while preserving provenance."""
    if not observations_by_horizon:
        raise ValueError("at least one horizon is required")
    horizons = sorted(observations_by_horizon)
    first = list(observations_by_horizon[horizons[0]])
    first_seed_mode = all(item["seed_id"] is not None for item in first)
    aggregate: list[dict[str, Any]] = []
    for position, first_observation in enumerate(first):
        horizon_observations = []
        for horizon in horizons:
            current = list(observations_by_horizon[horizon])
            if len(current) != len(first):
                raise ValueError("horizon run counts differ")
            if first_seed_mode:
                matching = [
                    item for item in current if str(item["seed_id"]) == str(first_observation["seed_id"])
                ]
                if len(matching) != 1:
                    raise ValueError("explicit seed sets differ across horizons")
                horizon_observations.append(matching[0])
            else:
                if current[position]["seed_id"] is not None:
                    raise ValueError("pairing basis differs across horizons")
                horizon_observations.append(current[position])
        reference_value = _mean([item["reference"] for item in horizon_observations])
        comparator_value = _mean([item["comparator"] for item in horizon_observations])
        aggregate.append(
            {
                "pair_index": position + 1,
                "seed_id": first_observation["seed_id"],
                "reference": reference_value,
                "comparator": comparator_value,
                "paired_difference": comparator_value - reference_value,
                "per_horizon": [
                    {
                        "horizon": horizon,
                        "reference": item["reference"],
                        "comparator": item["comparator"],
                        "paired_difference": item["paired_difference"],
                    }
                    for horizon, item in zip(horizons, horizon_observations)
                ],
            }
        )
    pairing = {
        "basis": "explicit_seed_id" if first_seed_mode else "legacy_positional_order",
        "seed_ids_available": first_seed_mode,
        "legacy": not first_seed_mode,
        "warning": None
        if first_seed_mode
        else (
            "The source records run order but no seed identifiers. Pair indices are retained "
            "for auditing and must not be interpreted as seed IDs."
        ),
    }
    return aggregate, pairing


def apply_bh_family(
    rows: Sequence[dict[str, Any]],
    family_id: str,
    member_ids: Iterable[str],
    description: str,
) -> dict[str, Any]:
    """Attach t-test and sign-flip BH q-values to an explicit test family."""
    row_by_id = {row["id"]: row for row in rows}
    members = list(member_ids)
    if len(members) != len(set(members)):
        raise ValueError(f"duplicate members in family {family_id}")
    missing = [member for member in members if member not in row_by_id]
    if missing:
        raise ValueError(f"unknown family members for {family_id}: {missing}")
    raw_p_values = [row_by_id[member]["statistics"]["p_value_raw"] for member in members]
    if any(value is None for value in raw_p_values):
        raise ValueError(f"family {family_id} contains tests with undefined t p-values")
    exact_p_values = [
        row_by_id[member]["statistics"]["exact_sign_flip_p_two_sided"] for member in members
    ]
    t_q_values = benjamini_hochberg(raw_p_values)
    exact_q_values = benjamini_hochberg(exact_p_values)
    for member, q_value, exact_q_value in zip(members, t_q_values, exact_q_values):
        row_by_id[member].setdefault("multiplicity", {})[family_id] = {
            "family_size": len(members),
            "q_bh": q_value,
            "q_bh_exact_sign_flip": exact_q_value,
        }
    return {
        "id": family_id,
        "description": description,
        "size": len(members),
        "members": members,
        "raw_p_value_field": "statistics.p_value_raw",
        "bh_q_value_field": f"multiplicity.{family_id}.q_bh",
        "sensitivity_p_value_field": "statistics.exact_sign_flip_p_two_sided",
        "sensitivity_bh_q_value_field": f"multiplicity.{family_id}.q_bh_exact_sign_flip",
    }


def write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write strict JSON and read it back as the sole source for reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)
