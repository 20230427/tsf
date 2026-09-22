"""Optuna hyperparameter optimization for DD-Mamba.

Ported from ``references/pamba/src/hpo`` (single-phase variant): a declarative
search space, a Pamba-style objective that runs full trainings per trial, a
file-locked GPU budget for single/multi-GPU environments, and SQLite-backed
study management with provenance-enabled exports.
"""
from .gpu_budget import GPUBudget, MultiGPUBudget, NoopBudget
from .memory_estimate import estimate_ddmamba_memory_mb
from .cleanup import plan_cleanup, run_cleanup
from .search_space import (
    anchor_params,
    load_spec,
    normalize_conditions,
    resolve_space,
    snap_to_space,
    suggest_params,
)
from .study_manager import create_study, export_results, load_study, study_name

__all__ = [
    "GPUBudget",
    "MultiGPUBudget",
    "NoopBudget",
    "estimate_ddmamba_memory_mb",
    "plan_cleanup",
    "run_cleanup",
    "anchor_params",
    "load_spec",
    "normalize_conditions",
    "resolve_space",
    "snap_to_space",
    "suggest_params",
    "create_study",
    "export_results",
    "load_study",
    "study_name",
]
