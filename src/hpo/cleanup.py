"""Reclaim storage used by HPO trial artifacts (Pamba ``hpo_clean`` counterpart).

The objective already deletes a finished trial's checkpoint and per-run
results JSON once its metrics are captured (``--keep-checkpoint`` opts
out), and failed trials are cleaned in its ``finally`` block. This module
reclaims everything else in one shot:

* ``checkpoints/<ds>_pl<pl>_hpo_t<n>_s<i>_best.pt``  per-trial checkpoints
* ``checkpoints/<ds>_hpo_t<n>_s<i>_best.pt``         legacy pre-fix naming
* ``output/hpo/pids/<tag>.pid``                      dead launcher pid files

Policy: **no JSON file is ever deleted** -- exported best-params records,
per-run results JSONs, CSVs and the SQLite study DBs all stay. There is
deliberately no liveness protection: every matched HPO trial checkpoint is
removed unconditionally, so run this only when no sweep should keep them
(a live trial whose checkpoint disappears simply evaluates its last-epoch
weights -- ``src.train`` guards the final load with ``os.path.exists``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_PID_DIR = os.path.join("output", "hpo", "pids")

# New naming carries pred_len (see src/hpo/objective.py); the legacy pattern
# predates the parallel-study checkpoint-collision fix of 2026-09-06.
_NEW_CHECKPOINT_RE = re.compile(
    r"^(?P<dataset>.+)_pl(?P<pred_len>\d+)"
    r"_hpo_t(?P<trial>\d+)_s(?P<seed>\d+)_best\.pt$")
_LEGACY_CHECKPOINT_RE = re.compile(
    r"^(?P<dataset>.+)_hpo_t(?P<trial>\d+)_s(?P<seed>\d+)_best\.pt$")


def classify_checkpoint(filename: str) -> dict | None:
    """Return trial metadata for an HPO checkpoint filename, else ``None``.

    Non-HPO checkpoints (regular experiment runs like ``ETTh1_best.pt``)
    and every non-``.pt`` artifact are not matched.
    """
    match = _NEW_CHECKPOINT_RE.match(filename)
    if match:
        return {
            "dataset": match.group("dataset"),
            "pred_len": int(match.group("pred_len")),
            "trial": int(match.group("trial")),
            "seed": int(match.group("seed")),
            "legacy": False,
        }
    match = _LEGACY_CHECKPOINT_RE.match(filename)
    if match:
        return {
            "dataset": match.group("dataset"),
            "pred_len": None,
            "trial": int(match.group("trial")),
            "seed": int(match.group("seed")),
            "legacy": True,
        }
    return None


def dead_pid_files(pid_dir: str | os.PathLike) -> list[Path]:
    """Pid files of batch launchers whose process is no longer alive."""
    stale = []
    for pid_file in sorted(Path(pid_dir).glob("*.pid")):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            stale.append(pid_file)  # unreadable/garbage pid file
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stale.append(pid_file)
        except PermissionError:
            continue  # alive but owned by another user
        except OSError:
            continue
    return stale


def plan_cleanup(
    checkpoint_dir: str | os.PathLike = DEFAULT_CHECKPOINT_DIR,
    pid_dir: str | os.PathLike = DEFAULT_PID_DIR,
    *,
    dataset: str | None = None,
    pred_len: int | None = None,
) -> dict:
    """List every cleanable artifact without touching the filesystem.

    Returns a report ``{"checkpoints": [(path, size_mb)],
    "pid_files": [path], "freed_mb": float}``. JSON files are never
    classified (preserve-by-policy).
    """
    report: dict = {"checkpoints": [], "pid_files": [], "freed_mb": 0.0}
    root = Path(checkpoint_dir)
    if root.is_dir():
        for path in sorted(root.iterdir()):
            info = classify_checkpoint(path.name)
            if info is None:
                continue  # regular experiment checkpoint / JSON / anything else
            if dataset is not None and info["dataset"] != dataset:
                continue
            if pred_len is not None and info["pred_len"] != pred_len:
                continue
            size_mb = path.stat().st_size / 1e6
            report["checkpoints"].append((str(path), round(size_mb, 2)))
            report["freed_mb"] += size_mb
    report["pid_files"] = [str(p) for p in dead_pid_files(pid_dir)]
    return report


def run_cleanup(confirm: bool = False, **kwargs) -> dict:
    """Plan the cleanup, then delete the artifacts if ``confirm``.

    Returns the :func:`plan_cleanup` report; with ``confirm=False`` this is
    a dry run and the filesystem is left untouched.
    """
    report = plan_cleanup(**kwargs)
    if not confirm:
        return report
    for path, _ in report["checkpoints"]:
        try:
            os.remove(path)
        except OSError:
            pass
    for pid_file in report["pid_files"]:
        try:
            os.remove(pid_file)
        except OSError:
            pass
    return report
