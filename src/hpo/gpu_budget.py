"""File-locked GPU memory budget for concurrent HPO studies.

Ported from ``references/pamba/src/hpo/gpu_budget.py`` (paths moved from
``temp/optuna`` to ``output/hpo``). Two flavours:

* :class:`GPUBudget` -- a single shared pool (MB accounting across all
  concurrent studies on the visible GPU). Trials wait until their estimate
  fits; stale reservations whose PID no longer exists are reaped.
* :class:`MultiGPUBudget` -- one pool per GPU; ``reserve`` picks the GPU with
  the most free budget (best-fit), so concurrent studies spread over cards.

All coordination happens through ``output/hpo/gpu_budget.lock`` (``fcntl``
exclusive lock) around JSON state files, so independent processes cooperate
without a server.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import fcntl

HPO_DIR = os.path.join("output", "hpo")
BUDGET_LOCK_FILE = os.path.join(HPO_DIR, "gpu_budget.lock")
BUDGET_JSON_FILE = os.path.join(HPO_DIR, "gpu_budget.json")
MULTI_BUDGET_FILE = os.path.join(HPO_DIR, "gpu_budget_multi.json")

_JSON_INDENT = 2
_POLL_SECONDS = 2.0


def _detect_gpu_memory_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return 24 * 1024


TOTAL_GPU_MEMORY_MB = _detect_gpu_memory_mb()


class NoopBudget:
    """Budget that never blocks (dedicated-GPU mode)."""

    def acquire(self, *args, **kwargs):
        return True

    def release(self, *args, **kwargs):
        pass


def _pid_alive(pid) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_json(path: str, default: dict) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        return default


class GPUBudget:
    """Single shared memory pool guarded by an flock'd JSON file."""

    def __init__(self, total_mb: int | None = None):
        self.total_mb = total_mb or TOTAL_GPU_MEMORY_MB
        self._pid = os.getpid()
        os.makedirs(HPO_DIR, exist_ok=True)

    def _state_path(self) -> str:
        return BUDGET_JSON_FILE

    def _default_state(self) -> dict:
        return {"used_mb": 0.0, "trials": {}}

    def _cleanup_stale(self, budget: dict) -> dict:
        stale = [tid for tid, entry in budget["trials"].items()
                 if not _pid_alive(entry.get("pid"))]
        for tid in stale:
            del budget["trials"][tid]
        budget["used_mb"] = sum(e.get("mb", 0) for e in budget["trials"].values())
        return budget

    def acquire(self, trial_id: str, needed_mb: float, timeout: int = 3600) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with open(BUDGET_LOCK_FILE, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    budget = _read_json(self._state_path(), self._default_state())
                    budget = self._cleanup_stale(budget)
                    if needed_mb <= self.total_mb - budget["used_mb"]:
                        budget["used_mb"] += needed_mb
                        budget["trials"][trial_id] = {"pid": self._pid, "mb": needed_mb}
                        with open(self._state_path(), "w") as f:
                            json.dump(budget, f, indent=_JSON_INDENT)
                        return True
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
            time.sleep(_POLL_SECONDS)
        return False

    def release(self, trial_id: str) -> None:
        with open(BUDGET_LOCK_FILE, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                budget = _read_json(self._state_path(), self._default_state())
                entry = budget["trials"].pop(trial_id, None)
                if entry is not None:
                    budget["used_mb"] = max(0.0, budget["used_mb"] - entry.get("mb", 0))
                with open(self._state_path(), "w") as f:
                    json.dump(budget, f, indent=_JSON_INDENT)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)


class MultiGPUBudget:
    """One pool per GPU; ``reserve`` returns the best-fitting GPU index."""

    def __init__(self, gpu_ids: list[int], total_mb: int = TOTAL_GPU_MEMORY_MB):
        self.gpu_ids = list(gpu_ids)
        self.total_mb = total_mb
        self._pid = os.getpid()
        os.makedirs(HPO_DIR, exist_ok=True)
        default = {"gpus": {str(g): {"reserved_mb": 0.0, "reservations": {}}
                            for g in self.gpu_ids}}
        with open(BUDGET_LOCK_FILE, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                budget = _read_json(MULTI_BUDGET_FILE, default)
                for g in self.gpu_ids:
                    budget["gpus"].setdefault(str(g), {"reserved_mb": 0.0, "reservations": {}})
                with open(MULTI_BUDGET_FILE, "w") as f:
                    json.dump(budget, f, indent=_JSON_INDENT)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def _cleanup_stale(self, budget: dict) -> dict:
        for state in budget["gpus"].values():
            stale = [tag for tag, entry in state.get("reservations", {}).items()
                     if not _pid_alive(entry.get("pid"))]
            for tag in stale:
                state["reserved_mb"] = max(
                    0.0, state.get("reserved_mb", 0) - state["reservations"].pop(tag).get("mb", 0))
        return budget

    def reserve(self, study_tag: str, needed_mb: float, timeout: int = 7200) -> int | None:
        """Reserve ``needed_mb`` on the freest GPU; returns its index or None."""
        start = time.time()
        while time.time() - start < timeout:
            with open(BUDGET_LOCK_FILE, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    budget = _cleanup_stale_multi(MULTI_BUDGET_FILE, self.gpu_ids)
                    best_gpu, best_available = None, -1.0
                    for g in self.gpu_ids:
                        state = budget["gpus"].get(str(g), {"reserved_mb": 0.0})
                        available = self.total_mb - state.get("reserved_mb", 0.0)
                        if available >= needed_mb and available > best_available:
                            best_gpu, best_available = g, available
                    if best_gpu is not None:
                        state = budget["gpus"][str(best_gpu)]
                        state["reserved_mb"] = state.get("reserved_mb", 0.0) + needed_mb
                        state.setdefault("reservations", {})[study_tag] = {
                            "pid": self._pid, "mb": needed_mb,
                        }
                        with open(MULTI_BUDGET_FILE, "w") as f:
                            json.dump(budget, f, indent=_JSON_INDENT)
                        return best_gpu
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
            time.sleep(2.0)
        return None

    def unreserve(self, study_tag: str) -> None:
        with open(BUDGET_LOCK_FILE, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                budget = _read_json(MULTI_BUDGET_FILE, {"gpus": {}})
                for state in budget["gpus"].values():
                    entry = state.get("reservations", {}).pop(study_tag, None)
                    if entry is not None:
                        state["reserved_mb"] = max(
                            0.0, state.get("reserved_mb", 0) - entry.get("mb", 0))
                with open(MULTI_BUDGET_FILE, "w") as f:
                    json.dump(budget, f, indent=_JSON_INDENT)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)


def _cleanup_stale_multi(path: str, gpu_ids: list[int]) -> dict:
    """Read/repair the multi-GPU state under the caller's lock."""
    default = {"gpus": {str(g): {"reserved_mb": 0.0, "reservations": {}} for g in gpu_ids}}
    budget = _read_json(path, default)
    for g in gpu_ids:
        budget["gpus"].setdefault(str(g), {"reserved_mb": 0.0, "reservations": {}})
    for state in budget["gpus"].values():
        stale = [tag for tag, entry in state.get("reservations", {}).items()
                 if not _pid_alive(entry.get("pid"))]
        for tag in stale:
            state["reserved_mb"] = max(
                0.0, state.get("reserved_mb", 0) - state["reservations"].pop(tag).get("mb", 0))
    return budget


def release_budget_by_pid(pid: int) -> float:
    """Free every reservation held by a dead/stopped process; returns MB freed."""
    freed = 0.0
    for path, trials_key in ((BUDGET_JSON_FILE, "trials"), (MULTI_BUDGET_FILE, "reservations")):
        if not os.path.exists(path):
            continue
        with open(BUDGET_LOCK_FILE, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                budget = _read_json(path, {})
                changed = False
                if trials_key == "trials":
                    for tid in [t for t, e in budget.get("trials", {}).items()
                                if e.get("pid") == pid]:
                        freed += budget["trials"].pop(tid).get("mb", 0)
                        changed = True
                    budget["used_mb"] = max(0.0, budget.get("used_mb", 0) - freed)
                else:
                    for state in budget.get("gpus", {}).values():
                        for tag in [t for t, e in state.get("reservations", {}).items()
                                    if e.get("pid") == pid]:
                            freed += state["reservations"].pop(tag).get("mb", 0)
                            state["reserved_mb"] = max(
                                0.0, state.get("reserved_mb", 0) - freed)
                            changed = True
                if changed:
                    with open(path, "w") as f:
                        json.dump(budget, f, indent=_JSON_INDENT)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    return freed
