"""Auditable experiment provenance and crash-safe JSON persistence.

The runner JSON files are scientific records, not just convenience summaries.
This module keeps their metadata format consistent and makes partial writes
unlikely by replacing the destination only after a complete temporary file has
been flushed to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = "ddmamba-experiment-v1"


def _jsonable(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of ``value``."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete resolved config using canonical JSON serialization."""
    canonical = json.dumps(
        _jsonable(config), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_state(cwd: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Return commit/dirty state, or explicit nulls outside a Git checkout."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=cwd,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(porcelain.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _runtime_environment(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    device_names: list[str] = []
    if cuda_available:
        try:
            device_names = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
        except (RuntimeError, AssertionError):
            device_names = []
    train_cfg = resolved_config.get("train", {})
    configured_device = (train_cfg.get("device")
                         if isinstance(train_cfg, Mapping) else None)
    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except (AttributeError, RuntimeError):
        pass
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "device": {
            "configured": configured_device,
            "cuda_device_count": len(device_names),
            "cuda_device_names": device_names,
        },
    }


def provenance_fields(
    resolved_config: Mapping[str, Any],
    *,
    config_path: str | os.PathLike[str] | None,
    seed_values: Sequence[int],
    cli_argv: Sequence[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build the required top-level provenance fields for a runner record."""
    resolved = _jsonable(resolved_config)
    argv = list(sys.argv if cli_argv is None else cli_argv)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        # Keep ``config`` for compatibility with the existing analysis scripts.
        "config": str(config_path) if config_path is not None else None,
        "resolved_config": resolved,
        "config_sha256": config_sha256(resolved),
        "git": _git_state(cwd),
        "cli": {"argv": argv},
        "environment": _runtime_environment(resolved),
        "seed_values": [int(seed) for seed in seed_values],
    }


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Atomically replace ``path`` with a fully flushed JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def validate_provenance_record(payload: Any) -> dict[str, Any]:
    """Classify a record as verified, legacy-unverifiable, or invalid.

    Older result files predate the provenance schema. They are deliberately
    reported as ``legacy_unverifiable`` rather than treated as corrupt or
    raising exceptions. A record that opts into this schema but violates it is
    ``invalid``.
    """
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "reasons": ["JSON root is not an object"]}
    if "schema_version" not in payload:
        return {
            "status": "legacy_unverifiable",
            "reasons": ["missing schema_version; provenance cannot be verified"],
        }
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {
            "status": "invalid",
            "reasons": [f"unsupported schema_version: {payload.get('schema_version')!r}"],
        }

    required = (
        "created_at_utc", "resolved_config", "config_sha256", "git", "cli",
        "environment", "seed_values", "run_records",
    )
    reasons = [f"missing required field: {field}"
               for field in required if field not in payload]
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping):
        try:
            expected = config_sha256(resolved)
            if payload.get("config_sha256") != expected:
                reasons.append("config_sha256 does not match resolved_config")
        except (TypeError, ValueError) as exc:
            reasons.append(f"resolved_config cannot be hashed: {exc}")
    elif "resolved_config" in payload:
        reasons.append("resolved_config is not an object")

    seeds = payload.get("seed_values")
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        reasons.append("seed_values must be a list of integers")

    run_records = payload.get("run_records")
    if not isinstance(run_records, list):
        reasons.append("run_records must be a list")
    else:
        for index, run in enumerate(run_records):
            if not isinstance(run, Mapping):
                reasons.append(f"run_records[{index}] is not an object")
                continue
            missing = [key for key in ("seed", "horizon", "arch") if key not in run]
            if missing:
                reasons.append(
                    f"run_records[{index}] missing: {', '.join(missing)}"
                )
            if isinstance(seeds, list) and run.get("seed") not in seeds:
                reasons.append(
                    f"run_records[{index}] seed is absent from seed_values"
                )

    environment = payload.get("environment")
    if isinstance(environment, Mapping):
        env_required = ("python_version", "torch_version", "cuda_version",
                        "host", "device")
        reasons.extend(f"environment missing: {field}" for field in env_required
                       if field not in environment)
    elif "environment" in payload:
        reasons.append("environment is not an object")

    git_state = payload.get("git")
    if isinstance(git_state, Mapping):
        reasons.extend(f"git missing: {field}" for field in ("commit", "dirty")
                       if field not in git_state)
    elif "git" in payload:
        reasons.append("git is not an object")

    return {"status": "invalid" if reasons else "verified", "reasons": reasons}
