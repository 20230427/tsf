#!/usr/bin/env python3
"""Cross-check ECL aggregate results, per-run JSONs and checkpoint metadata.

No extraction, Torch import, GPU use, foreign pickle globals, or tensor loading.
Only local shape/storage placeholders and collections.OrderedDict are allowed.
This validates saved metadata and archive CRCs, not numerical model predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import zipfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from scripts.analyze_lookback_results import (
        DEFAULT_LENGTHS, MODEL_LABELS, audit_records, canonical_hash,
        file_sha256, read_archive,
    )
except ModuleNotFoundError:
    from analyze_lookback_results import (
        DEFAULT_LENGTHS, MODEL_LABELS, audit_records, canonical_hash,
        file_sha256, read_archive,
    )


@dataclass
class StorageSpec:
    elements: int
    key: str


@dataclass
class TensorSpec:
    shape: tuple
    storage_key: str
    offset: int
    stride: tuple


def tensor_placeholder(storage, offset, size, stride, *unused):
    if not isinstance(storage, StorageSpec) or len(size) > 16:
        raise ValueError("Unsupported checkpoint tensor")
    if any(type(v) is not int or v < 0 or v > 10**9 for v in size):
        raise ValueError("Invalid checkpoint tensor shape")
    return TensorSpec(tuple(size), storage.key, int(offset), tuple(stride))


def parameter_placeholder(tensor, *unused):
    if not isinstance(tensor, TensorSpec):
        raise ValueError("Unsupported checkpoint parameter")
    return tensor


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("builtins", "set"):
            return set
        if module == "torch._utils" and name in (
                "_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"):
            return tensor_placeholder
        if module == "torch._utils" and name in (
                "_rebuild_parameter", "_rebuild_parameter_with_state"):
            return parameter_placeholder
        if module == "torch" and name in (
                "FloatStorage", "DoubleStorage", "HalfStorage", "BFloat16Storage",
                "LongStorage", "IntStorage", "ShortStorage", "ByteStorage",
                "CharStorage", "BoolStorage", "float32", "float64", "float16",
                "bfloat16", "int64", "int32", "int16", "uint8", "int8", "bool"):
            return name  # A local token, NOT a Torch class or callable.
        raise pickle.UnpicklingError(f"Forbidden checkpoint global: {module}.{name}")

    def persistent_load(self, identity):
        if (not isinstance(identity, tuple) or len(identity) != 5 or
                identity[0] != "storage" or type(identity[4]) is not int or
                identity[4] < 0):
            raise pickle.UnpicklingError("Unsupported checkpoint persistent ID")
        return StorageSpec(identity[4], str(identity[2]))


def json_keys(value):
    """Match JSON's string keys without concealing key collisions or value drift."""
    if isinstance(value, dict):
        converted = {str(k): json_keys(v) for k, v in value.items()}
        if len(converted) != len(value):
            raise ValueError("Checkpoint configuration has colliding JSON keys")
        return converted
    if isinstance(value, (list, tuple)):
        return [json_keys(v) for v in value]
    return value


def checkpoint_metadata(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1 or z.getinfo(names[0]).file_size > 16 * 1024 * 1024:
            raise ValueError("Expected one bounded PyTorch metadata pickle")
        saved = MetadataUnpickler(io.BytesIO(z.read(names[0]))).load()
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    required = {"config", "n_channels", "val_metrics", "epoch", "model_state"}
    if not required <= saved.keys():
        raise ValueError("Checkpoint is missing training metadata")
    state = saved["model_state"]
    if not isinstance(state, dict) or any(not isinstance(v, TensorSpec) for v in state.values()):
        raise ValueError("Unsupported state dictionary; no unsafe fallback is used")
    shapes = {k: list(v.shape) for k, v in state.items()}
    unique_views = {(v.storage_key, v.offset, v.shape, v.stride) for v in state.values()}
    return {"config": json_keys(saved["config"]), "n_channels": saved["n_channels"],
            "val_metrics": saved["val_metrics"], "epoch": saved["epoch"],
            "saved_keys": list(saved), "state_tensor_count": len(shapes),
            "state_tensor_elements": sum(math.prod(v) for v in shapes.values()),
            "unique_state_view_elements": sum(math.prod(v[2]) for v in unique_views),
            "example_tensor_shapes": dict(list(shapes.items())[:6])}


def audit_archive(path, verify_weights=True):
    aggregates = read_archive(path)
    matrix = audit_records(aggregates, DEFAULT_LENGTHS, list(MODEL_LABELS),
                           [2024], 96, "electricity")
    issues = list(matrix["issues"])
    result = {"schema": "ecl-training-archive-audit-v1",
              "archive": {"filename": path.name, "sha256": file_sha256(path)},
              "lookback_matrix_complete": matrix["complete"],
              "expected_runs": 117, "issues": issues, "runs": [],
              "checkpoint_inspections": [], "aggregate_copies": [],
              "nested_archives": [], "checkpoint_metadata_verified": verify_weights,
              "complete_abcd": False,
              "missing_abcd": {"b": "training ms/iteration and peak allocated GPU memory",
                               "c": "d_model=64/128/256 runs at L=96 (only d=512 anchor exists)",
                               "d": "LR=0.00025/0.00075 runs at L=96 (only LR=0.0005 anchor exists)"}}
    if not matrix["complete"]:
        result["complete_training_files"] = False
        return result
    with zipfile.ZipFile(path) as z:
        files = [f for f in z.infolist() if not f.is_dir()]
        if len({f.filename for f in files}) != len(files):
            issues.append("Duplicate ZIP member names")
        result["file_count"] = len(files)
        result["extensions"] = dict(Counter(PurePosixPath(f.filename).suffix.lower() for f in files))
        jsons, groups = [], {}
        for f in files:
            if f.filename.endswith(".json"):
                raw = z.read(f)
                d = json.loads(raw.decode("utf-8-sig"))
                if "results" in d:
                    groups.setdefault(canonical_hash(d), []).append(f.filename)
                else:
                    jsons.append((f.filename, d, file_digest(raw)))
        result["aggregate_copies"] = [{"members": members, "copy_count": len(members)}
                                      for members in groups.values()]
        result["unique_aggregate_jsons"] = len(aggregates)
        result["outer_aggregate_jsons"] = sum(map(len, groups.values()))
        result["individual_json_count"] = len(jsons)
        aggregate_contents = {canonical_hash(p) for _, p, _ in aggregates}
        for f in files:
            if f.filename.endswith(".zip"):
                blob = z.read(f)
                with zipfile.ZipFile(io.BytesIO(blob)) as nested:
                    nested_json = [n for n in nested.namelist() if n.endswith(".json")]
                    hashes = {canonical_hash(json.loads(nested.read(n).decode("utf-8-sig")))
                              for n in nested_json}
                same = hashes == aggregate_contents
                result["nested_archives"].append({"member": f.filename,
                    "sha256": file_digest(blob), "json_count": len(nested_json),
                    "matches_outer_aggregates": same})
                if not same:
                    issues.append(f"{f.filename}: nested archive contains different results")
        checkpoints = {}
        for f in files:
            if f.filename.endswith(".pt"):
                checkpoints.setdefault(PurePosixPath(f.filename).name, []).append(f)
        result["checkpoint_count"] = sum(map(len, checkpoints.values()))
        seen_runs = set()
        for member, d, json_digest in jsons:
            cfg = d.get("resolved_config", {})
            key = (cfg.get("data", {}).get("seq_len"), cfg.get("model", {}).get("arch"),
                   cfg.get("experiment", {}).get("seed"))
            if key in seen_runs:
                issues.append(f"{member}: duplicate individual run")
            seen_runs.add(key)
            length, arch, seed = key
            if length not in DEFAULT_LENGTHS or arch not in MODEL_LABELS or seed != 2024:
                issues.append(f"{member}: unexpected individual experiment")
                continue
            aggregate = matrix["cells"][str(length)][arch]["runs"][0]
            digest = canonical_hash(cfg)
            if d.get("schema_version") != "ddmamba-experiment-v1":
                issues.append(f"{member}: missing supported provenance schema")
            if digest != d.get("config_sha256") or digest != aggregate["run_config_sha256"]:
                issues.append(f"{member}: run configuration SHA mismatch")
            if d.get("test_metrics") != {k: aggregate[k] for k in ("mse", "mae", "loss")}:
                issues.append(f"{member}: test metrics disagree with aggregate")
            for k in ("best_val_metrics", "best_val_loss", "best_epoch", "param_count", "checkpoint"):
                if d.get(k) != aggregate[k]:
                    issues.append(f"{member}: aggregate mismatch for {k}")
            expected_phase = "validation_and_test"
            if d.get("evaluation_scope") != expected_phase:
                issues.append(f"{member}: unexpected evaluation scope")
            rr = d.get("run_records", [])
            if (len(rr) != 1 or rr[0].get("run_config_sha256") != digest or
                    rr[0].get("seed") != seed or rr[0].get("arch") != arch or
                    rr[0].get("horizon") != 96 or rr[0].get("phase") != expected_phase):
                issues.append(f"{member}: individual run-record identity mismatch")
            for k in ("best_val_metrics", "best_val_loss", "best_epoch", "epochs_ran", "param_count"):
                if len(rr) == 1 and rr[0].get(k) != d.get(k):
                    issues.append(f"{member}: run-record mismatch for {k}")
            if not 1 <= d["best_epoch"] <= d["epochs_ran"] <= cfg["train"]["epochs"]:
                issues.append(f"{member}: invalid training epoch bounds")
            checkpoint_name = PurePosixPath(d["checkpoint"].replace("\\", "/")).name
            candidates = checkpoints.get(checkpoint_name, [])
            if len(candidates) != 1:
                issues.append(f"{member}: missing or ambiguous checkpoint")
            run = {"lookback": length, "arch": arch, "seed": seed,
                   "json_member": member, "json_sha256": json_digest,
                   "config_sha256": digest, "d_model": cfg["model"]["d_model"],
                   "freq_hidden": cfg["model"]["freq_hidden"], "initial_lr": cfg["train"]["lr"],
                   "best_epoch": d["best_epoch"], "epochs_ran": d["epochs_ran"],
                   "param_count": d["param_count"], "test_metrics": d["test_metrics"],
                   "best_val_metrics": d["best_val_metrics"],
                   "runtime_fields_present": [k for k in (
                       "training_ms_per_iteration", "training_peak_memory_mb",
                       "peak_allocated_gpu_memory_mb") if k in d]}
            result["runs"].append(run)
            if verify_weights and len(candidates) == 1:
                f = candidates[0]
                try:
                    blob = z.read(f)  # Reading also verifies the outer member CRC.
                    metadata = checkpoint_metadata(blob)
                    cfg_ok = canonical_hash(metadata["config"]) == digest
                    val_ok = metadata["val_metrics"] == d["best_val_metrics"]
                    epoch_ok = metadata["epoch"] == d["best_epoch"]
                    channels_ok = metadata["n_channels"] == 321
                    if not all((cfg_ok, val_ok, epoch_ok, channels_ok)):
                        issues.append(f"{f.filename}: checkpoint metadata mismatch")
                    metadata.pop("config")
                    result["checkpoint_inspections"].append({"member": f.filename,
                        "sha256": file_digest(blob), "config_matches": cfg_ok,
                        "validation_matches": val_ok, "epoch_matches": epoch_ok,
                        "channels_match": channels_ok, "config_sha256": digest, **metadata})
                except (ValueError, TypeError, AttributeError, pickle.UnpicklingError,
                        zipfile.BadZipFile, EOFError) as exc:
                    issues.append(f"{f.filename}: restricted metadata inspection failed: {exc}")
        if len(seen_runs) != 117 or result["individual_json_count"] != 117 or result["checkpoint_count"] != 117:
            issues.append("Individual JSON/checkpoint matrix is not exactly 117 runs")
    result["complete_training_files"] = not issues
    result["checkpoint_metadata_match_count"] = sum(
        all(r[k] for k in ("config_matches", "validation_matches", "epoch_matches", "channels_match"))
        for r in result["checkpoint_inspections"])
    result["dd_epochs"] = sorted((r for r in result["runs"] if r["arch"] == "dual_domain"),
                                  key=lambda r: r["lookback"])
    result["notes"] = ["Duplicate aggregate copies are not additional seeds",
                       "Checkpoint integer horizon_overrides keys are normalized to JSON string keys before hashing; collisions are rejected",
                       "Saved model-state elements include buffers; they are not automatically trainable parameters",
                       "No actual dataset, training logs, per-epoch learning curves, or historical GPU allocator/timing records in this upload",
                       "A checkpoint records its best epoch, not the later training stop; epochs_ran comes from individual result JSON"]
    return result


def file_digest(blob):
    import hashlib
    return hashlib.sha256(blob).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-checkpoint-metadata", action="store_true")
    args = p.parse_args()
    report = audit_archive(args.archive, not args.skip_checkpoint_metadata)
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in (
        "complete_training_files", "complete_abcd", "file_count", "unique_aggregate_jsons",
        "individual_json_count", "checkpoint_count", "checkpoint_metadata_match_count", "issues")}, indent=2))
    if report["issues"]:
        raise SystemExit("Archive cross-check failed; do not publish a complete result")


if __name__ == "__main__":
    main()
