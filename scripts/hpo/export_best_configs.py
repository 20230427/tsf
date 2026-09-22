#!/usr/bin/env python3
"""Materialize per-horizon YAML configs from HPO best-params exports.

The generated bundle is intentionally separate from ``configs/``: source
configs stay untouched, while every ``DDMamba_<dataset>_pl<H>_best_params.json``
becomes one directly runnable YAML config plus an audit manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


BEST_PARAMS_RE = re.compile(
    r"^DDMamba_(?P<dataset>.+)_pl(?P<pred_len>\d+)_best_params\.json$",
    re.IGNORECASE,
)
VIRTUAL_PARAMS = {"model.dropout"}


def _get_dotted(config: dict[str, Any], key: str) -> tuple[bool, Any]:
    section, separator, field = key.partition(".")
    if not separator:
        raise ValueError(f"HPO parameter is not dotted: {key!r}")
    section_value = config.get(section)
    if not isinstance(section_value, dict) or field not in section_value:
        return False, None
    return True, section_value[field]


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    section, separator, field = key.partition(".")
    if not separator:
        raise ValueError(f"HPO parameter is not dotted: {key!r}")
    section_value = config.setdefault(section, {})
    if not isinstance(section_value, dict):
        raise ValueError(f"Config section {section!r} is not a mapping")
    section_value[field] = value


def _config_index(config_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    excluded = {"default", "hpo_search_spaces", "baseline_search_spaces"}
    for path in sorted(config_dir.glob("*.yaml")):
        if path.stem.casefold() in excluded:
            continue
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"Duplicate case-insensitive config stem: {path.stem}")
        index[key] = path
    return index


def _validate_export(path: Path, payload: dict[str, Any], dataset: str, pred_len: int) -> None:
    expected_study = f"DDMamba_{dataset}_pl{pred_len}"
    if payload.get("study_name", "").casefold() != expected_study.casefold():
        raise ValueError(
            f"{path}: study_name {payload.get('study_name')!r} does not match "
            f"filename ({expected_study!r})"
        )
    if payload.get("version") != "ddmamba-hpo-best-v2":
        raise ValueError(f"{path}: unsupported version {payload.get('version')!r}")
    if not isinstance(payload.get("best_params"), dict) or not payload["best_params"]:
        raise ValueError(f"{path}: best_params is missing or empty")


def export_bundle(hpo_dir: Path, config_dir: Path, output_dir: Path, zip_path: Path) -> dict[str, Any]:
    sources = sorted(hpo_dir.glob("DDMamba_*_best_params.json"))
    if not sources:
        raise FileNotFoundError(f"No best-params JSON files found in {hpo_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_by_dataset = _config_index(config_dir)
    entries: list[dict[str, Any]] = []

    for hpo_path in sources:
        match = BEST_PARAMS_RE.match(hpo_path.name)
        if match is None:
            continue
        dataset = match.group("dataset")
        pred_len = int(match.group("pred_len"))
        base_path = config_by_dataset.get(dataset.casefold())
        if base_path is None:
            raise FileNotFoundError(f"No configs/<dataset>.yaml match for {hpo_path.name}")

        payload = json.loads(hpo_path.read_text(encoding="utf-8"))
        _validate_export(hpo_path, payload, dataset, pred_len)
        config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"{base_path}: top-level YAML value is not a mapping")

        applied: dict[str, Any] = {}
        virtual: dict[str, Any] = {}
        differences: list[dict[str, Any]] = []
        for key, value in payload["best_params"].items():
            if key in VIRTUAL_PARAMS:
                virtual[key] = value
                continue
            existed, old_value = _get_dotted(config, key)
            if not existed or old_value != value:
                differences.append(
                    {"key": key, "base_value": old_value if existed else None, "hpo_value": value}
                )
            _set_dotted(config, key, value)
            applied[key] = value

        # One config corresponds to exactly one HPO study/horizon. A unique
        # experiment name prevents sequential runs from overwriting each other.
        config.setdefault("data", {})["pred_len"] = pred_len
        generated_name = f"{dataset}_pl{pred_len}_hpo_best"
        config.setdefault("experiment", {})["name"] = generated_name

        output_name = f"{base_path.stem}_pl{pred_len}.yaml"
        output_path = output_dir / output_name
        header = (
            f"# Generated from {hpo_path.as_posix()}\n"
            f"# Base config: {base_path.as_posix()}\n"
            f"# Study: {payload['study_name']} | {payload['metric']}="
            f"{payload['best_value']!r} | trial={payload['best_trial_number']}\n"
            "# model.dropout is an HPO-only alias; its value is materialized in "
            "time_dropout, freq_dropout, and head_dropout.\n\n"
        )
        output_path.write_text(
            header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        entries.append(
            {
                "output_config": output_name,
                "base_config": base_path.as_posix(),
                "hpo_file": hpo_path.as_posix(),
                "study_name": payload["study_name"],
                "pred_len": pred_len,
                "metric": payload["metric"],
                "best_value": payload["best_value"],
                "best_trial_number": payload["best_trial_number"],
                "applied_params": applied,
                "virtual_params": virtual,
                "generated_adjustments": {
                    "data.pred_len": pred_len,
                    "experiment.name": generated_name,
                },
                "differences_from_base": differences,
            }
        )

    manifest = {
        "schema_version": "ddmamba-hpo-config-bundle-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hpo_dir": hpo_dir.as_posix(),
        "source_config_dir": config_dir.as_posix(),
        "config_count": len(entries),
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            archive.write(path, arcname=f"{output_dir.name}/{path.name}")
    return manifest


def export_merged_bundle(
    hpo_dir: Path, config_dir: Path, output_dir: Path, zip_path: Path
) -> dict[str, Any]:
    """Write one base-like YAML per dataset with per-horizon override blocks."""
    sources = sorted(hpo_dir.glob("DDMamba_*_best_params.json"))
    if not sources:
        raise FileNotFoundError(f"No best-params JSON files found in {hpo_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {zip_path}")

    config_by_dataset = _config_index(config_dir)
    grouped: dict[str, list[tuple[Path, re.Match[str]]]] = {}
    for hpo_path in sources:
        match = BEST_PARAMS_RE.match(hpo_path.name)
        if match is not None:
            grouped.setdefault(match.group("dataset").casefold(), []).append((hpo_path, match))

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for dataset_key, source_group in sorted(grouped.items()):
        base_path = config_by_dataset.get(dataset_key)
        if base_path is None:
            names = ", ".join(path.name for path, _ in source_group)
            raise FileNotFoundError(f"No matching dataset config for: {names}")
        base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_config, dict):
            raise ValueError(f"{base_path}: top-level YAML value is not a mapping")

        horizon_overrides: dict[int, dict[str, Any]] = {}
        horizon_entries: list[dict[str, Any]] = []
        for hpo_path, match in sorted(
            source_group, key=lambda item: int(item[1].group("pred_len"))
        ):
            dataset = match.group("dataset")
            pred_len = int(match.group("pred_len"))
            payload = json.loads(hpo_path.read_text(encoding="utf-8"))
            _validate_export(hpo_path, payload, dataset, pred_len)

            override: dict[str, Any] = {
                "experiment": {"name": f"{dataset}_pl{pred_len}_hpo_best"},
                "data": {"pred_len": pred_len},
                "model": {},
                "train": {},
            }
            differences: list[dict[str, Any]] = []
            virtual: dict[str, Any] = {}
            for key, value in payload["best_params"].items():
                if key in VIRTUAL_PARAMS:
                    virtual[key] = value
                    continue
                section, _, field = key.partition(".")
                override.setdefault(section, {})[field] = value
                existed, old_value = _get_dotted(base_config, key)
                if not existed or old_value != value:
                    differences.append(
                        {
                            "key": key,
                            "base_value": old_value if existed else None,
                            "hpo_value": value,
                        }
                    )
            override = {section: values for section, values in override.items() if values}
            horizon_overrides[pred_len] = override
            horizon_entries.append(
                {
                    "pred_len": pred_len,
                    "hpo_file": hpo_path.as_posix(),
                    "study_name": payload["study_name"],
                    "metric": payload["metric"],
                    "best_value": payload["best_value"],
                    "best_trial_number": payload["best_trial_number"],
                    "virtual_params": virtual,
                    "differences_from_base": differences,
                }
            )

        merged_config = dict(base_config)
        merged_config["horizon_overrides"] = horizon_overrides
        output_path = output_dir / base_path.name
        header = (
            "# Consolidated HPO config: one file per dataset.\n"
            f"# Base config: {base_path.as_posix()}\n"
            "# For a run, recursively merge horizon_overrides[<pred_len>] over "
            "the base sections above.\n"
            "# Values in horizon_overrides come from output/hpo and take precedence.\n"
            "# model.dropout is an HPO-only alias; its value is materialized in "
            "time_dropout, freq_dropout, and head_dropout.\n\n"
        )
        output_path.write_text(
            header + yaml.safe_dump(merged_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        entries.append(
            {
                "dataset": base_path.stem,
                "output_config": base_path.name,
                "base_config": base_path.as_posix(),
                "horizon_count": len(horizon_entries),
                "horizons": horizon_entries,
            }
        )

    manifest = {
        "schema_version": "ddmamba-hpo-merged-config-bundle-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hpo_dir": hpo_dir.as_posix(),
        "source_config_dir": config_dir.as_posix(),
        "config_count": len(entries),
        "hpo_study_count": sum(entry["horizon_count"] for entry in entries),
        "merge_semantics": "recursive merge of horizon_overrides[pred_len] over base config",
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            archive.write(path, arcname=f"{output_dir.name}/{path.name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-dir", type=Path, default=Path("output/hpo"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("generated/hpo_configs_20260910")
    )
    parser.add_argument(
        "--zip-path", type=Path, default=Path("generated/hpo_configs_20260910.zip")
    )
    parser.add_argument(
        "--merge-by-dataset",
        action="store_true",
        help="write one YAML per dataset with a horizon_overrides mapping",
    )
    args = parser.parse_args()
    exporter = export_merged_bundle if args.merge_by_dataset else export_bundle
    manifest = exporter(args.hpo_dir, args.config_dir, args.output_dir, args.zip_path)
    print(f"Generated {manifest['config_count']} configs in {args.output_dir}")
    print(f"Created archive: {args.zip_path}")


if __name__ == "__main__":
    main()
