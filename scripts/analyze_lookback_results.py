#!/usr/bin/env python3
"""Audit an aggregate-result ZIP and render static lookback figures, without Torch.

The nested per-seed runs are authoritative; stored mean/std fields are not used.
Single-seed curves have no uncertainty bands. No ZIP member is extracted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODEL_LABELS = {
    "dual_domain": "DD-Mamba", "smamba": "S-Mamba",
    "itransformer": "iTransformer", "patchtst": "PatchTST",
    "crossformer": "Crossformer", "tide": "TiDE", "dlinear": "DLinear",
    "fedformer": "FEDformer", "autoformer": "Autoformer",
}
DEFAULT_LENGTHS = [24, 48, 72, 96, 144, 168, 192, 288, 336, 480, 672, 720, 1008]
PALETTE = ["#BE3535", "#8657A8", "#E58B2A", "#2C6FBB", "#208C71",
           "#94673A", "#687D93", "#C167A0", "#303B46"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def describe(values):
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_config(cfg):
    value = copy.deepcopy(cfg)
    value["data"].pop("seq_len", None)
    for key in ("name", "seed", "checkpoint_dir"):
        value["experiment"].pop(key, None)
    return value


def read_archive(path):
    records = []
    seen_bytes = set()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".json"):
                continue
            raw = archive.read(member)
            payload = json.loads(raw.decode("utf-8-sig"))
            # Non-result JSON is data, not a configuration or instruction source.
            if not isinstance(payload, dict) or "results" not in payload:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            # A complete upload can contain literal copies in two folders.
            # Only identical bytes are deduplicated; conflicting runs still
            # fail the matrix audit, including formatting-only differences.
            if digest not in seen_bytes:
                records.append((member.filename, payload, digest))
                seen_bytes.add(digest)
    if not records:
        raise ValueError("Archive contains no aggregate result JSON")
    return records


def audit_records(records, lengths, models, seeds, horizon, dataset):
    issues, warnings, normalized, environments = [], [], set(), set()
    cells = defaultdict(lambda: defaultdict(list))
    files, timestamps, configs = [], [], []
    run_keys = set()
    for member, payload, digest in records:
        cfg = payload.get("resolved_config")
        if not isinstance(cfg, dict):
            issues.append(f"{member}: missing resolved_config")
            continue
        configs.append(cfg)
        length = cfg.get("data", {}).get("seq_len")
        file_info = {"member": member, "sha256": digest,
                     "lookback": length, "seed_values": payload.get("seed_values"),
                     "config_sha256": payload.get("config_sha256"),
                     "created_at_utc": payload.get("created_at_utc"),
                     "git": payload.get("git"),
                     "baseline_selection_source": payload.get("baseline_selection_source"),
                     "architecture_overrides": payload.get("architecture_overrides")}
        files.append(file_info)
        if payload.get("schema_version") != "ddmamba-experiment-v1":
            issues.append(f"{member}: unsupported or legacy provenance schema")
        if canonical_hash(cfg) != payload.get("config_sha256"):
            issues.append(f"{member}: resolved config SHA-256 mismatch")
        normalized.add(canonical_hash(fixed_config(cfg)))
        environments.add(canonical_hash(payload.get("environment", {})))
        if length not in lengths:
            issues.append(f"{member}: unexpected lookback {length}")
        if cfg.get("data", {}).get("pred_len") != horizon:
            issues.append(f"{member}: unexpected configured prediction horizon")
        if Path(str(cfg.get("data", {}).get("csv_path", ""))).stem.lower() != dataset.lower():
            issues.append(f"{member}: dataset identity differs from {dataset}")
        if payload.get("horizons") != [horizon]:
            issues.append(f"{member}: aggregate horizons are not [{horizon}]")
        declared_seeds = payload.get("seed_values", [])
        if not declared_seeds or not set(declared_seeds).issubset(seeds):
            issues.append(f"{member}: unexpected declared seed values")
        if payload.get("seeds") != len(declared_seeds):
            issues.append(f"{member}: declared seed count mismatch")
        results = payload.get("results", {})
        if set(results) != set(models) or set(payload.get("architectures", [])) != set(models):
            issues.append(f"{member}: missing or unexpected architectures")
        if payload.get("failures") != {}:
            issues.append(f"{member}: failures={payload.get('failures')}")
        flat = payload.get("run_records", [])
        nested_runs = []
        for arch in models:
            runs = results.get(arch, {}).get(str(horizon), {}).get("runs", [])
            if sorted(run.get("seed", -1) for run in runs) != sorted(declared_seeds):
                issues.append(f"{member}: {arch} seed coverage mismatch")
            for index, run in enumerate(runs):
                nested_runs.append(run)
                key = (length, arch, run.get("seed"))
                if key in run_keys:
                    issues.append(f"{member}: duplicate run {key}")
                run_keys.add(key)
                if run.get("horizon") != horizon or run.get("arch") != arch:
                    issues.append(f"{member}: {arch} run identity mismatch")
                valid_metrics = True
                for metric in ("mse", "mae"):
                    value = run.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        issues.append(f"{member}: {key} invalid {metric}")
                        valid_metrics = False
                if not payload.get("architecture_overrides", {}).get(arch):
                    expected_cfg = copy.deepcopy(cfg)
                    expected_cfg["model"]["arch"] = arch
                    expected_cfg["experiment"]["seed"] = run.get("seed")
                    expected_cfg["experiment"]["name"] = (
                        f"{cfg['experiment']['name']}_{arch}_h{horizon}_s{index}"
                    )
                    if canonical_hash(expected_cfg) != run.get("run_config_sha256"):
                        issues.append(f"{member}: {key} reconstructed run config SHA-256 mismatch")
                if valid_metrics:
                    cells[length][arch].append(copy.deepcopy(run))
        # Invalid NaN/Inf metrics have already been flagged above. Comparing
        # their raw signatures must not crash before a failed audit is emitted.
        if Counter(json.dumps(run, sort_keys=True) for run in nested_runs) != Counter(json.dumps(run, sort_keys=True) for run in flat):
            issues.append(f"{member}: flat/nested run records differ")
        if payload.get("created_at_utc"):
            timestamps.append(payload["created_at_utc"])
    if len(normalized) != 1:
        issues.append(f"Fixed configuration changed across lookbacks ({len(normalized)} variants)")
    if len(environments) != 1:
        warnings.append(f"Runtime environment changed ({len(environments)} variants)")
    for length in lengths:
        for arch in models:
            actual = sorted(run["seed"] for run in cells[length][arch])
            if actual != sorted(seeds):
                issues.append(f"L={length}, {arch}: expected seeds {seeds}, observed {actual}")
    if any(not (p.get("git") or {}).get("commit") for _, p, _ in records):
        warnings.append("Git commit unavailable; exact code revision cannot be established")
    if any(not p.get("baseline_selection_source") for _, p, _ in records):
        warnings.append("No certified frozen validation-only baseline selection report")
    if any("horizon_overrides" in cfg for cfg in configs):
        warnings.append("Config includes horizon_overrides; run hashes are checked against top-level base config, not a presumed merged HPO recipe")
    warnings.append("Archive metrics do not verify the actual CSV/PT cache or contain learning curves")
    if len(seeds) == 1:
        warnings.append("Single seed: sample standard deviation is undefined; no uncertainty bands or significance claims")
    summary = {
        "analysis_schema": "ddmamba-lookback-analysis-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "horizon": horizon, "lengths": lengths,
        "models": models, "seed_values": seeds,
        "expected_runs": len(lengths) * len(models) * len(seeds),
        "observed_runs": len(run_keys), "result_json_count": len(records),
        "complete": not issues, "issues": issues, "warnings": warnings,
        "fixed_config_variants": len(normalized),
        "environment_variants": len(environments),
        "environment": records[0][1].get("environment"),
        "base_config": configs[0] if configs else None,
        "created_at_range_utc": [min(timestamps), max(timestamps)] if timestamps else None,
        "source_files": files, "cells": {}, "model_summary": {},
        "standard_deviation": "sample (n-1); null when n=1",
    }
    if issues:
        return summary
    for length in lengths:
        summary["cells"][str(length)] = {}
        for arch in models:
            runs = sorted(cells[length][arch], key=lambda run: run["seed"])
            cell = {metric: describe([run[metric] for run in runs])
                    for metric in ("mse", "mae")}
            cell["runs"] = runs
            summary["cells"][str(length)][arch] = cell
        for metric in ("mse", "mae"):
            values = {arch: summary["cells"][str(length)][arch][metric]["mean"]
                      for arch in models}
            for arch in models:
                summary["cells"][str(length)][arch][metric]["rank"] = (
                    1 + sum(value < values[arch] for value in values.values())
                )
    for arch in models:
        item = {"label": MODEL_LABELS.get(arch, arch)}
        for metric in ("mse", "mae"):
            best_l = min(lengths, key=lambda length: summary["cells"][str(length)][arch][metric]["mean"])
            series = [summary["cells"][str(length)][arch][metric] for length in lengths]
            item[metric] = {
                "best_lookback": best_l,
                "best_value": summary["cells"][str(best_l)][arch][metric]["mean"],
                "mean_across_lookbacks": statistics.mean(cell["mean"] for cell in series),
                "mean_rank": statistics.mean(cell["rank"] for cell in series),
                "wins": sum(cell["rank"] == 1 for cell in series),
            }
        epochs = [run["best_epoch"] for length in lengths for run in cells[length][arch]]
        item["best_epoch_counts"] = dict(sorted(Counter(epochs).items()))
        summary["model_summary"][arch] = item
    return summary


def save_figure(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(out_dir / f"{name}.svg", facecolor="white")


def style_axis(ax, lengths):
    ax.set_xlabel("Lookback length L (hourly steps)")
    ax.set_xlim(min(lengths) - 15, max(lengths) + 25)
    # Keep the true numeric spacing, including every densely spaced short window.
    ax.set_xticks(lengths, labels=[str(length) for length in lengths])
    ax.tick_params(axis="x", labelrotation=90, labelsize=8.5)
    ax.grid(axis="both", color="#D5DADE", alpha=0.75, linewidth=0.65,
            linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def render_figures(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "axes.edgecolor": "#35434F", "text.color": "#24323D",
                         "axes.labelcolor": "#24323D", "xtick.color": "#35434F",
                         "ytick.color": "#35434F", "svg.fonttype": "none"})
    lengths, models = summary["lengths"], summary["models"]
    meta = f"Electricity | H={summary['horizon']} | seeds={','.join(map(str, summary['seed_values']))} | n={len(summary['seed_values'])}"
    outputs = []
    for metric in ("mse", "mae"):
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
        handles = []
        for index, arch in enumerate(models):
            values = [summary["cells"][str(length)][arch][metric]["mean"] for length in lengths]
            stds = [summary["cells"][str(length)][arch][metric]["sample_std"] for length in lengths]
            for panel, ax in enumerate(axes):
                if panel == 1 and arch in ("fedformer", "autoformer"):
                    continue
                selected = [i for i, length in enumerate(lengths)
                            if panel == 0 or length >= 144]
                panel_lengths = [lengths[i] for i in selected]
                panel_values = [values[i] for i in selected]
                line, = ax.plot(panel_lengths, panel_values, label=MODEL_LABELS.get(arch, arch),
                                color=PALETTE[index], marker=MARKERS[index],
                                linewidth=2.25 if arch == "dual_domain" else 1.35,
                                markersize=4.6 if arch == "dual_domain" else 3.5,
                                linestyle="--" if index >= 7 else "-",
                                zorder=5 if arch == "dual_domain" else 3)
                if len(summary["seed_values"]) > 1:
                    y = np.array(panel_values)
                    sd = np.array([stds[i] for i in selected])
                    ax.fill_between(panel_lengths, y - sd, y + sd,
                                    color=PALETTE[index], alpha=0.10)
                if panel == 0:
                    handles.append(line)
        for panel, ax in enumerate(axes):
            panel_lengths = lengths if panel == 0 else [length for length in lengths if length >= 144]
            style_axis(ax, panel_lengths)
            ax.set_ylabel(f"Test {metric.upper()} (standardized data space)")
        axes[0].set_title("(a) All nine models", loc="left")
        axes[1].set_title("(b) Detail: L >= 144, excluding FEDformer / Autoformer", loc="left", fontsize=10.5)
        best = summary["model_summary"]["dual_domain"][metric]
        best_l, best_value = best["best_lookback"], best["best_value"]
        axes[1].scatter([best_l], [best_value], s=85, facecolors="none",
                        edgecolors=PALETTE[0], linewidths=1.6, zorder=7)
        axes[1].annotate(f"DD-Mamba minimum: L={best_l}\n{metric.upper()}={best_value:.6f}",
                         (best_l, best_value), xytext=(0.32, 0.78), textcoords="axes fraction",
                         fontsize=8.7, color=PALETTE[0],
                         arrowprops={"arrowstyle": "-", "color": PALETTE[0], "lw": 0.7})
        fig.suptitle(meta, fontsize=12, y=0.98)
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, 0.005))
        fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.255, wspace=0.29)
        name = f"electricity_lookback_{metric}"
        save_figure(fig, out_dir, name)
        plt.close(fig)
        outputs.extend([f"figures/{name}.png", f"figures/{name}.svg"])

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 7.6))
    for metric, ax in zip(("mse", "mae"), axes[0]):
        test = [summary["cells"][str(length)]["dual_domain"][metric]["mean"] for length in lengths]
        val = [statistics.mean(run["best_val_metrics"][metric] for run in summary["cells"][str(length)]["dual_domain"]["runs"]) for length in lengths]
        ax.plot(lengths, test, "o-", color=PALETTE[0], label="Test", lw=1.8, ms=4)
        ax.plot(lengths, val, "s--", color="#687D93", label="Validation", lw=1.35, ms=3.5)
        ax.set_ylabel(f"{metric.upper()} (standardized data space)")
        ax.set_title(f"DD-Mamba: {metric.upper()}", loc="left")
        ax.legend(frameon=False, fontsize=9)
    params = [statistics.mean(run["param_count"] for run in summary["cells"][str(length)]["dual_domain"]["runs"]) / 1e6 for length in lengths]
    epochs = [statistics.mean(run["best_epoch"] for run in summary["cells"][str(length)]["dual_domain"]["runs"]) for length in lengths]
    axes[1, 0].plot(lengths, params, "o-", color="#2C6FBB", lw=1.65, ms=4)
    axes[1, 0].set_ylabel("Trainable parameters (millions)")
    axes[1, 0].set_title("Model size changes with L", loc="left")
    axes[1, 1].plot(lengths, epochs, "o-", color="#208C71", lw=1.65, ms=4)
    axes[1, 1].set_ylim(0.5, summary["base_config"]["train"]["epochs"] + 0.5)
    axes[1, 1].set_ylabel("Best validation epoch")
    axes[1, 1].set_title("Checkpoint selection (not epochs actually run)", loc="left")
    for ax in axes.flat:
        style_axis(ax, lengths)
    fig.suptitle(meta + " | DD-Mamba diagnostics", fontsize=12)
    fig.tight_layout(rect=(0.015, 0.01, 0.99, 0.94), h_pad=2.0, w_pad=2.0)
    name = "electricity_dd_mamba_profile"
    save_figure(fig, out_dir, name)
    plt.close(fig)
    outputs.extend([f"figures/{name}.png", f"figures/{name}.svg"])
    return outputs


def print_tables(summary):
    models = summary["models"]
    for metric in ("mse", "mae"):
        print(f"\n## {metric.upper()}\n")
        print("| L | " + " | ".join(MODEL_LABELS.get(arch, arch) for arch in models) + " |")
        print("|---:" * (len(models) + 1) + "|")
        for length in summary["lengths"]:
            values = []
            for arch in models:
                cell = summary["cells"][str(length)][arch][metric]
                text = f"{cell['mean']:.6f}"
                if cell["sample_std"] is not None:
                    text += f" ± {cell['sample_std']:.6f}"
                values.append(f"**{text}**" if cell["rank"] == 1 else text)
            print(f"| {length} | " + " | ".join(values) + " |")
    print("\n## Model summary\n")
    print("| Model | Best MSE L | Best MSE | Best MAE L | Best MAE | Avg MSE | Mean MSE rank | MSE wins |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arch in models:
        item = summary["model_summary"][arch]
        a, b = item["mse"], item["mae"]
        print(f"| {item['label']} | {a['best_lookback']} | {a['best_value']:.6f} | {b['best_lookback']} | {b['best_value']:.6f} | {a['mean_across_lookbacks']:.6f} | {a['mean_rank']:.2f} | {a['wins']} |")
    print("\n## DD-Mamba detail\n")
    for length in summary["lengths"]:
        cell = summary["cells"][str(length)]["dual_domain"]
        runs = cell["runs"]
        print(json.dumps({"L": length, "mse": cell["mse"]["mean"],
                          "mae": cell["mae"]["mean"], "rank": cell["mse"]["rank"],
                          "best_epoch": [run["best_epoch"] for run in runs],
                          "param_count": [run["param_count"] for run in runs]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="electricity", choices=["electricity"])
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--expected-lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[2024])
    args = parser.parse_args()
    records = read_archive(args.archive)
    summary = audit_records(records, sorted(args.expected_lengths), list(MODEL_LABELS),
                            sorted(args.expected_seeds), args.horizon, args.dataset)
    summary["source_archive"] = {"filename": args.archive.name,
                                  "sha256": file_sha256(args.archive)}
    summary["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if summary["complete"]:
        summary["figures"] = render_figures(summary, args.out_dir / "figures")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "complete", "expected_runs", "observed_runs", "result_json_count",
        "fixed_config_variants", "environment_variants", "created_at_range_utc",
        "issues", "warnings")}, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit("Audit failed; figures were not generated")
    print_tables(summary)
    print("\nBest epoch counts:", json.dumps({arch: item["best_epoch_counts"]
          for arch, item in summary["model_summary"].items()}))


if __name__ == "__main__":
    main()
