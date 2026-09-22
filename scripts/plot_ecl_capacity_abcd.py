#!/usr/bin/env python3
"""Plot the reference paper's four ECL studies; never invent missing measurements.

An uploaded lookback archive supplies validation metrics for panel (a) only.
A fresh validation-only capacity-staircase report supplies all four panels.
Matplotlib is imported only during rendering; no training imports are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

try:
    from scripts.analyze_lookback_results import (
        DEFAULT_LENGTHS, MODEL_LABELS, audit_records, file_sha256, read_archive,
    )
except ModuleNotFoundError:
    from analyze_lookback_results import (
        DEFAULT_LENGTHS, MODEL_LABELS, audit_records, file_sha256, read_archive,
    )


def distribution(values):
    if not values or any(not isinstance(v, (int, float)) or
                         not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Measurements must be finite, nonnegative and nonempty")
    return {"mean": statistics.mean(values), "sample_std":
            statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def source_info(path):
    return {"path": str(path.resolve()),
            "sha256": file_sha256(path)}


def from_archive(path):
    records = read_archive(path)
    seeds = sorted({seed for _, payload, _ in records
                    for seed in payload.get("seed_values", [])})
    audit = audit_records(records, DEFAULT_LENGTHS, list(MODEL_LABELS), seeds,
                          96, "electricity")
    if not audit["complete"]:
        raise ValueError("Lookback archive audit failed: " + str(audit["issues"]))
    rows = []
    for length in DEFAULT_LENGTHS:
        runs = audit["cells"][str(length)]["dual_domain"]["runs"]
        rows.append({"x": length,
                     "mse": distribution([r["best_val_metrics"]["mse"] for r in runs]),
                     "mae": distribution([r["best_val_metrics"]["mae"] for r in runs])})
    anchor = next(row for row in rows if row["x"] == 96)
    anchor_runs = audit["cells"]["96"]["dual_domain"]["runs"]
    width = audit["base_config"]["model"]["d_model"]
    lr = audit["base_config"]["train"]["lr"]
    return {"schema": "ecl-capacity-abcd-plot-v1", "complete": False,
            "source": source_info(path), "dataset": "ECL / Electricity",
            "horizon": 96, "seeds": seeds, "anchor_lookback": 96,
            "base_d_model": audit["base_config"]["model"]["d_model"],
            "base_lr": audit["base_config"]["train"]["lr"],
            "lookback": rows,
            "width": [{"x": width, "mse": anchor["mse"],
                       "parameters_m": statistics.mean(r["param_count"] for r in anchor_runs) / 1e6}],
            "lr": [{"x": lr, "mse": anchor["mse"]}],
            "panel_status": {"a": "complete", "b": "missing", "c": "anchor_only", "d": "anchor_only"},
            "missing_panels": ["b", "c", "d"],
            "metric_scope": "best-checkpoint validation metrics",
            "notes": ["Historical source also evaluated test; not a new validation-only experiment",
                      "No training time or peak allocated GPU memory was recorded",
                      "No controlled width or learning-rate sweep was recorded",
                      "Single-seed sample standard deviation is undefined" if len(seeds) == 1
                      else "Uncertainty is sample standard deviation across seeds"]}


def from_capacity(payload):
    if payload.get("schema") != "ddmamba-capacity-staircase-v1":
        raise ValueError("Unsupported capacity report schema")
    p = payload["protocol"]
    if p["dataset"].lower() not in ("electricity", "ecl") or p["channels"] != 321:
        raise ValueError("The capacity report must be for ECL, not Weather")
    if p["evaluation_scope"] != "validation_only":
        raise ValueError("Reference panels require validation metrics, not final-test metrics")
    if payload.get("failures"):
        raise ValueError("Capacity report contains failed runs")
    records = payload.get("run_records", [])
    seed_sets, grouped, seen = [], {}, set()
    for r in records:
        key = (r["lookback"], r["d_model"], r["learning_rate"], r["seed"])
        if key in seen:
            raise ValueError("Duplicate capacity run: " + str(key))
        seen.add(key)
        if (r["dataset"].lower() not in ("electricity", "ecl") or
                r["prediction_length"] != p["prediction_length"] or
                r["phase"] != "validation_only"):
            raise ValueError("Mixed dataset, horizon, or metric scope")
        if not r.get("run_config_sha256"):
            raise ValueError("Missing run configuration hash")
        grouped.setdefault(key[:3], []).append(r)
    expected = {
        "lookback": [(v, p["base_d_model"], p["base_learning_rate"])
                     for v in p["lookbacks"]],
        "width": [(p["anchor_lookback"], v, p["base_learning_rate"])
                  for v in p["d_models"]],
        "lr": [(p["anchor_lookback"], p["base_d_model"],
                float(f"{p['base_learning_rate'] * v:.15g}"))
               for v in p["learning_rate_multipliers"]],
    }
    result = {"schema": "ecl-capacity-abcd-plot-v1", "complete": True,
              "dataset": "ECL / Electricity", "horizon": p["prediction_length"],
              "anchor_lookback": p["anchor_lookback"], "base_d_model": p["base_d_model"],
              "base_lr": p["base_learning_rate"], "missing_panels": [],
              "metric_scope": "validation-only selected-checkpoint metrics",
              "notes": ["Sample standard deviation (ddof=1); no error bars for one seed",
                        "Protocol and run fields checked; exact historical code/data cannot be reconstructed from metric records alone"]}
    freq_widths = set()
    for track, keys in expected.items():
        rows = []
        for key in keys:
            runs = grouped.get(key, [])
            if len(runs) != p["matched_seed_count"]:
                raise ValueError("Missing or incomplete capacity cell: " + str(key))
            seed_sets.append({r["seed"] for r in runs})
            freq_widths.update(r["freq_hidden"] for r in runs)
            if len({r["param_count"] for r in runs}) != 1:
                raise ValueError("Parameter count changes across seeds")
            row = {"x": key[0] if track == "lookback" else key[1] if track == "width" else key[2],
                   "mse": distribution([r["mse"] for r in runs]),
                   "mae": distribution([r["mae"] for r in runs]),
                   "parameters_m": runs[0]["param_count"] / 1e6}
            if track == "lookback":
                row["training_ms"] = distribution([r["training_ms_per_iteration"] for r in runs])
                row["memory_gib"] = distribution([r["peak_allocated_gpu_memory_mb"] / 1024 for r in runs])
            rows.append(row)
        result[track] = sorted(rows, key=lambda row: row["x"])
    if not seed_sets or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("All cells must use the same matched seeds")
    if len(freq_widths) != 1:
        raise ValueError("freq_hidden must stay fixed during the one-factor width sweep")
    result["seeds"] = sorted(seed_sets[0])
    return result


def plot(data, image_prefix, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, teal, grid = "#2772CA", "#EC7E32", "#21A398", "#D6DFEB"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))
    a, b, c, d = axes.flat
    for ax in axes.flat:
        ax.grid(True, color=grid, ls="--", lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    def ticks(ax, values, vertical=False):
        ax.set_xticks(values, labels=[f"{v:g}" for v in values])
        ax.tick_params(axis="x", labelrotation=90 if vertical else 0,
                       labelsize=8.5 if vertical else 9)
        ax.margins(x=0.035)

    def curve(ax, rows, key, color, marker, label=None):
        ys = [r[key]["mean"] for r in rows]
        std = [r[key]["sample_std"] for r in rows]
        if all(v is not None for v in std):
            label = label + " +/- 1 SD" if label else None
            return ax.errorbar([r["x"] for r in rows], ys, yerr=std,
                               color=color, marker=marker, ms=4, lw=1.5,
                               capsize=2.3, label=label)
        return ax.plot([r["x"] for r in rows], ys, color=color,
                       marker=marker, ms=4, lw=1.5, label=label)[0]

    def missing(ax, message):
        ax.text(0.5, 0.56, "NOT MEASURED", ha="center", va="center",
                transform=ax.transAxes, color="#606B76", fontsize=13)
        ax.text(0.5, 0.40, message, ha="center", va="center",
                transform=ax.transAxes, color="#606B76", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    lookback = data["lookback"]
    curve(a, lookback, "mse", blue, "o", "Validation MSE")
    curve(a, lookback, "mae", orange, "s", "Validation MAE")
    a.axvline(data["anchor_lookback"], color="#8657A8", ls=":", lw=1)
    ticks(a, [r["x"] for r in lookback], vertical=True)
    a.set_title("(a) Validation error versus lookback")
    a.set_xlabel("Lookback length L (hourly steps)")
    a.set_ylabel("Selected-checkpoint validation error")
    a.legend(frameon=False, loc="best", fontsize=9)

    b.set_title("(b) Lookback resource growth")
    b.set_xlabel("Lookback length L (hourly steps)")
    b.set_ylabel("Training time (ms/iteration)", color=blue)
    b2 = b.twinx()
    b2.set_ylabel("Peak allocated memory (GiB)", color=orange)
    if "b" in data["missing_panels"]:
        missing(b, "Need measured training time and peak GPU memory\nfor each lookback")
        b2.set_yticks([])
    else:
        h1 = curve(b, lookback, "training_ms", blue, "o", "Training time")
        h2 = curve(b2, lookback, "memory_gib", orange, "D", "Peak memory")
        ticks(b, [r["x"] for r in lookback], vertical=True)
        b.legend([h1, h2], ["Training time", "Peak memory"], frameon=False,
                 fontsize=9, loc="upper left")
    b.tick_params(axis="y", colors=blue)
    b2.tick_params(axis="y", colors=orange)

    c.set_title(f"(c) Width staircase at fixed L = {data['anchor_lookback']}")
    c.set_xlabel("Model width d_model")
    c.set_ylabel("Selected-checkpoint validation MSE", color=blue)
    c2 = c.twinx()
    c2.set_ylabel("Trainable parameters (M)", color=orange)
    if "c" in data["missing_panels"] and not data["width"]:
        missing(c, "Need a controlled d_model = 64 / 128 / 256 / 512 sweep\nwith lookback and learning rate held fixed")
        c2.set_yticks([])
    else:
        width = data["width"]
        h1 = curve(c, width, "mse", blue, "o", "Validation MSE")
        h2, = c2.plot([r["x"] for r in width], [r["parameters_m"] for r in width],
                      color=orange, marker="D", ls="--", lw=1.4, ms=4)
        c.set_xscale("log", base=2)
        ticks(c, [r["x"] for r in width])
        c.legend([h1, h2], ["Validation MSE", "Parameters"], frameon=False,
                 fontsize=9, loc="upper left")
        if "c" in data["missing_panels"]:
            # Single-point twin scales put both markers at their midpoint.
            # A larger hollow MSE marker keeps both observed quantities visible.
            mse_marker = h1.lines[0] if hasattr(h1, "lines") else h1
            mse_marker.set_markersize(8)
            mse_marker.set_markerfacecolor("none")
            c.set_xlim(56, 600)
            c.set_xticks([64, 128, 256, 512], labels=["64", "128", "256", "512"])
            c.text(0.45, 0.53, "ANCHOR ONLY\nWidth sweep is missing", transform=c.transAxes,
                   ha="center", fontsize=11, color="#606B76")
    c.tick_params(axis="y", colors=blue)
    c2.tick_params(axis="y", colors=orange)

    d.set_title("(d) Learning-rate micro-sweep")
    d.set_xlabel("Learning rate")
    d.set_ylabel("Selected-checkpoint validation MSE")
    if "d" in data["missing_panels"] and not data["lr"]:
        missing(d, f"Need a controlled LR sweep at fixed L = {data['anchor_lookback']}\nand fixed d_model = {data['base_d_model']}")
    else:
        curve(d, data["lr"], "mse", teal, "o")
        ticks(d, [r["x"] for r in data["lr"]])
        d.ticklabel_format(axis="y", useOffset=False)
        if "d" in data["missing_panels"]:
            d.set_xlim(data["base_lr"] * 0.4, data["base_lr"] * 1.6)
            ticks(d, [data["base_lr"] * v for v in (0.5, 1.0, 1.5)])
            d.text(0.5, 0.73, "ANCHOR ONLY - LR sweep is missing", transform=d.transAxes,
                   ha="center", fontsize=11, color="#606B76")

    status = "Complete validation studies" if data["complete"] else "DRAFT - (b) unmeasured; (c)/(d) have an anchor, not complete sweeps"
    fig.suptitle(f"DD-Mamba | ECL | H={data['horizon']}", fontsize=13)
    fig.text(0.5, 0.012, status + (" | Single seed: no SD error bars" if len(data["seeds"]) == 1 else ""),
             ha="center", fontsize=10, color="#606B76")
    fig.tight_layout(rect=(0.005, 0.04, 0.995, 0.95), h_pad=2.2, w_pad=3.2)
    image_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(image_prefix.with_suffix("." + ext), dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--capacity-results", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--pdf-path", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Refuse incomplete four-panel figures")
    args = parser.parse_args()
    if args.archive:
        data = from_archive(args.archive)
    else:
        data = from_capacity(json.loads(args.capacity_results.read_text(encoding="utf-8")))
        data["source"] = source_info(args.capacity_results)
    if args.strict and not data["complete"]:
        parser.error("Panels b/c/d require new ECL measurements; no complete figure was written")
    data["plot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plot(data, args.out_prefix, args.pdf_path)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": data["complete"], "missing_panels": data["missing_panels"],
                      "pdf": str(args.pdf_path), "image_prefix": str(args.out_prefix)}, indent=2))


if __name__ == "__main__":
    main()
