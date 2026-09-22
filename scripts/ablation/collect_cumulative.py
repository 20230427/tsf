#!/usr/bin/env python
"""Aggregate cumulative-ablation cells into one provenance JSON + markdown.

Scans ``checkpoints/cumulative_ablation/<ds>_cum_<variant>_h<H>_s<seed>_results.json``
(the provenance-schema JSONs written atomically by ``python -m src.train``
after each successful train + test) and rewrites, idempotently:

  - ``output/cumulative_ablation/cumulative_ablation.json`` -- provenance
    record wrapping every cell (schema ``ddmamba-experiment-v1``; validates
    with ``scripts/validate_provenance.py``)
  - ``output/cumulative_ablation/cumulative_ablation.md`` -- per-dataset
    markdown tables (variants x horizons, Avg + delta vs Full)

Verification per cell (the DD-Mamba counterpart of Pamba's ``des`` string
consistency check in update_ablation.py, made stronger -- the cell's whole
resolved config is checked, not a label):

  1. ``validate_provenance_record`` must classify it ``verified``.
  2. Every cumulative override expected for the cell's variant (from
     cumulative_registry) must match the run's ``resolved_config`` --
     a stale JSON from a renamed/reordered chain step fails loudly instead
     of silently poisoning the table.
  3. ``test_metrics`` must be present (the suite evaluates test directly).
  4. The run's seed and horizon must match its file name.

Exit code is non-zero when any cell fails verification (fail-closed:
tables must not be built from unverifiable records), after reporting every
offending cell and key. ``--allow-invalid`` downgrades to a warning for
inspection only.

Usage:
    python scripts/ablation/collect_cumulative.py
    python scripts/ablation/collect_cumulative.py --datasets ETTh1 weather \
        --checkpoint-dir checkpoints/cumulative_ablation --output-dir output/cumulative_ablation
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablation.cumulative_registry import (  # noqa: E402
    CUMULATIVE_STEPS,
    VARIANT_DISPLAYS,
    VARIANT_NAMES,
    config_path_for,
    noop_variants_for_dataset,
    pred_lens_for,
    resolve_overrides,
)
from src.utils import atomic_write_json, provenance_fields, validate_provenance_record  # noqa: E402

NAME_RE = re.compile(
    r"^(?P<dataset>.+)_cum_(?P<variant>[a-z0-9_]+)"
    r"_h(?P<pred_len>\d+)_s(?P<seed>\d+)$"
)


def _load_cells(ckpt_dir: Path) -> list[dict]:
    import json

    cells = []
    for path in sorted(ckpt_dir.glob("*_cum_*_h*_s*_results.json")):
        m = NAME_RE.match(path.name[: -len("_results.json")])
        if not m:
            print(f"[warn] name not matching the cumulative pattern, skipping: {path.name}")
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] unreadable JSON, skipping: {path.name} ({e})")
            continue
        cells.append({
            "dataset": m["dataset"],
            "variant": m["variant"],
            "pred_len": int(m["pred_len"]),
            "seed": int(m["seed"]),
            "path": path,
            "record": record,
        })
    return cells


def _verify_cell(cell: dict) -> list[str]:
    """Return a list of violation strings (empty = verified)."""
    problems = []
    record = cell["record"]

    verdict = validate_provenance_record(record)
    if verdict["status"] != "verified":
        problems.append(f"provenance {verdict['status']}: {verdict['reasons']}")

    if cell["variant"] not in VARIANT_NAMES:
        problems.append(f"unknown variant {cell['variant']!r} (stale chain?)")
        return problems

    idx = VARIANT_NAMES.index(cell["variant"])
    resolved = record.get("resolved_config") or {}
    for key, expected in resolve_overrides(idx).items():
        section, _, field = key.partition(".")
        actual = (resolved.get(section) or {}).get(field)
        if isinstance(expected, bool) != isinstance(actual, bool) or actual != expected:
            problems.append(
                f"config mismatch on {key}: expected {expected!r}, "
                f"record has {actual!r} (stale result for this variant?)"
            )

    run = (record.get("run_records") or [{}])[0]
    if run.get("seed") != cell["seed"]:
        problems.append(f"seed mismatch: name {cell['seed']}, record {run.get('seed')}")
    if run.get("horizon") != cell["pred_len"]:
        problems.append(f"horizon mismatch: name {cell['pred_len']}, record {run.get('horizon')}")
    if not record.get("test_metrics") or "mse" not in (record.get("test_metrics") or {}):
        problems.append("missing test_metrics (evaluation_scope != validation_and_test)")
    return problems


def _mean_std(values: list[float]):
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def _render_markdown(aggregated: dict, datasets: list[str], seeds_note: str) -> str:
    lines = [
        "# Cumulative step-down ablation -- DD-Mamba",
        "",
        "Test-split MSE/MAE. The chain peels components cumulatively from the",
        "per-dataset tuned config (Full); structural no-ops (component already off",
        "in the recipe) are marked '=' with the copy source in parentheses.",
        seeds_note,
        "",
    ]
    for dataset in datasets:
        ds_data = aggregated.get(dataset, {})
        noop = noop_variants_for_dataset(config_path_for(dataset))
        pred_lens = pred_lens_for(dataset)
        lines.append(f"## {dataset}")
        header = "| Variant | " + " | ".join(
            f"H{pl} MSE | H{pl} MAE" for pl in pred_lens) + " | Avg MSE | ΔMSE vs Full |"
        sep = "|---" * (2 + 2 * len(pred_lens) + 1) + "|---|"
        lines += [header, sep]
        full_avg = None
        for name in VARIANT_NAMES:
            row = ds_data.get(name)
            cells_txt, mse_vals = [], []
            for pl in pred_lens:
                entry = (row or {}).get(str(pl)) or (row or {}).get(pl)
                if entry is None:
                    cells_txt += ["—", "—"]
                else:
                    mse_vals.append(entry["mse_mean"])
                    cells_txt += [f"{entry['mse_mean']:.3f}", f"{entry['mae_mean']:.3f}"]
            avg = f"{sum(mse_vals) / len(mse_vals):.3f}" if len(mse_vals) == len(pred_lens) else "—"
            if name == "full" and len(mse_vals) == len(pred_lens):
                full_avg = sum(mse_vals) / len(pred_lens)
            if name in noop:
                label = f"{VARIANT_DISPLAYS[name]} (= {noop[name]})"
                delta = "—"
            else:
                label = VARIANT_DISPLAYS[name]
                if full_avg is not None and len(mse_vals) == len(pred_lens):
                    delta = f"{sum(mse_vals) / len(pred_lens) - full_avg:+.3f}"
                else:
                    delta = "—"
            lines.append(f"| {label} | " + " | ".join(cells_txt) + f" | {avg} | {delta} |")
        lines += ["", f"pred_lens: {'/'.join(map(str, pred_lens))}", ""]
    lines += [
        "Note: single-seed deltas below ~0.005 MSE are noise (AGENTS.md);",
        "rerun with `--seed` shifts and this collector's seed aggregation",
        "before concluding anything from small deltas.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import json

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint-dir", default="checkpoints/cumulative_ablation")
    cli.add_argument("--output-dir", default="output/cumulative_ablation")
    cli.add_argument("--datasets", nargs="+", default=None,
                     help="restrict to these dataset stems (default: discovered)")
    cli.add_argument("--allow-invalid", action="store_true",
                     help="report but do not fail on invalid cells (inspection only)")
    opts = cli.parse_args()

    ckpt_dir = Path(opts.checkpoint_dir)
    out_dir = Path(opts.output_dir)
    if not ckpt_dir.is_dir():
        print(f"No checkpoint dir at {ckpt_dir}; nothing to collect.")
        return 0

    cells = _load_cells(ckpt_dir)
    if opts.datasets:
        wanted = set(opts.datasets)
        cells = [c for c in cells if c["dataset"] in wanted]

    ok_cells, bad = [], []
    for cell in cells:
        problems = _verify_cell(cell)
        if problems:
            bad.append((cell, problems))
        else:
            ok_cells.append(cell)

    for cell, problems in bad:
        print(f"[invalid] {cell['path'].name}:")
        for p in problems:
            print(f"          - {p}")
    if bad and not opts.allow_invalid:
        print(f"\n{len(bad)} cell(s) failed verification; refusing to build tables "
              "(use --allow-invalid to inspect anyway).")
        return 1

    # ---- aggregate: results[dataset][variant][pred_len] -> seed-aggregated ----
    aggregated: dict = {}
    run_records = []
    all_seeds: set[int] = set()
    for cell in ok_cells:
        record = cell["record"]
        run = (record.get("run_records") or [{}])[0]
        metrics = record["test_metrics"]
        all_seeds.add(cell["seed"])
        entry = aggregated.setdefault(cell["dataset"], {}) \
            .setdefault(cell["variant"], {}).setdefault(str(cell["pred_len"]), {
                "seeds": [], "mse": [], "mae": [],
            })
        entry["seeds"].append(cell["seed"])
        entry["mse"].append(float(metrics["mse"]))
        entry["mae"].append(float(metrics["mae"]))
        run_records.append({
            "dataset": cell["dataset"],
            "variant": cell["variant"],
            "seed": cell["seed"],
            "horizon": cell["pred_len"],
            "arch": run.get("arch", "dual_domain"),
            "phase": run.get("phase"),
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
            "best_val_loss": run.get("best_val_loss"),
            "best_epoch": run.get("best_epoch"),
            "param_count": run.get("param_count"),
            "active_param_count": run.get("active_param_count"),
            "run_name": cell["path"].name[: -len("_results.json")],
            "created_at_utc": record.get("created_at_utc"),
            "run_config_sha256": run.get("run_config_sha256"),
        })

    for ds_data in aggregated.values():
        for variant_data in ds_data.values():
            for entry in variant_data.values():
                mse_mean, mse_std = _mean_std(entry.pop("mse"))
                mae_mean, mae_std = _mean_std(entry.pop("mae"))
                entry["n_seeds"] = len(entry["seeds"])
                entry["mse_mean"] = mse_mean
                entry["mse_std"] = mse_std
                entry["mae_mean"] = mae_mean
                entry["mae_std"] = mae_std

    datasets = sorted({c["dataset"] for c in ok_cells})
    plan = {
        "experiment": {"kind": "cumulative_ablation"},
        "chain": [
            {"name": s["name"], "display": s["display"], "overrides": s["overrides"]}
            for s in CUMULATIVE_STEPS
        ],
        "datasets": datasets,
        "checkpoint_dir": str(ckpt_dir),
    }
    payload = provenance_fields(plan, config_path=None, seed_values=sorted(all_seeds))
    payload["results"] = aggregated
    payload["run_records"] = run_records
    payload["verification"] = {
        "verified_cells": len(ok_cells),
        "invalid_cells": len(bad),
        "allow_invalid": bool(opts.allow_invalid),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cumulative_ablation.json"
    atomic_write_json(json_path, payload)

    seeds_note = (f"Seeds: {sorted(all_seeds)}"
                  + (" (single seed per cell)" if len(all_seeds) <= 1 else " (mean over seeds shown)"))
    md_path = out_dir / "cumulative_ablation.md"
    md_path.write_text(_render_markdown(aggregated, datasets, seeds_note), encoding="utf-8")

    verdict = validate_provenance_record(payload)
    print(f"Collected {len(ok_cells)} verified cell(s) into {json_path} "
          f"(provenance: {verdict['status']})")
    if verdict["status"] != "verified":
        print(f"[error] aggregated record failed validation: {verdict['reasons']}")
        return 1
    print(f"Markdown summary: {md_path}")
    if bad:
        print(f"[warn] {len(bad)} invalid cell(s) excluded (see above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
