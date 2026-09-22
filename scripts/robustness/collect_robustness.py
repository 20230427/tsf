#!/usr/bin/env python3
"""Aggregate robustness run records into per-dataset TSV files.

Scans ``checkpoints/robustness/<dataset>_robustness_pl<H>_nl<sigma>_s<seed>_results.json``
(the provenance-schema JSONs written atomically by ``python -m src.train``
after each successful train + noisy test evaluation) and rewrites
``output/robustness/robustness_results_<dataset>.txt`` from scratch on
every invocation (idempotent aggregation: re-running never duplicates
rows). The TSVs feed ``output/robustness/robustness_plot.ipynb``.

Row schema (tab-separated)::

    dataset  pred_len  noise_level  seed  mse  mae  status  timestamp

``mse``/``mae`` come from the JSON's ``test_metrics``; ``status`` is ``ok``
for every present record (src.train only writes the JSON after a successful
evaluation) or ``no_test_metrics`` if a JSON lacks test metrics (defensive;
should not happen for robustness runs). ``timestamp`` is the JSON's mtime.

Usage:
    python scripts/robustness/collect_robustness.py \
        [--checkpoint-dir checkpoints/robustness] \
        [--output-dir output/robustness]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NAME_RE = re.compile(
    r"^(?P<dataset>.+)_robustness_pl(?P<pred_len>\d+)"
    r"_nl(?P<noise_level>[\d.]+)_s(?P<seed>\d+)$"
)

TXT_COLUMNS = [
    "dataset",
    "pred_len",
    "noise_level",
    "seed",
    "mse",
    "mae",
    "status",
    "timestamp",
]

TXT_HEADER = "\t".join(TXT_COLUMNS)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint-dir", default="checkpoints/robustness")
    cli.add_argument("--output-dir", default="output/robustness")
    opts = cli.parse_args()

    ckpt_dir = Path(opts.checkpoint_dir)
    out_dir = Path(opts.output_dir)
    if not ckpt_dir.is_dir():
        print(f"No checkpoint dir at {ckpt_dir}; nothing to collect.")
        return 0

    rows = []
    for path in sorted(ckpt_dir.glob("*_robustness_*_results.json")):
        m = NAME_RE.match(path.name[: -len("_results.json")])
        if not m:
            print(f"[warn] unrecognized name pattern, skipping: {path.name}")
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] unreadable JSON, skipping: {path.name} ({e})")
            continue
        test_metrics = record.get("test_metrics") or {}
        mse_val = test_metrics.get("mse")
        mae_val = test_metrics.get("mae")
        has_metrics = mse_val is not None and mae_val is not None
        rows.append({
            "dataset": m["dataset"],
            "pred_len": int(m["pred_len"]),
            "noise_level": m["noise_level"],
            "seed": int(m["seed"]),
            "mse": f"{float(mse_val):.8f}" if has_metrics else "",
            "mae": f"{float(mae_val):.8f}" if has_metrics else "",
            "status": "ok" if has_metrics else "no_test_metrics",
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
            ),
        })

    rows.sort(key=lambda r: (r["dataset"], r["pred_len"],
                             float(r["noise_level"]), r["seed"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    total = 0
    for dataset, ds_rows in sorted(by_dataset.items()):
        tsv = out_dir / f"robustness_results_{dataset}.txt"
        with open(tsv, "w", encoding="utf-8") as f:
            f.write(TXT_HEADER + "\n")
            for row in ds_rows:
                f.write("\t".join(str(row[h]) for h in TXT_COLUMNS) + "\n")
        n_ok = sum(1 for r in ds_rows if r["status"] == "ok")
        total += n_ok
        print(f"{tsv}: {n_ok} ok row(s) of {len(ds_rows)}")

    print(f"Collected {total} ok row(s) into {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
