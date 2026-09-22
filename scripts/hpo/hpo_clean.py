#!/usr/bin/env python3
"""Clean stale HPO trial checkpoints and intermediate files.

Reclaims storage used by HPO trials in one shot: per-trial ``_best.pt``
checkpoints (both the current ``<ds>_pl<pl>_hpo_t<n>_s<i>`` naming and the
legacy pre-fix one) and dead launcher pid files. **JSON results are never
deleted**: exported best-params records, per-run results JSONs, CSVs and
the SQLite study DBs under ``output/hpo/`` all stay.

There is no liveness protection by design: every matched HPO trial
checkpoint is deleted, so run this only when no sweep needs them.

Examples
--------
    python scripts/hpo/hpo_clean.py                     # dry run (list only)
    python scripts/hpo/hpo_clean.py --confirm           # actually delete
    python scripts/hpo/hpo_clean.py --dataset electricity --confirm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.hpo.cleanup import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_PID_DIR,
    run_cleanup,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--confirm", action="store_true",
                        help="delete the listed files (default: dry run)")
    parser.add_argument("--dataset", default=None,
                        help="restrict to one dataset (e.g. electricity)")
    parser.add_argument("--pred-len", type=int, default=None, dest="pred_len",
                        help="restrict to one horizon (new naming only)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--pid-dir", default=DEFAULT_PID_DIR)
    args = parser.parse_args()

    report = run_cleanup(
        confirm=args.confirm,
        checkpoint_dir=args.checkpoint_dir,
        pid_dir=args.pid_dir,
        dataset=args.dataset,
        pred_len=args.pred_len,
    )

    mode = "DELETE" if args.confirm else "DRY RUN (use --confirm to delete)"
    print(f"[hpo_clean] {mode}")
    for path, size_mb in report["checkpoints"]:
        print(f"  checkpoint {size_mb:9.2f} MB  {path}")
    for pid_file in report["pid_files"]:
        print(f"  dead-pid                  {pid_file}")
    print(f"[hpo_clean] {len(report['checkpoints'])} checkpoint(s), "
          f"{report['freed_mb']:.1f} MB; "
          f"{len(report['pid_files'])} dead pid file(s); "
          "JSON results / CSVs / study DBs never touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
