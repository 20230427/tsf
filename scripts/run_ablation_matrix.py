#!/usr/bin/env python
"""Run one named ablation family across datasets and forecast horizons.

The wrapper gives every cell a horizon-specific experiment name and JSON path,
preventing checkpoint overwrites when the same configuration is rerun at
H=96/192/336/720. Unknown CLI options are forwarded to ``run_ablation.py``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True,
                        help="YAML files, e.g. configs/ETTh2.yaml")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[96, 192, 336, 720])
    parser.add_argument("--chain", default="core",
                        choices=["core", "smamba", "td", "placement", "init",
                                 "spectral", "fusion", "revin_alpha", "dispersion"])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--out-dir", default="checkpoints/ablation_matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args, forwarded = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_ablation.py"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for config_arg in args.configs:
        config = Path(config_arg)
        if not config.is_absolute():
            config = root / config
        if not config.exists():
            raise SystemExit(f"Config not found: {config}")
        dataset = config.stem
        for horizon in args.horizons:
            experiment_name = f"{dataset}_H{horizon}_{args.chain}"
            out_path = out_dir / f"{experiment_name}.json"
            command = [
                sys.executable, str(runner),
                "--config", str(config),
                "--chain", args.chain,
                "--seeds", str(args.seeds),
                "--pred_len", str(horizon),
                "--experiment.name", experiment_name,
                "--out", str(out_path),
                *forwarded,
            ]
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode:
                failures.append((dataset, horizon, completed.returncode))
                if not args.continue_on_error:
                    raise SystemExit(completed.returncode)

    if failures:
        details = ", ".join(f"{d}@{h} (exit {rc})" for d, h, rc in failures)
        raise SystemExit(f"Ablation cells failed: {details}")


if __name__ == "__main__":
    main()
