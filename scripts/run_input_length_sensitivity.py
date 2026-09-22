#!/usr/bin/env python
"""Run the frozen full model across look-back lengths without overwrites."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=[48, 96, 192, 336, 720])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--out-dir", default="checkpoints/input_length")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = root / "scripts" / "run_ablation.py"
    dataset = config.stem

    for seq_len in args.seq_lengths:
        for horizon in args.horizons:
            name = f"{dataset}_L{seq_len}_H{horizon}_input_length"
            command = [
                sys.executable, str(runner), "--config", str(config),
                "--variants", "full", "--seeds", str(args.seeds),
                "--seq_len", str(seq_len), "--pred_len", str(horizon),
                "--experiment.name", name,
                "--out", str(out_dir / f"{name}.json"),
                *forwarded,
            ]
            if args.validation_only:
                command.append("--validation-only")
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
