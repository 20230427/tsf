#!/usr/bin/env python
"""Quick smoke run: train a few epochs on synthetic data end-to-end.

    python scripts/run_synthetic.py

Handy for verifying the install (and GPU) work before pointing at real data.
"""
import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import load_config  # noqa: E402


def main():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    # Keep it fast for a smoke test.
    cfg["train"]["epochs"] = 3
    cfg["data"]["synthetic_length"] = 3000
    cfg["experiment"]["name"] = "synthetic_smoke"
    train(cfg)


if __name__ == "__main__":
    main()
