#!/usr/bin/env bash
# Convenience launcher. Edit the overrides below or pass extra flags:
#   ./scripts/train.sh --seq_len 192 --pred_len 96 --epochs 50
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.train \
  --config configs/default.yaml \
  --device auto \
  "$@"
