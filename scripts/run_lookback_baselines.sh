#!/usr/bin/env bash
# Weather look-back sensitivity sweep: DD-Mamba and eight matched baselines.
#
# Default experiment:
#   dataset:  Weather
#   horizon:  96
#   lookback: 24 48 72 96 144 192 288 336 480 720 1008
#   models:   DD-Mamba, S-Mamba, iTransformer, PatchTST, Crossformer,
#             TiDE, DLinear, FEDformer, Autoformer
#   seeds:    1 (base seed 2024 by default)
#
# The launcher detaches itself with setsid + nohup. Run it once from any
# directory and then follow the log path printed to the terminal:
#
#   bash scripts/run_lookback_baselines.sh
#   bash scripts/run_lookback_baselines.sh --gpu 1 --batch-size 4
#   bash scripts/run_lookback_baselines.sh --base-seed 2025
#   bash scripts/run_lookback_baselines.sh --foreground   # debugging
#   bash scripts/run_lookback_baselines.sh --force        # rerun valid outputs
#
# Arguments after "--" are forwarded to run_unified_baselines.py. For example:
#
#   bash scripts/run_lookback_baselines.sh -- --train.epochs 20
#
# Completed JSON files are validated and skipped on relaunch unless --force is
# present. A JSON containing any failed/missing architecture is not considered
# complete and will be rerun.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$PROJECT_ROOT"

MODE="launcher"
if [[ "${1:-}" == "--worker" ]]; then
    MODE="worker"
    shift
fi
ORIGINAL_ARGS=("$@")

CONFIG="configs/weather.yaml"
HORIZON=96
SEEDS=1
BASE_SEED="${LOOKBACK_BASE_SEED:-2024}"
BATCH_SIZE=8
GPU_ID="${LOOKBACK_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="checkpoints/lookback"
FOREGROUND=0
FORCE=0
LENGTHS=(24 48 72 96 144 192 288 336 480 720 1008)
ARCHS=(
    dual_domain
    smamba
    itransformer
    patchtst
    crossformer
    tide
    dlinear
    fedformer
    autoformer
)
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_lookback_baselines.sh [options] [-- extra overrides]

Options:
  --gpu ID             Physical/logical CUDA device id (default: 0)
  --seeds N            Seeds per model and lookback (default: 1)
  --base-seed N        First seed (default: 2024)
  --batch-size N       Common batch size for every run (default: 8)
  --python PATH        Python executable (default: $PYTHON_BIN or python)
  --output-dir DIR     Result JSON directory (default: checkpoints/lookback)
  --foreground         Do not detach; useful for debugging
  --force              Rerun cells whose aggregate JSON is already complete
  -h, --help           Show this help

The experiment uses configs/weather.yaml, H=96, and the predeclared lookback
and architecture arrays at the top of this file. Arguments after -- are passed
to scripts/run_unified_baselines.py, so training settings can be overridden
without editing the launcher.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="${2:?--gpu requires an id}"
            shift 2
            ;;
        --seeds)
            SEEDS="${2:?--seeds requires an integer}"
            shift 2
            ;;
        --base-seed)
            BASE_SEED="${2:?--base-seed requires an integer}"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="${2:?--batch-size requires an integer}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:?--python requires a path}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?--output-dir requires a directory}"
            shift 2
            ;;
        --foreground)
            FOREGROUND=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$SEEDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --seeds must be a positive integer, got: $SEEDS" >&2
    exit 2
fi
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --base-seed must be a non-negative integer, got: $BASE_SEED" >&2
    exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --batch-size must be a positive integer, got: $BATCH_SIZE" >&2
    exit 2
fi

CONFIG_ABS="$PROJECT_ROOT/$CONFIG"
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR_ABS="$OUTPUT_DIR"
else
    OUTPUT_DIR_ABS="$PROJECT_ROOT/$OUTPUT_DIR"
fi
LOG_ROOT="$PROJECT_ROOT/log/lookback"
PID_FILE="$LOG_ROOT/weather_H${HORIZON}_seed${BASE_SEED}.pid"

preflight() {
    if [[ ! -f "$CONFIG_ABS" ]]; then
        echo "ERROR: config not found: $CONFIG_ABS" >&2
        return 1
    fi
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
        return 1
    fi
    if ! command -v setsid >/dev/null 2>&1 && [[ "$FOREGROUND" -eq 0 ]]; then
        echo "ERROR: setsid is required for background launch" >&2
        return 1
    fi
    if ! command -v nohup >/dev/null 2>&1 && [[ "$FOREGROUND" -eq 0 ]]; then
        echo "ERROR: nohup is required for background launch" >&2
        return 1
    fi

    echo "[preflight] checking CUDA, official Mamba kernels, and Weather data"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - "$CONFIG_ABS" <<'PY'
import sys

import torch
from mamba_ssm import Mamba

from src.data.dataset import load_raw_series
from src.utils import load_config

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is not available on the selected device")

# S-Mamba has no pure-PyTorch fallback. A real forward catches an installed
# package whose compiled kernels do not support the selected GPU architecture.
probe = Mamba(d_model=16, d_state=2, d_conv=2, expand=1).cuda().eval()
x = torch.randn(1, 4, 16, device="cuda")
with torch.no_grad():
    probe(x)
torch.cuda.synchronize()

cfg = load_config(sys.argv[1])
d = cfg["data"]
series = load_raw_series(
    source=d["source"],
    csv_path=d.get("csv_path"),
    target_columns=d.get("target_columns"),
    synthetic_length=d.get("synthetic_length", 8000),
    synthetic_channels=d.get("synthetic_channels", 7),
    seed=cfg["experiment"]["seed"],
    npz_key=d.get("npz_key", "data"),
    npz_feature=d.get("npz_feature", 0),
)
if len(series) < 1008 + 96:
    raise SystemExit(
        f"Weather series is too short for L=1008, H=96: rows={len(series)}"
    )
print(
    f"[preflight] GPU={torch.cuda.get_device_name(0)}; "
    f"Weather shape={tuple(series.shape)}"
)
PY
}

result_is_complete() {
    local result_path="$1"
    [[ -s "$result_path" ]] || return 1
    "$PYTHON_BIN" - "$result_path" "$HORIZON" "$SEEDS" "$BASE_SEED" "${ARCHS[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
horizon = sys.argv[2]
expected_seeds = int(sys.argv[3])
base_seed = int(sys.argv[4])
expected_seed_values = list(range(base_seed, base_seed + expected_seeds))
expected_archs = sys.argv[5:]

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[incomplete] cannot read {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

failures = payload.get("failures") or {}
results = payload.get("results") or {}
problems = []
if failures:
    problems.append("recorded failures: " + ", ".join(sorted(failures)))
if payload.get("seed_values") != expected_seed_values:
    problems.append(
        f"seed_values={payload.get('seed_values')}, expected {expected_seed_values}"
    )

for arch in expected_archs:
    cell = (results.get(arch) or {}).get(horizon)
    if not isinstance(cell, dict):
        problems.append(f"{arch}: missing H={horizon}")
        continue
    runs = cell.get("runs")
    if not isinstance(runs, list) or len(runs) != expected_seeds:
        count = len(runs) if isinstance(runs, list) else 0
        problems.append(f"{arch}: expected {expected_seeds} runs, found {count}")
        continue
    actual_seeds = [run.get("seed") for run in runs]
    if actual_seeds != expected_seed_values:
        problems.append(
            f"{arch}: seeds={actual_seeds}, expected {expected_seed_values}"
        )

if problems:
    print(f"[incomplete] {path}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(1)
PY
}

if [[ "$MODE" == "launcher" ]]; then
    mkdir -p "$LOG_ROOT" "$OUTPUT_DIR_ABS"

    if [[ -f "$PID_FILE" ]]; then
        EXISTING_PID="$(<"$PID_FILE")"
        if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            echo "ERROR: lookback sweep is already running (PID $EXISTING_PID)" >&2
            echo "  PID file: $PID_FILE" >&2
            exit 1
        fi
        rm -f "$PID_FILE"
    fi

    preflight

    if [[ "$FOREGROUND" -eq 1 ]]; then
        exec bash "$SCRIPT_PATH" --worker "${ORIGINAL_ARGS[@]}"
    fi

    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_LOG="$LOG_ROOT/weather_H${HORIZON}_seed${BASE_SEED}_${TIMESTAMP}.log"

    # Detach the complete matrix, not individual training processes. This
    # keeps the nine models and eleven lookbacks sequential on one GPU while
    # surviving terminal/SSH disconnection.
    setsid nohup bash "$SCRIPT_PATH" --worker "${ORIGINAL_ARGS[@]}" \
        >"$RUN_LOG" 2>&1 < /dev/null &
    WORKER_PID=$!
    printf '%s\n' "$WORKER_PID" > "$PID_FILE"
    disown "$WORKER_PID" 2>/dev/null || true

    echo "[launched] Weather lookback sweep is running in the background"
    echo "  PID:     $WORKER_PID"
    echo "  Log:     $RUN_LOG"
    echo "  Results: $OUTPUT_DIR_ABS/weather_seed${BASE_SEED}_L*_H${HORIZON}.json"
    echo "  Follow:  tail -f \"$RUN_LOG\""
    exit 0
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR_ABS"
printf '%s\n' "$$" > "$PID_FILE"

cleanup() {
    if [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$$" ]]; then
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TOTAL_RUNS=$(( ${#LENGTHS[@]} * ${#ARCHS[@]} * SEEDS ))
echo "============================================================"
echo "[$(date '+%F %T')] Weather lookback sensitivity sweep"
echo "  Horizon:   $HORIZON"
echo "  Lookbacks: ${LENGTHS[*]}"
echo "  Models:    ${ARCHS[*]}"
echo "  Seeds:     $SEEDS"
echo "  Base seed: $BASE_SEED"
echo "  Batch:     $BATCH_SIZE"
echo "  GPU:       $GPU_ID"
echo "  Runs:      $TOTAL_RUNS"
echo "  Output:    $OUTPUT_DIR_ABS"
echo "============================================================"

for seq_len in "${LENGTHS[@]}"; do
    OUT_FILE="$OUTPUT_DIR_ABS/weather_seed${BASE_SEED}_L${seq_len}_H${HORIZON}.json"
    if [[ "$FORCE" -eq 0 ]] && result_is_complete "$OUT_FILE" >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] [skip] complete: L=$seq_len"
        continue
    fi

    EXPERIMENT_NAME="weather_lookback_seed${BASE_SEED}_L${seq_len}_H${HORIZON}"
    echo "[$(date '+%F %T')] [start] L=$seq_len"

    COMMAND=(
        "$PYTHON_BIN" scripts/run_unified_baselines.py
        --config "$CONFIG_ABS"
        --archs "${ARCHS[@]}"
        --horizons "$HORIZON"
        --seeds "$SEEDS"
        --experiment.seed "$BASE_SEED"
        --data.seq_len "$seq_len"
        --train.batch_size "$BATCH_SIZE"
        --experiment.name "$EXPERIMENT_NAME"
        --out "$OUT_FILE"
        "${EXTRA_ARGS[@]}"
    )
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "${COMMAND[@]}"

    # run_unified_baselines.py deliberately records an architecture failure
    # and continues. Fail closed here so a missing S-Mamba/FEDformer result is
    # never mistaken for a completed comparison matrix.
    result_is_complete "$OUT_FILE"
    echo "[$(date '+%F %T')] [done] L=$seq_len -> $OUT_FILE"
done

echo "============================================================"
echo "[$(date '+%F %T')] All lookback cells completed successfully"
echo "  Result files: $OUTPUT_DIR_ABS/weather_seed${BASE_SEED}_L*_H${HORIZON}.json"
echo "============================================================"
