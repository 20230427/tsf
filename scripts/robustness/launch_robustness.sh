#!/bin/bash
# ============================================================
# DD-Mamba Robustness Runner (multi-GPU)
# ============================================================
#
# Test-time Gaussian-noise robustness sweep: 2 datasets (ETTh2, weather)
# x noise levels {0, 0.1, 0.2, 0.3, 0.5} x pred_lens {96, 192, 336, 720}
# x seed(s). Each cell is one Pamba-style per-dataset shell script
# (scripts/robustness/<dataset>.sh) driving the project's default entry
# point (python -m src.train) with the tuned configs/<dataset>.yaml: train
# clean, then evaluate the test split under additive Gaussian noise on the
# standardized input window (sigma=0 is the clean anchor). Per-run
# provenance JSONs land in checkpoints/robustness/ and are aggregated into
# output/robustness/robustness_results_<dataset>.txt by
# scripts/robustness/collect_robustness.py at the end of the sweep.
#
# Idempotent: relaunching skips cells whose results.json already exists,
# so interrupted sweeps resume where they stopped.
#
# Ported from scripts/hyperparam/launch_sensitivity.sh (Pamba-style
# dispatch); the sweep dimensions are (dataset, noise_level, pred_len,
# seed) instead of (dataset, param, value, seed).
#
# Usage:
#   scripts/robustness/launch_robustness.sh                       # GPU 0, 1 slot/GPU
#   scripts/robustness/launch_robustness.sh --para 2              # GPU 0, 2 slots/GPU
#   scripts/robustness/launch_robustness.sh 0 2                   # GPUs 0,2, 1 slot/GPU
#   scripts/robustness/launch_robustness.sh --para 2 0 2          # GPUs 0,2, 2 slots each
#
# The script detaches immediately. All output is appended to:
#   log/robustness/all.log
#   log/robustness/robustness_<timestamp>/worker_*.log
#   log/robustness/robustness_<timestamp>/run.log
#   log/robustness/robustness_<timestamp>/summary.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# DATASETS must have a scripts/robustness/<dataset>.sh runner and a tuned
# configs/<dataset>.yaml. ETTh2 requires its canonical split_protocol (set
# in configs/ETTh2.yaml). Keep in lockstep with the plotting notebook.
DATASETS=(ETTh2 weather)

PARA=1
GPUS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --para) PARA="$2"; shift 2 ;;
        *)      GPUS+=("$1"); shift ;;
    esac
done

if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0)
fi

export ROBUSTNESS_DATASETS="${DATASETS[*]}"

mkdir -p log/robustness output/robustness

# GPU precheck -- refuse to dispatch if any requested GPU is not visible to
# torch. Without this, every cell fails in <1s with RuntimeError at
# model.to(device) and no results.json is ever written.
for _g in "${GPUS[@]}"; do
    if ! CUDA_VISIBLE_DEVICES="$_g" python3 -c "
import sys, torch
try:
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
except AssertionError:
    print('ERROR: GPU $_g not visible to torch (cuda.is_available()=False or device_count=0)', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        echo "[$(date '+%H:%M:%S')] GPU precheck failed for GPU $_g. Aborting." >&2
        exit 1
    fi
done

LOG_FILE="$PROJECT_ROOT/log/robustness/all.log"

nohup bash -c '
set -u
PROJECT_ROOT="'"${PROJECT_ROOT}"'"
SCRIPT_DIR="'"${SCRIPT_DIR}"'"
DATASETS_STR="'"${ROBUSTNESS_DATASETS}"'"
PARA="'"${PARA}"'"
GPUS_STR="'"${GPUS[*]}"'"

cd "$PROJECT_ROOT"

# PID lock to prevent concurrent robustness runs
exec 200>/tmp/_ddmamba_robustness.lock
if ! flock -n 200; then
    echo "[$(date "+%H:%M:%S")] ERROR: Another robustness run is in progress" >&2
    exit 1
fi

# Session directory for global FIFO semaphores + counter
export _ROBUSTNESS_SESSION="/tmp/_ddmamba_robustness_$$"
mkdir -p "$_ROBUSTNESS_SESSION"
trap '"'"'rm -rf "${_ROBUSTNESS_SESSION:-}"'"'"' EXIT

export GPU_LIST="$GPUS_STR"
export PARA_PER_GPU="$PARA"
export _GPU_COUNTER_FILE="${_ROBUSTNESS_SESSION}/counter"
echo 0 > "$_GPU_COUNTER_FILE"

export ROBUSTNESS_CKPT_DIR="$PROJECT_ROOT/checkpoints/robustness"

# Per-run log directory
LOG_DIR="$PROJECT_ROOT/log/robustness/robustness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
export _ROBUSTNESS_LOG_DIR="$LOG_DIR"

read -ra DATASETS <<< "$DATASETS_STR"

source "${SCRIPT_DIR}/robustness_common.sh"

TOTAL_CELLS=0
for ds in "${DATASETS[@]}"; do
    TOTAL_CELLS=$((TOTAL_CELLS + ${#NOISE_LEVELS[@]} * ${#PRED_LENS[@]} * ${#SEEDS[@]}))
done
TOTAL_WORKERS=$((${#_GPU_ARR[@]} * _PARA_PER_GPU))

echo "============================================================"
echo "[$(date "+%H:%M:%S")] Robustness (test-time Gaussian noise) — DD-Mamba"
echo "  Datasets:     ${DATASETS[*]}"
echo "  Pred lens:    ${PRED_LENS[*]}"
echo "  Seeds:        ${SEEDS[*]} (${#SEEDS[@]} per cell)"
echo "  Noise levels: ${NOISE_LEVELS[*]} (${#NOISE_LEVELS[@]} per cell, 0 = clean anchor)"
echo "  Total cells:  $TOTAL_CELLS (one src.train run each: train clean, test noisy)"
echo "  GPUs:         ${#_GPU_ARR[@]} x ${_PARA_PER_GPU} parallel = $TOTAL_WORKERS workers"
echo "  Per-run JSON: $ROBUSTNESS_CKPT_DIR/"
echo "  Aggregated:   $PROJECT_ROOT/output/robustness/robustness_results_*.txt"
echo "  Logs:         $LOG_DIR/"
echo "============================================================"

run_all_cells > "$LOG_DIR/run.log" 2>&1
EC=$?

# Aggregate the per-run provenance JSONs into the per-dataset TSVs consumed
# by output/robustness/robustness_plot.ipynb. Rewritten from scratch each
# time, so repeated launches never duplicate rows.
python -u "${SCRIPT_DIR}/collect_robustness.py" \
    --checkpoint-dir "$ROBUSTNESS_CKPT_DIR" \
    --output-dir "$PROJECT_ROOT/output/robustness" \
    >> "$LOG_DIR/run.log" 2>&1

DONE=0
for ds in "${DATASETS[@]}"; do
    f="$PROJECT_ROOT/output/robustness/robustness_results_${ds}.txt"
    if [[ -f "$f" ]]; then
        rows=$(wc -l < "$f")
        DONE=$((DONE + rows - 1))
    fi
done
echo "" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"
echo "[$(date "+%H:%M:%S")] All cells dispatched. Exit code: $EC" >> "$LOG_DIR/summary.log"
echo "  Result rows in txt: $DONE / $TOTAL_CELLS" >> "$LOG_DIR/summary.log"
echo "  Results: output/robustness/robustness_results_*.txt" >> "$LOG_DIR/summary.log"
echo "  Logs:    $LOG_DIR/" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"

# Completion marker -- plain stdout (no redirection) so it lands in $LOG_FILE
# (log/robustness/all.log) via the outer redirection.
echo "[$(date "+%H:%M:%S")] All robustness cells finished. Exit=$EC rows=$DONE/$TOTAL_CELLS"

rm -rf "${_ROBUSTNESS_SESSION:-}"
' >> "$LOG_FILE" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched robustness runner as PID $DISOWN_PID"
echo "  Datasets: ${DATASETS[*]}  pred_lens: 96 192 336 720  noise: 0 0.1 0.2 0.3 0.5  seeds: 2023"
echo "  GPUs: ${GPUS[*]}  slots/GPU: $PARA  total cells: $(( ${#DATASETS[@]} * 5 * 4 * 1 ))"
echo "  Log:  tail -f ${LOG_FILE}"
echo "  Results: output/robustness/robustness_results_*.txt (after collection)"
