#!/bin/bash
# ============================================================
# DD-Mamba Hyperparameter Sensitivity Runner (multi-GPU)
# ============================================================
#
# Sweeps 4 hyperparameters (d_model, freq_hidden, mamba_d_state, lr) across
# 2 datasets (ETTm1, weather) at pred_len=96, 5 seeds per cell. Each
# (dataset, param, value, seed) cell trains + tests exactly once via
# scripts/hyperparam/run_sensitivity.py; results are appended to
# output/hyperparam/sensitivity_results_<dataset>.txt (idempotent: relaunch
# skips cells whose row already has status "ok").
#
# Adapted from the Pamba sensitivity launcher (references/hyperparam_scripts).
#
# Usage:
#   scripts/hyperparam/launch_sensitivity.sh                       # GPU 0, 1 slot/GPU
#   scripts/hyperparam/launch_sensitivity.sh --para 2              # GPU 0, 2 slots/GPU
#   scripts/hyperparam/launch_sensitivity.sh 0 2                   # GPUs 0,2, 1 slot/GPU
#   scripts/hyperparam/launch_sensitivity.sh --para 2 0 2          # GPUs 0,2, 2 slots each
#
# Screen without touching the test split:
#   SENSITIVITY_EXTRA="--validation-only" scripts/hyperparam/launch_sensitivity.sh
#
# The script detaches immediately. All output is appended to:
#   log/hyperparam/sensitivity_<timestamp>/worker_*.log
#   log/hyperparam/sensitivity_<timestamp>/run.log
#   log/hyperparam/sensitivity_<timestamp>/summary.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# DATASETS are config stems resolved against configs/<stem>.yaml by
# run_sensitivity.py. Keep in lockstep with the plotting notebook.
DATASETS=(ETTm1 weather)
PRED_LEN=96

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

export SENSITIVITY_DATASETS="${DATASETS[*]}"
export SENSITIVITY_PRED_LEN="$PRED_LEN"

mkdir -p log/hyperparam output/hyperparam

RESULTS_TXT_DIR="$PROJECT_ROOT/output/hyperparam"

# Pre-create per-dataset TSV files with header. The launcher (parent shell)
# owns file creation; workers only append data rows. POSIX guarantees atomic
# appends for writes < PIPE_BUF (4KB on Linux), so no per-row lock is needed.
TXT_HEADER=$'dataset\tpred_len\tsweep_param\tsweep_value\tseed\tmse\tmae\tstatus\telapsed_s\ttimestamp'
for ds in "${DATASETS[@]}"; do
    f="$RESULTS_TXT_DIR/sensitivity_results_${ds}.txt"
    if [[ -f "$f" ]]; then
        echo "txt already exists: $f"
    else
        printf '%s\n' "$TXT_HEADER" > "$f"
        echo "Created txt: $f"
    fi
done
export RESULTS_TXT_DIR

# GPU precheck -- refuse to dispatch if any requested GPU is not visible to
# torch. Without this, every cell fails in <1s with RuntimeError at
# model.to(device) and the TSV fills with error rows.
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

LOG_FILE="$PROJECT_ROOT/log/hyperparam/all.log"

nohup bash -c '
set -u
PROJECT_ROOT="'"${PROJECT_ROOT}"'"
SCRIPT_DIR="'"${SCRIPT_DIR}"'"
DATASETS_STR="'"${SENSITIVITY_DATASETS}"'"
PRED_LEN="'"${SENSITIVITY_PRED_LEN}"'"
PARA="'"${PARA}"'"
GPUS_STR="'"${GPUS[*]}"'"

cd "$PROJECT_ROOT"

# PID lock to prevent concurrent sensitivity runs
exec 200>/tmp/_ddmamba_sensitivity.lock
if ! flock -n 200; then
    echo "[$(date "+%H:%M:%S")] ERROR: Another sensitivity run is in progress" >&2
    exit 1
fi

# Session directory for global FIFO semaphores + counter
export _SENSITIVITY_SESSION="/tmp/_ddmamba_sensitivity_$$"
mkdir -p "$_SENSITIVITY_SESSION"
trap '"'"'rm -rf "${_SENSITIVITY_SESSION:-}"'"'"' EXIT

export GPU_LIST="$GPUS_STR"
export PARA_PER_GPU="$PARA"
export _GPU_COUNTER_FILE="${_SENSITIVITY_SESSION}/counter"
echo 0 > "$_GPU_COUNTER_FILE"

export RESULTS_TXT_DIR="$PROJECT_ROOT/output/hyperparam"

# Per-run log directory
LOG_DIR="$PROJECT_ROOT/log/hyperparam/sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
export _SENSITIVITY_LOG_DIR="$LOG_DIR"

read -ra DATASETS <<< "$DATASETS_STR"

source "${SCRIPT_DIR}/sensitivity_common.sh"

TOTAL_CELLS=0
for ds in "${DATASETS[@]}"; do
    for param in "${PARAMS[@]}"; do
        TOTAL_CELLS=$((TOTAL_CELLS + $(echo ${RANGES[$param]} | wc -w) * ${#SEEDS[@]}))
    done
done
TOTAL_WORKERS=$((${#_GPU_ARR[@]} * _PARA_PER_GPU))

echo "============================================================"
echo "[$(date "+%H:%M:%S")] Hyperparameter Sensitivity — DD-Mamba"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Pred len:    $PRED_LEN"
echo "  Params:      ${PARAMS[*]}"
echo "  Seeds:       ${SEEDS[*]} (${#SEEDS[@]} per cell)"
echo "  Total cells: $TOTAL_CELLS"
echo "  GPUs:        ${#_GPU_ARR[@]} x ${_PARA_PER_GPU} parallel = $TOTAL_WORKERS workers"
echo "  Output:      $RESULTS_TXT_DIR/sensitivity_results_*.txt"
echo "  Logs:        $LOG_DIR/"
echo "============================================================"

run_all_cells "$PRED_LEN" > "$LOG_DIR/run.log" 2>&1
EC=$?

# Summary -- per-dataset txt files are pre-created by the parent shell.
# Count data rows (total lines minus header) across all dataset files.
DONE=0
for ds in "${DATASETS[@]}"; do
    f="$RESULTS_TXT_DIR/sensitivity_results_${ds}.txt"
    if [[ -f "$f" ]]; then
        rows=$(wc -l < "$f")
        DONE=$((DONE + rows - 1))
    fi
done
echo "" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"
echo "[$(date "+%H:%M:%S")] All cells dispatched. Exit code: $EC" >> "$LOG_DIR/summary.log"
echo "  Result rows in txt: $DONE / $TOTAL_CELLS" >> "$LOG_DIR/summary.log"
echo "  Results: $RESULTS_TXT_DIR/sensitivity_results_*.txt" >> "$LOG_DIR/summary.log"
echo "  Logs:    $LOG_DIR/" >> "$LOG_DIR/summary.log"
echo "============================================================" >> "$LOG_DIR/summary.log"

# Completion marker -- plain stdout (no redirection) so it lands in $LOG_FILE
# (log/hyperparam/all.log) via the outer redirection.
echo "[$(date "+%H:%M:%S")] All sensitivity cells finished. Exit=$EC rows=$DONE/$TOTAL_CELLS"

rm -rf "${_SENSITIVITY_SESSION:-}"
' >> "$LOG_FILE" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched sensitivity runner as PID $DISOWN_PID"
echo "  Datasets: ${DATASETS[*]}  pred_len: ${PRED_LEN}  seeds: 5"
echo "  GPUs: ${GPUS[*]}  slots/GPU: $PARA  total cells: $(( ${#DATASETS[@]} * 4 * 5 * 5 ))"
echo "  Log:  tail -f ${LOG_FILE}"
echo "  Results: output/hyperparam/sensitivity_results_*.txt"
