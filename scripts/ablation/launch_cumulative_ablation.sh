#!/bin/bash
# ============================================================
# DD-Mamba Cumulative Ablation Runner (detached, multi-GPU, resumable)
# ============================================================
#
# Trains the cumulative step-down ablation chain defined in
# scripts/ablation/cumulative_registry.py over the default 7-dataset matrix
# (ETTh1 ETTm2 weather electricity solar exchange_rate PEMS04; override
# with --datasets, e.g. add the slow traffic). Each (dataset, pred_len,
# variant, seed) cell is one `python -m src.train` run with the tuned
# configs/<dataset>.yaml plus cumulative dotted overrides; per-dataset
# seeds come from the config (single seed per cell -- single-seed deltas
# below ~0.005 MSE are noise, see AGENTS.md; extend later with --seed
# shifts and multi-seed aggregation in collect_cumulative.py).
#
# Variants run strictly serially (each variant occupies all GPUs); cells
# within a variant are dispatched round-robin with per-GPU FIFO
# semaphores. Structural no-op cells are skipped; completed cells (their
# provenance results.json exists) are skipped -- relaunch to resume.
#
# When the sweep drains, scripts/ablation/run_all_cumulative.sh (invoked
# below) aggregates results and rebuilds the xlsx automatically.
#
# Usage:
#   scripts/ablation/launch_cumulative_ablation.sh                      # GPU 0
#   scripts/ablation/launch_cumulative_ablation.sh --para 1 0 1 2 3     # 4 GPUs
#   scripts/ablation/launch_cumulative_ablation.sh --datasets ETTh1 traffic
#   scripts/ablation/launch_cumulative_ablation.sh --extra "--train.epochs 1"  # smoke
#
# The script detaches immediately. Output lands under:
#   log/ablation/cumulative_<ts>/worker_*.log + run.log
#   log/ablation/all.log          (completion marker)
#   checkpoints/cumulative_ablation/<name>_results.json  (durable records)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PARA=1
GPUS=()
DATASETS=""      # empty -> gen_cells default set
SEED=""          # empty -> per-config experiment.seed
EXTRA=""         # forwarded to src.train (e.g. "--train.epochs 1")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --para)     PARA="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --extra)    EXTRA="$2"; shift 2 ;;
        *)          GPUS+=("$1"); shift ;;
    esac
done

if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0)
fi

mkdir -p log/ablation

# GPU precheck -- refuse to dispatch if any requested GPU is not visible to
# torch. Without this every cell dies in <1s at model.to(device).
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

LOG_FILE="$PROJECT_ROOT/log/ablation/all.log"

nohup bash -c '
set -u
PROJECT_ROOT="'"${PROJECT_ROOT}"'"
SCRIPT_DIR="'"${SCRIPT_DIR}"'"
PARA="'"${PARA}"'"
GPUS_STR="'"${GPUS[*]}"'"
DATASETS="'"${DATASETS}"'"
SEED="'"${SEED}"'"
EXTRA="'"${EXTRA}"'"
CUM_CKPT_DIR="'"${CUM_CKPT_DIR:-checkpoints/cumulative_ablation}"'"

cd "$PROJECT_ROOT"

# PID lock to prevent concurrent cumulative-ablation runs
exec 200>/tmp/_ddmamba_cumablation.lock
if ! flock -n 200; then
    echo "[$(date "+%H:%M:%S")] ERROR: Another cumulative-ablation run is in progress" >&2
    exit 1
fi

# Session directory for global FIFO semaphores + counter (trap cleans up)
export _CUM_SESSION="/tmp/_ddmamba_cumablation_$$"
mkdir -p "$_CUM_SESSION"
trap '"'"'rm -rf "${_CUM_SESSION:-}"'"'"' EXIT

export GPU_LIST="$GPUS_STR"
export PARA_PER_GPU="$PARA"
export _GPU_COUNTER_FILE="${_CUM_SESSION}/counter"
echo 0 > "$_GPU_COUNTER_FILE"

export CUM_CKPT_DIR
export CUM_GEN_EXTRA=""
if [[ -n "$DATASETS" ]]; then
    export CUM_GEN_EXTRA="--datasets $DATASETS"
fi
if [[ -n "$SEED" ]]; then
    export CUM_GEN_EXTRA="$CUM_GEN_EXTRA --seed $SEED"
fi
export CUM_COLLECT_EXTRA=""
if [[ -n "$DATASETS" ]]; then
    export CUM_COLLECT_EXTRA="--datasets $DATASETS"
fi
export CUM_EXTRA="$EXTRA"

# Per-run log directory (workers write here via _CUM_LOG_DIR)
LOG_DIR="$PROJECT_ROOT/log/ablation/cumulative_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
export _CUM_LOG_DIR="$LOG_DIR"

bash "${SCRIPT_DIR}/run_all_cumulative.sh"
' >> "$LOG_FILE" 2>&1 &

DISOWN_PID=$!
disown $DISOWN_PID 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] Launched cumulative-ablation runner as PID $DISOWN_PID"
echo "  Datasets: ${DATASETS:-<default 7>}  GPUs: ${GPUS[*]}  slots/GPU: $PARA"
[[ -n "$EXTRA" ]] && echo "  Extra train flags: $EXTRA"
echo "  Log:  tail -f ${LOG_FILE}"
echo "  Results: ${CUM_CKPT_DIR:-checkpoints/cumulative_ablation}/*_results.json"
