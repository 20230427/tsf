#!/bin/bash
# ============================================================
# DD-Mamba Cumulative Ablation -- foreground orchestrator
# ============================================================
#
# Runs the full cumulative step-down chain (variant-serial, cell-parallel),
# then aggregates results and rebuilds the xlsx. Environment contract (all
# optional; sensible defaults for a single-GPU foreground run):
#
#   GPU_LIST          space-separated GPU ids (empty -> standalone mode,
#                     PARALLEL-wide batching on the current device)
#   PARA_PER_GPU      concurrent jobs per GPU in GPU_LIST mode (default 1)
#   _CUM_SESSION      session dir for the FIFO semaphores + counter
#                     (created+cleaned here if unset)
#   CUM_CKPT_DIR      per-cell checkpoint/results dir
#                     (default checkpoints/cumulative_ablation)
#   CUM_GEN_EXTRA     forwarded to gen_cells.py (--datasets ... --seed ...)
#   CUM_EXTRA         forwarded verbatim to src.train (e.g. --train.epochs 1)
#   CUM_SKIP_TABLE    set to 1 to skip collect + xlsx at the end
#
# Usage (foreground, resumable -- completed cells are skipped):
#   GPU_LIST=0 bash scripts/ablation/run_all_cumulative.sh
#   GPU_LIST="0 1" PARA_PER_GPU=1 CUM_GEN_EXTRA="--datasets exchange_rate" \
#       CUM_EXTRA="--train.epochs 1" bash scripts/ablation/run_all_cumulative.sh
#
# launch_cumulative_ablation.sh wraps this in nohup + a PID lock + log dirs.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUM_CKPT_DIR="${CUM_CKPT_DIR:-checkpoints/cumulative_ablation}"

# Own the session dir when launched standalone (the detached launcher
# creates it earlier so its trap cleans up).
if [[ -n "${GPU_LIST:-}" && -z "${_CUM_SESSION:-}" ]]; then
    export _CUM_SESSION="/tmp/_ddmamba_cumablation_$$"
    mkdir -p "$_CUM_SESSION"
    trap 'rm -rf "${_CUM_SESSION:-}"' EXIT
fi
if [[ -n "${GPU_LIST:-}" ]]; then
    export _GPU_COUNTER_FILE="${_GPU_COUNTER_FILE:-${_CUM_SESSION}/counter}"
    [[ -f "$_GPU_COUNTER_FILE" ]] || echo 0 > "$_GPU_COUNTER_FILE"
fi

source "${SCRIPT_DIR}/ablation_common.sh"

STATS=$(python3 "$CUM_GEN" --format stats --checkpoint-dir "$CUM_CKPT_DIR" ${CUM_GEN_EXTRA:-})
if [[ -n "${GPU_LIST:-}" ]]; then
    TOTAL_WORKERS=$((${#_GPU_ARR[@]} * _PARA_PER_GPU))
else
    TOTAL_WORKERS="${PARALLEL}"
fi

echo "============================================================"
echo "[$(date "+%H:%M:%S")] Cumulative Ablation -- DD-Mamba"
echo "  Cells:       $STATS"
echo "  Workers:     $TOTAL_WORKERS"
echo "  Checkpoints: $CUM_CKPT_DIR"
echo "  Extra train: ${CUM_EXTRA:-<none>}"
echo "============================================================"

run_all_cells
EC=$?

# Summary -- recount cells incl. those completed before this launch.
STATS=$(python3 "$CUM_GEN" --format stats --checkpoint-dir "$CUM_CKPT_DIR" ${CUM_GEN_EXTRA:-})
DONE=$(ls "$CUM_CKPT_DIR"/*_cum_*_results.json 2>/dev/null | wc -l)
echo "------------------------------------------------------------"
echo "[$(date "+%H:%M:%S")] Engine exit code: $EC"
echo "  $STATS"
echo "  Result JSONs present: $DONE"
echo "------------------------------------------------------------"

if [[ "${CUM_SKIP_TABLE:-0}" != "1" ]]; then
    # Aggregate + rebuild tables from the durable per-cell JSONs (idempotent).
    python3 scripts/ablation/collect_cumulative.py ${CUM_COLLECT_EXTRA:-} \
        || echo "[warn] collect_cumulative.py failed"
    python3 scripts/ablation/update_cumulative_table.py \
        || echo "[warn] update_cumulative_table.py failed (see output above for the reason; markdown summary remains available)"
fi

# Completion marker -- batch_cumulative_ablation.sh greps this line.
echo "[$(date "+%H:%M:%S")] All cumulative ablation cells finished. Exit=$EC $STATS"
exit $EC
