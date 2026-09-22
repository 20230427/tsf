#!/bin/bash

# Case-study experiment orchestrator: collect baseline recordings, then train
# DD-Mamba on ETTm1 (seq_len 96; pred_len 96/192/384) and dump test-set
# predictions + branch decomposition for the plotting notebook.
#
# Usage:
#   bash scripts/case_study/run_case_study.sh [--gpu N] [--skip-train]
#                                             [--extra "flags..."]
#
# Examples:
#   bash scripts/case_study/run_case_study.sh                          # DD-Mamba
#   bash scripts/case_study/run_case_study.sh --skip-train             # redump
#   bash scripts/case_study/run_case_study.sh \
#        --extra "--arch smamba --model-tag S_Mamba"                   # S-Mamba
#
# Logs are appended to output/case_study/log/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET=ETTm1
GPU_ID=0
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --skip-train) EXTRA_FLAGS="$EXTRA_FLAGS --skip-train"; shift ;;
        --extra) EXTRA_FLAGS="$EXTRA_FLAGS $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU_ID
LOG_DIR="output/case_study/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/case_study_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "Case study: ${DATASET} | GPU ${GPU_ID}"
echo "================================================================================"

{
    echo "==== $(date '+%Y-%m-%d %H:%M:%S') baseline collection ===="
    bash scripts/case_study/collect_baselines.sh "$DATASET"

    echo ""
    echo "==== $(date '+%Y-%m-%d %H:%M:%S') DD-Mamba case-study runs ===="
    python -u scripts/case_study/run_case_study.py \
        --config "configs/${DATASET}.yaml" \
        --dataset "$DATASET" \
        $EXTRA_FLAGS

    echo ""
    echo "==== $(date '+%Y-%m-%d %H:%M:%S') done ===="
    ls -lh "output/case_study/record_${DATASET}/" | grep -E "\.npy|manifest" || true
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log:       ${LOG_FILE}"
echo "Records:   output/case_study/record_${DATASET}/"
echo "Next:      run output/case_study/case_study.ipynb to plot"
