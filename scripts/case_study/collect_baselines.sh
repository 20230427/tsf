#!/bin/bash

# Copy the reference case-study baseline recordings (Pamba-repo dumps, same
# ETTm 12/4/4-month split and train-split standardization as dd-mamba) into
# output/case_study/record_ETTm1/ for the plotting notebook.
#
# Usage:
#   bash scripts/case_study/collect_baselines.sh [DATASET] [--force]
#
# Default DATASET=ETTm1 (the only dataset with reference recordings).

set -e

DATASET="${1:-ETTm1}"
FORCE=0
if [[ "${2:-}" == "--force" || "${1:-}" == "--force" ]]; then
    FORCE=1
    [[ "$1" == "--force" ]] && DATASET="${2:-ETTm1}" || DATASET="${1:-ETTm1}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SRC_DIR="references/case_study/record_${DATASET}"
RECORD_DIR="output/case_study/record_${DATASET}"

if [ ! -d "$SRC_DIR" ]; then
    echo "No reference recordings for dataset ${DATASET}: ${SRC_DIR} missing"
    exit 1
fi

mkdir -p "$RECORD_DIR"

if [ "$DATASET" == "ETTm1" ]; then
    BASELINE_MODELS=(iTransformer PatchTST TimeMixer Pamba)
else
    BASELINE_MODELS=(iTransformer PatchTST)
fi
PRED_LENS=(96 192 384)

COPIED=0
MISSING=0

for model in "${BASELINE_MODELS[@]}"; do
    for pred_len in "${PRED_LENS[@]}"; do
        SRC="${SRC_DIR}/${model}_96_${pred_len}_pred.npy"
        DST="${RECORD_DIR}/${model}_96_${pred_len}_pred.npy"
        if [ -f "$SRC" ]; then
            if [ -f "$DST" ] && [ "$FORCE" -eq 0 ]; then
                echo "  [keep] ${model}_96_${pred_len}_pred.npy (exists)"
            else
                cp "$SRC" "$DST"
                echo "  [copy] ${model}_96_${pred_len}_pred.npy"
                COPIED=$((COPIED + 1))
            fi
        else
            echo "  [miss] ${model}_96_${pred_len}_pred.npy"
            MISSING=$((MISSING + 1))
        fi
    done
done

# Ground truth is shared across models at a given pred_len; the reference
# recording wins over a locally dumped true (identical split, same scaling).
for pred_len in "${PRED_LENS[@]}"; do
    SRC="${SRC_DIR}/true_${pred_len}.npy"
    DST="${RECORD_DIR}/true_${pred_len}.npy"
    if [ -f "$SRC" ]; then
        if [ -f "$DST" ] && [ "$FORCE" -eq 0 ]; then
            echo "  [keep] true_${pred_len}.npy (exists)"
        else
            cp "$SRC" "$DST"
            echo "  [copy] true_${pred_len}.npy"
            COPIED=$((COPIED + 1))
        fi
    else
        echo "  [miss] true_${pred_len}.npy"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "Collected ${COPIED} files, ${MISSING} missing -> ${RECORD_DIR}/"
