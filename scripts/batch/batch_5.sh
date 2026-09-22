#!/bin/bash
#SBATCH --nodes=1 --gres=gpu:4 --time=48:00:00 --mail-type=ALL
#SBATCH --job-name=ddmamba-hpo-b5
#
# Submit:   sbatch scripts/batch/batch_5.sh
# Monitor:  squeue -u $USER
# Cancel:   scancel <jobid>

set -euo pipefail

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dd

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p log/hpo

STUDIES=(
    "PEMS08:12:200:3"
    "PEMS08:24:200:2"
    "PEMS08:48:200:1"
    "PEMS08:96:200:0"
    "exchange_rate:96:800:0"
    "exchange_rate:192:800:1"
    "exchange_rate:336:800:2"
    "exchange_rate:720:800:3"
)

PIDS=()
for line in "${STUDIES[@]}"; do
    ds=$(echo "$line" | cut -d: -f1)
    pl=$(echo "$line" | cut -d: -f2)
    trials=$(echo "$line" | cut -d: -f3)
    gpu=$(echo "$line" | cut -d: -f4)
    log_file="log/hpo/${ds}_${pl}_study.log"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/hpo/run_hpo.py \
        --config "configs/${ds}.yaml" --pred_len "$pl" --n_trials "$trials" \
        --metric mse \
        >> "$log_file" 2>&1 &
    study_pid=$!
    PIDS+=("$study_pid")
    echo "[$(date +%H:%M:%S)] Launched ${ds} pred_len=${pl} (${trials} trials) on GPU ${gpu} (PID ${study_pid})"
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "ERROR: ${FAIL}/${#STUDIES[@]} study/studies failed. Check log/hpo/*_study.log"
    exit 1
fi
echo "[$(date +%H:%M:%S)] All ${#STUDIES[@]} HPO studies completed."
