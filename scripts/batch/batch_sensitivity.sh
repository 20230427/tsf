#!/bin/bash
#SBATCH --nodes=1 --gres=gpu:4 --time=48:00:00 --mail-type=ALL
#SBATCH --job-name=ddmamba-sensitivity
#
# Submit:   sbatch scripts/batch/batch_sensitivity.sh
# Monitor:  squeue -u $USER
# Cancel:   scancel <jobid>

set -euo pipefail

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dd

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p log/hyperparam

# The launcher pre-creates per-dataset TSV files with headers on first run
# and skips cells whose row already has status "ok", so the sweep is
# resumable. To force a clean reset, delete
# output/hyperparam/sensitivity_results_*.txt before submitting.

PARA=1

# Use exactly the GPUs SLURM assigned: with --gres, SLURM exports
# CUDA_VISIBLE_DEVICES listing them. When it is unset (cgroup isolation or
# bare-metal bash), probe torch for the visible device count instead.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES//,/ }"
else
    NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    (( NGPU < 1 )) && NGPU=1
    GPUS=( $(seq 0 $((NGPU - 1))) )
fi

LOG_FILE="log/hyperparam/all.log"
touch "$LOG_FILE"
MARK_FROM_LINE=$(wc -l < "$LOG_FILE")

LAUNCH_LOG="$(mktemp)"
trap 'rm -f "$LAUNCH_LOG"' EXIT

echo "[$(date +%H:%M:%S)] Launching sensitivity runner (GPUs: ${GPUS[*]}, slots/GPU: ${PARA})"
bash scripts/hyperparam/launch_sensitivity.sh --para "$PARA" "${GPUS[@]}" 2>&1 | tee "$LAUNCH_LOG"

# The runner detaches via nohup + disown, so it is no longer a child of this
# shell and wait(1) cannot track it. Parse the PID it printed and poll
# /proc/$PID with `tail --pid` to keep the SLURM job alive until it finishes.
RUNNER_PID=$(grep -oE 'PID [0-9]+' "$LAUNCH_LOG" | tail -1 | awk '{print $2}' || true)

if [[ -z "${RUNNER_PID:-}" ]]; then
    echo "ERROR: could not parse sensitivity runner PID from launcher output."
    exit 1
fi

echo "[$(date +%H:%M:%S)] Sensitivity runner detached as PID ${RUNNER_PID}; waiting for completion..."
echo "  Log:  tail -f ${LOG_FILE}"

tail --pid="${RUNNER_PID}" -f /dev/null

# Only trust the completion marker if it was written by THIS run (all.log
# accumulates across launches), i.e. it appears after the pre-launch offset.
if tail -n +"$((MARK_FROM_LINE + 1))" "$LOG_FILE" | grep -q "All sensitivity cells finished"; then
    echo "[$(date +%H:%M:%S)] Sensitivity sweep completed."
    # Sanity check: count ok rows vs expected (2 datasets x 4 params x 5 values x 5 seeds).
    EXPECTED=200
    ACTUAL=$(awk -F'\t' 'NR > 1 && $8 ~ /^ok/ { n++ } END { print n+0 }' \
        output/hyperparam/sensitivity_results_*.txt 2>/dev/null || echo "?")
    echo "  Result rows (status=ok): ${ACTUAL} / ${EXPECTED}"
else
    echo "[$(date +%H:%M:%S)] Sensitivity runner exited without the completion marker. Check ${LOG_FILE}"
    exit 1
fi
