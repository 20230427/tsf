#!/bin/bash
#SBATCH --nodes=1 --gres=gpu:4 --time=48:00:00 --mail-type=ALL
#SBATCH --job-name=ddmamba-robustness
#
# Submit:   sbatch scripts/batch/batch_robustness.sh
# Monitor:  squeue -u $USER
# Cancel:   scancel <jobid>

set -euo pipefail

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dd

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p log/robustness

# Each cell (dataset x noise_level x pred_len x seed) is one src.train run
# (train clean, test noisy) whose results.json lands in
# checkpoints/robustness/; the launcher aggregates them into
# output/robustness/robustness_results_*.txt at the end of the sweep. The
# sweep is resumable: relaunching skips cells whose results.json already
# exists. To force a clean reset, delete checkpoints/robustness/ and
# output/robustness/robustness_results_*.txt before submitting.

PARA=2

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

LOG_FILE="log/robustness/all.log"
touch "$LOG_FILE"
MARK_FROM_LINE=$(wc -l < "$LOG_FILE")

LAUNCH_LOG="$(mktemp)"
trap 'rm -f "$LAUNCH_LOG"' EXIT

echo "[$(date +%H:%M:%S)] Launching robustness runner (GPUs: ${GPUS[*]}, slots/GPU: ${PARA})"
bash scripts/robustness/launch_robustness.sh --para "$PARA" "${GPUS[@]}" 2>&1 | tee "$LAUNCH_LOG"

# The runner detaches via nohup + disown, so it is no longer a child of this
# shell and wait(1) cannot track it. Parse the PID it printed and poll
# /proc/$PID with `tail --pid` to keep the SLURM job alive until it finishes.
RUNNER_PID=$(grep -oE 'PID [0-9]+' "$LAUNCH_LOG" | tail -1 | awk '{print $2}' || true)

if [[ -z "${RUNNER_PID:-}" ]]; then
    echo "ERROR: could not parse robustness runner PID from launcher output."
    exit 1
fi

echo "[$(date +%H:%M:%S)] Robustness runner detached as PID ${RUNNER_PID}; waiting for completion..."
echo "  Log:  tail -f ${LOG_FILE}"

tail --pid="${RUNNER_PID}" -f /dev/null

# Only trust the completion marker if it was written by THIS run (all.log
# accumulates across launches), i.e. it appears after the pre-launch offset.
if tail -n +"$((MARK_FROM_LINE + 1))" "$LOG_FILE" | grep -q "All robustness cells finished"; then
    echo "[$(date +%H:%M:%S)] Robustness sweep completed."
    # Sanity check: count ok rows vs expected (2 datasets x 5 noise levels
    # x 4 pred_lens x 1 seed).
    EXPECTED=40
    ACTUAL=$(awk -F'\t' 'NR > 1 && $7 ~ /^ok/ { n++ } END { print n+0 }' \
        output/robustness/robustness_results_*.txt 2>/dev/null || echo "?")
    echo "  Result rows (status=ok): ${ACTUAL} / ${EXPECTED}"
else
    echo "[$(date +%H:%M:%S)] Robustness runner exited without the completion marker. Check ${LOG_FILE}"
    exit 1
fi
