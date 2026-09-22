#!/bin/bash
#SBATCH --nodes=1 --gres=gpu:4 --time=72:00:00 --mail-type=ALL
#SBATCH --job-name=ddmamba-cumablation
#
# Submit:   sbatch scripts/batch/batch_cumulative_ablation.sh
# Monitor:  squeue -u $USER; tail -f log/ablation/all.log
# Cancel:   scancel <jobid>
#
# SLURM wrapper around scripts/ablation/launch_cumulative_ablation.sh.
# The launcher detaches via nohup + disown, so this wrapper parses the
# printed PID and polls /proc/$PID with `tail --pid` to keep the job alive
# until the sweep drains, then trusts only a completion marker written
# after its own launch offset in log/ablation/all.log. Expected cell count
# is derived from gen_cells.py (no hardcoding), so extra datasets submitted
# via SLURM_ARRAY or edited defaults stay consistent.

set -euo pipefail

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dd

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p log/ablation

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

PARA=1
# d512 electricity/traffic models are memory-heavy: keep 1 slot/GPU.
# Drop to more slots only for the small-dataset-only sweeps.

LOG_FILE="log/ablation/all.log"
touch "$LOG_FILE"
MARK_FROM_LINE=$(wc -l < "$LOG_FILE")

LAUNCH_LOG="$(mktemp)"
trap 'rm -f "$LAUNCH_LOG"' EXIT

echo "[$(date +%H:%M:%S)] Launching cumulative-ablation runner (GPUs: ${GPUS[*]}, slots/GPU: ${PARA})"
bash scripts/ablation/launch_cumulative_ablation.sh --para "$PARA" "${GPUS[@]}" 2>&1 | tee "$LAUNCH_LOG"

# The runner detaches via nohup + disown, so it is no longer a child of
# this shell and wait(1) cannot track it. Parse the PID it printed and poll
# /proc/$PID with `tail --pid` to keep the SLURM job alive until it ends.
RUNNER_PID=$(grep -oE 'PID [0-9]+' "$LAUNCH_LOG" | tail -1 | awk '{print $2}' || true)

if [[ -z "${RUNNER_PID:-}" ]]; then
    echo "ERROR: could not parse cumulative-ablation runner PID from launcher output."
    exit 1
fi

echo "[$(date +%H:%M:%S)] Cumulative-ablation runner detached as PID ${RUNNER_PID}; waiting for completion..."
echo "  Log:  tail -f ${LOG_FILE}"

tail --pid="${RUNNER_PID}" -f /dev/null

# Only trust the completion marker if it was written by THIS run (all.log
# accumulates across launches), i.e. it appears after the pre-launch offset.
if tail -n +"$((MARK_FROM_LINE + 1))" "$LOG_FILE" | grep -q "All cumulative ablation cells finished"; then
    echo "[$(date +%H:%M:%S)] Cumulative ablation completed."
    EXPECTED=$(python3 scripts/ablation/gen_cells.py --format stats | awk '{print $6}')
    ACTUAL=$(ls checkpoints/cumulative_ablation/*_cum_*_results.json 2>/dev/null | wc -l)
    echo "  Result JSONs: ${ACTUAL} / runnable ${EXPECTED} (+ any from earlier launches)"
    # The launcher scopes its in-run collect step to the datasets it ran
    # (CUM_COLLECT_EXTRA), which would drop earlier waves from the
    # aggregated JSON/markdown/xlsx. Rebuild the tables over ALL discovered
    # cells so the final artifacts cover every wave (datasets ordered by
    # channel-correlation PCC, documents/datasets.md names).
    echo "[$(date +%H:%M:%S)] Rebuilding cumulative tables over all cells..."
    python3 scripts/ablation/collect_cumulative.py \
        || echo "[warn] full collect_cumulative.py failed (see output above)"
    python3 scripts/ablation/update_cumulative_table.py \
        || echo "[warn] update_cumulative_table.py failed (see output above for the reason; markdown summary remains available)"
else
    echo "[$(date +%H:%M:%S)] Runner exited without the completion marker. Check ${LOG_FILE}"
    exit 1
fi
