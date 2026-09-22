#!/usr/bin/env bash
# Batch launcher for DD-Mamba HPO studies (ported from references/pamba
# scripts/hpo/hpo_batch.sh; per-study GPU pinning + wait-for-completion so a
# SLURM job stays alive until every study drains).
#
# Usage:
#   bash scripts/hpo/hpo_batch.sh --n_gpus 4 \
#     --study "ETTh1:96:200:0" --study "ETTh1:192:200:1" ...
#
# Each study is formatted as:  dataset:pred_len:trials[:target_gpu]
#   dataset    configs/<dataset>.yaml basename (e.g. ETTh1, weather, PEMS03)
#   target_gpu optional logical GPU id (0..n_gpus-1); omit for round-robin
#
# If SLURM exported CUDA_VISIBLE_DEVICES, logical GPU ids are mapped onto the
# assigned physical ids. Studies resume automatically from their SQLite DB
# under output/hpo/optuna, so relaunching a killed sweep just continues.

set -euo pipefail

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p log/hpo output/hpo/pids

N_GPUS=4
STUDIES=()
METRIC="mse"
EPOCHS=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/hpo/hpo_batch.sh --n_gpus 4 \
    --study "ETTh1:96:200" --study "ETTh1:720:200:3" ...

  Each study is formatted as:  dataset:pred_len:trials[:target_gpu]

Options:
  --n_gpus, -N   Number of GPUs (default: 4; <=0 auto-detect via nvidia-smi)
  --study, -s    A study definition (repeatable)
  --metric       Optimization metric: mse (default), mae or val_loss
  --epochs       Override train.epochs (0 = use the dataset config)
  --help, -h     Show help

GPU assignment:
  If target_gpu is given, the study is pinned to that logical GPU
  (mapped through SLURM's CUDA_VISIBLE_DEVICES when present).
  Otherwise round-robin across GPUs (0,1,...,N-1,0,1,...).
  Each study runs scripts/hpo/run_hpo.py pinned to one GPU and logs to
  log/hpo/<dataset>_<pl>_study.log. Pid files under output/hpo/pids guard
  against double-launching a study that is still running.
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_gpus|-N) N_GPUS="$2"; shift 2 ;;
        --study|-s) STUDIES+=("$2"); shift 2 ;;
        --metric) METRIC="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ${#STUDIES[@]} -eq 0 ]]; then
    echo "Error: no studies provided. Use --study <dataset:pred_len:trials[:target_gpu]>" >&2
    exit 1
fi

if [[ "$N_GPUS" -le 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t _gpu_indices < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
        N_GPUS=${#_gpu_indices[@]}
    fi
    if [[ "$N_GPUS" -le 0 ]]; then
        echo "Error: --n_gpus <= 0 and could not detect any GPU via nvidia-smi." >&2
        echo "       Pass --n_gpus N explicitly (N = number of available GPUs)." >&2
        exit 1
    fi
    echo "[auto] detected ${N_GPUS} GPU(s) via nvidia-smi"
fi

SLURM_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    read -ra SLURM_GPUS <<< "${CUDA_VISIBLE_DEVICES//,/ }"
fi

PID_DIR="output/hpo/pids"
echo "[$(date +%H:%M:%S)] === hpo_batch: ${#STUDIES[@]} studies across ${N_GPUS} GPUs ==="
echo ""

PIDS=()
TAGS=()

for ((i = 0; i < ${#STUDIES[@]}; i++)); do
    line="${STUDIES[$i]}"

    ds=$(echo "$line" | cut -d: -f1)
    pl=$(echo "$line" | cut -d: -f2)
    trials=$(echo "$line" | cut -d: -f3)
    target_gpu=$(echo "$line" | cut -d: -f4)

    config="configs/${ds}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "  [SKIP] ${ds}: no config ${config}"
        continue
    fi

    if [[ -n "$target_gpu" ]]; then
        if [[ "$target_gpu" -ge "$N_GPUS" || "$target_gpu" -lt 0 ]]; then
            echo "  [SKIP] ${ds}_pl${pl}: target_gpu=${target_gpu} out of range [0,$((N_GPUS - 1))]"
            continue
        fi
        gpu=$target_gpu
    else
        gpu=$((i % N_GPUS))
    fi
    if [[ ${#SLURM_GPUS[@]} -gt 0 ]]; then
        gpu="${SLURM_GPUS[$gpu]}"
    fi

    tag="${ds}_pl${pl}"
    pidfile="$PID_DIR/${tag}.pid"
    log_file="log/hpo/${ds}_${pl}_study.log"

    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "  [SKIP] ${tag} already running (PID $(cat "$pidfile"))"
        continue
    fi

    EXTRA_ARGS=()
    [[ "$EPOCHS" -gt 0 ]] && EXTRA_ARGS+=(--epochs "$EPOCHS")

    CUDA_VISIBLE_DEVICES="$gpu" python scripts/hpo/run_hpo.py \
        --config "$config" --pred_len "$pl" --n_trials "$trials" \
        --metric "$METRIC" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        >> "$log_file" 2>&1 &
    study_pid=$!
    echo "$study_pid" > "$pidfile"
    PIDS+=("$study_pid")
    TAGS+=("$tag")
    echo "  [GPU $gpu] ${tag} launched (PID $study_pid, ${trials} trials) -> $log_file"
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "ERROR: no study could be launched. Check the [SKIP] lines above."
    exit 1
fi

echo ""
echo "[$(date +%H:%M:%S)] ${#PIDS[@]} study(ies) launched; waiting for completion..."
echo "  Monitor: tail -f log/hpo/*_study.log ; python scripts/hpo/hpo_dump.py --list"

FAIL=0
FAILED_TAGS=()
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAIL=$((FAIL + 1))
        FAILED_TAGS+=("${TAGS[$i]}")
    fi
    rm -f "$PID_DIR/${TAGS[$i]}.pid"
done

if [[ "$FAIL" -gt 0 ]]; then
    echo "ERROR: ${FAIL}/${#PIDS[@]} study(ies) failed: ${FAILED_TAGS[*]}"
    echo "       Inspect log/hpo/<dataset>_<pl>_study.log; studies resume from their DB."
    exit 1
fi

echo "[$(date +%H:%M:%S)] All studies completed."
python scripts/hpo/hpo_dump.py --list || true
