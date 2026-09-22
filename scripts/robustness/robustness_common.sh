#!/bin/bash
# Shared functions for DD-Mamba robustness scripts.
# Source this file: source scripts/robustness/robustness_common.sh
#
# Ported from scripts/hyperparam/sensitivity_common.sh (itself ported from
# the Pamba sensitivity harness) with the dispatch engine kept verbatim:
#   - GPU_LIST mode: round-robin GPU assignment via a file-locked atomic
#     counter (_rotate_gpu) + per-GPU FIFO semaphore bounding concurrency
#     to PARA_PER_GPU tokens per GPU (_dispatch).
#   - Standalone (no GPU_LIST): batches under PARALLEL on the current
#     CUDA_VISIBLE_DEVICES.
#
# Differences vs sensitivity: each cell is one (dataset, pred_len,
# noise_level, seed) run of the Pamba-style per-dataset shell script
# scripts/robustness/<dataset>.sh, which drives the project's default entry
# point (python -m src.train) with the tuned configs/<dataset>.yaml. The
# final test evaluation is corrupted with additive Gaussian noise
# (--noise_level); training/validation stay clean. Idempotency is per-cell
# via the existence of checkpoints/robustness/<name>_results.json (written
# atomically by src.train only after a successful train+test).
#
# Keep NOISE_LEVELS in lockstep with the grid documented in
# scripts/robustness/ETTh2.sh / weather.sh and the plotting notebook
# output/robustness/robustness_plot.ipynb -- edit all together.

PARALLEL=2
CNT=0

NOISE_LEVELS=(0 0.1 0.2 0.3 0.5)
PRED_LENS=(96 192 336 720)
SEEDS=(2023)
ROBUSTNESS_CKPT_DIR="${ROBUSTNESS_CKPT_DIR:-checkpoints/robustness}"

# ============================================================
# GPU dispatch engine (ported from the Pamba harness).
# When GPU_LIST is set, opens one FIFO per GPU loaded with PARA_PER_GPU
# tokens so per-GPU concurrency is capped. When GPU_LIST is unset, falls
# back to PARALLEL-wide batching on the current CUDA_VISIBLE_DEVICES.
# ============================================================

if [[ -n "${GPU_LIST:-}" ]]; then
    read -ra _GPU_ARR <<< "${GPU_LIST}"
    _GPU_IDX=0
    _GPU_CNT=${#_GPU_ARR[@]}
    _PARA_PER_GPU="${PARA_PER_GPU:-1}"
    ((_PARA_PER_GPU < 1)) && _PARA_PER_GPU=1
    _CURRENT_GPU=""
    declare -A _GPU_FD=()
    _next_fd=20
    for _g in "${_GPU_ARR[@]}"; do
        if [[ -z "${_GPU_FD[$_g]+x}" ]]; then
            _sem="${_ROBUSTNESS_SESSION:-/tmp}/_ddmamba_rob_sem_${_g}_$$"
            mkfifo "$_sem"
            eval "exec ${_next_fd}<>\"$_sem\""
            rm -f "$_sem"
            _GPU_FD[$_g]=${_next_fd}
            for ((i = 0; i < _PARA_PER_GPU; i++)); do
                eval "echo >&${_next_fd}"
            done
            ((_next_fd++))
        fi
    done
fi

_rotate_gpu() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        if [[ -n "${_GPU_COUNTER_FILE:-}" && -f "${_GPU_COUNTER_FILE:-}" ]]; then
            local idx
            idx=$(flock "${_GPU_COUNTER_FILE}" bash -c 'read -r v < "$1"; echo $((v+1)) > "$1"; printf "%d" "$v"' _ "${_GPU_COUNTER_FILE}")
            _CURRENT_GPU="${_GPU_ARR[$((idx % _GPU_CNT))]}"
        else
            _CURRENT_GPU="${_GPU_ARR[$((_GPU_IDX % _GPU_CNT))]}"
            _GPU_IDX=$(( _GPU_IDX + 1 ))
        fi
    fi
}

# _dispatch -- GPU_LIST mode pins each job to _CURRENT_GPU (round-robin) and
# bounds per-GPU concurrency via that GPU's FIFO semaphore (PARA_PER_GPU
# tokens). Standalone mode batches under PARALLEL concurrency.
# Each worker's stdout/stderr is tee'd to a per-worker log file under
# $_ROBUSTNESS_LOG_DIR (if set) so cell-level errors are recoverable even
# when the parent run.log is lost.
_dispatch() {
    local cmd=("$@")
    if [[ -n "${GPU_LIST:-}" ]]; then
        local gpu="${_CURRENT_GPU}"
        local fd="${_GPU_FD[$gpu]}"
        (
            eval "read -r _ <&${fd}"
            if [[ -n "${_ROBUSTNESS_LOG_DIR:-}" && -d "${_ROBUSTNESS_LOG_DIR:-}" ]]; then
                local _wlog="${_ROBUSTNESS_LOG_DIR}/worker_gpu${gpu}_$$.log"
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$_wlog" 2>&1
            else
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
            fi
            eval "echo >&${fd}"
        ) &
    else
        if [[ -n "${_ROBUSTNESS_LOG_DIR:-}" && -d "${_ROBUSTNESS_LOG_DIR:-}" ]]; then
            local _wlog="${_ROBUSTNESS_LOG_DIR}/worker_standalone_$$.log"
            "${cmd[@]}" > "$_wlog" 2>&1 &
        else
            "${cmd[@]}" &
        fi
        CNT=$((CNT + 1))
        if [ $((CNT % PARALLEL)) -eq 0 ]; then wait; fi
    fi
}

_wait_all() {
    wait
}

_run() {
    _dispatch "$@"
}

# ============================================================
# Idempotency -- _robustness_cell_done checks whether the cell's
# results.json already exists. src.train writes it atomically only after a
# successful train + (possibly noisy) test evaluation, so existence means
# the cell completed; the per-dataset shell script double-checks and
# prints its own [skip] line when re-dispatched anyway.
# Usage: _robustness_cell_done <dataset> <pred_len> <noise_level> <seed>
# ============================================================
_robustness_cell_done() {
    local ds=$1 pl=$2 nl=$3 seed=$4
    local name="${ds}_robustness_pl${pl}_nl${nl}_s${seed}"
    [[ -f "${ROBUSTNESS_CKPT_DIR}/${name}_results.json" ]]
}

# ============================================================
# _run_robustness_cell -- dispatch one (dataset, pred_len, noise_level,
# seed) cell via the Pamba-style per-dataset shell script. Caller has
# already rotated the GPU; _dispatch pins CUDA_VISIBLE_DEVICES accordingly.
# ROBUSTNESS_EXTRA is forwarded verbatim as the script's extra_flags env.
# Usage: _run_robustness_cell <dataset> <pred_len> <noise_level> <seed>
# ============================================================
_run_robustness_cell() {
    local ds=$1 pl=$2 nl=$3 seed=$4
    _run env noise_level="$nl" seed="$seed" ckpt_dir="$ROBUSTNESS_CKPT_DIR" \
        extra_flags="${ROBUSTNESS_EXTRA:-}" \
        bash "scripts/robustness/${ds}.sh" "$pl"
}

# ============================================================
# run_all_cells -- loop over all (dataset, noise_level, pred_len, seed)
# cells, dispatching each via _dispatch. Completed cells are skipped.
# Usage: run_all_cells
# ============================================================
run_all_cells() {
    local ds nl pl seed
    for ds in "${DATASETS[@]}"; do
        for nl in "${NOISE_LEVELS[@]}"; do
            for pl in "${PRED_LENS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    if _robustness_cell_done "$ds" "$pl" "$nl" "$seed"; then
                        echo "[skip] ${ds} pl=${pl} nl=${nl} seed=${seed}: results.json exists"
                        continue
                    fi
                    _rotate_gpu
                    _run_robustness_cell "$ds" "$pl" "$nl" "$seed"
                done
            done
        done
    done
    _wait_all
}
