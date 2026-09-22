#!/bin/bash
# Shared functions for DD-Mamba hyperparameter-sensitivity scripts.
# Source this file: source scripts/hyperparam/sensitivity_common.sh
#
# Ported from the Pamba sensitivity harness (references/hyperparam_scripts)
# with the dispatch engine kept verbatim:
#   - GPU_LIST mode: round-robin GPU assignment via a file-locked atomic
#     counter (_rotate_gpu) + per-GPU FIFO semaphore bounding concurrency
#     to PARA_PER_GPU tokens per GPU (_dispatch).
#   - Standalone (no GPU_LIST): batches under PARALLEL on the current
#     CUDA_VISIBLE_DEVICES.
#
# Differences vs Pamba: datasets are config stems resolved by
# scripts/hyperparam/run_sensitivity.py (configs/<dataset>.yaml), and the
# sweep dimensions are d_model / freq_hidden / mamba_d_state / lr.
#
# Idempotency: _sensitivity_done reads the per-dataset TSV file
# sensitivity_results_<dataset>.txt and skips any (sweep_param, sweep_value,
# pred_len, seed) cell whose status already starts with "ok".

PARALLEL=2
CNT=0

# ============================================================
# Sweep configuration -- comment out any line below to skip
# that sweep dimension. When skipped, the hyperparameter keeps
# its tuned value from configs/<dataset>.yaml. Each `sweep <param>
# <values>` line registers one dimension: appends param to PARAMS
# (the active sweep list) and sets RANGES[param]. Keep in lockstep
# with SWEEP_RANGES in run_sensitivity.py and PARAMS in
# output/hyperparam/sensitivity_plot.ipynb.
# ============================================================
declare -A RANGES
PARAMS=()

sweep() {
    PARAMS+=("$1")
    RANGES[$1]="$2"
}

sweep d_model       "64 128 256 512 1024"
sweep freq_hidden   "16 32 64 128 256"
sweep mamba_d_state "2 4 8 16 32"
sweep lr            "1e-05 5e-05 0.0001 0.0005 0.001"

# Seeds for the multi-seed sensitivity band (min +/- 1 std) drawn by
# output/hyperparam/sensitivity_plot.ipynb.
SEEDS=(2022 2023 2024 2025 2026)

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
            _sem="${_SENSITIVITY_SESSION:-/tmp}/_ddmamba_sens_sem_${_g}_$$"
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
# $_SENSITIVITY_LOG_DIR (if set) so cell-level errors are recoverable even
# when the parent run.log is lost.
_dispatch() {
    local cmd=("$@")
    if [[ -n "${GPU_LIST:-}" ]]; then
        local gpu="${_CURRENT_GPU}"
        local fd="${_GPU_FD[$gpu]}"
        (
            eval "read -r _ <&${fd}"
            if [[ -n "${_SENSITIVITY_LOG_DIR:-}" && -d "${_SENSITIVITY_LOG_DIR:-}" ]]; then
                local _wlog="${_SENSITIVITY_LOG_DIR}/worker_gpu${gpu}_$$.log"
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$_wlog" 2>&1
            else
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
            fi
            eval "echo >&${fd}"
        ) &
    else
        if [[ -n "${_SENSITIVITY_LOG_DIR:-}" && -d "${_SENSITIVITY_LOG_DIR:-}" ]]; then
            local _wlog="${_SENSITIVITY_LOG_DIR}/worker_standalone_$$.log"
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
# Idempotency -- _sensitivity_done checks whether
# (sweep_param, sweep_value, pred_len, seed) already has a successful row
# in the per-dataset TSV file sensitivity_results_<dataset>.txt. Returns 0
# (true) if done, 1 otherwise. Uses awk for speed. sweep_value comparison
# is float-tolerant: $4+0 forces numeric context so "1e-5" matches
# "1e-05", "0.0001", or any other float spelling.
# ============================================================
_sensitivity_done() {
    local ds=$1 param=$2 val=$3 pl=$4 seed=$5
    local txt="${RESULTS_TXT_DIR:-./output/hyperparam}/sensitivity_results_${ds}.txt"
    [[ -f "$txt" ]] || return 1
    awk -F'\t' -v p="$param" -v v="$val" -v pl="$pl" -v sd="$seed" '
        NR == 1 { next }
        $3 == p && $2 == pl && $5 == sd && $8 ~ /^ok/ \
            && ($4 == v || ($4 + 0) == (v + 0)) { found = 1; exit }
        END { exit !found }
    ' "$txt"
}

# ============================================================
# _run_sensitivity_cell -- build the run_sensitivity.py invocation for one
# (dataset, sweep_param, sweep_value, pred_len, seed) cell. Caller has
# already rotated the GPU; _dispatch pins CUDA_VISIBLE_DEVICES accordingly.
# SENSITIVITY_EXTRA (e.g. "--validation-only") is forwarded verbatim.
# Usage: _run_sensitivity_cell <dataset> <pred_len> <param> <value> <seed>
# ============================================================
_run_sensitivity_cell() {
    local ds=$1 pl=$2 param=$3 val=$4 seed=$5
    # shellcheck disable=SC2086
    _run python -u scripts/hyperparam/run_sensitivity.py \
        --dataset "$ds" \
        --sweep_param "$param" \
        --sweep_value "$val" \
        --pred_len "$pl" \
        --seed "$seed" \
        ${SENSITIVITY_EXTRA:-}
}

# ============================================================
# run_all_cells -- loop over all (dataset, param, value, seed) cells for
# the given pred_len, dispatching each via _dispatch. Skips cells that are
# already done in the per-dataset TSV results files.
# Usage: run_all_cells <pred_len>
# ============================================================
run_all_cells() {
    local pl=$1
    local ds param val seed
    for ds in "${DATASETS[@]}"; do
        for param in "${PARAMS[@]}"; do
            for val in ${RANGES[$param]}; do
                for seed in "${SEEDS[@]}"; do
                    if _sensitivity_done "$ds" "$param" "$val" "$pl" "$seed"; then
                        echo "[skip] ${ds} pl=${pl} ${param}=${val} seed=${seed} already done"
                        continue
                    fi
                    _rotate_gpu
                    _run_sensitivity_cell "$ds" "$pl" "$param" "$val" "$seed"
                done
            done
        done
    done
    _wait_all
}
