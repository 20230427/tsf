#!/bin/bash
# Shared functions for the DD-Mamba cumulative-ablation suite.
# Source this file: source scripts/ablation/ablation_common.sh
#
# Ported from scripts/hyperparam/sensitivity_common.sh (itself ported from
# the Pamba sensitivity harness, and already reused once by
# scripts/robustness/robustness_common.sh) with the dispatch engine kept
# verbatim:
#   - GPU_LIST mode: round-robin GPU assignment via a file-locked atomic
#     counter (_rotate_gpu) + per-GPU FIFO semaphore bounding concurrency
#     to PARA_PER_GPU tokens per GPU (_dispatch).
#   - Standalone (no GPU_LIST): batches under PARALLEL on the current
#     CUDA_VISIBLE_DEVICES.
#
# Differences vs sensitivity: each cell is one (dataset, pred_len, variant,
# seed) run of the project's default entry point (python -m src.train) with
# the tuned configs/<dataset>.yaml plus the cumulative overrides produced by
# scripts/ablation/gen_cells.py from scripts/ablation/cumulative_registry.py
# (single source of truth for the chain). Variants run strictly serially --
# each variant occupies all GPUs (Pamba run_all_ablation.sh policy) -- while
# cells inside a variant are dispatched concurrently.
#
# Idempotency, two layers (Pamba semantics):
#   1. structural no-ops: gen_cells.py flags cells whose step changes
#      nothing for the dataset recipe; they are skipped here ([skip-noop]).
#   2. completed cells: checkpoints/cumulative_ablation/<name>_results.json
#      exists (written atomically by src.train only after a successful
#      train + test) -> [skip]. Re-launching the same command resumes.
#
# Checkpoint disk: per-cell .pt files are deleted after a successful run
# unless CUM_KEEP_CHECKPOINTS=1 (the provenance results.json is the durable
# record; tables are rebuilt from it).

PARALLEL=2
CNT=0

CUM_CKPT_DIR="${CUM_CKPT_DIR:-checkpoints/cumulative_ablation}"
CUM_CFG_DIR="${CUM_CFG_DIR:-${CUM_CKPT_DIR}/cfg}"
CUM_GEN="scripts/ablation/gen_cells.py"

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
            _sem="${_CUM_SESSION:-/tmp}/_ddmamba_cum_sem_${_g}_$$"
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
# Each worker's stdout/stderr goes to a per-worker log file under
# $_CUM_LOG_DIR (if set) so cell-level errors are recoverable even when the
# parent run.log is lost.
_dispatch() {
    local cmd=("$@")
    if [[ -n "${GPU_LIST:-}" ]]; then
        local gpu="${_CURRENT_GPU}"
        local fd="${_GPU_FD[$gpu]}"
        (
            eval "read -r _ <&${fd}"
            if [[ -n "${_CUM_LOG_DIR:-}" && -d "${_CUM_LOG_DIR:-}" ]]; then
                local _wlog="${_CUM_LOG_DIR}/worker_gpu${gpu}_$$.log"
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$_wlog" 2>&1
            else
                CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
            fi
            eval "echo >&${fd}"
        ) &
    else
        if [[ -n "${_CUM_LOG_DIR:-}" && -d "${_CUM_LOG_DIR:-}" ]]; then
            local _wlog="${_CUM_LOG_DIR}/worker_standalone_$$.log"
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
# Idempotency -- _cum_cell_done checks whether the cell's results.json
# already exists. src.train writes it atomically only after a successful
# train + test, so existence means the cell completed.
# Usage: _cum_cell_done <run_name>
# ============================================================
_cum_cell_done() {
    [[ -f "${CUM_CKPT_DIR}/$1_results.json" ]]
}

# ============================================================
# _cum_train_wrapper -- one cell: python -m src.train against the
# per-(dataset, variant) resolved config materialized by gen_cells.py
# (chain overrides are baked into the YAML -- string values such as
# freq_backbone 'none' cannot survive src.train's CLI _coerce), then
# post-success checkpoint cleanup (results.json is the durable record).
# Runs inside the dispatch subshell, so bash functions remain available.
# CUM_EXTRA (e.g. "--train.epochs 1") is forwarded verbatim.
# Usage: _cum_train_wrapper <config> <pred_len> <seed> <run_name>
# ============================================================
_cum_train_wrapper() {
    local config=$1 pl=$2 seed=$3 run_name=$4
    # shellcheck disable=SC2086
    python -u -m src.train \
        --config "$config" \
        --pred_len "$pl" \
        --experiment.seed "$seed" \
        --experiment.name "$run_name" \
        --experiment.checkpoint_dir "$CUM_CKPT_DIR" \
        ${CUM_EXTRA:-}
    local rc=$?
    if [[ $rc -eq 0 && "${CUM_KEEP_CHECKPOINTS:-0}" != "1" ]] \
        && [[ -f "${CUM_CKPT_DIR}/${run_name}_results.json" ]]; then
        rm -f "${CUM_CKPT_DIR}/${run_name}_best.pt"
    fi
    return $rc
}

# ============================================================
# run_all_cells -- iterate the gen_cells.py TSV. Cells are emitted in
# (variant, dataset, pred_len) order; when the variant field changes the
# engine drains all in-flight jobs first, so variants run strictly serially
# (each variant occupies all GPUs) while cells within a variant run
# concurrently. Structural no-ops and completed cells are skipped.
# CUM_GEN_EXTRA (e.g. "--datasets ETTh1 traffic --seed 2025") is forwarded
# to gen_cells.py.
# Usage: run_all_cells
# ============================================================
run_all_cells() {
    local tsv line cur_variant="__none__"
    tsv=$(python3 "$CUM_GEN" --format tsv --configs-dir "$CUM_CFG_DIR" \
        ${CUM_GEN_EXTRA:-}) \
        || { echo "[error] gen_cells.py failed"; return 1; }
    while IFS=$'\t' read -r action ds pl variant seed run_name config flags; do
        [[ -z "${action:-}" ]] && continue
        if [[ "$variant" != "$cur_variant" ]]; then
            if [[ "$cur_variant" != "__none__" ]]; then
                _wait_all
                echo "--- variant $cur_variant drained ---"
            fi
            cur_variant="$variant"
            echo "=== variant: $variant ==="
        fi
        if [[ "$action" == "noop" ]]; then
            echo "[skip-noop] ${ds} h=${pl} ${variant}: recipe already at target"
            continue
        fi
        if _cum_cell_done "$run_name"; then
            echo "[skip] ${run_name}: results.json exists"
            continue
        fi
        _rotate_gpu
        _run _cum_train_wrapper "$config" "$pl" "$seed" "$run_name"
    done <<< "$tsv"
    _wait_all
    if [[ "$cur_variant" != "__none__" ]]; then
        echo "--- variant $cur_variant drained ---"
    fi
}
