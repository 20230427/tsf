#!/bin/bash

# =============================================================================
# Robustness runner for weather (Pamba-style per-dataset script)
# =============================================================================
# Trains via the project's default entry point (``python -m src.train``) with
# the tuned per-dataset config ``configs/weather.yaml``, then evaluates the
# test split under additive Gaussian noise (``--noise_level``):
# x_tilde = x + N(0, sigma^2) on the standardized input window only. Targets
# / window stats are never corrupted; training and validation always see
# clean inputs (see src/train.py:evaluate).
#
# Environment:
#   noise_level  Gaussian sigma (default: 0.1); 0 = clean anchor
#   seed         model seed (default: 2023)
#   ckpt_dir     checkpoint/result dir (default: checkpoints/robustness)
#   extra_flags  forwarded verbatim to src.train (e.g. "--train.epochs 1")
#
# Usage:
#   noise_level=0.2 bash scripts/robustness/weather.sh [96|192|336|720|all]
#
# Skips a (pred_len, noise_level, seed) cell whose results.json already
# exists, so the sweep is resumable. Results are aggregated into
# output/robustness/ by scripts/robustness/collect_robustness.py.
# =============================================================================

dataset=weather
config=configs/weather.yaml
noise_level=${noise_level:-0.1}
seed=${seed:-2023}
ckpt_dir=${ckpt_dir:-checkpoints/robustness}
extra_flags=${extra_flags:-}

_robustness_common() {
    local pl=$1
    local name="${dataset}_robustness_pl${pl}_nl${noise_level}_s${seed}"
    local results="${ckpt_dir}/${name}_results.json"
    if [[ -f "$results" ]]; then
        echo "[skip] ${name}: ${results} already exists"
        return 0
    fi
    # shellcheck disable=SC2086
    python -u -m src.train \
        --config "$config" \
        --pred_len "$pl" \
        --train.noise_level "$noise_level" \
        --experiment.seed "$seed" \
        --experiment.name "$name" \
        --experiment.checkpoint_dir "$ckpt_dir" \
        $extra_flags
}

run_pl96()  { _robustness_common 96; }
run_pl192() { _robustness_common 192; }
run_pl336() { _robustness_common 336; }
run_pl720() { _robustness_common 720; }

if [[ "${1:-all}" == "all" ]]; then
    run_pl96
    run_pl192
    run_pl336
    run_pl720
else
    "run_pl${1}"
fi
