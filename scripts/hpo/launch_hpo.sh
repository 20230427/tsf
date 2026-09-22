#!/usr/bin/env bash
# Launch Optuna HPO sweeps over several datasets, detached with logs.
# Ported from scripts/hyperparam/launch_sensitivity.sh conventions.
#
# Usage:
#   ./scripts/hpo/launch_hpo.sh                                # all benchmark configs
#   ./scripts/hpo/launch_hpo.sh configs/ETTh1.yaml configs/weather.yaml
#   ./scripts/hpo/launch_hpo.sh configs/ETTh1.yaml --trials 30 --pred_len 96
#
# Extra args after the config list are forwarded to scripts/hpo/run_hpo.py.
# Runs are resumable: a study whose SQLite DB already holds n_trials finished
# trials is skipped by run_hpo itself, so relaunching just continues.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR%/scripts*}")"

CONFIGS=()
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -*) EXTRA+=("$1");;
        *)  CONFIGS+=("$1");;
    esac
    shift
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=(configs/ETTh1.yaml configs/ETTh2.yaml configs/ETTm1.yaml configs/ETTm2.yaml \
             configs/weather.yaml configs/electricity.yaml configs/exchange_rate.yaml \
             configs/solar.yaml)
fi

mkdir -p log/hpo

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "${cfg%.yaml}")"
    echo "[launch_hpo] ${cfg} -> log/hpo/${name}_hpo.log (${EXTRA[*]:-})"
    nohup python scripts/hpo/run_hpo.py --config "${cfg}" --bg "${EXTRA[@]}" \
        >> "log/hpo/${name}_hpo.log" 2>&1 &
    echo "[launch_hpo] pid $!"
done

echo "[launch_hpo] all launched; inspect with: python scripts/hpo/hpo_dump.py --list"
