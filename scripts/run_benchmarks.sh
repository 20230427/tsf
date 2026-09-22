#!/usr/bin/env bash
# Run the full benchmark suite: every dataset x its standard horizon set.
#   ./scripts/run_benchmarks.sh                 # all datasets
#   ./scripts/run_benchmarks.sh ETTh1 weather   # subset of datasets
#   ./scripts/run_benchmarks.sh PEMS04 PEMS08   # PEMS uses H in {12,24,48,96}
# Extra flags are forwarded to src.train, e.g.:
#   ./scripts/run_benchmarks.sh traffic --freq_encoder mamba
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity solar exchange_rate traffic)
PEMS_DATASETS=(PEMS03 PEMS04 PEMS07 PEMS08)
# The long-range benchmarks forecast 4-30 days ahead at hourly / 10-min / 15-min
# sampling. PEMS is sampled every 5 minutes, so its standard horizons are
# {12,24,48,96} steps = 1-8 hours; running it at 720 would forecast 2.5 days
# from an 8-hour window and is not the protocol other papers report.
HORIZONS=(96 192 336 720)
PEMS_HORIZONS=(12 24 48 96)

horizons_for() {
  case " ${PEMS_DATASETS[*]} " in
    *" $1 "*) echo "${PEMS_HORIZONS[@]}" ;;
    *)        echo "${HORIZONS[@]}" ;;
  esac
}

DATASETS=()
EXTRA=()
for arg in "$@"; do
  if [[ " ${ALL_DATASETS[*]} ${PEMS_DATASETS[*]} " == *" ${arg} "* ]]; then
    DATASETS+=("$arg")
  else
    EXTRA+=("$arg")
  fi
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

for ds in "${DATASETS[@]}"; do
  for h in $(horizons_for "${ds}"); do
    echo "=== ${ds} | pred_len=${h} ==="
    python -m src.train \
      --config "configs/${ds}.yaml" \
      --pred_len "${h}" \
      --experiment.name "${ds}_h${h}" \
      "${EXTRA[@]+"${EXTRA[@]}"}"
  done
done
