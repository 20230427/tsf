#!/usr/bin/env bash
# =============================================================================
# DD-Mamba revision experiments — run together, in one batch.
#
# Requires: a CUDA GPU and the dataset files under data/ (ETT*.csv, weather,
# solar_AL.txt.gz, electricity/traffic/exchange_rate, and PEMS0{3,4,7,8}.npz
# for the confirmatory step). No arguments; run from the repo root:
#     bash scripts/run_revision_batch.sh
#
# The branch-matrix stage resumes provenance-valid cells. Other stages retrain
# and atomically replace their output, so preserve completed outputs before a
# deliberate full rerun. Every command is teed to logs/revision_batch.log.
# The script stops immediately when a command or logging pipeline fails.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
LOG="logs/revision_batch.log"
run () { echo -e "\n\033[1m=== $* ===\033[0m" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

ALL_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather electricity solar traffic exchange_rate"
MATCHED_BASELINES="dlinear rlinear patchtst itransformer smamba msmamba"

# -----------------------------------------------------------------------------
# CONCERN 1 (most serious): kill the Solar RevIN test-set leakage.
# Re-select every tuned switch on VALIDATION only; report the predeclared fixed
# reference and the validation-selected candidate. After this, set each config
# to its *validation-selected* switch
# (printed in checkpoints/selection_<name>.json) BEFORE trusting Section-4 numbers
# — most importantly Solar's RevIN flag.
# -----------------------------------------------------------------------------
run python scripts/run_selection_protocol.py --config configs/solar.yaml \
    --horizons 96 192 336 720 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,1 \
    --fixed model.use_revin=true,model.channel_mixer_layers=1

run python scripts/run_selection_protocol.py --config configs/electricity.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.mixer_placement=both,shared \
    --fixed model.use_revin=true,model.mixer_placement=both

run python scripts/run_selection_protocol.py --config configs/weather.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,2 \
    --fixed model.use_revin=true,model.channel_mixer_layers=2

# Hands-off: write each VALIDATION-SELECTED switch back into its config
# (comments preserved), so the runs below use validation-chosen settings only
# — no test-informed choice survives. --dry-run first to log the diff.
run python scripts/apply_selection.py --dry-run \
    checkpoints/selection_solar.json \
    checkpoints/selection_electricity.json \
    checkpoints/selection_weather.json
run python scripts/apply_selection.py \
    checkpoints/selection_solar.json \
    checkpoints/selection_electricity.json \
    checkpoints/selection_weather.json

# -----------------------------------------------------------------------------
# CONCERN 3: validation-only tuning, followed by frozen same-pipeline baselines.
# The same six architecture families and the same eight-trial budget are used
# on all nine datasets. Final test evaluation is performed only after each
# validation winner has been frozen in its selection report.
# -----------------------------------------------------------------------------
for ds in $ALL_DATASETS; do
    run python scripts/run_baseline_selection.py \
        --config "configs/${ds}.yaml" \
        --search-space configs/baseline_search_spaces.yaml \
        --archs $MATCHED_BASELINES --seeds 5 \
        --out "checkpoints/baseline_selection_${ds}.json"
    run python scripts/run_unified_baselines.py \
        --config "configs/${ds}.yaml" \
        --archs dual_domain $MATCHED_BASELINES --seeds 5 \
        --selection-report "checkpoints/baseline_selection_${ds}.json" \
        --require-selected-baselines \
        --out "results/provenance/unified_${ds}.json"
done
run python scripts/analyze_unified_baselines.py \
    --results-dir results/provenance --output-dir generated/provenance_audit

# -----------------------------------------------------------------------------
# CONCERN 2 (support the narrowed claims): raise ablation power to 10 seeds and
# rerun the complete 9-dataset x 4-horizon branch matrix under one auditable
# provenance record, then recompute BH-corrected significance.
# -----------------------------------------------------------------------------
for ds in ETTh1 solar weather; do
    case "$ds" in
        ETTh1) out="results/provenance/Ablation_ETTh1.json" ;;
        solar) out="results/provenance/Ablation_solar_final.json" ;;
        weather) out="results/provenance/Ablation_weather.json" ;;
    esac
    run python scripts/run_ablation.py --config "configs/${ds}.yaml" \
        --seeds 10 --out "$out"
done
run python scripts/run_branch_matrix.py \
    --datasets $ALL_DATASETS --horizons 96 192 336 720 --seeds 3 \
    --out results/provenance/Branch_matrix_provenance_v1.json
run python scripts/compute_stats_correction.py \
    --results-dir results/provenance \
    --branch-file Branch_matrix_provenance_v1.json \
    --component-source Ablation_ETTh1.json=ETTh1 \
    --component-source Ablation_solar_final.json=Solar \
    --component-source Ablation_weather.json=Weather \
    --output-dir generated/provenance_audit

# -----------------------------------------------------------------------------
# Confirmatory datasets (does validation-only selection generalize?) — PEMS.
# Needs data/PEMS0{3,4,7,8}.npz. dual_domain only keeps it cheap; add more
# --archs to widen the comparison.
# -----------------------------------------------------------------------------
for ds in PEMS03 PEMS04 PEMS07 PEMS08; do
    run python scripts/run_unified_baselines.py --config "configs/${ds}.yaml" \
        --archs dual_domain --seeds 5 \
        --out "results/provenance/unified_${ds}.json"
done

# One JSON-sourced appendix after every provenance-enabled result is present.
run python scripts/generate_statistical_audit.py \
    --results-dir results/provenance \
    --branch-file Branch_matrix_provenance_v1.json \
    --component-source Ablation_ETTh1.json=ETTh1 \
    --component-source Ablation_solar_final.json=Solar \
    --component-source Ablation_weather.json=Weather \
    --output-dir generated/provenance_audit

# -----------------------------------------------------------------------------
# Matched efficiency (wall-clock / latency / throughput / peak memory) on GPU.
# -----------------------------------------------------------------------------
run python scripts/profile_efficiency.py --device cuda --batch 32 \
    --config configs/traffic.yaml \
    --archs dual_domain smamba itransformer patchtst crossformer dlinear rlinear msmamba

echo -e "\n\033[1mBATCH COMPLETE.\033[0m Outputs: results/provenance/*.json, "\
"checkpoints/{selection,baseline_selection}_*.json, generated/provenance_audit/. "\
"Fold them back into the paper per EXPERIMENTS.md." | tee -a "$LOG"
