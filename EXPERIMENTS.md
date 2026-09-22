# Revision experiment runbook (review items 2–6)

> **Run everything in one batch:** `bash scripts/run_revision_batch.sh`
> (needs a GPU + dataset files; resumable; logs to `logs/revision_batch.log`).
> It runs the validation-only selection protocol (fixes the Solar leakage),
> the unified same-pipeline baselines incl. Crossformer on all nine datasets,
> 10-seed ablations + branch-matrix completion + BH stats, PEMS confirmatory
> runs, and matched efficiency profiling — in that priority order. The
> per-command reference below explains each step.


Turnkey commands for the experiments the *Neurocomputing* review asks for.
All scaffolding is in place and has passed static and pure-statistics CPU
checks; the training runs themselves need
a CUDA GPU. Everything routes through the
**same** data pipeline, splits, training loop, schedule, and evaluation as
DD-Mamba. This controls pipeline and evaluation differences; repository-native
reimplementations can still differ from official baseline code, so
implementation fidelity remains a stated limitation.

Nothing here is run for you (no GPU / dataset files in the authoring
environment). Each command writes JSON under `checkpoints/`; the analysis
scripts turn those into the paper's tables and statistics.

The batch raises component-ablation runs to ten seeds. The much larger branch
matrix remains at three seeds unless the author increases `--seeds`; its
small-n exact sign-flip sensitivity is consequently low-powered and must be
reported alongside the paired-t/BH analysis.

---

## Item 2 — Unified, same-pipeline baselines

> **Status: legacy outputs exist for 6/9 datasets.** ETTh1/2, ETTm1/2,
> Weather, and Electricity have `results/unified_*.json`, but those files lack
> resolved configs/commit hashes and are therefore `legacy_unverifiable`.
> Provenance-enabled reruns are required for auditable result attribution;
> Solar, Traffic, and Exchange have no complete legacy sweep.

Replaces the quoted-from-S-Mamba numbers with reruns in one pipeline. Available
repository-native baselines: `dlinear`, `nlinear`, `rlinear`, `patchtst`, `itransformer`,
`smamba`, `msmamba`, and `crossformer` (DSW embedding + two-stage attention;
`src/models/baselines.py`). `crossformer` is now in the default sweep but has
**not been run yet** — its numbers are pending and not in the paper's Table 4.
`tf4tf` has **no** fabricated in-repo model — supply the authors' code through
the adapter.

```bash
# Freeze each baseline under the same eight-trial validation-only budget.
python scripts/run_baseline_selection.py --config configs/ETTh1.yaml \
    --search-space configs/baseline_search_spaces.yaml \
    --archs dlinear rlinear patchtst itransformer smamba msmamba \
    --seeds 5 --out checkpoints/baseline_selection_ETTh1.json

# Only then run the frozen baselines and DD-Mamba on test.
python scripts/run_unified_baselines.py --config configs/ETTh1.yaml --seeds 5 \
    --archs dual_domain dlinear rlinear patchtst itransformer smamba msmamba \
    --selection-report checkpoints/baseline_selection_ETTh1.json \
    --require-selected-baselines

# Repeat per dataset (ETTh1/2, ETTm1/2, weather, solar, electricity, traffic,
# exchange_rate, and the PEMS configs from item 6).

# TF4TF via the authors' implementation (installed separately):
python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
    --archs tf4tf --external_impl tf4tf=tf4tf_official.model:TF4TF --seeds 5
```

The predeclared search axes and equal eight-trial budget live in
`configs/baseline_search_spaces.yaml`; the selection report records every
candidate and refuses a changed dataset config at final evaluation. Output:
`checkpoints/unified_<name>.json` + a markdown table. New
JSON includes the resolved config/hash, runtime and Git provenance, explicit
seed set, and seed/horizon/architecture metadata for every run. Check it with
`python scripts/validate_provenance.py checkpoints/unified_<name>.json`.

## Item 3 — Validation-only model-selection protocol

Removes the model-selection bias (e.g. Solar RevIN-off chosen on test
ablations). The candidate phase fixes and records the complete search space,
metric, and budget, constructs no test dataset/DataLoader, and selects on
**validation only**. Only after the winner is frozen does an independent
final-confirmation phase enable test evaluation. No test-oracle is computed.

```bash
python scripts/run_selection_protocol.py --config configs/solar.yaml \
    --horizons 96 192 336 720 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,1 \
    --switch model.freq_backbone=fits,none \
    --fixed model.use_revin=true,model.channel_mixer_layers=0
```

Run per dataset; the FIXED assignment should be the SAME everywhere (that is
the point of a fixed reference). Output: `checkpoints/selection_<name>.json`
with validation-only candidate records and test metrics only under the frozen
candidate's `final_confirmation` block.

## Item 4 — Complete the branch-ablation matrix

The legacy file has 27 cells: four horizons for six datasets and only H=96
for Electricity/Traffic/Exchange. Because it has no immutable provenance, its
cells must not be spliced into a new result. Rerun the complete 9×4 grid into
one provenance-enabled file. This yields a 72-comparison branch family (36
cells × two removals), so the manuscript's legacy 54-test wording must be
updated only after this run finishes.

```bash
python scripts/run_branch_matrix.py \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather solar electricity traffic exchange_rate \
    --horizons 96 192 336 720 --seeds 3 \
    --out results/provenance/Branch_matrix_provenance_v1.json
```

Do not resume into the old `results/Branch_matrix.json`: it has no provenance.
The runner detects such legacy JSON, refuses to rewrite it, and asks for a new
output path. Schema-valid outputs are resumed with atomic replacement.

Then recompute significance **with multiplicity control** (feeds Table 6 and
the exploratory/confirmatory split in the paper):

```bash
python scripts/compute_stats_correction.py \
    --results-dir results/provenance \
    --branch-file Branch_matrix_provenance_v1.json \
    --component-source Ablation_ETTh1.json=ETTh1 \
    --component-source Ablation_solar_final.json=Solar \
    --component-source Ablation_weather.json=Weather \
    --output-dir generated/provenance_audit
```

## Item 5 — Matched efficiency (wall-clock / latency / throughput / memory)

Profiles DD-Mamba and the baselines on the **same** device and batch, so the
complexity table has matched runtime instead of params/MACs only. Peak GPU
memory needs CUDA.

```bash
python scripts/profile_efficiency.py --device cuda --batch 32 \
    --config configs/traffic.yaml \
    --archs dual_domain smamba itransformer patchtst dlinear rlinear msmamba
```

Emits a per-arch table and LaTeX rows (params, MACs, peak mem, latency,
throughput).

## Item 6 — Confirmatory datasets (PEMS03/04/07/08)

Untouched datasets that were **not** used to design DD-Mamba, to test whether
the validation-only selection logic (item 3) generalizes. Configs:
`configs/PEMS0{3,4,7,8}.yaml`; loader reads the standard `.npz`
(`source: npz`, key `data`, feature 0 = flow).

```bash
# Place the standard files at data/PEMS0{3,4,7,8}.npz, then:
python scripts/run_unified_baselines.py --config configs/PEMS04.yaml --seeds 5
python scripts/run_selection_protocol.py --config configs/PEMS04.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,2 \
    --fixed model.use_revin=true,model.channel_mixer_layers=0
python scripts/run_branch_matrix.py --datasets PEMS04 --horizons 96 192 336 720
```

(PEMS node counts: 03→358, 04→307, 07→883, 08→170; 5-min sampling;
0.6/0.2/0.2 split, matching the iTransformer/S-Mamba protocol.)

---

## After the runs: fold results back into the paper

1. `compute_stats_correction.py` → refresh the raw-vs-BH counts in
   §Experiments and the branch-matrix caption.
2. Unified table (item 2) → replace the quoted-baseline Table with rerun
   numbers and drop the "indicative, not confirmatory" caveat.
3. Selection report (item 3) → report the declared search space/budget,
   validation-selected frozen candidate, and its independent final test.
4. Efficiency table (item 5) → fill the matched-runtime columns.
5. PEMS (item 6) → add as confirmatory rows.

Generate the single JSON-sourced statistical appendix, then run the submission
gate. The second command is expected to fail until the authors replace all
identity, repository, funding, CRediT, conflict, and acknowledgement prompts.

```bash
python scripts/generate_statistical_audit.py \
    --results-dir results/provenance \
    --branch-file Branch_matrix_provenance_v1.json \
    --component-source Ablation_ETTh1.json=ETTh1 \
    --component-source Ablation_solar_final.json=Solar \
    --component-source Ablation_weather.json=Weather \
    --output-dir generated/provenance_audit
python scripts/audit_submission.py --json generated/submission_audit.json
```
