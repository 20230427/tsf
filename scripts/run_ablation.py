#!/usr/bin/env python
"""Component-ablation suite: train controlled variants and report deltas.

Each variant changes exactly ONE component relative to the dataset's base
config, so the metric delta isolates that component's contribution. All
variants share the split, schedule, seed(s), and every other hyperparameter.

Usage
-----
    # Full suite on a dataset (auto-selects the variants that apply):
    python scripts/run_ablation.py --config configs/ETTh1.yaml

    # Subset, multiple seeds, extra overrides forwarded to every variant:
    python scripts/run_ablation.py --config configs/weather.yaml \
        --variants full time_only freq_only no_revin --seeds 3 --epochs 5

Results are written to checkpoints/ablation_<name>.json and printed as a
markdown table (mean +/- std over seeds; delta vs the full model).
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import (  # noqa: E402
    apply_overrides,
    atomic_write_json,
    config_sha256,
    load_config,
    parse_overrides,
    provenance_fields,
)

# variant name -> (description, {(section, key): value})
VARIANTS = {
    "full": ("complete model (reference)", {}),
    # --- dual-branch structure ---
    "time_only": ("time branch alone (no frequency branch)",
                  {("model", "fusion"): "time_only"}),
    "freq_only": ("frequency branch alone (no time branch)",
                  {("model", "fusion"): "freq_only"}),
    "fusion_sum": ("plain average instead of the learned gate",
                   {("model", "fusion"): "sum"}),
    # --- convex vs. additive fusion ---
    # The gated/concat/sum rules are convex, so the output is confined to the
    # segment between the two branch forecasts and the branches act as
    # substitutes. These variants lift that constraint and let them superpose.
    "fusion_residual": ("additive fusion y = y_time + alpha * y_freq (scalar alpha)",
                        {("model", "fusion"): "residual"}),
    "fusion_affine": ("two independent gates (no sum-to-one constraint)",
                      {("model", "fusion"): "affine"}),
    "fusion_doubly_residual": ("freq branch fits the time branch's residual",
                               {("model", "fusion"): "doubly_residual"}),
    # --- normalization & linear anchors ---
    "no_revin": ("no instance normalization (RevIN off)",
                 {("model", "use_revin"): False}),
    "with_revin": ("enable standard instance normalization (RevIN on)",
                    {("model", "use_revin"): True}),
    # alpha-RevIN: normalization strength learned from training data instead of
    # chosen per dataset as an on/off switch (removes a test-informed choice).
    "revin_alpha_learned": ("alpha-RevIN: one learned global normalization strength",
                            {("model", "use_revin"): True,
                             ("model", "revin_alpha"): "learned"}),
    "revin_alpha_channel": ("alpha-RevIN: per-channel learned normalization strength",
                            {("model", "use_revin"): True,
                             ("model", "revin_alpha"): "channel"}),
    "no_linear_backbone": ("time branch without its DLinear backbone",
                           {("model", "time_linear_backbone"): False}),
    "rand_init": ("standard random init instead of zero-init heads/gate",
                  {("model", "zero_init"): False}),
    "no_fits": ("frequency branch without the FITS spectral backbone",
                {("model", "freq_backbone"): "none"}),
    "with_fits": ("add the FITS spectral backbone",
                  {("model", "freq_backbone"): "fits"}),
    # Separates the two things 'with_fits' conflates: silencing the frequency
    # head at init, and adding the spectral map. On backbone='none' datasets the
    # head is randomly initialized and injects noise; this variant silences it
    # at zero parameter cost and without the map's n_freq -> n_out_freq
    # extrapolation, whose ratio grows with the horizon.
    "freq_head_zero_init": ("silence the frequency head at init (no FITS map)",
                            {("model", "freq_zero_init_head"): "always"}),
    # --- Mamba components ---
    "time_mlp": ("replace the time-axis Mamba encoder with an MLP",
                 {("model", "time_encoder"): "mlp"}),
    "time_mamba": ("replace the time-axis MLP encoder with Mamba",
                   {("model", "time_encoder"): "mamba"}),
    "time_bimamba": ("replace the time encoder with bidirectional Mamba",
                      {("model", "time_encoder"): "bimamba"}),
    "time_attention": ("replace the time encoder with self-Attention",
                        {("model", "time_encoder"): "attention"}),
    "no_time_encoder": ("remove the temporal-dependency encoder; retain DLinear",
                        {("model", "time_encoder"): "none"}),
    "no_channel_mixer": ("no cross-channel (variate) Mamba mixing",
                         {("model", "channel_mixer_layers"): 0}),
    "with_channel_mixer": ("add 1 layer of cross-channel Mamba mixing",
                           {("model", "channel_mixer_layers"): 1}),
    "shared_mixer": ("variate mixer weight-tied across branches (~half mixer params)",
                     {("model", "mixer_placement"): "shared"}),
    "both_mixer": ("independent variate mixer in each branch (untie them)",
                   {("model", "mixer_placement"): "both"}),
    "time_mixer_only": ("variate mixer in the time branch only (freq unmixed)",
                        {("model", "mixer_placement"): "time"}),
    "freq_mixer_only": ("variate mixer in the frequency branch only (time unmixed)",
                         {("model", "mixer_placement"): "freq"}),
    # --- variate-encoder alternatives (S-Mamba Tab. 5, "VC Encoding") ---
    # S-Mamba benchmarks its bidirectional Mamba variate block against
    # Attention.  ``vc_unimamba`` is an additional DD-Mamba order-sensitivity
    # stress test; S-Mamba used uni-Mamba on TD, not as its VC replacement.
    "vc_unimamba": ("variate mixer: bi-Mamba -> uni-Mamba (one direction)",
                    {("model", "mixer_kind"): "unimamba"}),
    "vc_attention": ("variate mixer: bi-Mamba -> multi-head Attention",
                     {("model", "mixer_kind"): "attention"}),
    "freq_mamba": ("bidirectional Mamba over frequency bins instead of the linear filter",
                   {("model", "freq_encoder"): "mamba"}),
    "freq_linear": ("linear complex filter over frequency bins",
                    {("model", "freq_encoder"): "linear"}),
    "freq_sparsity_0": ("retain all frequency bins (rho=0.0)",
                        {("model", "freq_sparsity"): 0.0}),
    "freq_sparsity_20": ("drop highest 20% of frequency bins (rho=0.2)",
                         {("model", "freq_sparsity"): 0.2}),
    "freq_sparsity_40": ("drop highest 40% of frequency bins (rho=0.4)",
                         {("model", "freq_sparsity"): 0.4}),
    "freq_sparsity_60": ("drop highest 60% of frequency bins (rho=0.6)",
                         {("model", "freq_sparsity"): 0.6}),
    # --- isolated initialization controls ---
    "time_head_random_init": ("random-init temporal correction head only",
                              {("model", "time_zero_init_head"): False}),
    "time_head_zero_init": ("zero-init temporal correction head only",
                            {("model", "time_zero_init_head"): True}),
    "freq_backbone_random_init": ("random-init FITS spectral map only",
                                  {("model", "freq_zero_init_backbone"): False}),
    "freq_backbone_zero_init": ("zero-init FITS spectral map only",
                                {("model", "freq_zero_init_backbone"): True}),
    "freq_head_random_init": ("random-init frequency forecast head only",
                              {("model", "freq_zero_init_head"): "never"}),
    "freq_head_zero_init": ("zero-init frequency forecast head only",
                            {("model", "freq_zero_init_head"): "always"}),
    "fusion_random_init": ("standard random initialization of fusion parameters only",
                           {("model", "fusion_zero_init"): False}),
    "fusion_zero_init": ("time-biased zero initialization of fusion parameters only",
                         {("model", "fusion_zero_init"): True}),
    # --- dispersion chain (STD-style scale forecasting) ---
    "disp_base": ("base: no RevIN, no dispersion (raw de-norm)",
                  {("model", "use_revin"): False, ("model", "dispersion"): "none"}),
    "disp_revin": ("RevIN only (window mean/std de-norm)",
                   {("model", "use_revin"): True, ("model", "dispersion"): "none"}),
    "disp_fixed": ("fixed historical dispersion (longest-resolution std)",
                   {("model", "use_revin"): True, ("model", "dispersion"): "fixed"}),
    "disp_learned": ("learned dispersion head (per-horizon predicted scale)",
                     {("model", "use_revin"): True, ("model", "dispersion"): "learned"}),
}

# The dispersion ablation chain (base -> RevIN -> fixed -> learned), run as an
# explicit ordered subset: python scripts/run_ablation.py --config ... \
#     --variants disp_base disp_revin disp_fixed disp_learned --seeds 3
DISPERSION_CHAIN = ["disp_base", "disp_revin", "disp_fixed", "disp_learned"]

# Convex-vs-additive fusion study: does the frequency branch look inert because
# spectral modeling does not help here, or because the convex gate forces the
# branches to compete instead of superpose? Run as an explicit subset:
#   python scripts/run_ablation.py --config ... --variants full fusion_residual \
#       fusion_affine fusion_doubly_residual --seeds 5
FUSION_RULE_CHAIN = ["full", "fusion_sum", "fusion_residual",
                     "fusion_affine", "fusion_doubly_residual"]

# alpha-RevIN study: replaces the per-dataset on/off RevIN switch (a
# test-informed choice on Solar) with a strength learned on training data.
#   python scripts/run_ablation.py --config ... --variants full no_revin \
#       revin_alpha_learned revin_alpha_channel --seeds 5
REVIN_ALPHA_CHAIN = ["full", "@revin_flip", "revin_alpha_learned",
                     "revin_alpha_channel"]

# Submission-facing groups.  CORE_ABLATION contains the claim-bearing DD-Mamba
# components.  TD_REPLACEMENT_CHAIN mirrors S-Mamba's replace/remove logic for
# temporal-dependency encoding.  Placement and initialization are separated so
# their multiple-comparison families can be corrected independently.
CORE_ABLATION = [
    "full", "no_channel_mixer", "vc_attention", "@time_encoder_flip",
    "time_only", "freq_only", "freq_mamba", "fusion_sum",
    "no_linear_backbone", "@revin_flip",
]
TD_REPLACEMENT_CHAIN = [
    "full", "@time_encoder_flip", "time_bimamba", "time_attention",
    "no_time_encoder",
]
MIXER_PLACEMENT_CHAIN = [
    "full", "both_mixer", "shared_mixer", "time_mixer_only",
    "freq_mixer_only", "no_channel_mixer",
]
INITIALIZATION_CHAIN = [
    "full", "@time_head_init_flip", "@freq_head_init_flip",
    "@freq_backbone_init_flip", "@fusion_init_flip",
]
SPECTRAL_STRATEGY_CHAIN = [
    "full", "@fits_flip", "freq_linear", "freq_mamba",
    "freq_sparsity_0", "freq_sparsity_20", "freq_sparsity_40",
    "freq_sparsity_60",
]

# ---------------------------------------------------------------------------
# S-Mamba-style component ablation.
#
# S-Mamba (Wang et al., Neurocomputing 2025) validates its design by ablating
# two blocks: the VC (Variate Correlation) bidirectional Mamba across the
# variate axis, and the TD (Temporal Dependency) block over time. Each is
# removed or swapped for an alternative, and the accuracy delta attributed to
# it. DD-Mamba has structurally matching components plus two it does not share
# (the parallel frequency branch and the fusion rule), so the analogous
# ablation is the union below, grouped by which block each variant probes.
#
# Difference in method, deliberately: S-Mamba reports single-run deltas. We run
# multiple seeds and report paired confidence intervals, then apply
# Benjamini-Hochberg across the family (scripts/compute_stats_correction.py) --
# with several variants a few can clear an uncorrected threshold by chance.
# Structured as S-Mamba's Tab. 5: rows grouped by DESIGN ("reference",
# "Replace", "w/o"), each naming which block is altered and what it becomes.
# ms-Mamba's Tab. 4 adds the complementary idea of sweeping a *design choice*
# rather than only removing parts.  DD-Mamba applies that organization in the
# separate fusion, placement, spectral, normalization, and initialization
# chains; these are not mislabeled as Mamba sampling-rate scales.
#
# The three blocks, with S-Mamba's names in brackets:
#   VC   [Variate Correlation] -- the cross-variate mixer
#   TD   [Temporal Dependency] -- the time-axis encoder
#   FD                         -- the frequency branch, which S-Mamba lacks
#
# (design group, variant, VC cell, TD cell, FD cell)
SMAMBA_GRID = [
    ("reference", "full",             "bi-Mamba", "base",      "linear"),
    # -- Replace: swap a block for an alternative of the same role ----------
    ("Replace",   "vc_attention",     "Attention", "base",     "linear"),
    ("Replace",   "@time_encoder_flip", "bi-Mamba", "swapped", "linear"),
    ("Replace",   "time_bimamba",     "bi-Mamba", "bi-Mamba",  "linear"),
    ("Replace",   "time_attention",   "bi-Mamba", "Attention", "linear"),
    # -- w/o: remove a block entirely ---------------------------------------
    ("w/o",       "no_channel_mixer", "w/o",       "base",     "linear"),
    ("w/o",       "no_time_encoder",  "bi-Mamba", "w/o",       "linear"),
]

SMAMBA_STYLE_ABLATION = [v for _, v, _, _, _ in SMAMBA_GRID]


def resolve_chain(names: list, cfg: dict) -> list:
    """Turn '@...' placeholders into the variant that actually changes *this*
    config.

    Several switches are already set to one of their two values by a given
    dataset (PEMS ships time_encoder=mlp and mixer_placement=shared, ETT ships
    mamba/both), so a fixed variant list would silently include no-ops that
    duplicate the reference and waste a cell of the correction family. These
    placeholders always pick the direction that differs.
    """
    m = cfg["model"]

    def inherited_bool(key: str) -> bool:
        value = m.get(key)
        return bool(m.get("zero_init", True) if value is None else value)

    defaults = {
        "time_encoder": "mamba",
        "freq_encoder": "linear",
        "freq_backbone": "none",
        "freq_sparsity": 0.0,
        "freq_zero_init_head": "auto",
        "fusion": "gated",
        "channel_mixer_layers": 1,
        "mixer_placement": "both",
        "mixer_kind": "bimamba",
        "use_revin": True,
        "revin_alpha": "fixed",
        "time_linear_backbone": True,
        "zero_init": True,
        "dispersion": "none",
    }

    def effective_value(key: str):
        if key in ("time_zero_init_head", "freq_zero_init_backbone",
                   "fusion_zero_init"):
            return inherited_bool(key)
        return m.get(key, defaults.get(key))

    flip = {
        "@time_encoder_flip":
            "time_mlp" if m.get("time_encoder", "mamba") == "mamba" else "time_mamba",
        "@mixer_flip":
            "shared_mixer" if m.get("mixer_placement", "both") != "shared" else "both_mixer",
        "@fits_flip":
            "no_fits" if m.get("freq_backbone", "none") == "fits" else "with_fits",
        "@revin_flip":
            "no_revin" if m.get("use_revin", True) else "with_revin",
        "@time_head_init_flip":
            ("time_head_random_init" if inherited_bool("time_zero_init_head")
             else "time_head_zero_init"),
        "@freq_backbone_init_flip":
            ("freq_backbone_random_init" if
             inherited_bool("freq_zero_init_backbone")
             else "freq_backbone_zero_init"),
        "@fusion_init_flip":
            ("fusion_random_init" if inherited_bool("fusion_zero_init")
             else "fusion_zero_init"),
    }
    head_mode = m.get("freq_zero_init_head", "auto")
    head_is_zero = m.get("zero_init", True) and (
        head_mode == "always"
        or (head_mode == "auto" and m.get("freq_backbone", "none") == "fits")
    )
    flip["@freq_head_init_flip"] = (
        "freq_head_random_init" if head_is_zero else "freq_head_zero_init"
    )
    out = []
    for n in names:
        n = flip.get(n, n)
        # drop variants that cannot apply to this config
        if n in ("no_channel_mixer", "time_mixer_only", "freq_mixer_only", "shared_mixer",
                 "both_mixer", "vc_unimamba", "vc_attention") \
                and m.get("channel_mixer_layers", 1) == 0:
            continue
        if n in ("freq_backbone_random_init", "freq_backbone_zero_init") \
                and m.get("freq_backbone", "none") != "fits":
            continue
        # Direct variants can still be no-ops when a dataset config already
        # uses the requested value.  Never spend a run or a correction-family
        # cell on an unchanged configuration.
        if n != "full":
            changes = VARIANTS[n][1]
            if changes and all(effective_value(key) == value
                               for (section, key), value in changes.items()
                               if section == "model"):
                continue
        if n not in out:
            out.append(n)
    return out


def default_variants(cfg: dict) -> list:
    """Pick the variants that actually toggle something in this config."""
    m = cfg["model"]
    names = ["full", "time_only", "freq_only", "fusion_sum",
             "@revin_flip", "no_linear_backbone", "rand_init"]
    names.append("no_fits" if m.get("freq_backbone", "none") == "fits" else "with_fits")
    names.append("time_mlp" if m.get("time_encoder", "mamba") == "mamba" else "time_mamba")
    if m.get("channel_mixer_layers", 1) > 0:
        names.append("no_channel_mixer")
    else:
        names.append("with_channel_mixer")
    if m.get("freq_encoder", "linear") == "linear":
        names.append("freq_mamba")
    return names


def apply_variant(cfg: dict, changes: dict) -> dict:
    out = copy.deepcopy(cfg)
    for (section, key), value in changes.items():
        out[section][key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variants (default: auto-select).")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds per variant (base seed, +1, +2, ...).")
    parser.add_argument(
        "--out", default=None,
        help="output JSON path (default: checkpoint_dir/ablation_<dataset>.json)",
    )
    parser.add_argument(
        "--validation-only", action="store_true",
        help="screen variants without constructing or evaluating the test split",
    )
    parser.add_argument("--chain", default=None,
                        choices=["core", "smamba", "td", "placement", "init",
                                 "spectral", "fusion", "revin_alpha", "dispersion"],
                        help="Run a named ablation chain instead of --variants. "
                             "'smamba' mirrors the component ablation of "
                             "S-Mamba (Wang et al., Neurocomputing 2025), "
                             "resolved against this config so no variant is a "
                             "no-op.")
    args, unknown = parser.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    base_name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]
    dataset_name = Path(args.config).stem

    chains = {
        "core": CORE_ABLATION,
        "smamba": SMAMBA_STYLE_ABLATION,
        "td": TD_REPLACEMENT_CHAIN,
        "placement": MIXER_PLACEMENT_CHAIN,
        "init": INITIALIZATION_CHAIN,
        "spectral": SPECTRAL_STRATEGY_CHAIN,
        "fusion": FUSION_RULE_CHAIN,
        "revin_alpha": REVIN_ALPHA_CHAIN,
        "dispersion": DISPERSION_CHAIN,
    }
    if args.chain:
        names = resolve_chain(chains[args.chain], base_cfg)
    else:
        names = resolve_chain(args.variants or default_variants(base_cfg), base_cfg)
    unknown_names = [n for n in names if n not in VARIANTS]
    if unknown_names:
        raise SystemExit(f"Unknown variants: {unknown_names}. "
                         f"Available: {list(VARIANTS)}")
    if "full" not in names:
        names = ["full"] + names

    results, run_records = {}, []
    for name in names:
        desc, changes = VARIANTS[name]
        runs = []
        for s in range(args.seeds):
            cfg = apply_variant(base_cfg, changes)
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{base_name}_abl_{name}_s{s}"
            print(f"\n===== variant: {name} (seed {base_seed + s}) — {desc} =====")
            out = train(cfg, evaluate_test=not args.validation_only)
            metrics = (
                out["best_val_metrics"] if args.validation_only
                else out["test_metrics"]
            )
            run = {
                **metrics,
                "seed": base_seed + s,
                "horizon": cfg["data"]["pred_len"],
                "dataset": dataset_name,
                "arch": cfg["model"].get("arch", "dual_domain"),
                "variant": name,
                "phase": out["evaluation_scope"],
                "run_config_sha256": config_sha256(cfg),
                "best_val_loss": out["best_val_loss"],
                "best_val_metrics": out["best_val_metrics"],
                "best_epoch": out["best_epoch"],
                "param_count": out["param_count"],
                "active_param_count": out.get("active_param_count", out["param_count"]),
                "checkpoint": out["checkpoint"],
            }
            runs.append(run)
            run_records.append(run)
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("mse", "mae")}
        std = {
            k: (sum((r[k] - mean[k]) ** 2 for r in runs) / len(runs)) ** 0.5
            for k in ("mse", "mae")
        }
        results[name] = {
            "desc": desc,
            "mean": mean,
            "std": std,
            "param_count": runs[0]["param_count"],
            "active_param_count": runs[0]["active_param_count"],
            "runs": runs,
        }

    # Report.
    ref = results["full"]["mean"]
    lines = [
        f"\n## Ablation results — {base_name} "
        f"(pred_len={base_cfg['data']['pred_len']}, seeds={args.seeds})\n",
        "| variant | mse | mae | Δmse vs full | active params | component isolated |",
        "|---|---|---|---|---:|---|",
    ]
    for name in names:
        r = results[name]
        m, s = r["mean"], r["std"]
        dm = m["mse"] - ref["mse"]
        delta = "—" if name == "full" else f"{dm:+.4f}"
        mse_str = f"{m['mse']:.4f}" + (f" ±{s['mse']:.4f}" if args.seeds > 1 else "")
        mae_str = f"{m['mae']:.4f}" + (f" ±{s['mae']:.4f}" if args.seeds > 1 else "")
        lines.append(
            f"| {name} | {mse_str} | {mae_str} | {delta} | "
            f"{r['active_param_count']:,} | {r['desc']} |"
        )
    report = "\n".join(lines)
    print(report)

    ckpt_dir = base_cfg["experiment"]["checkpoint_dir"]
    out_path = args.out or os.path.join(ckpt_dir, f"ablation_{base_name}.json")
    record = provenance_fields(
        base_cfg, config_path=args.config,
        seed_values=[base_seed + s for s in range(args.seeds)],
        cwd=Path(__file__).resolve().parents[1],
    )
    record.update({
        "arch": base_cfg["model"].get("arch", "dual_domain"),
        "dataset": dataset_name,
        "evaluation_scope": (
            "validation_only" if args.validation_only else "validation_and_test"
        ),
        "horizons": [base_cfg["data"]["pred_len"]],
        "seeds": args.seeds,
        "variants": {
            name: {
                "description": VARIANTS[name][0],
                "changes": {f"{section}.{key}": value
                            for (section, key), value in VARIANTS[name][1].items()},
            }
            for name in names
        },
        "run_records": run_records,
        "results": results,
    })
    atomic_write_json(out_path, record)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
