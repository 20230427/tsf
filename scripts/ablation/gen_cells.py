#!/usr/bin/env python
"""Generate the cumulative-ablation cell matrix (single source: registry).

For every (dataset, variant, pred_len) cell this script resolves the
cumulative overrides against the dataset's tuned config, flags structural
no-ops (the step changes nothing vs the previous chain link), and emits the
exact ``python -m src.train`` invocation fields. The shell dispatch engine
(``scripts/ablation/ablation_common.sh``) consumes the TSV; humans use
``--dry-run`` to audit the matrix before spending GPU time.

Usage:
    python scripts/ablation/gen_cells.py --dry-run
    python scripts/ablation/gen_cells.py --dry-run --datasets ETTh1 traffic
    python scripts/ablation/gen_cells.py --format tsv            # for the engine
    python scripts/ablation/gen_cells.py --format stats          # counts only

Output formats:
    dry-run  human-readable matrix incl. per-dataset recipe keys and no-op
             reasoning (the pre-flight audit artifact)
    tsv      one line per cell, tab-separated:
             action<TAB>dataset<TAB>pred_len<TAB>variant<TAB>seed<TAB>
             run_name<TAB>config<TAB>flags   (action = run | noop; config =
             dataset yaml, or -- with --configs-dir -- the materialized
             per-(dataset, variant) resolved yaml the engine must train
             with; flags = the cumulative dotted overrides, informational
             audit trail -- the engine does NOT pass them because values
             like freq_backbone 'none' get mangled by src.train's _coerce)
    stats    "TOTAL <n> NOOP <n> RUNNABLE <n> DONE <n>" (DONE counts cells
             whose per-cell provenance JSON already exists -- resume status)
    jsonl    one JSON object per cell (tooling/debug)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ablation.cumulative_registry import (  # noqa: E402
    CUMULATIVE_STEPS,
    DEFAULT_DATASETS,
    OPTIONAL_DATASETS,
    VARIANT_NAMES,
    config_path_for,
    cumulative_flags,
    dataset_seed,
    effective_config,
    effective_value,
    is_noop,
    pred_lens_for,
    resolve_overrides,
    run_name,
)

CKPT_DIR_DEFAULT = "checkpoints/cumulative_ablation"
_RECIPE_KEYS = [
    "model.fusion", "model.freq_backbone", "model.freq_sparsity",
    "model.channel_mixer_layers", "model.time_linear_backbone",
    "model.use_revin",
]


def _cell(dataset: str, step_idx: int, pred_len: int, seed: int,
          configs_dir: Path | None = None) -> dict:
    config = config_path_for(dataset)
    noop = is_noop(config, step_idx)
    name = VARIANT_NAMES[step_idx]
    config_ref = str(config)
    if configs_dir is not None and not noop:
        # Materialize the resolved config so string values that src.train's
        # CLI _coerce would mangle (e.g. freq_backbone: "none" -> None)
        # stay properly typed; the shell then only passes run identity.
        cell_cfg = effective_config(config, step_idx)
        cfg_path = configs_dir / f"{dataset}_cum_{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            yaml.safe_dump(cell_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        config_ref = str(cfg_path)
    return {
        "action": "noop" if noop else "run",
        "dataset": dataset,
        "pred_len": pred_len,
        "variant": name,
        "variant_idx": step_idx,
        "seed": seed,
        "run_name": run_name(dataset, name, pred_len, seed),
        "config": config_ref,
        "flags": " ".join(cumulative_flags(step_idx)),
        "noop": noop,
    }


def _cells(datasets: list[str], variants: list[str], seed: int | None,
           configs_dir: Path | None = None) -> list[dict]:
    cells = []
    for dataset in datasets:
        config = config_path_for(dataset)
        base_seed = seed if seed is not None else dataset_seed(config)
        for name in variants:
            idx = VARIANT_NAMES.index(name)
            for pl in pred_lens_for(dataset):
                cells.append(_cell(dataset, idx, pl, base_seed, configs_dir))
    return cells


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _dry_run(cells: list[dict], datasets: list[str]) -> None:
    print("Cumulative step-down chain (overrides accumulate downwards):")
    for step in CUMULATIVE_STEPS:
        margin = " ".join(f"{k}={_fmt(v)}" for k, v in step["overrides"].items())
        print(f"  {step['name']:<18} {step['display']:<26} {margin or '(baseline)'}")
    print()
    for dataset in datasets:
        config = config_path_for(dataset)
        cfg = effective_config(config, 0)
        print(f"=== {dataset}  (config: {config.name}, "
              f"seed: {dataset_seed(config)}, "
              f"pred_lens: {'/'.join(map(str, pred_lens_for(dataset)))})")
        print("    recipe: " + ", ".join(
            f"{k.split('.')[-1]}={_fmt(effective_value(cfg, k))}" for k in _RECIPE_KEYS
        ))
        ds_cells = [c for c in cells if c["dataset"] == dataset]
        for name in VARIANT_NAMES:
            row = [c for c in ds_cells if c["variant"] == name]
            if not row:
                continue
            if row[0]["variant_idx"] == 0:
                print(f"    [base ] {name:<18} -> per-dataset tuned config")
            elif row[0]["noop"]:
                marginal = CUMULATIVE_STEPS[row[0]["variant_idx"]]["overrides"]
                why = ", ".join(
                    f"{k.split('.')[-1]} already {_fmt(v)}"
                    for k, v in marginal.items()
                    if _values_match(cfg, k, v)
                )
                print(f"    [noop ] {name:<18} -> skipped ({why})")
            else:
                flags = resolve_overrides(row[0]["variant_idx"])
                applied = ", ".join(f"{k.split('.')[-1]}={_fmt(v)}"
                                    for k, v in flags.items())
                print(f"    [run  ] {name:<18} -> {applied}")
        print()


def _values_match(cfg: dict, dotted_key: str, target) -> bool:
    v = effective_value(cfg, dotted_key)
    if isinstance(v, bool) != isinstance(target, bool):
        return False
    return v == target


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                     help=f"config stems (default: {' '.join(DEFAULT_DATASETS)}; "
                          f"optional extras: {' '.join(OPTIONAL_DATASETS)})")
    cli.add_argument("--variants", nargs="+", default=VARIANT_NAMES,
                     choices=VARIANT_NAMES, help="subset of chain steps")
    cli.add_argument("--seed", type=int, default=None,
                     help="force one seed for every dataset "
                          "(default: per-config experiment.seed)")
    cli.add_argument("--format", choices=["dry-run", "tsv", "jsonl", "stats"],
                     default="dry-run")
    cli.add_argument("--dry-run", action="store_true",
                     help="shorthand for --format dry-run (the default)")
    cli.add_argument("--configs-dir", default=None,
                     help="with --format tsv: materialize per-(dataset, variant) "
                          "resolved configs as YAML under this dir (relative to "
                          "the project root) and point each cell's config field "
                          "at them -- required for the shell engine, because "
                          "string values like freq_backbone 'none' cannot "
                          "survive src.train's CLI _coerce")
    cli.add_argument("--checkpoint-dir", default=CKPT_DIR_DEFAULT,
                     help="used by 'stats' to count resumable (done) cells")
    opts = cli.parse_args()

    known = set(DEFAULT_DATASETS) | set(OPTIONAL_DATASETS)
    unknown = [d for d in opts.datasets if d not in known]
    if unknown:
        cli.error(f"unknown dataset stem(s): {', '.join(unknown)}; "
                  f"known: {', '.join(sorted(known))}")
    missing = [d for d in opts.datasets if not config_path_for(d).is_file()]
    if missing:
        cli.error(f"missing config file(s): {', '.join(missing)}")

    cells = _cells(opts.datasets, opts.variants, opts.seed,
                   Path(opts.configs_dir) if opts.configs_dir else None)

    fmt = "dry-run" if opts.dry_run else opts.format
    if fmt == "dry-run":
        _dry_run(cells, opts.datasets)
    elif fmt == "tsv":
        if opts.configs_dir:
            Path(opts.configs_dir).mkdir(parents=True, exist_ok=True)
        for c in cells:
            print("\t".join(str(c[k]) for k in
                            ("action", "dataset", "pred_len", "variant", "seed",
                             "run_name", "config", "flags")))
    elif opts.format == "jsonl":
        for c in cells:
            print(json.dumps(c))
    else:  # stats
        done = sum(
            1 for c in cells
            if not c["noop"]
            and (Path(opts.checkpoint_dir) / f"{c['run_name']}_results.json").is_file()
        )
        noop = sum(1 for c in cells if c["noop"])
        print(f"TOTAL {len(cells)} NOOP {noop} "
              f"RUNNABLE {len(cells) - noop} DONE {done}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
