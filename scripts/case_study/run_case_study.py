#!/usr/bin/env python
"""Case-study runner: train a model per horizon and dump test predictions.

For each pred_len this script trains the model (DD-Mamba by default; any
registered baseline arch via ``--arch``), reloads the best checkpoint, and
dumps the full test-set forecasts as .npy files compatible with the reference
baseline recordings (``{tag}_96_{pred_len}_pred.npy``, ``true_{pred_len}.npy``).
For the dual-domain model it additionally dumps — for the first
``--component-samples`` windows — the per-branch (time / frequency)
decomposition used by the case-study notebook.

Layout (under ``output/case_study/record_<dataset>/``)::

    DDMamba_96_96_pred.npy        full test predictions (N, H, C), train-scaled
    S_Mamba_96_96_pred.npy        ... likewise for --model-tag runs
    true_96.npy                   ground truth (written only if absent, so a
                                   baseline-recording copy wins)
    components_96/                dual-domain branch dump for the first N samples
        fused.npy time.npy freq.npy true.npy gate.npy meta.json
    case_study_manifest_<TAG>.json  provenance record (ddmamba-experiment-v1)

Usage
-----
    python scripts/case_study/run_case_study.py --config configs/ETTm1.yaml

    # Official S-Mamba baseline (authors' ETTm1 recipe, see ARCH_PRESETS):
    python scripts/case_study/run_case_study.py --config configs/ETTm1.yaml \
        --arch smamba --model-tag S_Mamba

    # Reuse existing checkpoints (skip training), just dump predictions:
    python scripts/case_study/run_case_study.py --config configs/ETTm1.yaml \
        --skip-train
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.data import get_dataloaders
from src.models import build_model
from src.train import train
from src.utils import (
    all_metrics,
    apply_overrides,
    atomic_write_json,
    describe_device,
    get_device,
    load_config,
    parse_overrides,
    provenance_fields,
    set_seed,
)

MODEL_TAG = "DDMamba"

# Per-arch case-study recipes applied on top of the dataset config (each entry
# may contain "shared" overrides and per-horizon overrides keyed by pred_len).
# Values follow the official authors' scripts for the given dataset.
ARCH_PRESETS = {
    # Official S-Mamba ETTm1 script (scripts/multivariate_forecasting/ETT/
    # S_Mamba_ETTm1.sh): e_layers=2, d_state=2, lr=5e-5; d_model=d_ff=256 at
    # pl=96 and 128 at the longer horizons. Optimizer/schedule/AMP follow this
    # repo's shared training loop.
    "smamba": {
        "shared": {
            "model.baseline_layers": 2,
            "model.baseline_d_state": 2,
            "train.lr": 5e-5,
            "train.patience": 3,
        },
        "by_horizon": {
            96: {"model.baseline_d_model": 256, "model.baseline_d_ff": 256},
            192: {"model.baseline_d_model": 128, "model.baseline_d_ff": 128},
            384: {"model.baseline_d_model": 128, "model.baseline_d_ff": 128},
        },
    },
}


def _apply_preset(cfg, arch, pred_len):
    """Deep-apply an ARCH_PRESETS entry (dotted keys) onto cfg."""
    preset = ARCH_PRESETS.get(arch)
    if not preset:
        return cfg
    flat = dict(preset.get("shared", {}))
    flat.update(preset.get("by_horizon", {}).get(pred_len, {}))
    if flat:
        print(f"[preset:{arch}] {flat}")
        # apply_overrides coerces from string form; pass values as strings
        cfg = apply_overrides(cfg, [(k, str(v)) for k, v in flat.items()])
    return cfg


@torch.no_grad()
def dump_predictions(model, loader, device, component_samples):
    """One pass over the test loader.

    Returns ``(preds, trues, components)`` where ``preds``/``trues`` cover the
    whole split (N, H, C) and ``components`` holds the time/freq branch
    forecasts, the fused output, the ground truth and the fusion gate value
    for the first ``component_samples`` windows (dict of (n, H, C) arrays or
    None, plus ``gate`` of shape (n, C) when the fusion exposes a sigmoid
    gate).
    """
    model.eval()
    fusion = getattr(model, "fusion", None)
    fusion_mode = getattr(fusion, "mode", None)
    gate_outputs = []

    def _gate_hook(_module, _inputs, output):
        gate_outputs.append(output.detach().float().cpu())

    hook = None
    if fusion_mode in ("gated", "concat") and hasattr(fusion, "gate"):
        hook = fusion.gate.register_forward_hook(_gate_hook)

    preds, trues = [], []
    comp = {"fused": [], "time": [], "freq": [], "true": []}
    n_kept = 0
    try:
        for x, y, stats in loader:
            x_dev = x.to(device)
            stats_dev = stats.to(device) if stats is not None and stats.numel() else None
            out, components = model(x_dev, stats=stats_dev, return_components=True)
            out_np = out.cpu().numpy()
            preds.append(out_np)
            trues.append(y.numpy())
            if n_kept < component_samples:
                take = min(component_samples - n_kept, out_np.shape[0])
                comp["fused"].append(out_np[:take])
                comp["time"].append(components["time"].cpu().numpy()[:take])
                comp["freq"].append(components["freq"].cpu().numpy()[:take])
                comp["true"].append(y.numpy()[:take])
                n_kept += take
    finally:
        if hook is not None:
            hook.remove()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    trues = np.concatenate(trues, axis=0).astype(np.float32)
    if comp["fused"]:
        components = {
            k: np.concatenate(v, axis=0).astype(np.float32)
            for k, v in comp.items()
        }
    else:
        components = {}
    if gate_outputs:
        gate = torch.cat(gate_outputs, dim=0)  # (N, C, 1) or (N, C, H)
        components["gate"] = gate.numpy().astype(np.float32)
    return preds, trues, components


def run_horizon(base_cfg, pred_len, args, record_dir, cli_overrides=()):
    """Train/dump one horizon.

    Override precedence: explicit CLI overrides > arch preset > dataset config.
    """
    seq_len = base_cfg["data"]["seq_len"]
    arch = args.arch or base_cfg["model"].get("arch", "dual_domain")
    dump_components = arch == "dual_domain"
    # Backwards-compatible run name: the original dual-domain checkpoints are
    # case_<dataset>_h<h>; baselines are namespaced by arch.
    run_name = (f"case_{args.dataset}_h{pred_len}" if arch == "dual_domain"
                else f"case_{args.dataset}_{arch}_h{pred_len}")
    pred_path = record_dir / f"{args.model_tag}_{seq_len}_{pred_len}_pred.npy"

    ckpt_dir = base_cfg["experiment"]["checkpoint_dir"]
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    if args.skip_existing and pred_path.exists() and os.path.exists(ckpt_path):
        print(f"[skip] {run_name}: {pred_path} already exists")
        return None

    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["pred_len"] = pred_len
    cfg["experiment"]["name"] = run_name
    cfg = _apply_preset(cfg, args.arch, pred_len)
    if cli_overrides:
        cfg = apply_overrides(cfg, cli_overrides)

    print(f"\n===== {args.model_tag} | {args.dataset} | pred_len={pred_len} =====")
    if args.skip_train:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"--skip-train requires {ckpt_path}; run without the flag first"
            )
        print(f"[train] skipped, reusing {ckpt_path}")
    else:
        results = train(cfg)
        ckpt_path = results["checkpoint"]

    set_seed(cfg["experiment"]["seed"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = state["config"]
    n_channels = state["n_channels"]

    device = get_device(args.device)
    print(f"[device] {describe_device(device)}")
    _, _, test_loader, _, loaded_channels = get_dataloaders(model_cfg)
    if loaded_channels != n_channels:
        print(f"[warn] channel mismatch: ckpt={n_channels}, data={loaded_channels}")
    model = build_model(model_cfg, n_channels).to(device)
    model.load_state_dict(state["model_state"])

    preds, trues, components = dump_predictions(
        model, test_loader, device,
        args.component_samples if dump_components else 0,
    )
    metrics = all_metrics(preds, trues)
    print(
        f"[test] mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} "
        f"(n={preds.shape[0]}, dumped shape={preds.shape})"
    )

    np.save(pred_path, preds)
    true_path = record_dir / f"true_{pred_len}.npy"
    if not true_path.exists():
        np.save(true_path, trues)
        print(f"[true] wrote {true_path}")
    else:
        existing = np.load(true_path, mmap_mode="r")
        if existing.shape != trues.shape:
            print(
                f"[true] kept existing {true_path} {existing.shape} "
                f"(ours {trues.shape}); baseline GT wins by convention"
            )

    comp_dir = None
    if components:
        comp_dir = record_dir / f"components_{pred_len}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("fused", "time", "freq", "true", "gate"):
            if key in components:
                np.save(comp_dir / f"{key}.npy", components[key])
        meta = {
            "model": args.model_tag,
            "dataset": args.dataset,
            "seq_len": int(model_cfg["data"]["seq_len"]),
            "pred_len": int(pred_len),
            "arch": model_cfg["model"].get("arch", "dual_domain"),
            "fusion": model_cfg["model"].get("fusion", "gated"),
            "seed": int(model_cfg["experiment"]["seed"]),
            "component_samples": int(components["fused"].shape[0]),
            "test_windows": int(preds.shape[0]),
            "channels": int(n_channels),
            "checkpoint": ckpt_path,
            "test_metrics": metrics,
            "space": "train-split standardized (scale: true)",
            "align": (
                "test sample i targets data row (split_border + seq_len + i); "
                "dt-mamba windows may exceed baseline recordings by 1 "
                "(drop_last=False) — plotting uses sample 0"
            ),
        }
        with open(comp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return {
        "pred_len": int(pred_len),
        "seed": int(cfg["experiment"]["seed"]),
        "horizon": int(pred_len),
        "arch": cfg["model"].get("arch", "dual_domain"),
        "run_name": run_name,
        "checkpoint": ckpt_path,
        "test_metrics": metrics,
        "n_test_windows": int(preds.shape[0]),
        "component_samples": int(components["fused"].shape[0]) if components else 0,
        "pred_file": str(pred_path),
        "components_dir": str(comp_dir) if comp_dir else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ETTm1.yaml")
    parser.add_argument("--dataset", default="ETTm1",
                        help="Dataset tag used for record dir / run names.")
    parser.add_argument("--arch", default=None,
                        help="model.arch to run (default: dataset config's, "
                             "i.e. dual-domain DD-Mamba). Arch presets from "
                             "ARCH_PRESETS are applied on top.")
    parser.add_argument("--model-tag", default=MODEL_TAG,
                        help="Filename prefix for dumps and the manifest "
                             f"(default {MODEL_TAG}; e.g. S_Mamba).")
    parser.add_argument("--pred-lens", type=int, nargs="+",
                        default=[96, 192, 384])
    parser.add_argument("--component-samples", type=int, default=64,
                        help="First N test windows for branch decomposition.")
    parser.add_argument("--record-dir", default=None,
                        help="Default: output/case_study/record_<dataset>.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse existing case checkpoints (dump only).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a horizon entirely if its dump exists.")
    args, unknown = parser.parse_known_args()
    cli_overrides = parse_overrides(unknown)

    record_dir = Path(args.record_dir) if args.record_dir else (
        Path("output") / "case_study" / f"record_{args.dataset}"
    )
    record_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    if args.arch:
        base_cfg["model"]["arch"] = args.arch
    if args.seed is not None:
        base_cfg["experiment"]["seed"] = args.seed
    seq_len = base_cfg["data"]["seq_len"]

    run_records = []
    for pred_len in args.pred_lens:
        record = run_horizon(base_cfg, pred_len, args, record_dir, cli_overrides)
        if record is not None:
            run_records.append(record)

    manifest = provenance_fields(
        base_cfg, config_path=args.config,
        seed_values=[base_cfg["experiment"]["seed"]],
        cwd=Path(__file__).resolve().parents[2],
    )
    manifest.update({
        "case_study": args.model_tag,
        "dataset": args.dataset,
        "seq_len": int(seq_len),
        "pred_lens": [int(h) for h in args.pred_lens],
        "arch": base_cfg["model"].get("arch", "dual_domain"),
        "model_tag": args.model_tag,
        "run_records": run_records,
        "record_dir": str(record_dir),
        "plotting": "output/case_study/case_study.ipynb",
    })
    manifest_path = record_dir / f"case_study_manifest_{args.model_tag}.json"
    atomic_write_json(manifest_path, manifest)
    print(f"\n[saved] {manifest_path}")
    print(f"[done] {len(run_records)}/{len(args.pred_lens)} horizons dumped to "
          f"{record_dir}/")


if __name__ == "__main__":
    main()
