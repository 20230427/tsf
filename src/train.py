"""Training entry point for the dual-domain forecaster.

Usage
-----
    python -m src.train --config configs/default.yaml
    python -m src.train --epochs 5 --seq_len 192 --pred_len 96 --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_dataloaders
from .models import build_model
from .utils import (
    add_config_args,
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


def get_loss_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unknown loss: {name!r}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Per-batch schedulers. The 'halve' schedule is applied per-epoch in the
    training loop instead (see adjust_lr), so it returns None here.

    A note on 'halve'. Because it sets lr = base * 0.5^(epoch-1), the total
    learning budget is the geometric series sum(0.5^k) = 2*base -- *bounded no
    matter how many epochs are run*. epochs=10 and epochs=100 deliver the same
    budget, and only about four epochs ever run above 10% of the base rate.

    That reads like undertraining, and on electricity the training loss is
    indeed still falling at the final epoch. It was tested anyway
    (scripts/run_schedule_sweep.py) and the reading is wrong: cosine_warmup
    over 30 epochs, roughly 5x the integrated learning rate, lowered TRAIN loss
    by 16-31% while test error did not improve at any horizon (dataset average
    0.169 -> 0.171). The model can already fit the training distribution harder
    than it generalizes, so the sharp decay is acting as an implicit
    regularizer rather than starving the fit. Prefer 'halve' unless a sweep on
    your dataset says otherwise; a training loss that is still falling is by
    itself NOT evidence that a larger budget will help.
    """
    kind = cfg["train"].get("lr_scheduler", "none")
    epochs = cfg["train"]["epochs"]
    total = max(1, epochs * steps_per_epoch)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    if kind == "cosine_warmup":
        # Linear warmup then cosine decay, both per batch. Warmup matters when
        # the effective learning rate is raised: the variate mixer on
        # high-channel data is the part that destabilizes first.
        warm_epochs = float(cfg["train"].get("warmup_epochs", 1.0))
        warm = max(1, int(round(warm_epochs * steps_per_epoch)))
        floor = float(cfg["train"].get("lr_min_frac", 0.0))

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(1, total - warm)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return floor + (1.0 - floor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.5
        )
    return None


def adjust_lr(optimizer, base_lr: float, epoch: int) -> float:
    """'halve' schedule (Autoformer's type1): lr = base * 0.5^(epoch-1).

    Decaying hard from epoch 2 on is the standard recipe on small benchmarks
    like ETT: the model takes its big steps in epoch 1 and fine-tunes after,
    which suppresses the deep branches' tendency to overfit.
    """
    lr = base_lr * (0.5 ** (epoch - 1))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def add_gaussian_noise(batch_x, noise_level=0.0):
    """Add zero-mean i.i.d. Gaussian noise to a batch of input windows.

    Robustness-analysis corruption (Pamba protocol): applied to the
    standardized input window only. Targets, per-window stats and all
    training/validation inputs are never corrupted; noise_level=0 disables.
    """
    if noise_level <= 0:
        return batch_x
    noise = torch.randn_like(batch_x) * noise_level
    return batch_x + noise


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, noise_level: float = 0.0) -> dict:
    """Evaluate; ``noise_level > 0`` corrupts input windows (robustness runs).

    Only the final test evaluation ever passes a nonzero noise_level, so
    early stopping and checkpoint selection always see clean inputs.
    """
    model.eval()
    preds, trues, losses = [], [], []
    for x, y, stats in loader:
        x, y, stats = x.to(device), y.to(device), stats.to(device)
        if noise_level > 0:
            x = add_gaussian_noise(x, noise_level)
        out = model(x, stats=stats)
        losses.append(loss_fn(out, y).item())
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = all_metrics(preds, trues)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(cfg: dict, *, evaluate_test: bool = True) -> dict:
    """Train with validation checkpointing and optional final test evaluation.

    ``evaluate_test=False`` is intended for candidate search: it prevents the
    test dataset/DataLoader from being constructed and omits test metrics from
    the returned and persisted result. The default preserves the historical
    train-and-test behavior for ordinary final runs.
    """
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"[device] {describe_device(device)}")

    train_loader, val_loader, test_loader, scaler, n_channels = get_dataloaders(
        cfg, include_test=evaluate_test
    )
    test_summary = (f"test={len(test_loader.dataset)} windows"
                    if test_loader is not None else "test=disabled")
    print(
        f"[data] channels={n_channels} | "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"{test_summary}"
    )

    model = build_model(cfg, n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_n_params = (
        model.active_parameter_count()
        if hasattr(model, "active_parameter_count") else n_params
    )
    active_suffix = (
        f" | active forward-path params={active_n_params:,}"
        if active_n_params != n_params else ""
    )
    print(
        f"[model] {model.__class__.__name__} | trainable params={n_params:,}"
        f"{active_suffix}"
    )

    if cfg["train"].get("compile", False):
        try:
            model = torch.compile(model)
            print("[model] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[model] torch.compile failed ({e}); continuing eager.")

    loss_fn = get_loss_fn(cfg["train"]["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = cfg["train"].get("grad_clip", 0.0)

    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"{cfg['experiment']['name']}_best.pt")

    best_val = float("inf")
    best_val_metrics = None
    best_epoch = None
    patience = cfg["train"].get("patience", 10)
    min_delta = cfg["train"].get("min_delta", 0.0)
    bad_epochs = 0

    halve = cfg["train"].get("lr_scheduler") == "halve"
    # Log style: 'default' keeps the per-batch tqdm bar plus the repo's
    # one-line epoch summary; 'pamba' prints Pamba's plain per-epoch lines
    # and no bar (forced by the Optuna HPO objective so study logs stay
    # compact -- a tqdm bar redirected to a file emits one line per batch).
    pamba_log = cfg["train"].get("log_style", "default") == "pamba"
    train_hist, val_hist = [], []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if halve:
            adjust_lr(optimizer, cfg["train"]["lr"], epoch)
        model.train()
        running = 0.0
        n_batches = 0
        skipped = 0
        t0 = time.time()
        if pamba_log:
            lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            print(f"Epoch {epoch} : LR {lrs}")
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False,
                    disable=pamba_log)
        for x, y, stats in pbar:
            x, y, stats = x.to(device), y.to(device), stats.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x, stats=stats)
                loss = loss_fn(out, y)
            # Skip non-finite batches so a single diverged step cannot poison
            # the weights (without AMP there is no GradScaler to skip it, and
            # once the parameters are NaN no lr schedule can recover them).
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler_amp.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
            n_batches += 1
            if not pamba_log:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if skipped:
            print(f"[warn] epoch {epoch:02d}: skipped {skipped} non-finite "
                  f"training batch(es).")
        train_loss = running / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        train_hist.append(train_loss)
        val_hist.append(val_metrics["loss"])
        dt = time.time() - t0
        if pamba_log:
            # Pamba's verbatim per-epoch lines (exp_long_term_forecasting.py):
            # cost-time line + train/vali summary line.
            print("Epoch: {} cost time: {}".format(epoch, dt))
            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    epoch, len(train_loader), train_loss, val_metrics["loss"]
                )
            )
        else:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
                f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} "
                f"| lr={cur_lr:.2e} | {dt:.1f}s"
            )

        if not np.isfinite(val_metrics["loss"]):
            print(
                "[warn] validation loss is not finite. If this persists, "
                "training has diverged: try a lower lr, or set train.amp: "
                "false (the Mamba scan itself already runs in fp32)."
            )

        if val_metrics["loss"] < best_val - min_delta:
            if pamba_log:
                # Pamba's EarlyStopping verbose line on every save.
                print(
                    f"Validation loss decreased ({best_val:.7f} --> "
                    f"{val_metrics['loss']:.7f}).  Saving model ..."
                )
            best_val = val_metrics["loss"]
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "n_channels": n_channels,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if pamba_log:
                    print("Early stopping")
                else:
                    print(f"[early-stop] no val improvement for {patience} epochs.")
                break

    # Convergence diagnostic. A run whose *training* loss is still falling when
    # it stops was budget-limited, not converged -- the usual cause is a
    # schedule that drove the learning rate to zero while there was still
    # signal to fit. Distinguish that from genuine overfitting, where the
    # validation loss turned up early while training loss kept dropping.
    if len(train_hist) >= 3:
        still_falling = train_hist[-1] < train_hist[-2] < train_hist[-3]
        diagnostic_best_epoch = int(np.argmin(val_hist)) + 1
        ran = len(val_hist)
        lr_frac = optimizer.param_groups[0]["lr"] / cfg["train"]["lr"]
        if still_falling and lr_frac < 0.05 and diagnostic_best_epoch >= ran - 1:
            print(
                f"[diag] train loss was still decreasing at epoch {ran} with lr "
                f"at {lr_frac:.1%} of base: this run was UNDERTRAINED, not "
                f"converged. Consider train.lr_scheduler: cosine_warmup with "
                f"more epochs ('halve' caps the total budget at 2x base lr no "
                f"matter how many epochs are run)."
            )
        elif diagnostic_best_epoch <= max(2, ran // 3) and still_falling:
            print(
                f"[diag] best validation was epoch {diagnostic_best_epoch} of {ran} while "
                f"train loss kept falling: this run OVERFIT. Consider more "
                f"regularization (dropout/weight decay) rather than more epochs."
            )

    results = {
        "best_val_loss": best_val,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "epochs_ran": len(val_hist),
        "param_count": n_params,
        "active_param_count": active_n_params,
        "checkpoint": best_path,
        "evaluation_scope": "validation_and_test" if evaluate_test else "validation_only",
    }
    if evaluate_test:
        if test_loader is None:  # defensive: the two controls must stay coupled
            raise RuntimeError("test evaluation requested without a test DataLoader")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"])
        test_metrics = evaluate(
            model, test_loader, loss_fn, device,
            noise_level=float(cfg["train"].get("noise_level", 0.0)),
        )
        print(f"[test] mse={test_metrics['mse']:.4f} mae={test_metrics['mae']:.4f}"
              + (f" (noise_level={cfg['train'].get('noise_level', 0.0)})" if cfg["train"].get("noise_level", 0.0) else ""))
        results["test_metrics"] = test_metrics
    else:
        print("[test] disabled for validation-only candidate evaluation")

    run_record = {
        "seed": cfg["experiment"]["seed"],
        "horizon": cfg["data"]["pred_len"],
        "arch": cfg["model"].get("arch", "dual_domain"),
        "phase": results["evaluation_scope"],
        "best_val_loss": best_val,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "epochs_ran": len(val_hist),
        "param_count": n_params,
        "active_param_count": active_n_params,
        "checkpoint": best_path,
    }
    if evaluate_test:
        run_record.update(results["test_metrics"])
    persisted = provenance_fields(
        cfg, config_path=None, seed_values=[cfg["experiment"]["seed"]],
        cwd=Path(__file__).resolve().parents[1],
    )
    run_record["run_config_sha256"] = persisted["config_sha256"]
    persisted.update(results)
    persisted["run_records"] = [run_record]
    atomic_write_json(
        os.path.join(ckpt_dir, f"{cfg['experiment']['name']}_results.json"),
        persisted,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the dual-domain forecaster.")
    add_config_args(parser)
    args, unknown = parser.parse_known_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(unknown))
    train(cfg)


if __name__ == "__main__":
    main()
