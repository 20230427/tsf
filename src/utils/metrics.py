"""Forecasting metrics computed on NumPy arrays of shape (N, H, C).

The project's evaluation metrics are MSE and MAE.
"""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def all_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "mse": mse(pred, true),
        "mae": mae(pred, true),
    }
