"""Dispersion head: predict a positive, per-horizon future scale.

Motivation
----------
RevIN de-normalization reconstructs a forecast as ``Y = mu + sigma * Y_norm``,
where ``mu`` and ``sigma`` are the mean and standard deviation of the *input
window*, held constant across the whole horizon. That is a reversible,
window-level normalization --- it does not *forecast* how the scale will
evolve over the horizon.

The dispersion head generalizes the de-normalization to

    Y_hat = T_hat + D_hat (.) S_hat ,

where ``S_hat`` is the normalized-space forecast (the "shape"), ``T_hat`` is a
predicted per-horizon location (trend), and ``D_hat`` is a *positive*
per-horizon, per-variate scale predicted from multi-resolution historical
statistics. Unlike RevIN's single ``sigma`` per channel, ``D_hat`` varies with
the forecast step. This is inspired by the seasonal-trend-dispersion view
(STD, Dudek 2023): trend + scale * seasonal-shape --- but here the scale is
*forecast*, not extracted by a decomposition.

Design
------
``D_hat = sigma (.) r_hat``   with   ``r_hat = softplus(head(stats)) >= 0``
``T_hat = mu + t_hat``        with   ``t_hat = head_trend(stats)``

The ratio/trend heads are initialized so that ``r_hat = 1`` and ``t_hat = 0``
at start, hence at initialization the head is *exactly* RevIN
de-normalization; it then learns per-horizon multiplicative scale corrections
``r_hat`` and additive location corrections ``t_hat``. The multiplicative
parameterization keeps ``D_hat`` positive and anchored to the input scale.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# softplus(b0) = 1  ->  b0 = log(e - 1)
_UNIT_SOFTPLUS_BIAS = math.log(math.e - 1.0)


class DispersionHead(nn.Module):
    """Predict per-horizon scale ratio r_hat and trend correction t_hat.

    Parameters
    ----------
    stats_dim : int
        Number of historical-statistic features per channel (input dim).
    pred_len : int
        Forecast horizon H (output dim of each head).
    hidden : int
        Width of the shared MLP.
    learn_trend : bool
        If False, t_hat is fixed at 0 (only the scale is learned).
    """

    def __init__(self, stats_dim: int, pred_len: int, hidden: int = 64,
                 dropout: float = 0.0, learn_trend: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.learn_trend = learn_trend
        self.body = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ratio = nn.Linear(hidden, pred_len)
        # Zero the weights and set the bias so softplus(bias) = 1: r_hat starts
        # at exactly 1, i.e. D_hat starts at the input-window sigma (RevIN).
        nn.init.zeros_(self.ratio.weight)
        nn.init.constant_(self.ratio.bias, _UNIT_SOFTPLUS_BIAS)
        if learn_trend:
            self.trend = nn.Linear(hidden, pred_len)
            nn.init.zeros_(self.trend.weight)
            nn.init.zeros_(self.trend.bias)

    def forward(self, stats: torch.Tensor):
        """stats: (B, C, S) -> (r_hat, t_hat), each (B, C, H).

        r_hat > 0 (multiplicative scale ratio, ~1 at init); t_hat additive
        location correction (~0 at init).
        """
        h = self.body(stats)
        r_hat = nn.functional.softplus(self.ratio(h))            # (B, C, H) > 0
        if self.learn_trend:
            t_hat = self.trend(h)
        else:
            t_hat = torch.zeros_like(r_hat)
        return r_hat, t_hat
