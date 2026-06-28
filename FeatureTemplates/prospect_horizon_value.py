"""
prospect_horizon_value.py - Cumulative-Prospect-Theory value at multi-day horizons (Tier-2).

The 1-day CPT value (prospect_theory_value) was a top screen result. But investors evaluate a
stock as a gamble over their HOLDING horizon, not a single day. This computes the same
cumulative-prospect-theory value over OVERLAPPING multi-day (5d, 10d) return distributions:
the longer-horizon lottery appeal. A stock that looks like an attractive 1-2 week gamble is
overpriced and underperforms (Barberis-Mu-Wang logic extended in horizon). Distinct from the
1-day CPT block (different return horizon -> different skew/tail shape entering the valuation).

  phv_pt_5d  : CPT value of the 5-day overlapping-return distribution (W=60)
  phv_pt_10d : CPT value of the 10-day overlapping-return distribution (W=60)

Tversky-Kahneman value v(x)=x^a / -lambda*(-x)^a (a=0.88, lambda=2.25) with inverse-S
probability weighting (g+=0.61, g-=0.69). Vectorised; equal-probability empirical outcomes give
fixed rank-dependent decision weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_A, _LAM, _GP, _GL = 0.88, 2.25, 0.61, 0.69
_W = 60


def _wfn(p, g):
    pg = np.power(p, g)
    return pg / np.power(pg + np.power(1.0 - p, g), 1.0 / g)


METADATA = {
    "name":        "prospect_horizon_value",
    "description": "Cumulative-Prospect-Theory value of the trailing 5-day and 10-day overlapping-return distributions (Tversky-Kahneman value + inverse-S tail weighting). Multi-horizon extension of the screen-winning 1-day CPT value; high value -> overpriced lottery.",
    "requires":    ["Close"],
    "produces":    ["phv_pt_5d", "phv_pt_10d", "phv_pt_loss_5d"],
    "tags":        ["behavioral", "tail", "lottery", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 behavioral build (multi-horizon CPT, Barberis-Mu-Wang)",
}


def _cpt(r: np.ndarray, W: int, loss_only: bool = False) -> np.ndarray:
    n = r.shape[0]
    out = np.full(n, np.nan)
    if n < W:
        return out
    rw = np.nan_to_num(sliding_window_view(r, W).copy(), nan=0.0)
    v = np.where(rw >= 0, np.power(np.abs(rw), _A), -_LAM * np.power(np.abs(rw), _A))
    order = np.argsort(rw, axis=1)
    v_s = np.take_along_axis(v, order, axis=1)
    rw_s = np.take_along_axis(rw, order, axis=1)
    pos = np.arange(W)
    wmin = _wfn((pos + 1) / W, _GL)
    wplus = _wfn((W - pos) / W, _GP)
    pi_loss = wmin - np.concatenate([[0.0], wmin[:-1]])
    pi_gain = wplus - np.concatenate([wplus[1:], [0.0]])
    pi = np.where(rw_s < 0, pi_loss[None, :], pi_gain[None, :])
    contrib = np.where(rw_s < 0, pi * v_s, 0.0) if loss_only else pi * v_s
    val = contrib.sum(axis=1)
    out[W - 1:] = val[: n - W + 1]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    r5 = (close / close.shift(5) - 1.0).values
    r10 = (close / close.shift(10) - 1.0).values
    df["phv_pt_5d"] = np.clip(_cpt(r5, _W), -2, 2)
    df["phv_pt_10d"] = np.clip(_cpt(r10, _W), -2, 2)
    df["phv_pt_loss_5d"] = np.clip(_cpt(r5, _W, loss_only=True), -2, 2)
    return df
