"""
prospect_theory_value.py - Cumulative-Prospect-Theory value of the return path (Tier-2).

Barberis, Mu & Wang (2016, JF) "Prospect Theory and Stock Returns: An Empirical Test".
An investor evaluating a stock as a standalone gamble over its recent return distribution
assigns it a CUMULATIVE-PROSPECT-THEORY value: outcomes pass through a value function with
loss aversion and diminishing sensitivity, and probabilities are reweighted by an inverse-S
function that OVERWEIGHTS the tails. Stocks with HIGH prospect value are attractive lotteries,
get overpriced, and earn LOW subsequent returns -- a clean cross-sectional sorter that is
distinct from the salience kernel (salience weights by distinctiveness; CPT weights by tail
cumulative probability + value curvature) and orthogonal to plain volatility.

Tversky-Kahneman (1992) parameters: value v(x)=x^a (x>=0), -lambda*(-x)^a (x<0) with a=0.88,
lambda=2.25; weighting w(p)=p^g/(p^g+(1-p)^g)^(1/g) with g+=0.61 (gains), g-=0.69 (losses).
With equal-probability empirical outcomes the rank-dependent decision weights are a FIXED
function of sorted position, so the whole thing is one vectorized pass per window.

  rdd_pt_value_60/20 : CPT value of the trailing return distribution (high -> overpriced)
  rdd_pt_loss_60     : the loss-side (dread) contribution alone
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_A, _LAM, _GP, _GL = 0.88, 2.25, 0.61, 0.69


def _wfn(p, g):
    pg = np.power(p, g)
    return pg / np.power(pg + np.power(1.0 - p, g), 1.0 / g)


METADATA = {
    "name":        "prospect_theory_value",
    "description": "Cumulative-Prospect-Theory value of the trailing return distribution (Tversky-Kahneman value + inverse-S tail weighting) at 60d/20d, plus the loss-side component, per Barberis-Mu-Wang 2016. High value -> attractive lottery -> overpriced.",
    "requires":    ["Close"],
    "produces":    ["rdd_pt_value_60", "rdd_pt_value_20", "rdd_pt_loss_60"],
    "tags":        ["behavioral", "tail", "lottery", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 behavioral build (Barberis-Mu-Wang 2016)",
}


def _cpt(r: np.ndarray, W: int, loss_only: bool = False) -> np.ndarray:
    n = r.shape[0]
    out = np.full(n, np.nan)
    if n < W:
        return out
    rw = sliding_window_view(r, W).copy()
    rw = np.nan_to_num(rw, nan=0.0)                       # warmup NaNs -> neutral outcome
    v = np.where(rw >= 0, np.power(np.abs(rw), _A), -_LAM * np.power(np.abs(rw), _A))
    order = np.argsort(rw, axis=1)                        # ascending by outcome
    v_s = np.take_along_axis(v, order, axis=1)
    rw_s = np.take_along_axis(rw, order, axis=1)

    pos = np.arange(W)
    cdf_bot = (pos + 1) / W                               # P(X <= x_(i))
    cdf_top = (W - pos) / W                               # P(X >= x_(i))
    wmin = _wfn(cdf_bot, _GL)
    wplus = _wfn(cdf_top, _GP)
    pi_loss = wmin - np.concatenate([[0.0], wmin[:-1]])   # cumulative weight from the worst
    pi_gain = wplus - np.concatenate([wplus[1:], [0.0]])  # cumulative weight from the best
    pi = np.where(rw_s < 0, pi_loss[None, :], pi_gain[None, :])
    if loss_only:
        contrib = np.where(rw_s < 0, pi * v_s, 0.0)
    else:
        contrib = pi * v_s
    val = contrib.sum(axis=1)
    out[W - 1:] = val[: n - W + 1]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change().values
    df["rdd_pt_value_60"] = np.clip(_cpt(r, 60), -1, 1)
    df["rdd_pt_value_20"] = np.clip(_cpt(r, 20), -1, 1)
    df["rdd_pt_loss_60"] = np.clip(_cpt(r, 60, loss_only=True), -1, 1)
    return df
