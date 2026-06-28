"""
salience_theory_value.py - Bordalo-Gennaioli-Shleifer salience theory value (Tier-2).

Bordalo, Gennaioli & Shleifer (2012, QJE) "Salience Theory of Choice Under Risk", applied
to the cross-section by Cosemans & Frehen (2021, JFE) "Salience Theory and Stock Prices".
Investors overweight a stock's SALIENT payoffs (the days whose return stands out most from
the average), and they overpay for salient upside -> those stocks subsequently UNDERPERFORM.
Over a trailing window each day's return r_s gets a salience

    sigma_s = |r_s - rbar| / (|r_s| + |rbar| + theta),     theta = 0.1

is ranked (most salient -> rank 0), assigned a decaying salience weight omega_s = delta^rank
(delta = 0.7, normalized), and the salience value is the covariance of weight with return:

    ST = sum_s omega_s * r_s  -  mean_s r_s
    signal = -ST     (overweighted salient upside -> negative forward return)

The salience kernel is bounded and DOWN-WEIGHTS the loudest days, so unlike MAX/lottery it is
not a volatility proxy (Cosemans-Frehen show the premium survives MAX/idio-vol/skew controls).
Validated at the monthly horizon; emitted at 21d and a shorter 10d window (closer to the 1-day
target) plus a within-ticker z-scored variant to strip residual return-unit vol loading. The
1-day sign should be confirmed empirically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_THETA = 0.1
_DELTA = 0.7
_TSZ = 252

METADATA = {
    "name":        "salience_theory_value",
    "description": "Salience-theory value (-ST) per Bordalo-Gennaioli-Shleifer 2012 / Cosemans-Frehen 2021: salience-rank-weighted return covariance over 21d and 10d windows, plus a 252d within-ticker z-score. Bounded salience kernel down-weights loud days (not a vol proxy).",
    "requires":    ["Close"],
    "produces":    ["xdm_salience_21", "xdm_salience_10", "xdm_salience_tsz"],
    "tags":        ["behavioral", "tail", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (BGS 2012 / Cosemans-Frehen 2021)",
}


def _salience(r: np.ndarray, W: int) -> np.ndarray:
    n = r.shape[0]
    out = np.full(n, np.nan)
    if n < W:
        return out
    rw = sliding_window_view(r, W)                      # (m, W)
    rbar = np.nanmean(rw, axis=1, keepdims=True)
    sal = np.abs(rw - rbar) / (np.abs(rw) + np.abs(rbar) + _THETA)
    sal = np.nan_to_num(sal, nan=0.0)                   # warmup NaNs -> least salient
    ranks = np.argsort(np.argsort(-sal, axis=1), axis=1)  # 0 = most salient
    w = _DELTA ** (ranks + 1)
    w = w / np.nansum(w, axis=1, keepdims=True)
    st = np.nansum(w * rw, axis=1) - np.nanmean(rw, axis=1)
    sig = -st
    out[W - 1:] = sig[: n - W + 1]                      # window ending at t -> row t-W+1
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change().values
    s21 = _salience(r, 21)
    s10 = _salience(r, 10)

    df["xdm_salience_21"] = np.clip(s21, -1, 1)
    df["xdm_salience_10"] = np.clip(s10, -1, 1)

    s = pd.Series(s21, index=df.index)
    mu = s.rolling(_TSZ, min_periods=120).mean()
    sd = s.rolling(_TSZ, min_periods=120).std().replace(0, np.nan)
    df["xdm_salience_tsz"] = np.clip(((s - mu) / sd).values, -4, 4)

    return df
