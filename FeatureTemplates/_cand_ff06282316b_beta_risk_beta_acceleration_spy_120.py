"""
Feature block: ff06282316b_beta_risk_beta_acceleration_spy_120
Vein: beta_risk
Batch: 06282316b

Computes the rolling 30-day OLS beta vs SPY on a fixed-from-start i%5==0 stride
over a 120-day trailing span, forward-fills, then measures the acceleration
(second difference) of that beta series averaged over ~40 days.

Per-ticker proxy: fully causal, uses SPY as market index, no cross-sectional data.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# Load index helper
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_beta_risk_beta_acceleration_spy_120",
    "description": (
        "Rolling 30-day OLS beta vs SPY computed on a fixed-from-start i%5==0 stride "
        "(forward-filled), then beta acceleration = second difference of the beta series, "
        "averaged over the trailing ~40 days. Captures whether market exposure is "
        "accelerating or decelerating — orthogonal to level-beta features."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_level",
        "ff06282316b_beta_accel",
        "ff06282316b_beta_accel_zscore",
    ],
    "tags": ["beta", "acceleration", "market_exposure", "spy", "momentum"],
    "version": "1.0.0",
    "author": "feature-factory",
}

_EPS = 1e-10
_BETA_WINDOW = 30       # days of returns for each OLS beta estimate
_STRIDE = 5             # compute beta every 5 bars on fixed-from-start grid
_ACCEL_WINDOW = 40      # trailing bars to average the second-diff (acceleration)
_ZSCORE_WINDOW = 60     # rolling window for z-scoring the acceleration


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on all code paths
    df["ff06282316b_beta_level"] = np.nan
    df["ff06282316b_beta_accel"] = np.nan
    df["ff06282316b_beta_accel_zscore"] = np.nan

    n = len(df)
    if n < _BETA_WINDOW + 2:
        return df

    # Fetch SPY close aligned to df dates
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Merge SPY onto df dates (backward-safe: index_close returns a Date-indexed Series)
    df_dates = pd.to_datetime(df["Date"])
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_aligned = spy_close.reindex(df_dates).values  # align by date

    # If too many SPY values are missing, fall back to NaN
    spy_valid = ~np.isnan(spy_aligned)
    if spy_valid.sum() < _BETA_WINDOW + 2:
        return df

    # Compute log returns for stock and SPY
    close_vals = df["Close"].values.astype(float)
    # stock log returns (length n-1, indexed [1..n-1] -> return i is ret from bar i-1 to bar i)
    stock_ret = np.full(n, np.nan)
    stock_ret[1:] = np.log(np.where(close_vals[:-1] > 0, close_vals[1:] / np.maximum(close_vals[:-1], _EPS), np.nan))

    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = np.log(np.where(spy_aligned[:-1] > 0, spy_aligned[1:] / np.maximum(spy_aligned[:-1], _EPS), np.nan))

    # Compute OLS beta on a fixed-from-start stride (i % stride == 0 in bar-index space)
    beta_sparse = np.full(n, np.nan)

    for i in range(n):
        if i % _STRIDE != 0:
            continue
        if i < _BETA_WINDOW:
            continue
        # window: bars [i - BETA_WINDOW + 1 .. i]
        x = spy_ret[i - _BETA_WINDOW + 1: i + 1]
        y = stock_ret[i - _BETA_WINDOW + 1: i + 1]
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < _BETA_WINDOW // 2:
            continue
        xm = x[mask]
        ym = y[mask]
        xvar = np.var(xm)
        if xvar < _EPS:
            continue
        beta_sparse[i] = np.cov(xm, ym, ddof=1)[0, 1] / xvar

    # Forward-fill the sparse beta series
    beta_ff = pd.Series(beta_sparse).ffill().values

    df["ff06282316b_beta_level"] = beta_ff

    # Second difference (acceleration) of the forward-filled beta
    # first diff: d1[i] = beta_ff[i] - beta_ff[i-1]
    # second diff: d2[i] = d1[i] - d1[i-1]
    beta_series = pd.Series(beta_ff)
    d1 = beta_series.diff()
    d2 = d1.diff()

    # Rolling mean of d2 over the past ACCEL_WINDOW bars = average acceleration
    accel = d2.rolling(window=_ACCEL_WINDOW, min_periods=_ACCEL_WINDOW // 2).mean()
    df["ff06282316b_beta_accel"] = accel.values

    # Z-score the acceleration for comparability across tickers
    accel_mean = accel.rolling(window=_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW // 3).mean()
    accel_std = accel.rolling(window=_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW // 3).std()
    accel_z = (accel - accel_mean) / np.where(accel_std.values > _EPS, accel_std.values, np.nan)
    df["ff06282316b_beta_accel_zscore"] = accel_z

    return df
