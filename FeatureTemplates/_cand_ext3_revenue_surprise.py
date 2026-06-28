"""
ext3_revenue_surprise — Revenue surprise vs own trend.

Fits a rolling 504-day linear trend on PIT revenue_ttm (per-ticker),
then produces the standardised residual of the latest value ("revenue
surprise") plus a 252-day revenue growth rate.  Both signals are
forward-looking-free because _fundamentals.as_of() uses backward
merge on filed_date.

Coverage: ~84 % (ETFs / foreign tickers return NaN — expected).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (import by file path)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_revenue_surprise",
    "description": (
        "Revenue surprise vs own rolling linear trend: "
        "(1) standardised residual of revenue_ttm from its 504-day rolling OLS trend "
        "(captures unexpected revenue deviation); "
        "(2) 252-day revenue TTM growth rate. "
        "Per-ticker proxy using PIT fundamentals (filed_date safe). "
        "ETFs/foreign names yield NaN — expected."
    ),
    "requires": [],   # uses PIT fundamentals; no raw OHLCV columns needed
    "produces": [
        "ext3_revenue_surprise_resid",   # standardised residual vs 504d trend
        "ext3_revenue_surprise_growth",  # 252-day revenue TTM YoY growth
    ],
    "tags": ["fundamentals", "revenue", "surprise", "trend", "growth"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_orgcap)",
}


# ---------------------------------------------------------------------------
# Rolling linear-trend helpers (vectorised, no O(n^2) Python loops)
# ---------------------------------------------------------------------------

def _rolling_ols_resid_std(series: pd.Series, window: int) -> pd.Series:
    """
    For each position t, fit OLS(y ~ 1 + x) over the previous `window`
    observations (causal window, current bar included), then return the
    standardised residual of the LAST observation in each window.

    Uses numpy.lib.stride_tricks.sliding_window_view for speed.
    Returns pd.Series aligned to the original index.
    """
    n = len(series)
    vals = series.to_numpy(dtype=float)
    out = np.full(n, np.nan)

    if n < window:
        return pd.Series(out, index=series.index)

    from numpy.lib.stride_tricks import sliding_window_view  # numpy ≥ 1.20

    # windows shape: (n - window + 1, window)
    wins = sliding_window_view(vals, window_shape=window)  # read-only view
    W = wins.shape[0]

    # x = [0, 1, ..., window-1]
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_c = x - x_mean                     # centred x for numerical stability
    x_ss = (x_c ** 2).sum()              # sum of squares of x_c

    # For each window:  y_c = y - mean(y);  beta = sum(x_c * y_c) / x_ss
    y_mean = wins.mean(axis=1)            # (W,)
    y_c = wins - y_mean[:, None]         # (W, window)
    beta = (x_c * y_c).sum(axis=1) / x_ss  # (W,)
    alpha = y_mean - beta * x_mean       # (W,)

    # fitted value at last point (x = window-1)
    x_last = float(window - 1)
    fitted_last = alpha + beta * x_last  # (W,)
    resid_last = wins[:, -1] - fitted_last  # (W,)

    # standard deviation of residuals over the window
    fitted_all = alpha[:, None] + beta[:, None] * x[None, :]  # (W, window)
    resid_all = wins - fitted_all                              # (W, window)
    resid_std = resid_all.std(axis=1)                         # (W,)

    # standardise the last residual; guard zero std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std_resid = np.where(resid_std > 0, resid_last / resid_std, np.nan)

    out[window - 1:] = std_resid
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- pull PIT fundamentals (revenue_ttm) --------------------------------
    df = _fundamentals.as_of(df, fields=["revenue_ttm"])

    rev = df["fund_revenue_ttm"].copy()

    # ---- feature 1: standardised residual from 504-day rolling trend --------
    df["ext3_revenue_surprise_resid"] = _rolling_ols_resid_std(rev, window=504)

    # ---- feature 2: 252-day revenue TTM growth rate -------------------------
    rev_lag = rev.shift(252)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        growth = np.where(
            (rev_lag.notna()) & (rev_lag != 0),
            (rev - rev_lag) / rev_lag.abs(),
            np.nan,
        )
    df["ext3_revenue_surprise_growth"] = growth

    # ---- drop scratch fundamental columns -----------------------------------
    df.drop(columns=["fund_revenue_ttm"], inplace=True)

    return df
