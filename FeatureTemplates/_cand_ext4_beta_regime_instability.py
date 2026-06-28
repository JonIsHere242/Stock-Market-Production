"""
ext4_beta_regime_instability — Beta instability conditioned on market regime.

Extends beta_instability: computes the 120-day rolling std of a 40-day SPY-beta
separately within up-market and down-market regimes (SPY ≥ 0 vs SPY < 0 day),
then aggregates the regime-conditional std over a trailing 120-day window.

Produces:
  ext4_beta_regime_instability_down  — 120d std of rolling 40d beta on down-market days
  ext4_beta_regime_instability_up    — 120d std of rolling 40d beta on up-market days
  ext4_beta_regime_instability_diff  — down_instability minus up_instability

Per-ticker OHLCV proxy; uses _indexes.index_close('SPY') for market returns.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the _indexes helper by file path (mandatory pattern)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_beta_regime_instability",
    "description": (
        "Beta instability conditioned on market regime. "
        "Computes rolling 40-day SPY-beta at each date, then over a 120-day "
        "trailing window estimates the standard deviation of those beta values "
        "separately on days when SPY had a non-negative return (up-regime) and "
        "days when SPY had a negative return (down-regime). "
        "ext4_beta_regime_instability_down: how erratic loading is in stress. "
        "ext4_beta_regime_instability_up: erratic loading in calm markets. "
        "ext4_beta_regime_instability_diff: down minus up (does stress amplify "
        "beta instability?). Per-ticker proxy faithful to the cross-sectional "
        "concept; requires at least 40 stock and SPY observations."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_beta_regime_instability_down",
        "ext4_beta_regime_instability_up",
        "ext4_beta_regime_instability_diff",
    ],
    "tags": ["beta", "regime", "instability", "market", "risk", "stress"],
    "version": "1.0.0",
    "author": "Round-5 expansion (ext3_beta_instability)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rolling_beta_40(stock_ret: np.ndarray, mkt_ret: np.ndarray, win: int = 40) -> np.ndarray:
    """
    For each index t, compute OLS beta of stock_ret[t-win+1:t+1] on mkt_ret[t-win+1:t+1].
    Returns an array of length n; first (win-1) values are NaN.
    Uses the formula: beta = cov(r_s, r_m) / var(r_m), computed via sliding sums.
    """
    n = len(stock_ret)
    beta = np.full(n, np.nan)

    if n < win:
        return beta

    # Use sliding_window_view for efficiency
    from numpy.lib.stride_tricks import sliding_window_view

    sw_s = sliding_window_view(stock_ret, win)   # shape (n-win+1, win)
    sw_m = sliding_window_view(mkt_ret, win)     # shape (n-win+1, win)

    mean_s = sw_s.mean(axis=1)
    mean_m = sw_m.mean(axis=1)

    # cov = mean of products minus product of means
    cov_sm = (sw_s * sw_m).mean(axis=1) - mean_s * mean_m
    var_m = sw_m.var(axis=1)               # denominator

    # Guard divide-by-zero
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(var_m > 0, cov_sm / var_m, np.nan)

    # Place results starting at index win-1
    beta[win - 1:] = b
    return beta


def _regime_rolling_std(beta_arr: np.ndarray, regime_mask: np.ndarray,
                         win: int = 120) -> np.ndarray:
    """
    For each index t, compute std of beta values within the past `win` rows where
    regime_mask is True.  Returns NaN if fewer than 5 valid observations.
    """
    n = len(beta_arr)
    result = np.full(n, np.nan)

    for t in range(win - 1, n):
        window_beta = beta_arr[t - win + 1: t + 1]
        window_mask = regime_mask[t - win + 1: t + 1]
        vals = window_beta[window_mask & np.isfinite(window_beta)]
        if len(vals) >= 5:
            result[t] = vals.std(ddof=1) if len(vals) > 1 else 0.0

    return result


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    ROLL_BETA = 40
    ROLL_STD = 120
    COL_DOWN = "ext4_beta_regime_instability_down"
    COL_UP = "ext4_beta_regime_instability_up"
    COL_DIFF = "ext4_beta_regime_instability_diff"

    n = len(df)

    # Default outputs to NaN
    df[COL_DOWN] = np.nan
    df[COL_UP] = np.nan
    df[COL_DIFF] = np.nan

    if n < ROLL_BETA + 1:
        return df

    # ----- 1. Load SPY close, align to df --------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("SPY_Close").reset_index().rename(columns={"index": "Date", "Date": "Date"})
    # Ensure Date column name is correct regardless of reset_index() label
    if "Date" not in spy_df.columns:
        spy_df.columns = ["Date", "SPY_Close"]

    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    work = work.set_index(work.index)  # keep positional after sort
    spy_aligned = work["SPY_Close"].values  # same length as df after sort; need original order

    # Reconstruct in original df index order
    orig_order = df.index
    work2 = df[["Date"]].copy()
    work2["Date"] = pd.to_datetime(work2["Date"])
    work2 = pd.merge_asof(
        work2.reset_index().sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    ).sort_values("index").set_index("index")
    spy_aligned = work2["SPY_Close"].values

    # ----- 2. Compute daily returns (stock and SPY) -----------------------
    close_arr = df["Close"].values.astype(float)

    # Stock returns
    stock_ret = np.full(n, np.nan)
    stock_ret[1:] = np.where(
        close_arr[:-1] > 0,
        (close_arr[1:] - close_arr[:-1]) / close_arr[:-1],
        np.nan,
    )

    # SPY returns
    spy_arr = spy_aligned.astype(float)
    mkt_ret = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt_ret[1:] = np.where(
            spy_arr[:-1] > 0,
            (spy_arr[1:] - spy_arr[:-1]) / spy_arr[:-1],
            np.nan,
        )

    # ----- 3. Rolling 40-day beta series ----------------------------------
    # Treat NaN returns as 0 for beta window computation (sparse NaN okay)
    sr = np.where(np.isfinite(stock_ret), stock_ret, np.nan)
    mr = np.where(np.isfinite(mkt_ret), mkt_ret, np.nan)

    beta_series = _rolling_beta_40(sr, mr, win=ROLL_BETA)

    # ----- 4. Regime mask (based on SPY return on that day) ---------------
    # down-market day: mkt_ret < 0
    # up-market day  : mkt_ret >= 0
    down_mask = np.where(np.isfinite(mkt_ret), mkt_ret < 0, False)
    up_mask = np.where(np.isfinite(mkt_ret), mkt_ret >= 0, False)

    # ----- 5. Regime-conditional rolling std of beta ----------------------
    down_std = _regime_rolling_std(beta_series, down_mask, win=ROLL_STD)
    up_std = _regime_rolling_std(beta_series, up_mask, win=ROLL_STD)

    with np.errstate(invalid="ignore"):
        diff = np.where(
            np.isfinite(down_std) & np.isfinite(up_std),
            down_std - up_std,
            np.nan,
        )

    df[COL_DOWN] = down_std
    df[COL_UP] = up_std
    df[COL_DIFF] = diff

    return df
