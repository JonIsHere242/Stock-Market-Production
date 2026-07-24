"""
Downside-beta excess over full SPY beta (120-day trailing window).

For each bar: compute full beta vs SPY and downside beta (SPY-down days only)
over a 120-day trailing window. Feature = (downside_beta - full_beta) / (|full_beta| + eps).
This captures the PROPORTIONAL extra sensitivity a stock shows specifically on down-market days.
Positive values = stock amplifies losses more than the full-period beta suggests (bad risk).
Negative values = stock is more defensive on down days than its average beta implies.

Requires >=20 SPY-negative days in the window; otherwise NaN.
Per-ticker proxy — fully causal, no cross-sectional data needed.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path -- required by sandbox rules)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_beta_risk_downside_beta_excess_over_full_spy_120",
    "description": (
        "120-day trailing downside-beta minus full-beta, normalised by |full-beta|+eps. "
        "Measures proportional extra exposure on SPY-down days vs average. "
        "Also emits the raw downside-beta and slope of the excess metric over a 20-bar "
        "rolling window for momentum in asymmetry. Per-ticker causal proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_risk_downside_beta_excess_over_full_spy_120",
        "ff06282316b_beta_risk_downside_beta_raw_120",
        "ff06282316b_beta_risk_excess_slope_20",
    ],
    "tags": ["beta", "downside_risk", "market_sensitivity", "asymmetry"],
    "version": "1.0.0",
    "author": "feature-factory",
}

# ---------------------------------------------------------------------------
_WINDOW = 120
_MIN_DOWN_OBS = 20
_SLOPE_WIN = 20
_EPS = 1e-8


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    col_excess = "ff06282316b_beta_risk_downside_beta_excess_over_full_spy_120"
    col_down_beta = "ff06282316b_beta_risk_downside_beta_raw_120"
    col_slope = "ff06282316b_beta_risk_excess_slope_20"

    df[col_excess] = np.nan
    df[col_down_beta] = np.nan
    df[col_slope] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY close and align to ticker dates (backward merge = causal)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]

    # Ensure dates are datetime
    ticker_dates = pd.to_datetime(df["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        pd.DataFrame({"Date": ticker_dates}).sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df index order
    merged = merged.set_index(df.index if len(merged) == len(df) else merged.index)
    spy_aligned = merged["spy_close"].values  # shape (n,)

    n = len(df)
    close_vals = df["Close"].values.astype(float)

    # Daily log returns for ticker and SPY (index 0 = NaN)
    ticker_ret = np.full(n, np.nan)
    spy_ret = np.full(n, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticker_ret[1:] = np.where(
            close_vals[:-1] > 0,
            np.log(close_vals[1:] / close_vals[:-1]),
            np.nan,
        )
        spy_aligned_f = spy_aligned.astype(float)
        spy_ret[1:] = np.where(
            spy_aligned_f[:-1] > 0,
            np.log(spy_aligned_f[1:] / spy_aligned_f[:-1]),
            np.nan,
        )

    excess_arr = np.full(n, np.nan)
    down_beta_arr = np.full(n, np.nan)

    # Rolling window — compute from bar _WINDOW onward
    for i in range(_WINDOW, n):
        t_ret = ticker_ret[i - _WINDOW + 1 : i + 1]  # length _WINDOW
        s_ret = spy_ret[i - _WINDOW + 1 : i + 1]

        # Mask valid pairs
        valid = ~(np.isnan(t_ret) | np.isnan(s_ret))
        tv = t_ret[valid]
        sv = s_ret[valid]

        if len(tv) < 2:
            continue

        # Full beta: cov(t, s) / var(s)
        sv_mean = np.mean(sv)
        tv_mean = np.mean(tv)
        cov_full = np.mean((tv - tv_mean) * (sv - sv_mean))
        var_s_full = np.mean((sv - sv_mean) ** 2)
        if var_s_full < _EPS:
            continue
        full_beta = cov_full / var_s_full

        # Downside beta: restrict to SPY-negative days
        down_mask = sv < 0.0
        tv_down = tv[down_mask]
        sv_down = sv[down_mask]
        if len(tv_down) < _MIN_DOWN_OBS:
            continue

        sv_d_mean = np.mean(sv_down)
        tv_d_mean = np.mean(tv_down)
        cov_down = np.mean((tv_down - tv_d_mean) * (sv_down - sv_d_mean))
        var_s_down = np.mean((sv_down - sv_d_mean) ** 2)
        if var_s_down < _EPS:
            continue
        down_beta = cov_down / var_s_down

        # Normalised excess
        excess = (down_beta - full_beta) / (abs(full_beta) + _EPS)

        excess_arr[i] = excess
        down_beta_arr[i] = down_beta

    df[col_excess] = excess_arr
    df[col_down_beta] = down_beta_arr

    # Slope of the excess metric over last _SLOPE_WIN bars (simple linear OLS slope)
    # Use a fixed-stride approach: compute at every bar using the rolling series
    excess_series = pd.Series(excess_arr, index=df.index)
    slope_arr = np.full(n, np.nan)
    x = np.arange(_SLOPE_WIN, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    ss_x = np.dot(x_dev, x_dev)

    if ss_x > 0:
        for i in range(_SLOPE_WIN - 1 + _WINDOW, n):
            window_vals = excess_arr[i - _SLOPE_WIN + 1 : i + 1]
            if np.sum(~np.isnan(window_vals)) < _SLOPE_WIN // 2:
                continue
            # Fill NaN with window mean for slope estimation
            wv = window_vals.copy()
            wv_mean = np.nanmean(wv)
            wv[np.isnan(wv)] = wv_mean
            slope_arr[i] = np.dot(x_dev, wv) / ss_x

    df[col_slope] = slope_arr

    return df
