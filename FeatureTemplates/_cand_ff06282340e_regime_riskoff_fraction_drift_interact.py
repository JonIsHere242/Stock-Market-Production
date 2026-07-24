"""
ff06282340e_regime_riskoff_fraction_drift_interact
---------------------------------------------------
Fraction of the last 60 days spent in a risk-off regime (VIX in the top
tercile of its trailing-252d distribution), multiplied by the ticker's
relative drift (ticker 60d return minus SPY 60d return).

Captures relative-performance weighted by stress exposure: a stock that
outperformed during high-VIX periods gets a high positive value; one that
underperformed during stress gets a large negative value.

A slope variant (20-day change in the interaction) measures how this
stress-weighted alpha is evolving recently.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load optional index helper (VIX + SPY)
# ---------------------------------------------------------------------------
_idx = None
try:
    _s = _ilu.spec_from_file_location(
        "_indexes", _P(__file__).resolve().parent / "_indexes.py"
    )
    _idx = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_idx)
except Exception:
    _idx = None

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282340e_regime_riskoff_fraction_drift_interact",
    "description": (
        "Risk-off fraction × relative-drift interaction. "
        "Fraction of trailing 60 bars where VIX was in the top tercile of "
        "its own trailing 252-bar distribution, multiplied by (ticker 60d "
        "return − SPY 60d return). Stress-weighted alpha proxy. "
        "Slope variant = 20-bar change in the interaction."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282340e_riskoff_relret_interact",   # level: fraction × relative drift
        "ff06282340e_riskoff_fraction",           # intermediate: fraction of risk-off bars
        "ff06282340e_riskoff_relret_slope",       # 20-bar change in interaction
    ],
    "tags": ["regime", "vix", "relative-return", "stress", "interaction"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WIN_REGIME = 252    # lookback for VIX tercile boundaries
_WIN_INTERACT = 60  # window for fraction + return calc
_WIN_SLOPE = 20     # lookback for slope of interaction


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns [Date, Ticker, Open, High, Low, Close, Volume],
         one ticker, ascending by Date.

    Returns
    -------
    df with produced columns added.
    """
    # Initialise outputs to NaN on every code path
    df["ff06282340e_riskoff_relret_interact"] = np.nan
    df["ff06282340e_riskoff_fraction"] = np.nan
    df["ff06282340e_riskoff_relret_slope"] = np.nan

    n = len(df)
    if n < _WIN_INTERACT + 1:
        return df

    # ------------------------------------------------------------------
    # 1. Pull VIX and SPY series aligned to df's dates
    # ------------------------------------------------------------------
    if _idx is None:
        return df

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vix_series = _idx.index_close("VIX")   # DatetimeIndex → float
            spy_series = _idx.index_close("SPY")
    except Exception:
        return df

    if vix_series is None or vix_series.empty:
        return df
    if spy_series is None or spy_series.empty:
        return df

    # Align dates: merge_asof requires sorted DatetimeIndex
    df_dates = pd.to_datetime(df["Date"])

    vix_df = vix_series.rename("vix").reset_index()
    vix_df.columns = ["Date", "vix"]
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    spy_df = spy_series.rename("spy").reset_index()
    spy_df.columns = ["Date", "spy"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = pd.DataFrame({"Date": df_dates}).reset_index(drop=True)

    work = pd.merge_asof(
        work.sort_values("Date"),
        vix_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    work = pd.merge_asof(
        work,
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    work = work.sort_values("Date").reset_index(drop=True)

    vix_vals = work["vix"].values.astype(float)
    spy_vals = work["spy"].values.astype(float)
    close_vals = df["Close"].values.astype(float)

    nv = len(vix_vals)

    # ------------------------------------------------------------------
    # 2. Rolling VIX tercile boundary (top tercile = risk-off)
    # ------------------------------------------------------------------
    # For each bar t, the 66.67th percentile of vix over [t-252, t-1]
    # (past-only). We use pandas rolling quantile (causal).
    vix_s = pd.Series(vix_vals)
    # rolling window of _WIN_REGIME, giving the 0.6667 quantile
    vix_q67 = vix_s.rolling(_WIN_REGIME, min_periods=max(30, _WIN_REGIME // 4)).quantile(0.6667)

    # risk-off flag: 1 where vix >= vix_q67 boundary (of PREVIOUS day)
    # shift(1) ensures we only use data available at open of bar t
    vix_q67_lag = vix_q67.shift(1)
    riskoff_flag = (vix_s >= vix_q67_lag).astype(float)
    riskoff_flag[vix_q67_lag.isna()] = np.nan

    # ------------------------------------------------------------------
    # 3. Compute rolling quantities over _WIN_INTERACT = 60 bars
    # ------------------------------------------------------------------
    ro_series = pd.Series(riskoff_flag.values)
    close_series = pd.Series(close_vals)
    spy_series_aligned = pd.Series(spy_vals)

    # Rolling fraction of risk-off bars
    ro_fraction = ro_series.rolling(_WIN_INTERACT, min_periods=_WIN_INTERACT // 2).mean()

    # Ticker 60d return: close[t] / close[t-60] - 1  (causal)
    ticker_ret60 = close_series.pct_change(_WIN_INTERACT)

    # SPY 60d return
    spy_ret60 = spy_series_aligned.pct_change(_WIN_INTERACT)

    # Relative drift: ticker return − spy return
    rel_drift = ticker_ret60 - spy_ret60

    # Interaction: fraction × relative drift
    interact = ro_fraction * rel_drift

    # ------------------------------------------------------------------
    # 4. Slope: 20-bar change in interaction
    # ------------------------------------------------------------------
    interact_slope = interact.diff(_WIN_SLOPE)

    # ------------------------------------------------------------------
    # 5. Guard inf
    # ------------------------------------------------------------------
    def _clean(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 6. Write back
    # ------------------------------------------------------------------
    df["ff06282340e_riskoff_relret_interact"] = _clean(interact).values
    df["ff06282340e_riskoff_fraction"] = _clean(ro_fraction).values
    df["ff06282340e_riskoff_relret_slope"] = _clean(interact_slope).values

    return df
