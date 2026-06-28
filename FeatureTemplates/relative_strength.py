"""
relative_strength.py — Relative-strength line, its slope, excess return, and
beta-adjusted residual drift of the stock versus SPY (the broad market).

These capture WHETHER a stock is out/under-performing the market and at what
PACE — orthogonal to plain price momentum because the market component is
removed. All joins are inner-join on Date (price/return known at the close of
the same session), so they are lookahead-safe.

Conventions (mirroring beta_metrics.py):
  - log returns
  - inner-join alignment of stock vs SPY on shared trading dates
  - reindex back to the original df row order via .values (df order untouched)
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared index helper (auto-skipped by the framework; imported by file path)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "relative_strength",
    "description": (
        "Relative-strength line vs SPY, its 20/60d slope, 5/20/60d excess return "
        "over SPY, and 60d cumulative beta-adjusted residual drift."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "xas_rs_line_spy",
        "xas_rs_slope_20",
        "xas_rs_slope_60",
        "xas_excess_ret_5",
        "xas_excess_ret_20",
        "xas_excess_ret_60",
        "xas_resid_drift_60",
    ],
    "tags":    ["market_regime", "relative_strength", "cross_asset"],
    "version": "1.0",
    "author": "feature-gen",
}

_BETA_WINDOW = 60
_BETA_MIN    = 30


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """OLS slope of s against a 0..window-1 time index, per rolling window.

    slope = cov(t, y) / var(t).  Constant denominator => cheap closed form.
    """
    t = np.arange(window, dtype="float64")
    t_mean = t.mean()
    t_dev = t - t_mean
    denom = float((t_dev * t_dev).sum())
    if denom == 0:
        return pd.Series(np.nan, index=s.index)

    def _slope(arr: np.ndarray) -> float:
        # arr has length == window (raw=True ndarray)
        y_dev = arr - arr.mean()
        return float((t_dev * y_dev).sum() / denom)

    return s.rolling(window, min_periods=window).apply(_slope, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-create columns as NaN so the produces contract always holds.
    for col in METADATA["produces"]:
        df[col] = np.nan

    df_dates = pd.to_datetime(df["Date"]).values

    stock_close = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    stock_ret = np.log(stock_close / stock_close.shift(1))

    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close.empty:
            return df
        spy_ret = np.log(spy_close / spy_close.shift(1))
    except Exception:
        return df

    # Inner-join on shared trading dates.
    s_ret, m_ret = stock_ret.align(spy_ret, join="inner")
    if len(s_ret) < _BETA_MIN:
        return df

    # --- Relative-strength line: cumulative (stock - market) log return -------
    excess_daily = (s_ret - m_ret).fillna(0.0)
    rs_line = excess_daily.cumsum()
    rs_line[s_ret.isna() | m_ret.isna()] = np.nan  # keep leading NaN honest

    # Slope of the RS line over 20 / 60 days (pace of out/under-performance).
    rs_slope_20 = _rolling_slope(rs_line, 20)
    rs_slope_60 = _rolling_slope(rs_line, 60)

    # --- Excess return over SPY over fixed horizons ---------------------------
    # Sum of daily log-return differences == log(stock cum) - log(market cum).
    def _excess(window: int) -> pd.Series:
        return (s_ret - m_ret).rolling(window, min_periods=window).sum()

    excess_5  = _excess(5)
    excess_20 = _excess(20)
    excess_60 = _excess(60)

    # --- Beta-adjusted residual drift -----------------------------------------
    # residual_t = stock_ret_t - beta_t * market_ret_t ; cumulative over 60d.
    rolling_cov = s_ret.rolling(_BETA_WINDOW, min_periods=_BETA_MIN).cov(m_ret)
    rolling_var = m_ret.rolling(_BETA_WINDOW, min_periods=_BETA_MIN).var()
    beta = (rolling_cov / rolling_var).clip(-5, 5)
    resid = s_ret - beta * m_ret
    resid_drift_60 = resid.rolling(_BETA_WINDOW, min_periods=_BETA_MIN).sum()

    # --- Assign back in df row order -----------------------------------------
    df["xas_rs_line_spy"]    = rs_line.reindex(df_dates).values
    df["xas_rs_slope_20"]    = rs_slope_20.reindex(df_dates).values
    df["xas_rs_slope_60"]    = rs_slope_60.reindex(df_dates).values
    df["xas_excess_ret_5"]   = excess_5.reindex(df_dates).values
    df["xas_excess_ret_20"]  = excess_20.reindex(df_dates).values
    df["xas_excess_ret_60"]  = excess_60.reindex(df_dates).values
    df["xas_resid_drift_60"] = resid_drift_60.reindex(df_dates).values

    # Clip pathological values.
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).clip(-50, 50)

    return df
