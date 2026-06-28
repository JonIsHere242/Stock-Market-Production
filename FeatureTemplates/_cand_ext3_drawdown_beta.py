"""
ext3_drawdown_beta — Drawdown sensitivity to market drawdowns.

Computes rolling drawdown series for the stock and for SPY, then produces:
  - ext3_drawdown_beta_slope : rolling 120-day OLS slope of stock drawdown on
    SPY drawdown (how much the stock amplifies market drawdowns).
  - ext3_drawdown_beta_corr  : rolling 120-day Pearson correlation between the
    two drawdown series (co-movement strength).
  - ext3_drawdown_beta_level : the stock's own trailing drawdown from its
    120-day rolling high (standalone depth signal).

Per-ticker proxy note: slope and correlation are computed in pure time-series
(no cross-sectional rank), which is a faithful per-ticker proxy for the
cross-sectional downside-beta concept.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY drawdown)
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
    "name": "ext3_drawdown_beta",
    "description": (
        "Rolling 120-day drawdown sensitivity to SPY market drawdowns. "
        "Produces: (1) OLS slope of stock drawdown regressed on SPY drawdown "
        "(amplification), (2) Pearson correlation of the two drawdown series, "
        "(3) stock's own trailing drawdown from 120-day rolling high. "
        "Per-ticker time-series proxy for cross-sectional downside-beta. "
        "Uses _indexes(SPY). Source: Round-4 expansion (xdom2_downside_beta)."
    ),
    "requires": ["Close", "High"],
    "produces": [
        "ext3_drawdown_beta_slope",
        "ext3_drawdown_beta_corr",
        "ext3_drawdown_beta_level",
    ],
    "tags": ["drawdown", "beta", "downside", "market_sensitivity", "risk"],
    "version": "1.0.0",
    "author": "Round-4 expansion spec (xdom2_downside_beta / ext3_drawdown_beta)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120          # rolling window in trading days
_MIN_PERIODS = 30      # minimum observations before emitting a value


def _rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Rolling drawdown: (close - rolling_max) / rolling_max, range (-inf, 0]."""
    roll_max = close.rolling(window, min_periods=1).max()
    dd = (close - roll_max) / roll_max.replace(0, np.nan)
    return dd


def _rolling_ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y ~ x (no intercept needed for drawdown-on-drawdown).
    Returns NaN if insufficient variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < _MIN_PERIODS:
        return np.nan
    xm = x[mask]
    ym = y[mask]
    xvar = np.dot(xm, xm)
    if xvar == 0.0:
        return np.nan
    return float(np.dot(xm, ym) / xvar)


def _rolling_corr_manual(
    stock_dd: pd.Series, spy_dd: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """Pearson correlation using pandas rolling (vectorised)."""
    return stock_dd.rolling(window, min_periods=min_periods).corr(spy_dd)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Stock drawdown from rolling 120-day high                         #
    # ------------------------------------------------------------------ #
    close = df["Close"].copy()
    stock_dd = _rolling_drawdown(close, _WINDOW)

    # ------------------------------------------------------------------ #
    # 2. SPY drawdown (market)                                            #
    # ------------------------------------------------------------------ #
    spy_dd_full: pd.Series | None = None
    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close is not None and len(spy_close) > 0:
            # Ensure DatetimeIndex on both sides for merge_asof
            df_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"])})
            spy_df = spy_close.reset_index()
            spy_df.columns = ["Date", "spy_close"]
            spy_df["Date"] = pd.to_datetime(spy_df["Date"])
            spy_df = spy_df.sort_values("Date")
            merged = pd.merge_asof(
                df_dates.sort_values("Date"),
                spy_df,
                on="Date",
                direction="backward",
            )
            # Re-align to original df index
            merged = merged.set_index(df_dates.sort_values("Date").index)
            # Restore original order
            spy_aligned = merged["spy_close"].reindex(df.index)
            spy_dd_full = _rolling_drawdown(spy_aligned, _WINDOW)
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # 3. Rolling OLS slope (stock_dd ~ spy_dd)                            #
    # ------------------------------------------------------------------ #
    slope_vals = np.full(len(df), np.nan)
    corr_vals = np.full(len(df), np.nan)

    if spy_dd_full is not None:
        stock_arr = stock_dd.to_numpy(dtype=float)
        spy_arr = spy_dd_full.to_numpy(dtype=float)

        # Vectorised correlation via pandas
        s_dd_s = pd.Series(stock_arr, index=df.index)
        m_dd_s = pd.Series(spy_arr, index=df.index)
        corr_series = _rolling_corr_manual(s_dd_s, m_dd_s, _WINDOW, _MIN_PERIODS)
        corr_vals = corr_series.to_numpy(dtype=float)

        # Rolling OLS slope — iterate over windows (window is 120 rows)
        # Use numpy stride tricks for efficiency: sliding_window_view
        n = len(stock_arr)
        if n >= _WINDOW:
            from numpy.lib.stride_tricks import sliding_window_view

            s_wins = sliding_window_view(stock_arr, _WINDOW)   # shape (n-W+1, W)
            m_wins = sliding_window_view(spy_arr, _WINDOW)

            # Vectorised OLS: slope = dot(x, y) / dot(x, x) per row
            # Handle NaNs: mask them out per window
            for i in range(s_wins.shape[0]):
                sw = s_wins[i]
                mw = m_wins[i]
                mask = np.isfinite(sw) & np.isfinite(mw)
                if mask.sum() < _MIN_PERIODS:
                    continue
                xm = mw[mask]
                ym = sw[mask]
                xvar = np.dot(xm, xm)
                if xvar == 0.0:
                    continue
                slope_vals[i + _WINDOW - 1] = float(np.dot(xm, ym) / xvar)

    # ------------------------------------------------------------------ #
    # 4. Assign produced columns                                          #
    # ------------------------------------------------------------------ #
    df["ext3_drawdown_beta_slope"] = slope_vals
    df["ext3_drawdown_beta_corr"] = corr_vals
    df["ext3_drawdown_beta_level"] = stock_dd.to_numpy(dtype=float)

    return df
