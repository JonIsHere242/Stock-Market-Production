"""
Abdi-Ranaldo close-high-low spread estimator (causal, per-ticker).

Abdi & Ranaldo (2017) propose estimating the bid-ask spread from daily OHLC
data using the mid-range eta = (log(H) + log(L)) / 2 as a proxy for the
quote midpoint:

    S_t = sqrt( max(4 * (c_t - eta_t) * (c_t - eta_{t-1}), 0) )

where c_t = log(Close_t).  The original paper uses eta_{t+1} (next day's
mid-range), which would introduce lookahead.  We adapt CAUSALLY by using the
previous day's mid-range (eta_{t-1}) instead of the next day's, preserving
the economic intuition that the current close's distance from the *prior*
mid-range reveals the spread component.

Produces:
  ext3_abdi_ranaldo_spread   : instantaneous per-bar spread estimate
  ext3_abdi_ranaldo_roll21   : 21-day rolling mean spread (monthly)
  ext3_abdi_ranaldo_trend60  : 60-day linear slope of the spread (scaled by mean)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_abdi_ranaldo",
    "description": (
        "Causal Abdi-Ranaldo (2017) OHLC bid-ask spread estimator. "
        "Uses eta=(logH+logL)/2 lagged one period as mid-range proxy so "
        "no lookahead occurs. S_t = sqrt(max(4*(c_t-eta_t)*(c_t-eta_{t-1}),0)). "
        "Produces instantaneous spread, 21d rolling mean, and 60d normalised trend."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ext3_abdi_ranaldo_spread",
        "ext3_abdi_ranaldo_roll21",
        "ext3_abdi_ranaldo_trend60",
    ],
    "tags": ["microstructure", "spread", "liquidity", "ohlc"],
    "version": "1.0",
    "author": "Abdi & Ranaldo (2017) 'A new estimation of transaction costs' — causal OHLCV adaptation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add Abdi-Ranaldo spread columns to df (single ticker, ascending Date)."""

    # ------------------------------------------------------------------ #
    # 1. Log-price inputs                                                  #
    # ------------------------------------------------------------------ #
    # Guard: replace non-positive prices with NaN before log
    high_safe = df["High"].where(df["High"] > 0)
    low_safe  = df["Low"].where(df["Low"]  > 0)
    close_safe = df["Close"].where(df["Close"] > 0)

    log_h = np.log(high_safe)
    log_l = np.log(low_safe)
    log_c = np.log(close_safe)

    # Mid-range proxy for quote midpoint
    eta = (log_h + log_l) / 2.0        # shape: (n,)
    eta_lag = eta.shift(1)              # causal: previous bar's mid-range

    # ------------------------------------------------------------------ #
    # 2. Per-bar spread estimate                                           #
    # ------------------------------------------------------------------ #
    # S_t = sqrt( max(4 * (c_t - eta_t) * (c_t - eta_{t-1}), 0) )
    inner = 4.0 * (log_c - eta) * (log_c - eta_lag)
    inner_floored = inner.clip(lower=0.0)       # max(..., 0) — no negative sqrt
    spread = np.sqrt(inner_floored)             # NaN propagated automatically

    # ------------------------------------------------------------------ #
    # 3. Rolling 21-day mean spread                                        #
    # ------------------------------------------------------------------ #
    roll21 = spread.rolling(window=21, min_periods=10).mean()

    # ------------------------------------------------------------------ #
    # 4. 60-day trend of spread (slope / mean — scale-free)               #
    # ------------------------------------------------------------------ #
    # Fit a linear slope over the last 60 bars; normalise by rolling mean
    # so the output is a dimensionless "relative trend".
    window = 60
    min_p  = 30

    def _slope(arr: np.ndarray) -> float:
        """OLS slope via closed-form; returns NaN if insufficient data."""
        n = len(arr)
        mask = ~np.isnan(arr)
        valid = mask.sum()
        if valid < min_p:
            return np.nan
        x = np.where(mask)[0].astype(float)
        y = arr[mask]
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0.0:
            return np.nan
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    spread_arr = spread.to_numpy(dtype=float)
    n = len(spread_arr)
    slope_vals = np.full(n, np.nan, dtype=float)

    # Vectorise outer loop using stride tricks for the rolling window
    for i in range(window - 1, n):
        chunk = spread_arr[i - window + 1: i + 1]
        slope_vals[i] = _slope(chunk)

    slope_series = pd.Series(slope_vals, index=df.index)

    # Normalise by rolling mean to get a relative trend (handle near-zero mean)
    roll60_mean = spread.rolling(window=window, min_periods=min_p).mean()
    roll60_mean_safe = roll60_mean.where(roll60_mean.abs() > 1e-12)
    trend60 = slope_series / roll60_mean_safe

    # Replace any inf/-inf (shouldn't occur after guard, but be safe)
    trend60 = trend60.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #
    # 5. Assign to df                                                      #
    # ------------------------------------------------------------------ #
    df["ext3_abdi_ranaldo_spread"]  = spread
    df["ext3_abdi_ranaldo_roll21"]  = roll21
    df["ext3_abdi_ranaldo_trend60"] = trend60

    return df
