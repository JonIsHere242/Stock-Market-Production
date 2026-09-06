"""
Kunitomo (1992) Brownian-bridge range estimator feature block.

Strips the open-to-close drift out of the squared high-low range so only the
mean-reverting "bridge" component of intraday variance remains:

    r = ln(High/Low)          (total range)
    k = ln(Close/Open)        (drift / directional move)
    BE = max(r^2 - k^2, 0)    (bridge energy, non-directional range variance)
    sigmaK2 = BE / (4*ln2)    (Kunitomo bridge variance estimator)

LEVEL:   est30   = sqrt(mean(sigmaK2) over trailing 30 days)
DYNAMIC: purity  = mean over trailing 30 days of (BE / r^2)
                    = 1 - k^2/r^2, the fraction of the day's range that is
                    non-directional bridge volatility (Parkinson-efficiency-
                    style purity ratio).
         trend60 = OLS slope of est30 over the trailing 60 days (is bridge
                    volatility rising or falling).

Per-ticker, causal, OHLC-only. No cross-sectional or index/fundamentals
dependency needed -- this is a direct, faithful implementation of the
Kunitomo bridge estimator (not a proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703g_ohlc_range_kunitomo_bridge",
    "description": (
        "Kunitomo (1992) Brownian-bridge range estimator: strips the open-to-close "
        "drift out of the squared high-low range (r=ln(H/L), k=ln(C/O), "
        "BE=max(r^2-k^2,0), sigmaK2=BE/(4*ln2)) so only the mean-reverting bridge "
        "variance remains. Produces (1) ff0703g_ohlc_range_kunitomo_bridge_est30 = "
        "sqrt(mean(sigmaK2) over trailing 30d) -- the bridge-volatility level; "
        "(2) ff0703g_ohlc_range_kunitomo_bridge_purity = mean over 30d of BE/r^2 "
        "(=1-k^2/r^2) -- the fraction of daily range that is non-directional "
        "(bridge) rather than drift, a Parkinson-efficiency-style purity ratio; "
        "(3) ff0703g_ohlc_range_kunitomo_bridge_trend60 = trailing-60d OLS slope of "
        "est30, capturing whether bridge volatility is rising or falling. Direct "
        "per-ticker OHLC implementation, faithful to the Kunitomo (1992) estimator."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703g_ohlc_range_kunitomo_bridge_est30",
        "ff0703g_ohlc_range_kunitomo_bridge_purity",
        "ff0703g_ohlc_range_kunitomo_bridge_trend60",
    ],
    "tags": ["ohlc_range", "volatility", "range-estimator", "bridge", "kunitomo"],
    "version": "1.0.0",
    "author": "ff0703g batch codegen (direct implementation, not a proxy)",
}

_EST_WIN = 30
_TREND_WIN = 60
_FOUR_LN2 = 4.0 * np.log(2.0)


def _rolling_ols_slope(y: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling OLS slope of y against a local time index (0..window-1).

    Uses a fixed-x closed-form (x = 0..window-1 each window), vectorised via
    pandas rolling sums -- no python-level per-row loop.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return out

    s = pd.Series(y)
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = window * sum_x2 - sum_x * sum_x  # constant, > 0 for window >= 2

    # rolling sum of y
    sum_y = s.rolling(window).sum()
    # rolling sum of x*y : weight the *most recent* window sample with the
    # largest x (causal: x=0 is oldest in window, x=window-1 is current bar)
    weighted = s.rolling(window).apply(
        lambda w: np.dot(w, x) if np.all(np.isfinite(w)) else np.nan,
        raw=True,
    )

    slope = (window * weighted - sum_x * sum_y) / denom if denom != 0 else pd.Series(np.full(n, np.nan))
    return slope.to_numpy()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    col_est = "ff0703g_ohlc_range_kunitomo_bridge_est30"
    col_purity = "ff0703g_ohlc_range_kunitomo_bridge_purity"
    col_trend = "ff0703g_ohlc_range_kunitomo_bridge_trend60"

    df[col_est] = np.nan
    df[col_purity] = np.nan
    df[col_trend] = np.nan

    if n == 0:
        return df

    high = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=np.float64)
    open_ = pd.to_numeric(df["Open"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=np.float64)

    valid = (high > 0) & (low > 0) & (open_ > 0) & (close > 0) & (high > low)

    r = np.full(n, np.nan)
    k = np.full(n, np.nan)
    r[valid] = np.log(high[valid] / low[valid])
    k[valid] = np.log(close[valid] / open_[valid])

    r2 = r * r
    k2 = k * k

    be = r2 - k2
    be = np.where(np.isfinite(be), np.maximum(be, 0.0), np.nan)

    sigma_k2 = be / _FOUR_LN2

    # purity ratio, guard divide-by-zero on r2
    purity_bar = np.where(
        (r2 > 0) & np.isfinite(be) & np.isfinite(r2), be / np.where(r2 > 0, r2, np.nan), np.nan
    )

    sigma_s = pd.Series(sigma_k2)
    purity_s = pd.Series(purity_bar)

    est30_mean = sigma_s.rolling(_EST_WIN, min_periods=_EST_WIN).mean()
    est30 = np.sqrt(est30_mean.to_numpy())
    est30 = np.where(np.isfinite(est30), est30, np.nan)

    purity30 = purity_s.rolling(_EST_WIN, min_periods=_EST_WIN).mean().to_numpy()

    trend60 = _rolling_ols_slope(est30, _TREND_WIN)

    # guard inf
    est30 = np.where(np.isinf(est30), np.nan, est30)
    purity30 = np.where(np.isinf(purity30), np.nan, purity30)
    trend60 = np.where(np.isinf(trend60), np.nan, trend60)

    df[col_est] = est30
    df[col_purity] = purity30
    df[col_trend] = trend60

    return df
