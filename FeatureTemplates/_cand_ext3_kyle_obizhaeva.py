"""
Kyle-Obizhaeva market-impact invariance proxy.

Per Kyle & Obizhaeva (2016) "Market Microstructure Invariance", trading activity
is proportional to dollar-volume * return-volatility ("bet" flow W), and the
implied market-impact cost scales as vol / dollar_volume^(1/3).  This block
implements per-ticker OHLCV proxies for those quantities, together with a
60-day trend of the impact proxy to capture regime shifts in liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_kyle_obizhaeva",
    "description": (
        "Kyle-Obizhaeva market-impact invariance proxy (per-ticker OHLCV). "
        "W = rolling-21d(dollar_volume * return_vol) captures 'bet' activity; "
        "impact_proxy = return_vol / dollar_volume^(1/3) estimates implicit "
        "transaction cost; trading_activity = 1 / W^(1/3) is a liquidity "
        "intensity inverse. A 60-day OLS slope of impact_proxy captures the "
        "trend in market-impact regime. All values are per-stock proxies -- "
        "the original model is cross-sectional; this is a faithful per-ticker "
        "adaptation capturing the same economic signal (liquidity/impact)."
    ),
    "requires": ["Close", "Volume", "High", "Low"],
    "produces": [
        "ext3_kyle_obizhaeva_impact",       # vol / dolvol^(1/3), rolling 21d
        "ext3_kyle_obizhaeva_activity",     # 1 / W^(1/3), rolling 21d
        "ext3_kyle_obizhaeva_impact_trend", # 60d OLS slope of impact
    ],
    "tags": ["microstructure", "liquidity", "market_impact", "kyle_obizhaeva"],
    "version": "1.0.0",
    "author": "Kyle & Obizhaeva (2016) 'Market Microstructure Invariance'; impl. by agent",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Base series (per row)                                             #
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    n = len(close)

    # Dollar volume
    dollar_vol = close * volume  # shape (n,)

    # Daily return volatility proxy: Parkinson range-based estimator
    # sigma_t = (H - L) / (2 * sqrt(ln2) * Close)  (abs, not annualised)
    _sqrt_4ln2 = 2.0 * np.sqrt(np.log(2.0))
    hl_range = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_vol = np.where(close > 0, hl_range / (close * _sqrt_4ln2), np.nan)

    # ------------------------------------------------------------------ #
    # 2. Rolling 21-day averages                                           #
    # ------------------------------------------------------------------ #
    window = 21

    def _roll_mean(arr: np.ndarray, w: int) -> np.ndarray:
        """Causal rolling mean; NaN-safe via pandas."""
        return (
            pd.Series(arr).rolling(w, min_periods=w).mean().to_numpy(dtype=np.float64)
        )

    avg_dolvol = _roll_mean(dollar_vol, window)   # shape (n,)
    avg_retvol = _roll_mean(ret_vol, window)       # shape (n,)

    # W = dollar_volume * return_vol (Kyle-Obizhaeva "bet" activity)
    W = avg_dolvol * avg_retvol

    # ------------------------------------------------------------------ #
    # 3. Impact proxy and trading activity (invariance relationships)      #
    # ------------------------------------------------------------------ #
    with np.errstate(invalid="ignore", divide="ignore"):
        # impact_proxy ~ sigma / dolvol^(1/3)
        dolvol_cbrt = np.where(avg_dolvol > 0, avg_dolvol ** (1.0 / 3.0), np.nan)
        impact = np.where(
            np.isfinite(dolvol_cbrt) & (dolvol_cbrt > 0),
            avg_retvol / dolvol_cbrt,
            np.nan,
        )

        # trading_activity ~ 1 / W^(1/3)
        W_cbrt = np.where(W > 0, W ** (1.0 / 3.0), np.nan)
        activity = np.where(
            np.isfinite(W_cbrt) & (W_cbrt > 0),
            1.0 / W_cbrt,
            np.nan,
        )

    # Replace any inf with NaN
    impact = np.where(np.isinf(impact), np.nan, impact)
    activity = np.where(np.isinf(activity), np.nan, activity)

    # ------------------------------------------------------------------ #
    # 4. 60-day OLS slope of the impact proxy (regime trend)              #
    # ------------------------------------------------------------------ #
    trend_window = 60

    impact_s = pd.Series(impact)
    # Use a rolling OLS slope: cov(x, y) / var(x) where x = 0..w-1
    # Vectorised: slope = (w*sum(i*y) - sum(i)*sum(y)) / (w*sum(i^2) - sum(i)^2)
    # Pre-compute x weights once
    x = np.arange(trend_window, dtype=np.float64)
    sum_x = x.sum()                    # w*(w-1)/2
    sum_x2 = (x * x).sum()            # w*(w-1)*(2w-1)/6
    denom = trend_window * sum_x2 - sum_x * sum_x  # scalar

    def _ols_slope(series: pd.Series, w: int) -> np.ndarray:
        """Rolling OLS slope (causal, min_periods=w)."""
        n_s = len(series)
        slopes = np.full(n_s, np.nan)
        if denom == 0 or n_s < w:
            return slopes
        arr = series.to_numpy(dtype=np.float64)
        for t in range(w - 1, n_s):
            window_vals = arr[t - w + 1 : t + 1]
            if np.any(~np.isfinite(window_vals)):
                continue
            sum_y = window_vals.sum()
            sum_xy = (x * window_vals).sum()
            slopes[t] = (w * sum_xy - sum_x * sum_y) / denom
        return slopes

    impact_trend = _ols_slope(impact_s, trend_window)
    # Guard against any residual inf
    impact_trend = np.where(np.isinf(impact_trend), np.nan, impact_trend)

    # ------------------------------------------------------------------ #
    # 5. Assign to df                                                      #
    # ------------------------------------------------------------------ #
    df["ext3_kyle_obizhaeva_impact"] = impact
    df["ext3_kyle_obizhaeva_activity"] = activity
    df["ext3_kyle_obizhaeva_impact_trend"] = impact_trend

    return df
