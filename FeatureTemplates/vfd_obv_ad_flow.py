"""
vfd_obv_ad_flow.py — Accumulation / distribution money-flow DYNAMICS.

Theme: the *slope* and *divergence* of cumulative money-flow lines, not their
raw levels. The model already has raw `obv` (volume_flow_metrics) and VWAP
distance — here we extract the directional information those lines carry that
plain price momentum does NOT:

  - OBV 20d slope MINUS price 20d slope  -> classic OBV/price divergence
    (accumulation while price flat = bullish; distribution while price up =
    bearish). Slopes are normalised so the difference is scale-free.
  - Accumulation/Distribution (A/D) line slope — uses the close-location-value
    so it reacts to *where in the bar* the close sits, orthogonal to OBV which
    only looks at close-to-close sign.
  - Chaikin Money Flow (CMF) 20d/60d — volume-weighted CLV, a bounded
    [-1, 1] accumulation pressure gauge.
  - Chaikin Oscillator — MACD(3,10) of the A/D line, a momentum-of-flow signal.

All within-ticker, lookahead-safe (only uses bars up to and including t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "vfd_obv_ad_flow",
    "description": (
        "Accumulation/distribution money-flow dynamics: OBV-vs-price slope "
        "divergence, A/D line slope, Chaikin Money Flow (20/60d), and the "
        "Chaikin oscillator."
    ),
    "requires":    ["High", "Low", "Close", "Volume"],
    "produces": [
        "vfd_obv_slope_20",
        "vfd_price_slope_20",
        "vfd_obv_price_divergence_20",
        "vfd_ad_slope_20",
        "vfd_cmf_20",
        "vfd_cmf_60",
        "vfd_chaikin_osc",
    ],
    "tags":    ["volume", "flow", "money_flow", "divergence"],
    "version": "1.0",
    "author":  "feature-gen",
}


def _rolling_slope(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """OLS slope of s against an integer time index, per rolling window.

    Closed-form cov(x,y)/var(x). The window array length varies during the
    warm-up region (min_periods <= len < window), so x is rebuilt from the
    actual array length on every call.
    """

    def _slope(arr: np.ndarray) -> float:
        n = arr.size
        if n < 2:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        x_dev = x - x_mean
        x_var = np.dot(x_dev, x_dev)
        if x_var == 0:
            return np.nan
        return float(np.dot(x_dev, arr) / x_var)

    return s.rolling(window, min_periods=min_periods).apply(_slope, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # ---- On-Balance Volume (computed locally; never overwrite shared `obv`) --
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * volume).cumsum()

    # Normalise OBV by its own rolling scale so the slope is comparable to the
    # price slope (both become "per-day fractional drift").
    obv_scale = obv.rolling(20, min_periods=10).std().replace(0, np.nan)
    obv_norm = obv / obv_scale
    df["vfd_obv_slope_20"] = _rolling_slope(obv_norm, 20, 10).clip(-50, 50)

    # Price slope measured on log-price so it is a fractional drift too.
    log_price = np.log(close.replace(0, np.nan))
    df["vfd_price_slope_20"] = _rolling_slope(log_price, 20, 10).clip(-1, 1)

    # Divergence: flow trend minus price trend. Standardise each leg by its own
    # 60d dispersion before subtracting so neither dominates by raw units.
    obv_sl = df["vfd_obv_slope_20"]
    px_sl = df["vfd_price_slope_20"]
    obv_z = (obv_sl - obv_sl.rolling(60, min_periods=20).mean()) / (
        obv_sl.rolling(60, min_periods=20).std().replace(0, np.nan)
    )
    px_z = (px_sl - px_sl.rolling(60, min_periods=20).mean()) / (
        px_sl.rolling(60, min_periods=20).std().replace(0, np.nan)
    )
    df["vfd_obv_price_divergence_20"] = (obv_z - px_z).clip(-10, 10)

    # ---- Accumulation / Distribution line ----------------------------------
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng  # close-location value [-1,1]
    clv = clv.fillna(0.0)
    mf_vol = clv * volume
    ad_line = mf_vol.cumsum()

    ad_scale = ad_line.rolling(20, min_periods=10).std().replace(0, np.nan)
    df["vfd_ad_slope_20"] = _rolling_slope(ad_line / ad_scale, 20, 10).clip(-50, 50)

    # ---- Chaikin Money Flow (bounded accumulation pressure) ----------------
    for w in (20, 60):
        mfv_sum = mf_vol.rolling(w, min_periods=max(5, w // 2)).sum()
        vol_sum = volume.rolling(w, min_periods=max(5, w // 2)).sum().replace(0, np.nan)
        df[f"vfd_cmf_{w}"] = (mfv_sum / vol_sum).clip(-1, 1)

    # ---- Chaikin oscillator = EMA3(A/D) - EMA10(A/D) -----------------------
    ema_fast = ad_line.ewm(span=3, min_periods=3, adjust=False).mean()
    ema_slow = ad_line.ewm(span=10, min_periods=10, adjust=False).mean()
    chaikin = ema_fast - ema_slow
    # Scale by trailing dollar-volume so it is comparable across tickers.
    dv_scale = (close * volume).rolling(20, min_periods=10).mean().replace(0, np.nan)
    df["vfd_chaikin_osc"] = (chaikin / dv_scale).clip(-50, 50)

    return df
