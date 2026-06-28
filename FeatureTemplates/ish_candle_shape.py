"""
ish_candle_shape.py — Candle geometry: close location, body/wick fractions,
shadow asymmetry, range expansion/contraction and opening-drive persistence.

Daily OHLC carries a "shape" beyond its net return. This block summarizes that
shape over windows in ways that are largely orthogonal to close-to-close
momentum/vol:

  - CLOSE LOCATION VALUE (CLV): where the close sits inside the day's range.
        clv = ((Close-Low) - (High-Close)) / (High-Low)  in [-1, 1]
    +1 = closed on the high (buyers in control), -1 = closed on the low.
  - BODY FRACTION: |Close-Open| / (High-Low) — how decisive the session was
    (large body = trend day, small body = indecision/doji).
  - WICK FRACTIONS: upper wick = (High-max(O,C))/(H-L), lower wick = (min(O,C)-Low)/(H-L).
  - SHADOW ASYMMETRY: lower-wick minus upper-wick share — rejection of lows (+)
    vs rejection of highs (-), a classic exhaustion/absorption tell.
  - RANGE EXPANSION: today's true range vs its own 20d average (volatility-of-shape).
  - CONTRACTION STREAK: consecutive bars with range below the 20d average (coil).
  - OPENING-DRIVE PERSISTENCE: lag-1 autocorrelation of the sign of the intraday
    (Open->Close) move — does today's session direction predict tomorrow's?

Guard: bars with High == Low (zero range) produce NaN for shape ratios.
All fractions clipped to valid ranges; leading rolling NaN expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "ish_candle_shape",
    "description": (
        "Candle geometry over windows: close-location value, body/upper/lower wick "
        "fractions, shadow asymmetry, range expansion vs 20d, contraction streak and "
        "opening-drive sign autocorrelation."
    ),
    "requires":    ["Open", "High", "Low", "Close"],
    "produces":    [
        "ish_clv",
        "ish_clv_mean_20",
        "ish_clv_trend_60",
        "ish_body_frac_mean_20",
        "ish_upper_wick_mean_20",
        "ish_lower_wick_mean_20",
        "ish_shadow_asym_20",
        "ish_range_expansion_20",
        "ish_contraction_streak",
        "ish_open_drive_autocorr_60",
    ],
    "tags":    ["candle", "intraday_shape", "range", "mean_reversion"],
    "version": "1.0",
    "author": "feature-gen",
}

_EPS = 1e-9


def compute(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["Open"].astype("float64")
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    close = df["Close"].astype("float64")

    # day range; guard zero-range bars (High == Low) -> NaN, never divide by 0
    rng = (high - low)
    rng = rng.where(rng > _EPS, np.nan)

    # --- close location value in [-1, 1] -----------------------------------
    clv = ((close - low) - (high - close)) / rng
    clv = clv.replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    df["ish_clv"] = clv.values

    clv_mean_20 = clv.rolling(20, min_periods=10).mean()
    df["ish_clv_mean_20"] = clv_mean_20.values

    # CLV trend: are closes drifting toward the high (accumulation) over 60d?
    # slope proxy = recent 20d mean minus prior 40d mean of CLV.
    clv_mean_60 = clv.rolling(60, min_periods=30).mean()
    clv_trend = clv_mean_20 - clv_mean_60
    df["ish_clv_trend_60"] = clv_trend.replace([np.inf, -np.inf], np.nan).clip(-2, 2).values

    # --- body and wick fractions (each in [0, 1]) --------------------------
    body = (close - open_).abs() / rng
    upper_wick = (high - np.maximum(open_, close)) / rng
    lower_wick = (np.minimum(open_, close) - low) / rng

    body = body.replace([np.inf, -np.inf], np.nan).clip(0, 1)
    upper_wick = upper_wick.replace([np.inf, -np.inf], np.nan).clip(0, 1)
    lower_wick = lower_wick.replace([np.inf, -np.inf], np.nan).clip(0, 1)

    df["ish_body_frac_mean_20"] = body.rolling(20, min_periods=10).mean().values
    uw_mean_20 = upper_wick.rolling(20, min_periods=10).mean()
    lw_mean_20 = lower_wick.rolling(20, min_periods=10).mean()
    df["ish_upper_wick_mean_20"] = uw_mean_20.values
    df["ish_lower_wick_mean_20"] = lw_mean_20.values

    # shadow asymmetry: +ve => persistently rejecting lows (long lower shadows)
    shadow_asym = (lw_mean_20 - uw_mean_20)
    df["ish_shadow_asym_20"] = shadow_asym.replace([np.inf, -np.inf], np.nan).clip(-1, 1).values

    # --- range expansion vs own 20d average true range ---------------------
    prev_close = close.shift(1)
    true_range = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    avg_tr_20 = true_range.rolling(20, min_periods=10).mean()
    range_exp = true_range / (avg_tr_20 + _EPS)
    df["ish_range_expansion_20"] = range_exp.replace([np.inf, -np.inf], np.nan).clip(0, 10).values

    # --- contraction streak: consecutive bars with TR below its 20d avg ----
    below = (true_range < avg_tr_20)
    # running count of consecutive True, reset at each False
    grp = (~below).cumsum()
    streak = below.groupby(grp).cumsum()
    # only valid once the 20d average exists
    streak = streak.where(avg_tr_20.notna())
    df["ish_contraction_streak"] = streak.clip(0, 60).astype("float64").values

    # --- opening-drive persistence: lag-1 autocorr of intraday move sign ---
    intraday_sign = np.sign((close / open_.replace(0, np.nan) - 1.0))
    s = pd.Series(intraday_sign.values, index=df.index, dtype="float64")
    s_lag = s.shift(1)
    # rolling correlation of sign_t with sign_{t-1} over 60 bars
    autocorr = s.rolling(60, min_periods=30).corr(s_lag)
    df["ish_open_drive_autocorr_60"] = autocorr.replace([np.inf, -np.inf], np.nan).clip(-1, 1).values

    return df
