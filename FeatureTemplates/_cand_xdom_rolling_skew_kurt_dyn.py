"""
Tail-asymmetry dynamics: rolling skewness and kurtosis SLOPE (change, not level).

Instead of emitting the level of rolling skew/kurt (common in many blocks), this
block emits the 20-day CHANGE in the 60-day rolling skewness and kurtosis of
daily log-returns.  The rate of change in tail shape captures whether the return
distribution is becoming more or less skewed / fat-tailed, which is a leading
regime-change signal orthogonal to the levels themselves.

Per-ticker proxy: computed entirely from this stock's own daily log-returns.
Cross-sectional ranking is performed at the portfolio level during feature
assembly and is not required here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_rolling_skew_kurt_dyn",
    "description": (
        "Tail-asymmetry dynamics: 20-day change in 60-day rolling skewness "
        "(xdom_rolling_skew_kurt_dyn_skew_slope) and 20-day change in 60-day "
        "rolling excess kurtosis (xdom_rolling_skew_kurt_dyn_kurt_slope) of "
        "daily log-returns.  Captures the DIRECTION in which the return "
        "distribution's tails are evolving -- a leading regime signal. "
        "Per-ticker proxy; inherently single-stock, no cross-sectional component."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_rolling_skew_kurt_dyn_skew_lvl",
        "xdom_rolling_skew_kurt_dyn_skew_slope",
        "xdom_rolling_skew_kurt_dyn_kurt_slope",
    ],
    "tags": ["cross-domain", "econophysics", "tail", "skew", "kurtosis", "regime"],
    "version": "1.0",
    "author": (
        "Spec: Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "Tail-asymmetry dynamics: skew & kurtosis SLOPE (not level)"
    ),
}

_WINDOW = 60   # rolling window for moment estimation
_SLOPE_LAG = 20  # look-back for the slope (change over this many days)
_MIN_OBS = 30  # minimum non-NaN observations inside the rolling window


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- log returns (no lookahead: uses only current and past Close) ---
    log_ret = np.log(
        df["Close"].replace(0, np.nan) /
        df["Close"].replace(0, np.nan).shift(1)
    )

    # --- rolling skewness and kurtosis (excess) ---
    roll_skew = log_ret.rolling(window=_WINDOW, min_periods=_MIN_OBS).skew()
    roll_kurt = log_ret.rolling(window=_WINDOW, min_periods=_MIN_OBS).kurt()

    # --- 20-day slope = current level minus the level _SLOPE_LAG days ago ---
    skew_slope = roll_skew - roll_skew.shift(_SLOPE_LAG)
    kurt_slope = roll_kurt - roll_kurt.shift(_SLOPE_LAG)

    # --- guard against inf (can arise if variance collapses to 0) ---
    def _clean(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan)

    df["xdom_rolling_skew_kurt_dyn_skew_lvl"]   = _clean(roll_skew)
    df["xdom_rolling_skew_kurt_dyn_skew_slope"] = _clean(skew_slope)
    df["xdom_rolling_skew_kurt_dyn_kurt_slope"] = _clean(kurt_slope)

    return df
