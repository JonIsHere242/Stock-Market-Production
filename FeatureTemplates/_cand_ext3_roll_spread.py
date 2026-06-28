"""
ext3_roll_spread — Roll (1984) implied effective spread
Per-ticker proxy for the bid-ask spread using the serial covariance of returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_roll_spread",
    "description": (
        "Roll (1984) implied effective spread estimated from the rolling 21-day "
        "first-order serial autocovariance of log returns: S = 2*sqrt(-cov(r_t, r_{t-1})) "
        "when the 21-day rolling autocovariance is negative, else 0. "
        "Also produces the 60-day change in the spread (widening/narrowing trend). "
        "Pure Close-based per-ticker proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": ["ext3_roll_spread_level", "ext3_roll_spread_chg60"],
    "tags": ["microstructure", "liquidity", "spread", "roll1984"],
    "version": "1.0",
    "author": "Roll (1984) 'A Simple Implicit Measure of the Effective Bid-Ask Spread'; spec: Round-4 expansion (NEW: microstructure)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns: r_t = log(Close_t / Close_{t-1})
    log_ret = np.log(df["Close"].replace(0, np.nan)).diff()

    # Lag-1 returns for rolling autocovariance
    log_ret_lag1 = log_ret.shift(1)

    # Rolling 21-day autocovariance: cov(r_t, r_{t-1}) over a 21-bar window.
    # We compute this manually so we can handle the sign check properly.
    # cov(x, y) = mean(x*y) - mean(x)*mean(y), min_periods=21.
    window = 21

    xy = log_ret * log_ret_lag1
    x_mean = log_ret.rolling(window, min_periods=window).mean()
    y_mean = log_ret_lag1.rolling(window, min_periods=window).mean()
    xy_mean = xy.rolling(window, min_periods=window).mean()

    # Population-style rolling covariance (consistent with Roll's definition)
    rolling_autocov = xy_mean - x_mean * y_mean

    # Roll spread: 2*sqrt(-autocov) when autocov < 0, else 0
    negative_cov = rolling_autocov.where(rolling_autocov < 0, other=0.0)
    spread_level = 2.0 * np.sqrt(-negative_cov)

    df["ext3_roll_spread_level"] = spread_level

    # 60-day change (spread widening = liquidity worsening)
    df["ext3_roll_spread_chg60"] = spread_level.diff(60)

    return df
