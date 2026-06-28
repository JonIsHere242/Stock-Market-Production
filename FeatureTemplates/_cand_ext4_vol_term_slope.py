from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext4_vol_term_slope",
    "description": (
        "Realized-vol term-structure slope: ratio of short realized vol (10d std of log-returns) "
        "to long realized vol (60d std of log-returns). Ratio > 1 signals vol backwardation / stress "
        "(near-term risk elevated vs long-term); ratio < 1 signals contango / calm. "
        "Also computes the 20-day change of this ratio to capture whether the term structure is "
        "steepening or flattening. Pure per-ticker OHLCV — no cross-sectional dependency."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_vol_term_slope_ratio",   # short_vol / long_vol
        "ext4_vol_term_slope_chg20",   # 20d change of ratio (slope of slope)
    ],
    "tags": ["volatility", "term_structure", "realized_vol", "stress"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom_allan_variance); spec: ext4_vol_term_slope",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns (shift(1) is past, no lookahead)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Short realized vol: 10-day rolling std of log returns
    short_vol = log_ret.rolling(window=10, min_periods=5).std()

    # Long realized vol: 60-day rolling std of log returns
    long_vol = log_ret.rolling(window=60, min_periods=30).std()

    # Vol term-structure ratio: backwardation (>1) vs contango (<1)
    ratio = short_vol / long_vol.replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # 20-day change of the ratio (momentum of term structure)
    chg20 = ratio - ratio.shift(20)

    df["ext4_vol_term_slope_ratio"] = ratio
    df["ext4_vol_term_slope_chg20"] = chg20

    return df
