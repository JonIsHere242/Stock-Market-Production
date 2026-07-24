"""
_cand_ff0703f_distribution_pearson_median_skew.py -- Pearson's second skewness coefficient
of trailing daily-return distribution (mean-vs-median robust skew).

METHOD (spec ff0703f_distribution_pearson_median_skew):
  On simple daily returns over a 90-trading-day trailing window compute mean m,
  median med, and std sd. LEVEL = 3*(m - med)/sd, guarded to NaN when sd==0 or
  fewer than 60 valid returns are available in the window.
  DYNAMIC = level minus its value 20 trading days earlier.

This is a robust (mean-vs-median) skew measure, distinct from the classic third
standardized moment skew already present elsewhere in the feature set -- it is
far less sensitive to single extreme-return outliers and instead captures
persistent asymmetry in the bulk of the return distribution.

Per-ticker, causal (trailing rolling windows only), fully vectorized with pandas
rolling ops. No lookahead: window at time t only uses returns up to and including t.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703f_distribution_pearson_median_skew",
    "description": (
        "Pearson's second skewness coefficient (3*(mean-median)/std) of simple daily "
        "returns over a trailing 90-day window, plus its 20-day change. Robust "
        "mean-vs-median skew measure, orthogonal to classic third-moment skew "
        "features since it is much less dominated by single outlier returns."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703f_dist_pearson_med_skew",
        "ff0703f_dist_pearson_med_skew_slope20",
    ],
    "tags": ["distribution", "skewness", "robust-stats", "returns"],
    "version": "1.0",
    "author": "feature-factory (faithful per-ticker implementation of spec)",
}

_WINDOW = 90
_MIN_VALID = 60
_SLOPE_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    level_col = "ff0703f_dist_pearson_med_skew"
    slope_col = "ff0703f_dist_pearson_med_skew_slope20"

    df[level_col] = np.nan
    df[slope_col] = np.nan

    n = len(df)
    if n == 0:
        return df

    close = df["Close"].astype(np.float64)
    prev_close = close.shift(1)
    ret = (close - prev_close) / prev_close.replace(0, np.nan)
    ret = ret.replace([np.inf, -np.inf], np.nan)

    if ret.notna().sum() < _MIN_VALID:
        return df

    roll = ret.rolling(window=_WINDOW, min_periods=_MIN_VALID)
    m = roll.mean()
    med = roll.median()
    sd = roll.std(ddof=1)
    valid_count = ret.rolling(window=_WINDOW, min_periods=1).count()

    sd_safe = sd.where(sd != 0, np.nan)
    level = 3.0 * (m - med) / sd_safe
    level = level.where(valid_count >= _MIN_VALID, np.nan)
    level = level.replace([np.inf, -np.inf], np.nan)

    df[level_col] = level
    df[slope_col] = level - level.shift(_SLOPE_LAG)

    return df
