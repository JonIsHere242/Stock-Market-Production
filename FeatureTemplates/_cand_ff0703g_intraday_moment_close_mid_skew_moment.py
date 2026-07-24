from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703g_intraday_moment_close_mid_skew_moment",
    "description": (
        "Distributional asymmetry of intraday close-position over a quarter window. "
        "Per-bar close-position cp = (Close - (High+Low)/2) / (High-Low), with zero-range "
        "bars set to NaN. Computes the rolling standardized skewness of cp over a 60-bar "
        "window: skew60 = mean((cp-mu)^3) / sigma^3 using the 60-bar mean/std of cp "
        "(sigma==0 -> NaN). LEVEL = skew60 captures whether closes have recently clustered "
        "asymmetrically near the high or low of the daily range (fat tail toward one side), "
        "distinct from mean close-location (CLV) which only captures the average level. "
        "DYNAMIC = skew60 minus its value 20 bars earlier, capturing the recent change in "
        "that distributional asymmetry (e.g. regime shift from bottom-fishing closes to "
        "top-of-range closes or vice versa). Pure per-ticker OHLC, no cross-sectional data."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff0703g_intraday_moment_close_mid_skew_moment_level",
        "ff0703g_intraday_moment_close_mid_skew_moment_dyn",
    ],
    "tags": ["intraday", "skewness", "distribution", "close_position", "asymmetry"],
    "version": "1.0",
    "author": "ff0703g batch - faithful implementation of spec (skew of close-position, per-ticker OHLC)",
}

_WIN = 60
_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df["ff0703g_intraday_moment_close_mid_skew_moment_level"] = np.nan
    df["ff0703g_intraday_moment_close_mid_skew_moment_dyn"] = np.nan

    if n == 0:
        return df

    hl = df["High"] - df["Low"]
    hl_safe = hl.replace(0, np.nan)
    mid = (df["High"] + df["Low"]) / 2.0
    cp = (df["Close"] - mid) / hl_safe

    mu = cp.rolling(_WIN, min_periods=_WIN).mean()
    sigma = cp.rolling(_WIN, min_periods=_WIN).std(ddof=0)
    sigma_safe = sigma.replace(0, np.nan)

    m3 = (cp - mu).pow(3).rolling(_WIN, min_periods=_WIN).mean()
    skew60 = m3 / sigma_safe.pow(3)

    df["ff0703g_intraday_moment_close_mid_skew_moment_level"] = skew60
    df["ff0703g_intraday_moment_close_mid_skew_moment_dyn"] = skew60 - skew60.shift(_LAG)

    return df
