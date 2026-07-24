from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703k_intraday_moment_body_ratio_third_moment",
    "description": (
        "Distribution-shape feature on the intraday body ratio br_t = (Close-Open)/(High-Low) "
        "(NaN when High==Low), rather than its mean. LEVEL = rolling standardized third moment "
        "(true skewness) of br over 60 trading days: skew = mean((br-mu)^3)/sigma^3 using the 60d "
        "rolling mean/std, NaN when sigma<=1e-9. Captures whether the body sits asymmetrically "
        "toward the top/bottom of the daily range over time (fat tail of bullish or bearish closes), "
        "orthogonal to the mean-of-close-position features already in the pipeline. DYNAMIC = "
        "short-minus-long term-structure of that skew: skew_20 - skew_60 (same third-moment "
        "computed on a 20d window), capturing recent acceleration/reversal of body-asymmetry regime. "
        "Pure per-ticker OHLC construction, fully causal (rolling windows only)."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703k_intraday_moment_body_ratio_third_moment_skew60",
        "ff0703k_intraday_moment_body_ratio_third_moment_ts2060",
    ],
    "tags": ["intraday_moment", "candle", "skewness", "distribution", "body_ratio"],
    "version": "1.0",
    "author": "ff0703k spec codegen (faithful implementation of body-ratio 3rd-moment skew + 20-60 term structure)",
}

_WIN_SHORT = 20
_WIN_LONG = 60


def _rolling_skew(br: pd.Series, window: int) -> pd.Series:
    mu = br.rolling(window, min_periods=window).mean()
    sigma = br.rolling(window, min_periods=window).std(ddof=0)
    m3 = (br - mu).pow(3).rolling(window, min_periods=window).mean()
    sigma_safe = sigma.where(sigma > 1e-9, np.nan)
    skew = m3 / sigma_safe.pow(3)
    return skew.replace([np.inf, -np.inf], np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_level = "ff0703k_intraday_moment_body_ratio_third_moment_skew60"
    col_dyn = "ff0703k_intraday_moment_body_ratio_third_moment_ts2060"

    df[col_level] = np.nan
    df[col_dyn] = np.nan

    if len(df) == 0:
        return df

    hl = df["High"] - df["Low"]
    hl_safe = hl.where(hl != 0, np.nan)
    br = (df["Close"] - df["Open"]) / hl_safe

    skew_60 = _rolling_skew(br, _WIN_LONG)
    skew_20 = _rolling_skew(br, _WIN_SHORT)

    df[col_level] = skew_60
    df[col_dyn] = skew_20 - skew_60

    return df
