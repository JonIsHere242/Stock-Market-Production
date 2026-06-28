from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "xdom2_gini_returns",
    "description": (
        "Rolling Gini coefficient of absolute daily returns. "
        "Measures how concentrated the period's total price movement is across days. "
        "High Gini => a few jump days dominate (jumpy/gap regime); "
        "low Gini => volatility is smoothly spread across all days. "
        "Per-ticker time-series proxy -- captures the same economic signal as the "
        "cross-sectional version (return-concentration regime) but is computed "
        "independently for each stock from its own OHLCV history. "
        "Produces: 60-day level, 20-day short-window level, and a 20-vs-60 slope "
        "that detects regime transitions."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_gini_returns_60",   # Gini of |ret| over rolling 60-day window
        "xdom2_gini_returns_20",   # Gini of |ret| over rolling 20-day window (shorter-horizon)
        "xdom2_gini_returns_slope", # 20-day Gini minus 60-day Gini (rising => jump regime emerging)
    ],
    "tags": ["volatility", "regime", "concentration", "cross-domain", "gini"],
    "version": "1.0",
    "author": "Spec: Cross-domain / practitioner method transfer (batch 2) — Gini concentration of absolute moves",
}


def _gini_series(arr: np.ndarray) -> float:
    """
    Gini coefficient of a 1-D non-negative array.
    Returns NaN if all values are zero or array is empty.
    """
    n = len(arr)
    if n == 0:
        return np.nan
    total = arr.sum()
    if total == 0.0:
        return np.nan
    # Sort ascending then use the closed-form cumsum formula
    s = np.sort(arr)
    # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n+1)/n
    ranks = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (ranks * s).sum()) / (n * total) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))


def _rolling_gini(abs_ret: pd.Series, window: int) -> pd.Series:
    """
    Apply _gini_series over a rolling window using .rolling().apply().
    Fast enough for window<=60 on ~700 rows.
    """
    return abs_ret.rolling(window=window, min_periods=window).apply(
        _gini_series, raw=True
    )


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Absolute daily log returns (close-to-close)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    abs_ret = log_ret.abs()

    df["xdom2_gini_returns_60"] = _rolling_gini(abs_ret, 60)
    df["xdom2_gini_returns_20"] = _rolling_gini(abs_ret, 20)
    df["xdom2_gini_returns_slope"] = (
        df["xdom2_gini_returns_20"] - df["xdom2_gini_returns_60"]
    )

    return df
