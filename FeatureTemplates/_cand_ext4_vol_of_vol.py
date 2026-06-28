from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_vol_of_vol",
    "description": (
        "Volatility-of-volatility: computes the 20-day rolling realized vol series "
        "(annualised std of log-returns), then measures its own 60-day rolling standard "
        "deviation normalised by the 60-day mean vol (VoV ratio) and its 60-day linear "
        "trend slope (VoV trend). High VoV ratio signals unstable, regime-changing vol; "
        "the trend captures whether vol uncertainty is expanding or contracting. "
        "Per-ticker OHLCV proxy; no cross-sectional information."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_vol_of_vol_ratio",   # rolling std(20d-vol, 60d) / rolling mean(20d-vol, 60d)
        "ext4_vol_of_vol_level",   # rolling std(20d-vol, 60d) -- unnormalised level
        "ext4_vol_of_vol_trend",   # 60-day OLS slope of the 20d-vol series, normalised by mean vol
    ],
    "tags": ["volatility", "regime", "risk", "ohlcv"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom_allan_variance)",
}

_ANN = np.sqrt(252)
_SHORT_WIN = 20   # window for base realized vol
_LONG_WIN = 60    # window for vol-of-vol statistics


def _ols_slope_series(s: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope via the closed-form formula for equally-spaced x."""
    n = window
    # x = [0, 1, ..., n-1]; precompute x - x_mean
    x = np.arange(n, dtype=np.float64)
    x_dm = x - x.mean()
    ss_x = (x_dm ** 2).sum()  # scalar

    def _slope(arr: np.ndarray) -> float:
        if np.any(np.isnan(arr)):
            return np.nan
        return float(np.dot(x_dm, arr) / ss_x)

    return s.rolling(window).apply(_slope, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)

    # --- 1. 20-day rolling realised vol (annualised) ---
    log_ret = np.log(close / close.shift(1))
    # std of log-returns over 20 days, annualised; min_periods=15 to avoid very noisy early windows
    rv20 = log_ret.rolling(_SHORT_WIN, min_periods=max(10, _SHORT_WIN // 2)).std() * _ANN

    # --- 2. 60-day rolling statistics of the rv20 series ---
    rv_mean60 = rv20.rolling(_LONG_WIN, min_periods=_LONG_WIN // 2).mean()
    rv_std60 = rv20.rolling(_LONG_WIN, min_periods=_LONG_WIN // 2).std()

    # VoV ratio: std / mean (coefficient of variation of vol)
    vov_ratio = rv_std60 / rv_mean60.replace(0, np.nan)
    vov_ratio = vov_ratio.replace([np.inf, -np.inf], np.nan)

    # VoV level: raw std (unnormalised)
    vov_level = rv_std60

    # VoV trend: 60-day OLS slope of the rv20 series, normalised by mean vol
    slope_raw = _ols_slope_series(rv20, _LONG_WIN)
    vov_trend = slope_raw / rv_mean60.replace(0, np.nan)
    vov_trend = vov_trend.replace([np.inf, -np.inf], np.nan)

    df["ext4_vol_of_vol_ratio"] = vov_ratio.values
    df["ext4_vol_of_vol_level"] = vov_level.values
    df["ext4_vol_of_vol_trend"] = vov_trend.values

    return df
