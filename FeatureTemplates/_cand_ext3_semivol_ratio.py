"""
Candidate feature block: ext3_semivol_ratio
Up/down realized-semivolatility ratio & trend.

Rolling 40-day downside semideviation (RMS of negative daily returns) divided
by upside semideviation (RMS of positive daily returns), plus its 60-day trend
and the raw downside semideviation level.

Source: Round-4 expansion (xdom2_downside_beta)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_semivol_ratio",
    "description": (
        "Rolling 40-day up/down realized-semivolatility ratio and trend. "
        "Downside semideviation = RMS of strictly negative log-returns; "
        "upside semideviation = RMS of strictly positive log-returns. "
        "Ratio > 1 means recent downside moves are larger in magnitude than "
        "upside moves (bearish asymmetry). 60-day slope of the ratio captures "
        "whether the asymmetry is worsening or improving. Pure OHLCV, per-ticker. "
        "Distinct from semivariance-share: uses RMS of signed deviations, not "
        "squared-share of total variance."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_semivol_ratio",        # 40d down_semivol / up_semivol
        "ext3_semivol_ratio_slope",  # 60d linear trend of the ratio
        "ext3_semivol_down",         # 40d downside semideviation level
    ],
    "tags": ["volatility", "asymmetry", "semivolatility", "downside", "risk"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom2_downside_beta)",
}

_WINDOW = 40       # rolling window for semideviation
_SLOPE_WIN = 60    # window for trend (slope) of ratio
_MIN_OBS = 10      # minimum observations to emit a value


def _rolling_rms(series: pd.Series, window: int, min_obs: int, mask_positive: bool) -> pd.Series:
    """
    For each position t, compute RMS of the subset of values in
    series[t-window+1 : t+1] that are positive (mask_positive=True)
    or negative (mask_positive=False).

    Uses a vectorised sliding-window approach via numpy stride tricks so
    that the loop is over windows, not over every row.
    """
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    result = np.full(n, np.nan)

    if n < window:
        return pd.Series(result, index=series.index)

    # Build a (n-window+1) x window matrix of rolling windows
    from numpy.lib.stride_tricks import sliding_window_view
    wins = sliding_window_view(arr, window)  # shape (n-window+1, window)

    for i, w in enumerate(wins):
        t = i + window - 1  # index of the last element of this window
        if mask_positive:
            sub = w[w > 0.0]
        else:
            sub = w[w < 0.0]

        if len(sub) < min_obs:
            continue
        result[t] = np.sqrt(np.mean(sub ** 2))

    return pd.Series(result, index=series.index)


def _slope_rolling(series: pd.Series, window: int) -> pd.Series:
    """
    For each position t compute the OLS slope of series[t-window+1:t+1]
    vs a [0..window-1] time index. Vectorised via sliding_window_view.
    """
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    result = np.full(n, np.nan)

    if n < window:
        return pd.Series(result, index=series.index)

    from numpy.lib.stride_tricks import sliding_window_view
    wins = sliding_window_view(arr, window)  # (n-window+1, window)

    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)  # scalar, same for every window

    for i, w in enumerate(wins):
        t = i + window - 1
        # skip if any NaN in the window
        if np.any(np.isnan(w)):
            continue
        y_mean = w.mean()
        cov = np.dot(x - x_mean, w - y_mean)
        result[t] = cov / x_var

    return pd.Series(result, index=series.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Log returns (no lookahead: shift(1) only)
    # ------------------------------------------------------------------
    log_ret = np.log(df["Close"] / df["Close"].shift(1))  # NaN at row 0

    # ------------------------------------------------------------------
    # 2. Rolling 40d downside and upside semideviation
    # ------------------------------------------------------------------
    down_semivol = _rolling_rms(log_ret, _WINDOW, _MIN_OBS, mask_positive=False)
    up_semivol   = _rolling_rms(log_ret, _WINDOW, _MIN_OBS, mask_positive=True)

    # ------------------------------------------------------------------
    # 3. Ratio: down / up (guard zero denominator)
    # ------------------------------------------------------------------
    safe_up = up_semivol.replace(0.0, np.nan)
    ratio = down_semivol / safe_up

    # Replace inf/-inf with NaN (shouldn't happen after guard, but be safe)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 4. 60d rolling slope of the ratio
    # ------------------------------------------------------------------
    ratio_slope = _slope_rolling(ratio, _SLOPE_WIN)
    ratio_slope = ratio_slope.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 5. Assign produced columns
    # ------------------------------------------------------------------
    df["ext3_semivol_ratio"]       = ratio.values
    df["ext3_semivol_ratio_slope"] = ratio_slope.values
    df["ext3_semivol_down"]        = down_semivol.values

    return df
