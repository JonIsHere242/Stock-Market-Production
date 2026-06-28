from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_ljung_box",
    "description": (
        "Rolling Ljung-Box Q statistic on log-returns for a single ticker. "
        "Computes Q5 over a 90-day trailing window using the first 5 autocorrelations "
        "of daily log-returns: Q = n(n+2) * sum_{k=1..5} rho_k^2 / (n-k). "
        "High Q => significant serial dependence (predictable structure); near 0 => white noise. "
        "Also produces a 20-day short-window Q5 for recency contrast and a slope "
        "(difference of 90d minus 20d normalized by 90d, capturing regime transition). "
        "Per-ticker time-series proxy; no cross-sectional computation required."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_ljung_box_q5_90",   # 90-day rolling Ljung-Box Q(5)
        "xdom2_ljung_box_q5_20",   # 20-day rolling Ljung-Box Q(5) -- recency signal
        "xdom2_ljung_box_slope",   # (q5_90 - q5_20) / (q5_90 + 1e-8) -- regime shift
    ],
    "tags": ["serial-correlation", "autocorrelation", "ljung-box", "cross-domain"],
    "version": "1.0",
    "author": "Ljung-Box serial-correlation statistic (Ljung & Box, 1978); batch xdom2 spec",
}

# Minimum observations needed inside a window to produce a valid result
_MIN_OBS = 10
_LAGS = 5


def _ljung_box_q5(returns_arr: np.ndarray) -> float:
    """
    Compute the Ljung-Box Q statistic for lags 1..5 on a 1-D array of returns.
    Returns NaN if insufficient data.
    """
    n = len(returns_arr)
    if n < _MIN_OBS + _LAGS:
        return np.nan

    # Demean
    x = returns_arr - np.nanmean(returns_arr)
    # c0: variance (lag-0 autocovariance)
    c0 = np.nanmean(x ** 2)
    if c0 == 0 or np.isnan(c0):
        return np.nan

    q = 0.0
    for k in range(1, _LAGS + 1):
        # autocorrelation at lag k
        xpast = x[:-k]
        xfut  = x[k:]
        # use only valid (non-nan) pairs
        mask = np.isfinite(xpast) & np.isfinite(xfut)
        if mask.sum() < _MIN_OBS:
            return np.nan
        rho_k = np.sum(xpast[mask] * xfut[mask]) / (mask.sum() * c0)
        q += rho_k ** 2 / (n - k)

    q *= n * (n + 2)
    return float(q)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns; first value will be NaN (expected)
    log_ret = np.log(df["Close"].replace(0, np.nan)).diff().values.astype(float)

    n = len(log_ret)
    q90 = np.full(n, np.nan)
    q20 = np.full(n, np.nan)

    win90 = 90
    win20 = 20

    # Rolling 90-day window
    for i in range(win90 - 1, n):
        window = log_ret[i - win90 + 1: i + 1]
        q90[i] = _ljung_box_q5(window)

    # Rolling 20-day window
    for i in range(win20 - 1, n):
        window = log_ret[i - win20 + 1: i + 1]
        q20[i] = _ljung_box_q5(window)

    df["xdom2_ljung_box_q5_90"] = q90
    df["xdom2_ljung_box_q5_20"] = q20

    # Slope: positive => long-window more autocorrelated than short (persistent structure)
    # negative => structure decaying / short-term mean-reverting
    denom = np.where(np.isfinite(q90) & (q90 != 0), q90, np.nan)
    slope = np.where(
        np.isfinite(q90) & np.isfinite(q20),
        (q90 - q20) / (np.abs(denom) + 1e-8),
        np.nan,
    )
    df["xdom2_ljung_box_slope"] = slope.astype(float)

    return df
