"""
Long-memory features for realized volatility derived from:
  "Memory, Roughness, and Information Persistence in Financial Markets:
   A Structural Approach to Volatility Forecasting" (arXiv 2605.24285).

The paper estimates the long-memory parameter d of volatility using:
  1. Geweke-Porter-Hudak (GPH) semi-parametric log-periodogram OLS on |returns|.
  2. Local-Whittle (LW) estimator on |returns| via approximate spectral likelihood.
  3. HAR-X-style regression: realized variance predicted by its own lags (1d, 5d, 22d).

All features are rolling per-ticker (no cross-section).

Key outputs:
  - mem_gph_d_*d     : rolling GPH memory parameter (0.5=long memory, near 0=short)
  - mem_lw_d_*d      : rolling local-Whittle memory parameter
  - mem_har_resid_*d : residual of current RV from HAR prediction (surprise / shock)
  - mem_rv_ratio_*d  : ratio of short- to long-run realized variance (roughness proxy)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2605_24285_gph_lw_vol_memory",
    "description": (
        "Rolling GPH log-periodogram and local-Whittle long-memory parameter estimates "
        "on absolute returns, plus HAR-X residuals; proxy per-ticker implementation of "
        "cross-sectional panel study arXiv 2605.24285."
    ),
    "requires":    ["Close"],
    "produces": [
        "mem_gph_d_64d",
        "mem_gph_d_128d",
        "mem_lw_d_64d",
        "mem_lw_d_128d",
        "mem_har_resid_22d",
        "mem_rv_ratio_5_22d",
        "mem_gph_spread",
    ],
    "tags":        ["volatility", "mean_reversion", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2605.24285",
}


def _gph_estimate(abs_ret: np.ndarray, m: int) -> float:
    """
    GPH log-periodogram OLS estimator for memory parameter d.
    Uses the m lowest Fourier frequencies (m ~ n^0.5 is classic).
    Returns d in [-0.5, 1.5] or NaN on failure.
    """
    n = len(abs_ret)
    if n < 16 or m < 4:
        return np.nan
    # Periodogram at frequencies j*2pi/n for j=1..m
    # Use FFT of the series
    ft = np.fft.rfft(abs_ret - abs_ret.mean())
    freq_idx = np.arange(1, min(m + 1, len(ft)))
    if len(freq_idx) < 4:
        return np.nan
    # Periodogram = |FFT|^2 / n
    periodogram = (np.abs(ft[freq_idx]) ** 2) / n
    freqs = freq_idx * (2.0 * np.pi / n)
    # GPH: log(I_j) = const - 2d * log(freq_j) + eps
    log_freq = np.log(freqs)
    log_I = np.log(periodogram + 1e-16)
    # OLS: d = -slope/2
    lf_mean = log_freq.mean()
    lI_mean = log_I.mean()
    ss_xx = ((log_freq - lf_mean) ** 2).sum()
    if ss_xx < 1e-12:
        return np.nan
    slope = ((log_freq - lf_mean) * (log_I - lI_mean)).sum() / ss_xx
    d = -slope / 2.0
    if not np.isfinite(d):
        return np.nan
    return float(np.clip(d, -0.5, 1.5))


def _lw_estimate(abs_ret: np.ndarray, m: int) -> float:
    """
    Simplified local-Whittle estimator: minimise spectral likelihood over d.
    Approximate grid search over d in [-0.5, 1.5] with 40 steps.
    Returns d or NaN on failure.
    """
    n = len(abs_ret)
    if n < 16 or m < 4:
        return np.nan
    ft = np.fft.rfft(abs_ret - abs_ret.mean())
    freq_idx = np.arange(1, min(m + 1, len(ft)))
    if len(freq_idx) < 4:
        return np.nan
    periodogram = (np.abs(ft[freq_idx]) ** 2) / n
    freqs = freq_idx * (2.0 * np.pi / n)
    log_freq = np.log(freqs)

    best_d = np.nan
    best_Q = np.inf
    for d_try in np.linspace(-0.5, 1.5, 41):
        # Spectral density ~ freq^{-(2d-0)}; G estimated by average(I * freq^{2d})
        w = freqs ** (2.0 * d_try)
        G_hat = (periodogram * w).mean()
        if G_hat <= 0:
            continue
        Q = np.log(G_hat) - 2.0 * d_try * log_freq.mean()
        if Q < best_Q:
            best_Q = Q
            best_d = d_try
    return float(best_d) if np.isfinite(best_d) else np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Close"].pct_change()
    abs_ret = ret.abs()
    rv = ret ** 2  # realized variance proxy

    n = len(df)
    gph_64  = np.full(n, np.nan)
    gph_128 = np.full(n, np.nan)
    lw_64   = np.full(n, np.nan)
    lw_128  = np.full(n, np.nan)

    abs_arr = abs_ret.values

    for i in range(63, n):
        # 64-day window
        seg64 = abs_arr[max(0, i - 63): i + 1]
        m64 = max(4, int(len(seg64) ** 0.5))
        gph_64[i]  = _gph_estimate(seg64, m64)
        lw_64[i]   = _lw_estimate(seg64, m64)

    for i in range(127, n):
        # 128-day window
        seg128 = abs_arr[max(0, i - 127): i + 1]
        m128 = max(4, int(len(seg128) ** 0.5))
        gph_128[i] = _gph_estimate(seg128, m128)
        lw_128[i]  = _lw_estimate(seg128, m128)

    df["mem_gph_d_64d"]  = gph_64
    df["mem_gph_d_128d"] = gph_128
    df["mem_lw_d_64d"]   = lw_64
    df["mem_lw_d_128d"]  = lw_128

    # HAR residual: RV_t - (beta_1*RV_{t-1} + beta_5*RV_{1..5}.mean() + beta_22*RV_{1..22}.mean())
    # Rolling HAR: use fixed rolling averages as predictors; actual HAR OLS is too slow row-wise.
    # Instead, use the HAR forecast = 0.4*rv_lag1 + 0.3*rv_lag5_avg + 0.3*rv_lag22_avg (typical coefs)
    rv_lag1   = rv.shift(1)
    rv_lag5   = rv.shift(1).rolling(5,  min_periods=3).mean()
    rv_lag22  = rv.shift(1).rolling(22, min_periods=10).mean()
    har_pred  = 0.4 * rv_lag1 + 0.3 * rv_lag5 + 0.3 * rv_lag22
    har_resid = rv - har_pred
    # Normalise by rolling std of rv
    rv_std    = rv.rolling(22, min_periods=10).std()
    df["mem_har_resid_22d"] = (har_resid / rv_std.replace(0, np.nan)).clip(-5, 5)

    # Roughness proxy: ratio of 5-day avg RV to 22-day avg RV
    rv5  = rv.rolling(5,  min_periods=3).mean()
    rv22 = rv.rolling(22, min_periods=10).mean()
    df["mem_rv_ratio_5_22d"] = (rv5 / rv22.replace(0, np.nan)).clip(0, 10)

    # Spread between two GPH windows (regime indicator: rising spread = memory regime shift)
    gph_64_s  = pd.Series(gph_64,  index=df.index)
    gph_128_s = pd.Series(gph_128, index=df.index)
    df["mem_gph_spread"] = gph_64_s - gph_128_s

    return df
