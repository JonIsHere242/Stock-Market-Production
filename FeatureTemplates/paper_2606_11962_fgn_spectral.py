"""
Fractional Gaussian Noise (fGn) spectral features derived from:
  "Composite likelihood inference of fractional Gaussian processes with
   sequentially optimal subset selection" (arXiv 2606.11962)

The paper fits fractional Brownian motion / fractional Gaussian noise to
volatility series using composite likelihood with Godambe information.
The core representation: fGn has a power-law spectral density
  S(f) ~ f^{-(2H-1)}  for  H in (0,1).

We extract this via rolling periodogram regression (Geweke-Porter-Hudak
style log-periodogram OLS), giving a rolling estimate of the spectral
exponent beta = 2H - 1 per window.  Features include:
  - Rolling spectral exponent (log-log OLS slope) on log-returns
  - Residual from log-linear fit (spectral roughness)
  - Multi-window exponent spread (regime indicator)
  - Spectral coherence: fraction of variance in low vs high frequencies

IMPLEMENTATION: fully vectorised via stride tricks + batch FFT.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_11962_fgn_spectral",
    "description": (
        "Fractional Gaussian noise spectral exponent (Geweke-Porter-Hudak log-periodogram OLS) "
        "and derived multi-window features; based on arXiv 2606.11962 fGn composite likelihood."
    ),
    "requires":    ["Close"],
    "produces": [
        "fgn_spec_exp_32d",
        "fgn_spec_exp_64d",
        "fgn_spec_exp_128d",
        "fgn_spec_roughness_64d",
        "fgn_spec_spread",
        "fgn_lowfreq_ratio_64d",
        "fgn_spec_exp_zscore",
    ],
    "tags":        ["volatility", "spectral", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.11962",
}


def _strided_windows(x: np.ndarray, window: int) -> np.ndarray:
    """2-D view of rolling windows.  Shape: (n-window+1, window)."""
    n = len(x)
    if n < window:
        return np.empty((0, window), dtype=x.dtype)
    shape = (n - window + 1, window)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _batch_gph(wins: np.ndarray) -> np.ndarray:
    """
    Vectorised Geweke-Porter-Hudak (GPH) log-periodogram OLS.
    wins: (n_wins, w) — rows are fixed-length segments of log-returns.
    Returns: (n_wins,) spectral slope beta (2H-1 for fGn).
    """
    n_wins, w = wins.shape
    # Demean each row
    x = wins - wins.mean(axis=1, keepdims=True)
    # Batch FFT (pad to next power of 2)
    nfft = 1
    while nfft < w:
        nfft <<= 1
    X = np.fft.rfft(x, n=nfft, axis=1)          # (n_wins, nfft//2+1)
    period = (np.abs(X[:, 1:]) ** 2) / w         # skip DC; shape (n_wins, nfft//2)
    # GPH: use first m = floor(sqrt(w)) components
    m = max(4, int(np.floor(np.sqrt(w))))
    m = min(m, period.shape[1])
    freqs = np.arange(1, m + 1, dtype=np.float64)
    log_f = np.log(freqs / nfft)                  # (m,)
    log_I = np.log(period[:, :m] + 1e-30)         # (n_wins, m)
    lf_c = log_f - log_f.mean()
    lI_c = log_I - log_I.mean(axis=1, keepdims=True)
    num = (lI_c * lf_c).sum(axis=1)
    den = (lf_c ** 2).sum()
    return np.where(den > 1e-12, num / den, np.nan)


def _batch_roughness(wins: np.ndarray) -> np.ndarray:
    """Std of log-periodogram residuals after GPH fit (power-law deviation)."""
    n_wins, w = wins.shape
    x = wins - wins.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < w:
        nfft <<= 1
    X = np.fft.rfft(x, n=nfft, axis=1)
    period = (np.abs(X[:, 1:]) ** 2) / w
    m = max(4, int(np.floor(np.sqrt(w))))
    m = min(m, period.shape[1])
    freqs = np.arange(1, m + 1, dtype=np.float64)
    log_f = np.log(freqs / nfft)
    log_I = np.log(period[:, :m] + 1e-30)
    lf_c = log_f - log_f.mean()
    lI_c = log_I - log_I.mean(axis=1, keepdims=True)
    num = (lI_c * lf_c).sum(axis=1)
    den = (lf_c ** 2).sum()
    beta = np.where(den > 1e-12, num / den, 0.0)
    pred = log_I.mean(axis=1, keepdims=True) + beta[:, None] * lf_c
    resid = lI_c - (pred - log_I.mean(axis=1, keepdims=True))
    return resid.std(axis=1)


def _batch_lowfreq_ratio(wins: np.ndarray) -> np.ndarray:
    """Fraction of spectral power in lower half of non-DC frequencies."""
    n_wins, w = wins.shape
    x = wins - wins.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < w:
        nfft <<= 1
    X = np.fft.rfft(x, n=nfft, axis=1)
    power = (np.abs(X[:, 1:]) ** 2)
    total = power.sum(axis=1)
    mid = max(1, power.shape[1] // 2)
    lo = power[:, :mid].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 1e-30, lo / total, np.nan)


def _rolling_batch(lr_vals: np.ndarray, window: int,
                   func, min_frac: float = 0.5) -> np.ndarray:
    """Apply a batch function over all rolling windows of given size."""
    n = len(lr_vals)
    result = np.full(n, np.nan)
    if n < window:
        return result
    min_obs = max(16, int(window * min_frac))
    finite = np.isfinite(lr_vals).astype(np.float64)
    x_filled = np.where(np.isfinite(lr_vals), lr_vals, 0.0)
    wins = _strided_windows(x_filled, window)           # (n-w+1, w) view
    fmask = _strided_windows(finite, window)            # same shape
    valid = fmask.sum(axis=1) >= min_obs
    vals = func(wins.copy())   # copy avoids stride-tricks mutation issues
    vals[~valid] = np.nan
    result[window - 1:] = vals
    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    log_ret = np.log(close / close.shift(1))
    lr_vals = log_ret.values.astype(np.float64)

    # Vectorised rolling GPH at three window sizes
    df["fgn_spec_exp_32d"]  = _rolling_batch(lr_vals, 32,  _batch_gph)
    df["fgn_spec_exp_64d"]  = _rolling_batch(lr_vals, 64,  _batch_gph)
    df["fgn_spec_exp_128d"] = _rolling_batch(lr_vals, 128, _batch_gph)

    # Spectral roughness and low-freq ratio at 64-bar window
    df["fgn_spec_roughness_64d"] = _rolling_batch(lr_vals, 64, _batch_roughness)
    df["fgn_lowfreq_ratio_64d"]  = _rolling_batch(lr_vals, 64, _batch_lowfreq_ratio)

    # Multi-window spread (long minus short exponent)
    s32  = pd.Series(df["fgn_spec_exp_32d"].values,  index=df.index)
    s128 = pd.Series(df["fgn_spec_exp_128d"].values, index=df.index)
    df["fgn_spec_spread"] = s128 - s32

    # Z-score of 64d exponent vs its own trailing 126-day distribution
    s64 = pd.Series(df["fgn_spec_exp_64d"].values, index=df.index)
    rm = s64.rolling(126, min_periods=32).mean()
    rs = s64.rolling(126, min_periods=32).std()
    df["fgn_spec_exp_zscore"] = (s64 - rm) / rs.replace(0, np.nan)

    return df
