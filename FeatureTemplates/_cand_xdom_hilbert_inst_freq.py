"""
Hilbert instantaneous frequency & bandwidth (analytic-signal demodulation).

Per-ticker rolling implementation: for each trading day t, we slide a 64-bar
window of detrended log-close prices, compute the analytic signal via the
FFT-based Hilbert transform (numpy), extract the instantaneous phase, diff to
get instantaneous frequency, then record its rolling mean (dominant cycle
speed) and std (bandwidth / cycle-stability proxy).

Cross-sectional note: the spec is inherently per-ticker (time-series DSP);
no cross-sectional proxy is needed -- this IS the faithful implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

METADATA = {
    "name": "xdom_hilbert_inst_freq",
    "description": (
        "Analytic-signal demodulation of detrended log-price on a rolling 64-bar window. "
        "Produces: instantaneous frequency (mean phase-derivative over window), its std "
        "(bandwidth proxy), and a normalised cycle-irregularity index. "
        "Implements the Hilbert transform via numpy FFT (equivalent to scipy.signal.hilbert). "
        "Per-ticker time-series; no cross-sectional proxy required."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_hilbert_inst_freq_mean",   # mean instantaneous frequency (cycles / bar)
        "xdom_hilbert_inst_freq_std",    # std of inst. freq = bandwidth / stability proxy
        "xdom_hilbert_freq_irregularity",# std / (|mean| + eps) = normalised irregularity
    ],
    "tags": ["cross-domain", "signal-processing", "cycle", "hilbert", "dsp", "econophysics"],
    "version": "1.0.0",
    "author": (
        "Hilbert instantaneous frequency & bandwidth (analytic-signal demodulation); "
        "Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Spec SOURCE: xdom_hilbert_inst_freq."
    ),
}

# ---------------------------------------------------------------------------
# Numpy FFT-based Hilbert transform (no scipy dependency)
# ---------------------------------------------------------------------------

def _hilbert_np(x: np.ndarray) -> np.ndarray:
    """Return the analytic signal of real 1-D array x (complex).
    Follows the same algorithm as scipy.signal.hilbert:
      1. FFT of x
      2. Zero negative frequencies, double positive ones
      3. IFFT -> complex analytic signal
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=complex)
    X = np.fft.fft(x, n=n)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[1: n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def _inst_freq_stats(log_prices: np.ndarray) -> tuple[float, float]:
    """Compute (mean_inst_freq, std_inst_freq) for a window of log prices."""
    if len(log_prices) < 4:
        return np.nan, np.nan
    # Linear detrend
    n = len(log_prices)
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, log_prices, 1)
    detrended = log_prices - (slope * t + intercept)
    # Analytic signal
    analytic = _hilbert_np(detrended)
    # Instantaneous phase (unwrapped)
    phase = np.unwrap(np.angle(analytic))
    # Instantaneous frequency = phase derivative (cycles per bar / 2pi)
    inst_freq = np.diff(phase) / (2.0 * np.pi)
    if len(inst_freq) == 0:
        return np.nan, np.nan
    return float(np.mean(inst_freq)), float(np.std(inst_freq, ddof=0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    WINDOW = 64

    n = len(df)
    mean_freq = np.full(n, np.nan, dtype=float)
    std_freq  = np.full(n, np.nan, dtype=float)
    irreg     = np.full(n, np.nan, dtype=float)

    if n < WINDOW:
        df["xdom_hilbert_inst_freq_mean"]    = mean_freq
        df["xdom_hilbert_inst_freq_std"]     = std_freq
        df["xdom_hilbert_freq_irregularity"] = irreg
        return df

    # Pre-compute log close; guard non-positive prices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        close_arr = df["Close"].to_numpy(dtype=float)

    pos_mask = close_arr > 0
    log_close = np.where(pos_mask, np.log(np.where(pos_mask, close_arr, 1.0)), np.nan)

    # Use numpy sliding_window_view for efficiency; still O(n*W) but avoids
    # python loop overhead for the outer iteration (inner FFT is unavoidable).
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(log_close, window_shape=WINDOW)  # shape: (n-W+1, W)
    start = WINDOW - 1  # index into df where first full window ends

    for i, win in enumerate(windows):
        # If any NaN in window, skip
        if np.any(~np.isfinite(win)):
            continue
        mf, sf = _inst_freq_stats(win)
        idx = start + i
        mean_freq[idx] = mf
        std_freq[idx]  = sf
        if np.isfinite(mf) and np.isfinite(sf):
            denom = abs(mf) + 1e-10
            irreg[idx] = sf / denom

    df["xdom_hilbert_inst_freq_mean"]    = mean_freq
    df["xdom_hilbert_inst_freq_std"]     = std_freq
    df["xdom_hilbert_freq_irregularity"] = irreg

    return df
