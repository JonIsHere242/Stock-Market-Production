"""
Spectral centroid & bandwidth of rolling return periodogram.

Rolling 64-day periodogram of log-returns; computes:
  - spectral_centroid: frequency 'center of mass' (sum(freq*power)/sum(power))
  - spectral_bandwidth: spread around centroid (sqrt(sum(power*(freq-centroid)^2)/sum(power)))

High centroid -> return energy at high frequencies (choppy, mean-reverting regime).
Low centroid  -> return energy at low frequencies (trending regime).
bandwidth measures how concentrated vs dispersed the spectral energy is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_spectral_centroid",
    "description": (
        "Per-ticker rolling 64-day periodogram of log-returns. "
        "spectral_centroid_64 = sum(freq*power)/sum(power) is the 'center of mass' "
        "of return energy in frequency space; high values indicate choppy/high-freq "
        "return structure, low values indicate trending/low-freq structure. "
        "spectral_bandwidth_64 = sqrt(sum(power*(freq-centroid)^2)/sum(power)) measures "
        "dispersion of spectral energy. "
        "spectral_centroid_delta = 10-day change in centroid captures shifts in regime. "
        "Faithful per-ticker proxy: cross-sectional ranking not available in this block; "
        "signal captures same economic content (trend vs chop) on a per-stock basis."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_spectral_centroid_64",
        "xdom_spectral_bandwidth_64",
        "xdom_spectral_centroid_delta",
    ],
    "tags": ["spectral", "cross-domain", "dsp", "regime", "frequency"],
    "version": "1.0",
    "author": "Spectral centroid & bandwidth (audio DSP timbre features); cross-domain method transfer from signal processing / econophysics / HRV / DSP",
}

_WINDOW = 64
_DELTA = 10


def _rolling_spectral(log_ret: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling spectral centroid and bandwidth over a 1-D log-return array.

    Returns two arrays of the same length (NaN for the first window-1 entries).
    Uses a sliding window and real FFT; only positive frequencies are used.
    """
    n = len(log_ret)
    centroid = np.full(n, np.nan)
    bandwidth = np.full(n, np.nan)

    # Precompute frequency bins for a window of length `window`
    # rfft gives window//2 + 1 non-negative freqs; freq[0] is DC (0).
    # We exclude DC (index 0) to avoid anchoring centroid at zero when series is flat.
    freqs = np.fft.rfftfreq(window)  # shape: (window//2 + 1,)
    # exclude DC bin (freq=0) to keep centroid meaningful
    freq_idx = slice(1, None)
    f = freqs[freq_idx]  # positive frequencies only

    if len(f) == 0:
        return centroid, bandwidth

    from numpy.lib.stride_tricks import sliding_window_view  # numpy >= 1.20

    if n < window:
        return centroid, bandwidth

    windows = sliding_window_view(log_ret, window_shape=window)  # shape: (n-window+1, window)
    # rfft along last axis
    fft_vals = np.fft.rfft(windows, axis=1)  # shape: (n-window+1, window//2+1)
    power = (np.abs(fft_vals[:, freq_idx]) ** 2)  # shape: (n-window+1, len(f))

    total_power = power.sum(axis=1)  # (n-window+1,)
    safe_total = np.where(total_power == 0, np.nan, total_power)

    # centroid: weighted mean frequency
    c = (power * f[np.newaxis, :]).sum(axis=1) / safe_total  # (n-window+1,)

    # bandwidth: weighted std of frequencies around centroid
    diff = f[np.newaxis, :] - c[:, np.newaxis]  # (n-window+1, len(f))
    bw = np.sqrt((power * diff ** 2).sum(axis=1) / safe_total)  # (n-window+1,)

    # Place results at the LAST index of each window (no lookahead)
    start = window - 1
    centroid[start:] = c
    bandwidth[start:] = bw

    return centroid, bandwidth


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)

    # log-returns; first entry is NaN
    log_ret = np.empty(len(close))
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(np.where(close[:-1] == 0, np.nan, close[1:] / close[:-1]))

    centroid, bandwidth = _rolling_spectral(log_ret, _WINDOW)

    # Replace inf/-inf with NaN (guard)
    centroid = np.where(np.isfinite(centroid), centroid, np.nan)
    bandwidth = np.where(np.isfinite(bandwidth), bandwidth, np.nan)

    # Delta: change in centroid over _DELTA days
    delta = np.full(len(close), np.nan)
    if len(centroid) > _DELTA:
        delta[_DELTA:] = centroid[_DELTA:] - centroid[:-_DELTA]
    delta = np.where(np.isfinite(delta), delta, np.nan)

    df["xdom_spectral_centroid_64"] = centroid
    df["xdom_spectral_bandwidth_64"] = bandwidth
    df["xdom_spectral_centroid_delta"] = delta

    return df
