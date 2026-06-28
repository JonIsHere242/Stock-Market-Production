"""
xdom_spectral_entropy — Spectral entropy / Wiener entropy on rolling returns.

Per-ticker proxy for a cross-sectional signal-processing concept.
On a rolling 64-day window of log returns, compute the normalized power
spectral density (squared rFFT magnitudes / sum), then Shannon spectral
entropy = -sum(p * log(p)) / log(K).

High entropy = white/flat spectrum (no dominant cycle, random-walk-like).
Low entropy  = concentrated power (a dominant oscillation / trend).

Also produces a 20-day EMA of the entropy level as a smoothed variant,
and a slope (diff over 5 bars) to capture directional change in spectral
character.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_spectral_entropy",
    "description": (
        "Rolling 64-day spectral entropy of log returns (per-ticker). "
        "Normalized power spectral density via rFFT -> Shannon entropy. "
        "High = flat/random spectrum; Low = dominant cycle/trend. "
        "Cross-sectional ranking is inherently lost; this is a faithful "
        "per-ticker time-series proxy of the same economic signal."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_spectral_entropy_64",       # raw normalized spectral entropy
        "xdom_spectral_entropy_ema20",    # 20-bar EMA smoothed
        "xdom_spectral_entropy_slope5",   # 5-bar difference (momentum of entropy)
    ],
    "tags": ["cross-domain", "spectral", "entropy", "econophysics", "dsp", "information-theory"],
    "version": "1.0",
    "author": (
        "Spec: Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "Spectral entropy / Wiener entropy (information theory on the periodogram)"
    ),
}

_WINDOW = 64
_MIN_PERIODS = 32  # require at least half window to emit a value


def _spectral_entropy(returns_arr: np.ndarray) -> float:
    """
    Compute normalized Shannon spectral entropy for a 1-D array of returns.
    Returns NaN if data is insufficient or degenerate.
    """
    n = len(returns_arr)
    if n < 4:
        return np.nan

    # rFFT — only positive frequencies (DC + unique freqs)
    fft_vals = np.fft.rfft(returns_arr)
    power = (fft_vals.real ** 2 + fft_vals.imag ** 2)  # squared magnitudes

    total_power = power.sum()
    if total_power == 0.0 or not np.isfinite(total_power):
        return np.nan

    # Normalized PSD (probability distribution over frequencies)
    psd_norm = power / total_power

    # Number of frequency bins = K
    K = len(psd_norm)
    if K < 2:
        return np.nan

    # Shannon entropy: -sum(p * log(p)), ignoring zero-probability bins
    log_K = np.log(float(K))
    if log_K == 0.0:
        return np.nan

    # Mask zeros to avoid log(0)
    mask = psd_norm > 0.0
    entropy = -np.sum(psd_norm[mask] * np.log(psd_norm[mask])) / log_K

    if not np.isfinite(entropy):
        return np.nan

    return float(entropy)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- log returns (shift(1) uses only past data, no lookahead) ---
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    n = len(df)
    entropy_vals = np.full(n, np.nan)

    # Rolling window: for each position t, use returns[t-WINDOW+1 : t+1]
    # We need at least _MIN_PERIODS valid (non-NaN) returns in the window.
    # Using a numpy sliding approach for speed; scipy.signal not in allowlist
    # so we use np.fft directly (already imported via numpy).

    ret_arr = log_ret.to_numpy(dtype=np.float64)

    for t in range(n):
        start = max(0, t - _WINDOW + 1)
        window_data = ret_arr[start : t + 1]

        # Filter NaNs
        valid = window_data[np.isfinite(window_data)]
        if len(valid) < _MIN_PERIODS:
            continue

        entropy_vals[t] = _spectral_entropy(valid)

    # Assign raw entropy
    df["xdom_spectral_entropy_64"] = entropy_vals

    # EMA-20 smoothed version (exponential moving average of the entropy series)
    ent_series = pd.Series(entropy_vals, index=df.index)
    df["xdom_spectral_entropy_ema20"] = (
        ent_series.ewm(span=20, min_periods=5, adjust=False).mean()
    )

    # 5-bar slope (difference)
    df["xdom_spectral_entropy_slope5"] = ent_series.diff(5)

    return df
