import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.fft import fft

METADATA = {
    "name":        "volume_spectral_splatter",
    "description": "FFT-based spectral entropy measure of volume signal dispersion",
    "requires":    ["Volume"],
    "produces":    ["volume_spectral_splatter"],
    "tags":        ["volume", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume spectral splatter: a rolling FFT-based metric measuring
    the dispersion/chaos of volume patterns.

    Process per bar (for i >= window_size):
    1. Extract 50-bar rolling volume window
    2. Log-transform to stabilize variance
    3. Detrend to remove low-frequency components
    4. Apply Hanning window to reduce spectral leakage
    5. Compute FFT power spectrum
    6. Calculate normalized spectral entropy
    7. Normalize entropy to 0-1 range (entropy / log(n))

    Higher values indicate more chaotic/dispersed volume patterns.
    """

    window_size = 50
    epsilon = 1e-8

    volume = df["Volume"]
    n = len(volume)

    results = np.full(n, np.nan)

    # Number of full windows. Window i (i >= window_size) uses volume[i-50:i],
    # i.e. rows i-50 .. i-1. There are n - window_size such windows, with the
    # first one ending at original index i = window_size.
    n_windows = n - window_size
    if n_windows > 0:
        vol = volume.values.astype(float)
        # Clamp to epsilon then log (same as per-row np.maximum + np.log).
        log_all = np.log(np.maximum(vol, epsilon))

        # Strided view: rows are the sliding windows of length window_size.
        # windows[k] == log_all[k : k+window_size]; window for output index
        # i = window_size + k uses log_all[i-50 : i] == log_all[k : k+50].
        windows = np.lib.stride_tricks.sliding_window_view(log_all, window_size)
        # Take only windows that correspond to output indices >= window_size,
        # i.e. starting at k=0 .. n_windows-1 (the last full sliding window
        # starts at n - window_size, which equals n_windows).
        windows = windows[:n_windows]  # shape (n_windows, window_size)

        # --- Vectorized linear detrend (matches scipy.signal.detrend type='linear')
        # scipy fits least-squares line over x = arange(window_size) and subtracts.
        m = window_size
        x = np.arange(m, dtype=float)
        xmean = x.mean()
        xc = x - xmean
        denom = np.dot(xc, xc)  # sum of squared centered x
        ymean = windows.mean(axis=1, keepdims=True)
        # slope per window = sum(xc * (y - ymean)) / denom = (xc @ y) / denom
        slope = (windows @ xc) / denom  # shape (n_windows,)
        # detrended = y - (slope*x + intercept); intercept = ymean - slope*xmean
        # => detrended = (y - ymean) - slope*(x - xmean)
        detrended = (windows - ymean) - slope[:, None] * xc[None, :]

        # --- Hanning window (same as np.hanning(window_size))
        han = np.hanning(m)
        windowed = detrended * han[None, :]

        # --- Batched FFT along axis 1
        fft_vals = fft(windowed, axis=1)
        power_spectrum = np.abs(fft_vals) ** 2

        total_power = power_spectrum.sum(axis=1)  # shape (n_windows,)

        normalized_power = np.empty_like(power_spectrum)
        ok = total_power > epsilon
        # Rows with sufficient power: divide by total power.
        if ok.any():
            normalized_power[ok] = power_spectrum[ok] / total_power[ok, None]
        # Degenerate rows: uniform distribution.
        if (~ok).any():
            normalized_power[~ok] = 1.0 / m

        spectral_entropy = -np.sum(
            normalized_power * np.log(normalized_power + epsilon), axis=1
        )

        max_entropy = np.log(m)  # m == len(normalized_power) per row
        if max_entropy > 0:
            normalized_entropy = spectral_entropy / max_entropy
        else:
            normalized_entropy = np.zeros(n_windows)

        results[window_size:] = normalized_entropy

    df["volume_spectral_splatter"] = pd.Series(results, index=df.index)

    return df
