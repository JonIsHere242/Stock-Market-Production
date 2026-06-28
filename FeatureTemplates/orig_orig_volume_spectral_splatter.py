import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import detrend

METADATA = {
    "name":        "orig_orig_volume_spectral_splatter",
    "description": "Spectral entropy (splatter) of detrended log-volume over a rolling FFT window -- faithful AlphaSensitivity port",
    "requires":    [],
    "produces": [
        "Volume_Spectral_Splatter",
    ],
    "tags": [
        "volume",
        "spectral",
        "fft",
        "alphasens_port",
    ],
    "version": "1.0",
    "author":  "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    window = 50
    epsilon = 1e-8

    def calculate_spectral_splatter(volume_series, window_size):
        """Calculate spectral splatter for a volume series."""
        results = []

        for i in range(len(volume_series)):
            if i < window_size:
                results.append(np.nan)
                continue

            # Get window of volume data
            vol_window = volume_series.iloc[i - window_size:i].values

            # Handle zero/negative volumes
            vol_window = np.maximum(vol_window, epsilon)

            # Log transform to stabilize variance
            log_vol = np.log(vol_window)

            # Remove trend (detrend)
            detrended = detrend(log_vol)

            # Apply window function to reduce spectral leakage
            windowed = detrended * np.hanning(len(detrended))

            # Compute FFT
            fft_vals = fft(windowed)
            power_spectrum = np.abs(fft_vals) ** 2

            # Normalize power spectrum
            total_power = np.sum(power_spectrum)
            if total_power > epsilon:
                normalized_power = power_spectrum / total_power
            else:
                normalized_power = np.ones_like(power_spectrum) / len(power_spectrum)

            # Calculate spectral entropy (measure of signal dispersion)
            # Higher entropy = more dispersed/chaotic signal
            spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + epsilon))

            # Normalize to 0-1 range approximately
            max_entropy = np.log(len(normalized_power))
            if max_entropy > 0:
                normalized_entropy = spectral_entropy / max_entropy
            else:
                normalized_entropy = 0

            results.append(normalized_entropy)

        return pd.Series(results, index=volume_series.index)

    # Calculate spectral splatter
    df['Volume_Spectral_Splatter'] = calculate_spectral_splatter(df['Volume'], window)

    return df
