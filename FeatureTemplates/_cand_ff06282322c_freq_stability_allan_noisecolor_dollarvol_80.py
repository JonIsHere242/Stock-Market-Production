from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80",
    "description": (
        "Allan-deviation noise-color slope on log dollar-volume over an 80-bar window. "
        "Computes Allan deviation at short tau (avg of tau=1,2) and long tau (avg of tau=8,16), "
        "then takes log(sigma_short / sigma_long). Positive = white/blue noise (short-tau dominated); "
        "negative = red/persistent noise (long-tau dominated). Stride i%5==0 from series start, "
        "forward-filled. Per-ticker proxy; causal, no lookahead."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_slope",
        "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_short",
        "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_long",
    ],
    "tags": ["frequency", "allan_deviation", "noise_color", "dollar_volume", "stability"],
    "version": "1.0",
    "author": "feature-factory ff06282322c",
}

_COL_SLOPE = "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_slope"
_COL_SHORT = "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_short"
_COL_LONG  = "ff06282322c_freq_stability_allan_noisecolor_dollarvol_80_long"

_WINDOW = 80
_STRIDE = 5
_TAU_SHORT = (1, 2)
_TAU_LONG  = (8, 16)


def _allan_dev(x: np.ndarray, tau: int) -> float:
    """Allan deviation for a 1-D array x and averaging time tau."""
    # Allan variance = mean of squared successive differences of tau-length averages
    # = (1/(2*(N-tau))) * sum( (avg[i+tau] - avg[i])^2 )
    # We use overlapping averages for efficiency but keep causal (within window).
    n = len(x)
    if n < 2 * tau + 1:
        return np.nan
    # Cumulative sum for block averages
    cs = np.cumsum(x)
    # block average starting at index i of length tau: (cs[i+tau] - cs[i]) / tau
    # valid i from tau to n-tau (inclusive end = n-tau, gives pair average at i+tau)
    # pairs: avg_a = block [i, i+tau), avg_b = block [i+tau, i+2*tau)
    end = n - 2 * tau
    if end < 1:
        return np.nan
    # avg_b[i] - avg_a[i] for i in [0, end)
    # avg_a[i] = (cs[i+tau] - cs[i]) / tau
    # avg_b[i] = (cs[i+2*tau] - cs[i+tau]) / tau
    i_arr = np.arange(end)
    avg_a = (cs[i_arr + tau] - cs[i_arr]) / tau
    avg_b = (cs[i_arr + 2 * tau] - cs[i_arr + tau]) / tau
    diff_sq = (avg_b - avg_a) ** 2
    return float(np.sqrt(0.5 * np.mean(diff_sq)))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN
    df[_COL_SLOPE] = np.nan
    df[_COL_SHORT] = np.nan
    df[_COL_LONG]  = np.nan

    n = len(df)
    if n < _WINDOW:
        return df

    # Build log dollar-volume; guard non-positive values
    close  = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    dv = close * volume
    # Replace non-positive with NaN before log
    with np.errstate(invalid="ignore", divide="ignore"):
        log_dv = np.where(dv > 0.0, np.log(dv), np.nan)

    # Arrays to fill
    slope_arr = np.full(n, np.nan)
    short_arr = np.full(n, np.nan)
    long_arr  = np.full(n, np.nan)

    # Compute on stride grid anchored from series start (i % 5 == 0)
    for i in range(_WINDOW - 1, n):
        if i % _STRIDE != 0:
            continue
        window = log_dv[i - _WINDOW + 1 : i + 1]  # length == WINDOW
        # Skip if too many NaNs (require at least WINDOW/2 valid)
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < _WINDOW // 2:
            continue
        # Use only valid portion for Allan computation (fill-forward NaNs within window)
        # Simple approach: replace NaN with linearly interpolated / ffill within window
        w = window.copy()
        # forward-fill NaNs
        last_valid = np.nan
        for k in range(len(w)):
            if np.isnan(w[k]):
                w[k] = last_valid  # may remain nan at start
            else:
                last_valid = w[k]
        # If still NaN at start, backward fill
        last_valid = np.nan
        for k in range(len(w) - 1, -1, -1):
            if np.isnan(w[k]):
                w[k] = last_valid
            else:
                last_valid = w[k]
        if np.any(np.isnan(w)):
            continue

        # Allan deviation for each tau, then average short and long groups
        ad_short_vals = [_allan_dev(w, t) for t in _TAU_SHORT]
        ad_long_vals  = [_allan_dev(w, t) for t in _TAU_LONG]

        ad_short_vals = [v for v in ad_short_vals if not np.isnan(v)]
        ad_long_vals  = [v for v in ad_long_vals  if not np.isnan(v)]

        if not ad_short_vals or not ad_long_vals:
            continue

        sigma_short = float(np.mean(ad_short_vals))
        sigma_long  = float(np.mean(ad_long_vals))

        short_arr[i] = sigma_short
        long_arr[i]  = sigma_long

        if sigma_short > 0.0 and sigma_long > 0.0:
            slope_arr[i] = np.log(sigma_short / sigma_long)

    # Forward-fill stride gaps (causal)
    def _ffill(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        last = np.nan
        for k in range(len(out)):
            if not np.isnan(out[k]):
                last = out[k]
            else:
                out[k] = last
        return out

    df[_COL_SLOPE] = _ffill(slope_arr)
    df[_COL_SHORT] = _ffill(short_arr)
    df[_COL_LONG]  = _ffill(long_arr)

    return df
