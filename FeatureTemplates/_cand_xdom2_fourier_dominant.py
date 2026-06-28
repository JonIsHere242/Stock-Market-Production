"""
Candidate feature block: xdom2_fourier_dominant
Rolling dominant-cycle extraction via periodogram on detrended returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_fourier_dominant",
    "description": (
        "Rolling 96-day periodogram of detrended log-returns. "
        "xdom2_fourier_dominant_period: period (in days) of the highest-power "
        "frequency bin, excluding the DC component. "
        "xdom2_fourier_dominant_strength: fraction of total spectral power held "
        "by that peak bin (dominant-cycle concentration). "
        "xdom2_fourier_dominant_period_chg: signed change in dominant period vs "
        "20 bars ago (captures regime shifts in cycle length). "
        "Per-ticker proxy -- inherently single-stock spectral analysis, "
        "no cross-sectional component needed. "
        "Detrending = subtract linear OLS trend from the 96-bar return window "
        "before applying rfft, so the DC bin is clean."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_fourier_dominant_period",
        "xdom2_fourier_dominant_strength",
        "xdom2_fourier_dominant_period_chg",
    ],
    "tags": ["spectral", "cycle", "fourier", "cross-domain", "price"],
    "version": "1.0",
    "author": "Cross-domain / practitioner method transfer (batch 2) — dominant cycle via periodogram",
}

_WINDOW = 96        # periodogram window length
_MIN_OBS = 32       # require at least this many non-NaN bars
_CHGLOOK = 20       # lookback for period-change slope variant


def _dominant_cycle(prices: np.ndarray) -> tuple[float, float]:
    """
    Given a 1-D array of prices of length == _WINDOW, compute:
      - dominant period (days) of the peak power bin (excluding DC / freq=0)
      - strength = peak power / total power (excluding DC)

    Returns (np.nan, np.nan) on any failure.
    """
    n = len(prices)
    if n < _MIN_OBS:
        return np.nan, np.nan

    # Log-returns
    lp = np.log(prices)
    rets = np.diff(lp)  # length n-1
    m = len(rets)
    if m < _MIN_OBS:
        return np.nan, np.nan

    # Linear detrend: subtract least-squares line
    x = np.arange(m, dtype=np.float64)
    xm = x - x.mean()
    rm = rets - rets.mean()
    slope = (xm * rm).sum() / (xm * xm).sum() if (xm * xm).sum() != 0.0 else 0.0
    intercept = rets.mean() - slope * x.mean()
    detrended = rets - (slope * x + intercept)

    # rfft -- frequencies 0 .. m//2
    fft_out = np.fft.rfft(detrended)
    power = (fft_out.real ** 2 + fft_out.imag ** 2)  # |FFT|^2, length m//2+1

    # Exclude DC bin (index 0)
    ac_power = power[1:]          # indices 1..m//2
    if len(ac_power) == 0 or ac_power.sum() == 0.0:
        return np.nan, np.nan

    peak_idx = int(np.argmax(ac_power))   # index into ac_power
    freq_idx = peak_idx + 1               # actual frequency index (1-based)

    # Frequency in cycles-per-sample = freq_idx / m  =>  period = m / freq_idx
    period = float(m) / float(freq_idx)
    strength = float(ac_power[peak_idx]) / float(ac_power.sum())

    return period, strength


def compute(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["Close"].to_numpy(dtype=np.float64)
    n = len(closes)

    periods = np.full(n, np.nan)
    strengths = np.full(n, np.nan)

    # Rolling window: need _WINDOW prices to get _WINDOW-1 returns
    for i in range(_WINDOW - 1, n):
        window = closes[i - _WINDOW + 1: i + 1]
        if np.any(np.isnan(window)) or np.any(window <= 0.0):
            continue
        p, s = _dominant_cycle(window)
        periods[i] = p
        strengths[i] = s

    df["xdom2_fourier_dominant_period"] = periods
    df["xdom2_fourier_dominant_strength"] = strengths

    # Period change: current period minus period _CHGLOOK bars ago
    period_series = pd.Series(periods, index=df.index)
    df["xdom2_fourier_dominant_period_chg"] = period_series - period_series.shift(_CHGLOOK)

    return df
