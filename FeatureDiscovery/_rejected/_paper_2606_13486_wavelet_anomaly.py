"""
Analytic wavelet cross-resolution anomaly features derived from:
  "CRAFTIIF: Cross-Resolution Analytic Four-Type Interpretable Isolation Forest
   for Multivariate Time Series Anomaly Detection" (arXiv 2606.13486)

The paper uses K=500 random analytic wavelet draws from four families
(Morlet, DOG, Haar, Coiflet) targeting four anomaly types:
  1. Point/spike anomalies  (Haar-like: abrupt changes)
  2. Distributional shifts  (Coiflet-like: level/mean shifts)
  3. Temporal rhythm changes (Morlet: oscillation period changes)
  4. Collective/correlation breakdowns (DOG: derivative-of-Gaussian, curvature)

We implement a deterministic subset of these wavelet convolutions as
OHLCV features, measuring local anomaly scores across scales.
Features capture: spike magnitude z-score, level-shift score, rhythm
disruption (instantaneous frequency), and curvature anomaly.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13486_wavelet_anomaly",
    "description": (
        "Cross-resolution analytic wavelet anomaly scores (Morlet, DOG, Haar, Coiflet-style) "
        "for four anomaly types; based on arXiv 2606.13486 CRAFTIIF framework."
    ),
    "requires":    ["Close", "Volume", "High", "Low"],
    "produces": [
        "wav_haar_spike_8d",
        "wav_haar_spike_21d",
        "wav_dog_curv_13d",
        "wav_dog_curv_34d",
        "wav_morlet_rhythm_21d",
        "wav_coiflet_shift_21d",
        "wav_coiflet_shift_55d",
        "wav_cross_res_anomaly",
    ],
    "tags":        ["volatility", "anomaly", "spectral", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13486",
}


def _haar_spike_score(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Haar wavelet detail coefficient: difference of local means over two adjacent half-windows.
    Captures abrupt level jumps (point anomalies, regime breaks).
    Score = |coeff| / rolling_std (MAD-normalised anomaly score).
    """
    n = len(x)
    half = scale // 2
    out = np.full(n, np.nan)
    for i in range(scale - 1, n):
        left = x[i - scale + 1: i - half + 1]
        right = x[i - half + 1: i + 1]
        if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
            continue
        coeff = right.mean() - left.mean()
        out[i] = coeff
    coeff_s = pd.Series(out)
    roll_std = coeff_s.rolling(3 * scale, min_periods=scale).std()
    return (coeff_s / roll_std.replace(0, np.nan)).values


def _dog_curvature_score(x: np.ndarray, scale: int) -> np.ndarray:
    """
    DOG (Derivative-of-Gaussian) approximation: second difference (discrete Laplacian)
    smoothed at given scale.  Captures local curvature / inflection anomalies.
    Score is normalised by rolling std.
    """
    n = len(x)
    # Smooth with Gaussian-like kernel (rolling mean) then second-difference
    ser = pd.Series(x)
    smoothed = ser.rolling(scale, min_periods=scale // 2, center=False).mean()
    # Second derivative approximation
    curv = smoothed.diff().diff()
    roll_std = curv.rolling(3 * scale, min_periods=scale).std()
    return (curv / roll_std.replace(0, np.nan)).values


def _morlet_rhythm_score(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Morlet wavelet approximation via rolling autocorrelation at lag ~scale/4.
    Morlet captures oscillation amplitude changes (rhythm/periodicity disruption).
    A sudden drop in local autocorrelation = rhythm change.
    """
    n = len(x)
    ser = pd.Series(x)
    # Local autocorrelation at lag = scale//4 over rolling window = scale
    lag = max(1, scale // 4)
    # Rolling corr between x and x.shift(lag)
    shifted = ser.shift(lag)
    roll_corr = ser.rolling(scale, min_periods=scale // 2).corr(shifted)
    # First diff of autocorrelation = disruption signal
    rhythm_disruption = roll_corr.diff()
    roll_std = rhythm_disruption.rolling(3 * scale, min_periods=scale).std()
    return (rhythm_disruption / roll_std.replace(0, np.nan)).values


def _coiflet_shift_score(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Coiflet-style level-shift detector: comparison of rolling mean vs longer baseline.
    Coiflets have vanishing moments (flat frequency response at zero) → ideal for
    detecting distributional shifts (level changes in local mean).
    Score = (short_mean - long_mean) / long_std.
    """
    ser = pd.Series(x)
    short_mean = ser.rolling(scale, min_periods=scale // 2).mean()
    long_mean = ser.rolling(scale * 4, min_periods=scale).mean()
    long_std = ser.rolling(scale * 4, min_periods=scale).std()
    shift = (short_mean - long_mean) / long_std.replace(0, np.nan)
    return shift.values


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    log_ret = np.log(close / close.shift(1)).values.astype(np.float64)

    # ── 1. Haar spike (point anomalies) ─────────────────────────────────────
    df["wav_haar_spike_8d"] = _haar_spike_score(log_ret, 8)
    df["wav_haar_spike_21d"] = _haar_spike_score(log_ret, 21)

    # ── 2. DOG curvature (collective/correlation anomalies) ──────────────────
    df["wav_dog_curv_13d"] = _dog_curvature_score(log_ret, 13)
    df["wav_dog_curv_34d"] = _dog_curvature_score(log_ret, 34)

    # ── 3. Morlet rhythm disruption (temporal anomalies) ─────────────────────
    df["wav_morlet_rhythm_21d"] = _morlet_rhythm_score(log_ret, 21)

    # ── 4. Coiflet level shift (distributional anomalies) ────────────────────
    df["wav_coiflet_shift_21d"] = _coiflet_shift_score(log_ret, 21)
    df["wav_coiflet_shift_55d"] = _coiflet_shift_score(log_ret, 55)

    # ── 5. Cross-resolution compound anomaly: max abs across all 4 types ────
    types = pd.DataFrame({
        "h8": df["wav_haar_spike_8d"].abs(),
        "h21": df["wav_haar_spike_21d"].abs(),
        "d13": df["wav_dog_curv_13d"].abs(),
        "d34": df["wav_dog_curv_34d"].abs(),
        "m21": df["wav_morlet_rhythm_21d"].abs(),
        "c21": df["wav_coiflet_shift_21d"].abs(),
        "c55": df["wav_coiflet_shift_55d"].abs(),
    })
    # Compound: mean of rank across anomaly types (0=no anomaly of any type, 1=extreme everywhere)
    ranks = types.rank(pct=True)
    df["wav_cross_res_anomaly"] = ranks.mean(axis=1)

    return df
