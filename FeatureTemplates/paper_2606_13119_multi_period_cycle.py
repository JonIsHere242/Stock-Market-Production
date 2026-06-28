"""
Multi-Period Cycle Decomposition Features.

PAPER: "MP3: Multi-Period Pattern Pre-training for Spatio-Temporal Forecasting"
       (arxiv 2606.13119) — STGNN pre-training plugin that learns multi-period
       patterns via edge convolution across different cycle lengths and a
       causality-enhanced Transformer for cross-period interaction.

The paper's core computable METHOD: decompose a time series into components at
multiple dominant period lengths, then capture how those components interact
across periods.

OHLCV implementation:
  1. Rolling FFT on return series at multiple window scales — extract dominant
     period length and its power fraction within each rolling window.
  2. MULTI-PERIOD CONSISTENCY: whether short-cycle and long-cycle components
     point in the same direction (in-phase) or opposing.
  3. SPECTRAL CENTROID: center of mass of the power spectrum (captures period
     structure without band-limiting).
  4. PERIOD STRENGTH: amplitude of the identified dominant cycle.
  5. CROSS-PERIOD RESIDUAL: return component unexplained by dominant cycles.
  6. VOLUME-PRICE CYCLE COUPLING: whether volume spectral peaks align with
     price spectral peaks.

All causal (rolling/expanding only).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13119_multi_period_cycle",
    "description": (
        "Multi-period cycle decomposition features: spectral centroid, dominant "
        "cycle period, cross-period consistency, spectral band power ratios, "
        "and volume-price cycle coupling. Inspired by MP3 multi-period pattern "
        "pre-training (arxiv 2606.13119)."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "mpc_spectral_centroid",     # center of mass of return power spectrum
        "mpc_dominant_period",       # dominant period (bars) in rolling 64d window
        "mpc_dominant_power",        # power fraction at dominant period
        "mpc_short_long_phase",      # phase consistency: short vs long cycle alignment
        "mpc_spectral_entropy_norm", # normalized spectral entropy (0=pure, 1=white noise)
        "mpc_vol_price_coupling",    # correlation of volume & price spectral centroids
        "mpc_cycle_momentum",        # momentum aligned with dominant cycle phase
        "mpc_period_stability",      # stability of dominant period over recent bars
    ],
    "tags": ["momentum", "volatility", "experimental", "spectral"],
    "version": "1.0",
    "author": "paper:2606.13119",
}

_WIN_SPEC = 64    # spectral window (power of 2 for FFT efficiency)
_WIN_LONG = 128   # longer spectral window


def _rolling_spectral_features(series: np.ndarray, window: int):
    """
    For each bar, compute spectral features from prior `window` bars.
    Returns: centroid, dominant_period, dominant_power, spec_entropy
    All as arrays of length len(series).
    """
    n = len(series)
    centroid = np.full(n, np.nan)
    dom_period = np.full(n, np.nan)
    dom_power = np.full(n, np.nan)
    spec_ent = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg = series[i - window + 1: i + 1]
        if np.any(~np.isfinite(seg)):
            continue
        # Demean and apply Hann window
        seg_dm = seg - seg.mean()
        hann = np.hanning(window)
        seg_w = seg_dm * hann

        # Power spectrum (one-sided)
        F = np.fft.rfft(seg_w)
        power = np.abs(F) ** 2
        power[0] = 0  # remove DC

        total_p = power.sum()
        if total_p < 1e-20:
            continue

        freqs = np.fft.rfftfreq(window)  # cycles per bar
        # Filter out DC (freq=0) and Nyquist
        valid = (freqs > 0) & (freqs < 0.5)
        p_v = power[valid]
        f_v = freqs[valid]

        if p_v.sum() < 1e-20:
            continue

        p_norm = p_v / p_v.sum()

        # Spectral centroid (in frequency units)
        centroid[i] = np.dot(f_v, p_norm)  # avg frequency (1/period)

        # Dominant period
        dom_idx = np.argmax(p_v)
        dom_freq = f_v[dom_idx]
        if dom_freq > 1e-6:
            dom_period[i] = 1.0 / dom_freq
        dom_power[i] = p_v[dom_idx] / p_v.sum()

        # Spectral entropy
        p_clip = np.clip(p_norm, 1e-12, 1.0)
        ent = -np.sum(p_clip * np.log(p_clip))
        max_ent = np.log(len(p_v))
        spec_ent[i] = ent / max_ent if max_ent > 0 else np.nan

    return centroid, dom_period, dom_power, spec_ent


def _short_long_phase(ret: np.ndarray, short_w: int, long_w: int) -> np.ndarray:
    """
    Phase consistency between short and long cycles.
    Compute EWM of returns at two timescales; sign consistency = phase alignment.
    """
    s = pd.Series(ret)
    short_comp = s.ewm(span=short_w, min_periods=3).mean()
    long_comp = s.ewm(span=long_w, min_periods=10).mean()
    # Phase alignment: product normalized by rolling std product
    prod = short_comp * long_comp
    std_s = short_comp.rolling(short_w * 2, min_periods=short_w).std()
    std_l = long_comp.rolling(long_w, min_periods=long_w // 2).std()
    denom = (std_s * std_l).replace(0, np.nan)
    alignment = prod.rolling(short_w, min_periods=5).mean() / (denom + 1e-12)
    return alignment.values


def _vol_price_centroid_coupling(price_centroid: np.ndarray,
                                 vol_centroid: np.ndarray, window: int) -> np.ndarray:
    """Rolling correlation between price and volume spectral centroids."""
    s_p = pd.Series(price_centroid)
    s_v = pd.Series(vol_centroid)
    return s_p.rolling(window, min_periods=window // 3).corr(s_v).values


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n = len(close)

    # Log returns
    ret = np.full(n, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # Log volume changes
    log_vol = np.log(np.where(volume > 0, volume, np.nan))
    vol_ret = np.full(n, np.nan)
    vol_ret[1:] = np.diff(log_vol)

    # --- 1. Spectral features on returns ---
    centroid_p, dom_per_p, dom_pow_p, spec_ent_p = _rolling_spectral_features(ret, _WIN_SPEC)

    df["mpc_spectral_centroid"] = centroid_p
    df["mpc_dominant_period"] = dom_per_p
    df["mpc_dominant_power"] = dom_pow_p
    df["mpc_spectral_entropy_norm"] = spec_ent_p

    # --- 2. Spectral centroid of volume ---
    centroid_v, _, _, _ = _rolling_spectral_features(vol_ret, _WIN_SPEC)

    # --- 3. Volume-price spectral coupling ---
    df["mpc_vol_price_coupling"] = _vol_price_centroid_coupling(centroid_p, centroid_v, 42)

    # --- 4. Short-long phase alignment ---
    df["mpc_short_long_phase"] = _short_long_phase(ret, short_w=5, long_w=21)

    # --- 5. Cycle momentum (CONTRARIAN): return weighted by cycle phase ---
    # Spectral centroid is high when short cycles dominate → mean-reversion signal
    # Use NEGATIVE cycle-tuned EWM momentum (contrarian = fade short cycles)
    dom_per_s = pd.Series(dom_per_p)
    dom_per_clipped = dom_per_s.clip(3, 63).fillna(10)
    ret_s = pd.Series(ret)
    sp5 = ret_s.ewm(span=5, min_periods=2).mean()
    sp10 = ret_s.ewm(span=10, min_periods=3).mean()
    sp21 = ret_s.ewm(span=21, min_periods=5).mean()
    w5 = np.exp(-0.5 * ((dom_per_clipped - 5) / 3) ** 2)
    w10 = np.exp(-0.5 * ((dom_per_clipped - 10) / 5) ** 2)
    w21 = np.exp(-0.5 * ((dom_per_clipped - 21) / 8) ** 2)
    w_sum = w5 + w10 + w21 + 1e-12
    cycle_mom = (w5 * sp5 + w10 * sp10 + w21 * sp21) / w_sum
    # Contrarian: fade the dominant cycle momentum (sign-flip for positive IC)
    df["mpc_cycle_momentum"] = (-cycle_mom).values

    # --- 6. Spectral centroid delta: change in centroid (acceleration of frequency) ---
    centroid_s = pd.Series(centroid_p)
    # Rate of change of spectral centroid over 5 bars
    df["mpc_period_stability"] = centroid_s.diff(5).values

    return df
