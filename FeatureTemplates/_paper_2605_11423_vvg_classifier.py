"""
Volatility-Volume-Gap (VVG) classifier features derived from:
  "A Validated Volatility-Volume-Gap Classifier for Regime Identification
   in MNQ Intraday Data" (arXiv 2605.11423).

The paper uses three pre-market observable conditions on MNQ futures daily bars:
  1. First-30-minute return magnitude (morning drift strength)
  2. Overnight gap magnitude (open vs prior close)
  3. Abnormal opening-bar volume relative to rolling baseline

On daily OHLCV we can faithfully approximate (2) and (3); (1) is approximated
by the open-to-high/open-to-low intraday range as a morning volatility proxy.
Classifier-positive days (all three conditions triggered) exhibit directional
morning drift followed by late-session reversal.

Features produced:
  - vvg_gap_mag:     |Open_t / Close_{t-1} - 1| (overnight gap magnitude)
  - vvg_open_range:  (High - Low) / Open (intraday range as morning drift proxy)
  - vvg_vol_ratio:   Volume_t / rolling_mean_volume (abnormal volume flag)
  - vvg_composite:   z-score aggregate of all three (rolling expanding normalization)
  - vvg_signal:      1 if all three exceed rolling 67th pct (classifier-positive)
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2605_11423_vvg_classifier",
    "description": (
        "Volatility-Volume-Gap classifier: overnight gap magnitude, open-to-range "
        "morning proxy, and abnormal volume ratio combined into a composite regime "
        "signal; daily OHLCV proxy for arXiv 2605.11423 VVG intraday framework."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "vvg_gap_mag",
        "vvg_open_range",
        "vvg_vol_ratio",
        "vvg_composite",
        "vvg_signal",
    ],
    "tags":        ["volatility", "volume", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "paper:2605.11423",
}

_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n      = len(df)
    close  = df["Close"].values.astype(np.float64)
    open_  = df["Open"].values.astype(np.float64)
    high   = df["High"].values.astype(np.float64)
    low    = df["Low"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    # --- (1) Overnight gap magnitude: |Open_t / Close_{t-1} - 1| ---
    gap_mag = np.full(n, np.nan)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    gap_mag = np.abs(open_ / np.maximum(prev_close, _EPS) - 1.0)
    gap_mag[0] = np.nan

    # --- (2) Open-to-high/low range (morning drift proxy) ---
    open_range = (high - low) / np.maximum(open_, _EPS)

    # --- (3) Abnormal volume: ratio vs rolling 20-bar mean ---
    vol_ratio = np.full(n, np.nan)
    vol_window = 20
    for i in range(vol_window, n):
        past_vol = volume[i - vol_window: i]  # exclude current bar (no lookahead)
        mean_vol = np.nanmean(past_vol)
        if mean_vol > _EPS:
            vol_ratio[i] = volume[i] / mean_vol

    # --- Expanding rolling normalization of each component ---
    # Use expanding window so thresholds are computed from past data only
    gap_z    = np.full(n, np.nan)
    range_z  = np.full(n, np.nan)
    volrat_z = np.full(n, np.nan)

    min_hist = 30
    for i in range(min_hist, n):
        # Gap mag
        past_g = gap_mag[: i]
        past_g = past_g[~np.isnan(past_g)]
        if len(past_g) >= 10:
            m, s = past_g.mean(), past_g.std()
            if s > _EPS and not np.isnan(gap_mag[i]):
                gap_z[i] = (gap_mag[i] - m) / s

        # Open range
        past_r = open_range[: i]
        past_r = past_r[~np.isnan(past_r)]
        if len(past_r) >= 10:
            m, s = past_r.mean(), past_r.std()
            if s > _EPS and not np.isnan(open_range[i]):
                range_z[i] = (open_range[i] - m) / s

        # Vol ratio
        past_v = vol_ratio[: i]
        past_v = past_v[~np.isnan(past_v)]
        if len(past_v) >= 10:
            m, s = past_v.mean(), past_v.std()
            if s > _EPS and not np.isnan(vol_ratio[i]):
                volrat_z[i] = (vol_ratio[i] - m) / s

    # Composite: equal-weight average of z-scores
    composite = np.full(n, np.nan)
    for i in range(n):
        vals = [gap_z[i], range_z[i], volrat_z[i]]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if len(valid_vals) == 3:
            composite[i] = np.mean(valid_vals)

    # VVG signal: 1 if all three z-scores exceed 0.43 (approx 67th pct of std normal)
    # (threshold calibrated to mimic the paper's upper-tercile cut for each component)
    threshold = 0.43
    vvg_signal = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(gap_z[i]) or np.isnan(range_z[i]) or np.isnan(volrat_z[i])):
            vvg_signal[i] = float(
                gap_z[i] > threshold and
                range_z[i] > threshold and
                volrat_z[i] > threshold
            )

    df["vvg_gap_mag"]    = gap_mag
    df["vvg_open_range"] = open_range
    df["vvg_vol_ratio"]  = vol_ratio
    df["vvg_composite"]  = composite
    df["vvg_signal"]     = vvg_signal

    return df
