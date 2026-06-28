"""
Hjorth Mobility on log-volume (signal-complexity of the volume process).

Hjorth (1970) descriptors are used in EEG/HRV signal processing to characterise
the "busyness" or dominant frequency of a time-series without fitting a model.
Applied per-ticker to rolling log-volume, mobility captures how rapidly trading
activity changes direction — orthogonal to simple volume z-scores or trends.

Per-ticker proxy: the block operates on each individual stock's volume series,
so it is inherently per-ticker rather than cross-sectional.  The cross-sectional
predicted sign is left blank in the spec; this block produces a raw descriptor
and a short-term slope for downstream use in models / rubrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_hjorth_volume",
    "description": (
        "Hjorth Mobility descriptor applied to a rolling 40-day window of "
        "log(Volume) changes (detrended within each window). "
        "mobility = sqrt(var(diff(logvol_window)) / var(detrended_logvol_window)). "
        "A complexity measure of trading-activity dynamics, orthogonal to simple "
        "volume z-scores.  Produces: "
        "xdom_hjorth_volume_mob40 (raw mobility, 40d window), "
        "xdom_hjorth_volume_mob40_z (z-score of mobility over trailing 60 bars), "
        "xdom_hjorth_volume_mob40_slope (5d slope of mobility). "
        "Per-ticker proxy — same economic signal as cross-sectional Hjorth but "
        "computed on each stock's own volume series."
    ),
    "requires": ["Volume"],
    "produces": [
        "xdom_hjorth_volume_mob40",
        "xdom_hjorth_volume_mob40_z",
        "xdom_hjorth_volume_mob40_slope",
    ],
    "tags": ["volume", "complexity", "cross-domain", "hjorth", "signal-processing"],
    "version": "1.0",
    "author": (
        "Hjorth, B. (1970). EEG analysis based on time domain properties. "
        "Electroencephalography and Clinical Neurophysiology, 29(3), 306-310. "
        "Cross-domain transfer to log-volume per spec xdom_hjorth_volume."
    ),
}

_WINDOW = 40       # Hjorth mobility window (spec)
_Z_WINDOW = 60     # rolling normalisation window
_SLOPE_LAG = 5     # slope lookback (bars)
_MIN_OBS = 10      # minimum non-NaN obs inside a window to emit a value


def _hjorth_mobility_1d(logvol_window: np.ndarray) -> float:
    """Compute Hjorth mobility for a single window (1-D numpy array).

    1. Linearly detrend the window in-place (removes drift so variance is
       centred; equivalent to the original Hjorth 'activity' baseline).
    2. mobility = sqrt( var(diff(w)) / var(w_detrended) )

    Returns np.nan if variance is zero or window is too short.
    """
    n = len(logvol_window)
    # remove linear trend (index 0..n-1)
    x = np.arange(n, dtype=np.float64)
    # least-squares fit: slope & intercept
    xm = x.mean()
    ym = logvol_window.mean()
    ss_xx = ((x - xm) ** 2).sum()
    if ss_xx == 0.0:
        return np.nan
    slope = ((x - xm) * (logvol_window - ym)).sum() / ss_xx
    intercept = ym - slope * xm
    detrended = logvol_window - (slope * x + intercept)

    var_w = np.var(detrended, ddof=1)
    if var_w == 0.0 or np.isnan(var_w):
        return np.nan

    diffs = np.diff(detrended)
    var_d = np.var(diffs, ddof=1)
    if np.isnan(var_d):
        return np.nan

    return float(np.sqrt(var_d / var_w))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- log-volume (guard against zero / negative volume) ---
    vol = df["Volume"].values.astype(np.float64)
    vol_safe = np.where(vol > 0, vol, np.nan)
    logvol = np.log(vol_safe)           # NaN where volume == 0

    n = len(logvol)
    mob = np.full(n, np.nan, dtype=np.float64)

    if n >= _WINDOW:
        # Use numpy stride tricks for a fast rolling view
        from numpy.lib.stride_tricks import sliding_window_view
        views = sliding_window_view(logvol, window_shape=_WINDOW)
        # views shape: (n - WINDOW + 1, WINDOW); first valid output index = WINDOW - 1
        for i, w in enumerate(views):
            valid = w[~np.isnan(w)]
            if len(valid) < _MIN_OBS:
                continue
            # If there are NaNs in window, work with valid subset only
            # (detrend over the full window if no NaNs, partial otherwise)
            if len(valid) == _WINDOW:
                mob[i + _WINDOW - 1] = _hjorth_mobility_1d(w)
            else:
                mob[i + _WINDOW - 1] = _hjorth_mobility_1d(valid)

    # --- z-score of mobility over trailing _Z_WINDOW bars ---
    mob_s = pd.Series(mob, index=df.index)
    roll = mob_s.rolling(window=_Z_WINDOW, min_periods=max(5, _Z_WINDOW // 4))
    mob_mean = roll.mean()
    mob_std = roll.std(ddof=1)
    # guard zero std
    mob_std_safe = mob_std.where(mob_std > 0, other=np.nan)
    mob_z = (mob_s - mob_mean) / mob_std_safe

    # --- short-term slope (5-bar) ---
    mob_slope = mob_s.diff(_SLOPE_LAG) / _SLOPE_LAG

    df["xdom_hjorth_volume_mob40"] = mob
    df["xdom_hjorth_volume_mob40_z"] = mob_z.values
    df["xdom_hjorth_volume_mob40_slope"] = mob_slope.values

    return df
