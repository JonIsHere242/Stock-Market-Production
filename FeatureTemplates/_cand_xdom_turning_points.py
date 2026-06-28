"""
xdom_turning_points — Turning-point rate vs IID expectation (Kendall randomness test).

Per-ticker: counts local maxima/minima (turning points) in Close over a rolling window,
standardises vs the IID expectation (Kendall 1945), and also provides a shorter window
and a slow-moving Z-score of Z-scores for multi-scale context.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_turning_points",
    "description": (
        "Rolling turning-point rate vs IID expectation (Kendall 1945 randomness test). "
        "A turning point occurs at index i when Close[i-1]<Close[i]>Close[i+1] (local max) "
        "or Close[i-1]>Close[i]<Close[i+1] (local min). "
        "Under IID, expected count in window n is mu=2(n-2)/3, variance=sigma^2=(16n-29)/90. "
        "xdom_turning_points_z60: Z-score (60-day window). Positive Z = more oscillation than "
        "random (choppy/mean-reverting price); negative Z = fewer turning points (trending). "
        "xdom_turning_points_z20: same test on 20-day window (faster). "
        "xdom_turning_points_z60_ma: 10-day rolling mean of z60 (smoothed regime signal). "
        "Per-ticker proxy — faithful implementation of the cross-domain DSP/econophysics method."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_turning_points_z60",
        "xdom_turning_points_z20",
        "xdom_turning_points_z60_ma",
    ],
    "tags": ["cross-domain", "turning-point", "randomness", "kendall", "regime", "oscillation"],
    "version": "1.0.0",
    "author": (
        "Spec: Turning-point rate vs IID expectation (Kendall time-series randomness); "
        "Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Kendall (1945) turning-point test. Implemented per-ticker from OHLCV Close."
    ),
}


def _tp_z(close_vals: np.ndarray, w: int) -> np.ndarray:
    """
    For each end-position t compute the Z-score of turning-point count in the
    window close_vals[t-w+1 : t+1].

    A turning point at interior index i (1 <= i <= w-2) means:
        (x[i-1] < x[i] > x[i+1])  OR  (x[i-1] > x[i] < x[i+1])
    Ties are NOT turning points (strict inequalities).

    E[T] = 2*(n-2)/3     where n = window length
    Var[T] = (16*n - 29) / 90

    Returns array of length len(close_vals), NaN for first (w-1) positions.
    """
    n = len(close_vals)
    out = np.full(n, np.nan)

    if w < 3:
        return out

    mu = 2.0 * (w - 2) / 3.0
    var = (16.0 * w - 29.0) / 90.0
    if var <= 0.0:
        return out
    sigma = np.sqrt(var)

    # Precompute turning-point indicator for each interior position (global array)
    # tp_flag[i] = 1 if close_vals[i] is a turning point (needs i-1 and i+1)
    # We'll use a sliding-sum approach:
    # Build tp flags for indices 1..n-2, then use sliding window sum.

    x = close_vals
    # tp_flag[i] = 1 if x[i-1]<x[i]>x[i+1] or x[i-1]>x[i]<x[i+1], for i in [1, n-2]
    # shape: length n, but valid only for indices 1..n-2
    tp_flag = np.zeros(n, dtype=np.float64)
    if n >= 3:
        prev_ = x[:-2]   # x[i-1], i=1..n-2
        curr_ = x[1:-1]  # x[i],   i=1..n-2
        next_ = x[2:]    # x[i+1], i=1..n-2
        local_max = (curr_ > prev_) & (curr_ > next_)
        local_min = (curr_ < prev_) & (curr_ < next_)
        tp_flag[1:-1] = (local_max | local_min).astype(np.float64)

    # Cumulative sum for O(n) rolling sum
    cum = np.concatenate(([0.0], np.cumsum(tp_flag)))
    # For window [t-w+1, t], interior turning-point indices are [t-w+2, t-1]
    # (the first and last of the window cannot be turning points in Kendall sense)
    # In the global tp_flag the valid interior positions are indices [t-w+2 .. t-1]
    # sum = cum[t] - cum[t-w+2]  ... but we need to be careful with boundaries
    # Actually the rolling sum of tp_flag[t-w+1 .. t] covers interior indices
    # correctly because tp_flag is 0 at 0 and n-1 by construction, so:
    # sum_tp = cum[t+1] - cum[t-w+1]
    # This counts tp_flag for indices t-w+1 .. t, which = tp_flag for interior
    # positions of the window (the endpoints have 0 by construction only if they
    # happen to be at global position 0 or n-1; but interior window endpoints
    # may have non-zero tp_flag if they are local extrema relative to the GLOBAL
    # neighbors, not the window neighbors).
    #
    # Correction: the window endpoints (positions t-w+1 and t in global index)
    # do NOT count as turning points for THIS window (no left/right neighbor
    # within window). We must zero them out for the window sum.
    # Since we built tp_flag based on global neighbors, the endpoints of any
    # sub-window may have been assigned 1 if they are turning points globally,
    # but they shouldn't count for THIS sub-window (they lack one neighbor).
    # However, for large w, the endpoints rarely change the count significantly
    # and the standard Kendall formula applies to the interior.
    # We handle this by subtracting tp_flag at the two window endpoints:

    for t in range(w - 1, n):
        left = t - w + 1
        right = t
        # Sum of tp_flag for indices left+1 .. right-1 (interior of window)
        # = cum[right] - cum[left+1]
        count = cum[right] - cum[left + 1]
        # Also exclude tp_flag at right itself from the window sum
        # (cum[right+1]-cum[left+1] would include right; we use cum[right]-cum[left+1]
        # which already excludes index right — correct)
        z = (count - mu) / sigma
        out[t] = z

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)

    # 60-day turning-point Z
    z60 = _tp_z(close, w=60)

    # 20-day turning-point Z
    z20 = _tp_z(close, w=20)

    # Guard against inf (shouldn't happen but be safe)
    z60 = np.where(np.isfinite(z60), z60, np.nan)
    z20 = np.where(np.isfinite(z20), z20, np.nan)

    df["xdom_turning_points_z60"] = z60
    df["xdom_turning_points_z20"] = z20

    # Smoothed regime signal: 10-day rolling mean of z60
    df["xdom_turning_points_z60_ma"] = (
        df["xdom_turning_points_z60"].rolling(10, min_periods=5).mean()
    )

    return df
