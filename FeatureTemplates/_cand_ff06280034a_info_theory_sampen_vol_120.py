"""
Rolling Sample Entropy of log-Volume first-differences.

SampEn(m=2, r=0.2*std) measures the unpredictability / complexity of the
volume-change process over a 120-day window.  High entropy = irregular /
hard-to-predict volume dynamics.  Low entropy = stereotyped / regime-locked
volume behaviour.

Per-ticker proxy: faithfully implements SampEn per stock in rolling fashion.
Because O(n^2) per window is too slow, the block runs on a fixed-from-start
stride grid (every 5 bars from i=0) and forward-fills between grid points,
which is causal and passes the truncation test.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06280034a_info_theory_sampen_vol_120",
    "description": (
        "Rolling 120-bar Sample Entropy (SampEn m=2, r=0.2*std) of the "
        "first-differences of log(Volume).  Measures complexity / irregularity "
        "of the volume-change process. Produces: level, 20-bar change, 60-bar "
        "z-score.  Computed on a causal fixed-from-start grid (stride 5) then "
        "forward-filled to avoid O(n^2) cost and maintain causality under "
        "truncation."
    ),
    "requires": ["Volume"],
    "produces": [
        "ff06280034a_info_theory_sampen_vol_120_level",
        "ff06280034a_info_theory_sampen_vol_120_change20",
        "ff06280034a_info_theory_sampen_vol_120_zscore60",
    ],
    "tags": ["info_theory", "entropy", "volume", "complexity", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory/ff06280034a",
}

_WIN = 120
_M = 2
_R_FACTOR = 0.2
_STRIDE = 5
_CHANGE_LAG = 20
_ZSCORE_WIN = 60


def _sampen_m2(x: np.ndarray, r: float) -> float:
    """
    Sample entropy SampEn(m=2, r) for a 1-D array x.
    B = number of template-match pairs of length m=2
    A = number of template-match pairs of length m+1=3
    SampEn = -ln(A/B); returns np.nan if B==0.
    Uses Chebyshev (max-norm) distance, self-matches excluded.
    """
    n = len(x)
    if n < 4:
        return np.nan

    # Build template matrix for m=2 and m+1=3 at once
    # templates of length 3 (longest needed): shape (n-2, 3)
    n3 = n - _M  # n - 2
    if n3 < 2:
        return np.nan

    # use numpy broadcasting; for n=120 this is 118×118 = ~14k comparisons
    # shape (n3, 3)
    templates = np.lib.stride_tricks.sliding_window_view(x, window_shape=3)
    # templates[:, :2] -> m=2 portion; templates[:, :3] -> m+1=3 portion

    tm2 = templates[:, :2]  # (n3, 2)
    tm3 = templates          # (n3, 3)

    B_count = 0
    A_count = 0
    for i in range(n3 - 1):
        # Chebyshev distance from template i to all j>i
        diff2 = np.max(np.abs(tm2[i+1:] - tm2[i]), axis=1)
        match2 = diff2 <= r
        b = int(match2.sum())
        B_count += b
        if b > 0:
            diff3 = np.max(np.abs(tm3[i+1:] - tm3[i]), axis=1)
            match3 = diff3 <= r
            A_count += int(match3.sum())

    if B_count == 0:
        return np.nan
    if A_count == 0:
        return np.nan  # infinite entropy; return nan to avoid -inf
    return -np.log(A_count / B_count)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_level = "ff06280034a_info_theory_sampen_vol_120_level"
    col_chg = "ff06280034a_info_theory_sampen_vol_120_change20"
    col_z = "ff06280034a_info_theory_sampen_vol_120_zscore60"

    # Pre-initialise all produced columns to NaN (required on every code path)
    df[col_level] = np.nan
    df[col_chg] = np.nan
    df[col_z] = np.nan

    n = len(df)
    if n < _WIN + 1:
        return df

    vol = df["Volume"].values.astype(np.float64)

    # Guard: replace non-positive volume with NaN before log
    vol = np.where(vol > 0, vol, np.nan)

    log_vol = np.log(vol)
    # First differences of log(Volume)
    dlog = np.diff(log_vol, prepend=np.nan)  # length n, dlog[0]=nan

    # Compute SampEn on causal fixed-from-start grid
    # Grid indices: bars where (i % _STRIDE == 0), i measured from 0
    # We need at least _WIN bars ending at index i, so i >= _WIN - 1
    level_arr = np.full(n, np.nan)

    for i in range(n):
        if i % _STRIDE != 0:
            continue
        if i < _WIN - 1:
            continue
        # Window: indices [i - _WIN + 1 .. i] inclusive
        window = dlog[i - _WIN + 1 : i + 1]
        # Drop NaN values within window
        valid = window[~np.isnan(window)]
        if len(valid) < _M + 2:
            continue
        r_thresh = _R_FACTOR * np.std(valid, ddof=0)
        if r_thresh <= 0:
            continue
        level_arr[i] = _sampen_m2(valid, r_thresh)

    # Forward-fill between stride grid points (causal: only past grid value)
    # Use pandas ffill
    level_series = pd.Series(level_arr)
    level_series = level_series.ffill()
    level_arr = level_series.values

    df[col_level] = level_arr

    # 20-bar change (difference)
    level_s = pd.Series(level_arr, index=df.index)
    df[col_chg] = level_s.diff(_CHANGE_LAG)

    # 60-bar z-score of the level
    roll = level_s.rolling(_ZSCORE_WIN, min_periods=max(10, _ZSCORE_WIN // 4))
    mu = roll.mean()
    sigma = roll.std(ddof=0)
    z = (level_s - mu) / sigma.replace(0, np.nan)
    # Guard inf
    z = z.replace([np.inf, -np.inf], np.nan)
    df[col_z] = z.values

    return df
