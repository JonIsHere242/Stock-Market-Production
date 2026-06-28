"""
Magnitude-based complexity features derived from:
  "Magnitude-Based Features for Multispecies Spatial Data" (arxiv 2606.11775)

The paper uses the *magnitude* of a finite metric space — a real-valued invariant
that measures "effective number of distinct points" incorporating spatial
configuration and scale — as a feature.

We treat a trailing window of daily OHLCV observations as a finite metric space
in ℝ^2 (log_ret, hl_range), and compute its magnitude at multiple scales t.

Magnitude of X = {x_1,...,x_n} at scale t:
    Mag(t·X) = 1ᵀ ζ(t)    where  Z(t)_{ij} = exp(-t·‖xᵢ−xⱼ‖)  and  Zζ = 1

Key insight: the profile Mag(t) across scales encodes effective dimensionality
and clustering of the point-cloud without needing a label.

FAST IMPLEMENTATION: window=10, k=2 features.
Each solve is 10×10, giving <1ms per row.  All windows batched using strides.

Features:
  mag_scale1    — magnitude at t=1 (coarse scale)
  mag_scale5    — magnitude at t=5 (fine scale)
  mag_ratio     — mag_scale5 / mag_scale1
  mag_delta     — 5-day difference in mag_scale5 (complexity trend)
  mag_pctrank   — percentile rank of mag_scale5 in trailing 60 rows
  mag_entropy   — entropy of normalised magnitude-vector at t=2
"""

import numpy as np
import pandas as pd
from scipy import linalg as la

METADATA = {
    "name": "paper_2606_11775_magnitude_complexity",
    "description": (
        "Magnitude of rolling OHLCV point-cloud at multiple length scales — "
        "measures effective structural complexity; from arxiv 2606.11775 "
        "(Magnitude-Based Features for Multispecies Spatial Data)."
    ),
    "requires": ["High", "Low", "Close", "Volume"],
    "produces": [
        "mag_scale1",
        "mag_scale5",
        "mag_ratio",
        "mag_delta",
        "mag_pctrank",
        "mag_entropy",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11775",
}

_W  = 10    # rolling window
_K  = 2     # feature dimensions
_R  = 1e-5  # ridge for solve


def _batch_magnitude_and_entropy(feat: np.ndarray) -> tuple:
    """
    Compute magnitude at t in {1, 2, 5} and entropy at t=2
    for all (n - W + 1) rolling windows using stride trick.

    feat: (n, K) float array, already normalised, no NaN
    Returns: m1, m5, ent  each of length (n - W + 1)
    """
    n = len(feat)
    B = n - _W + 1
    if B <= 0:
        empty = np.array([])
        return empty, empty, empty

    # Build (B, W, K) windows via strides
    s0, s1 = feat.strides
    wins = np.lib.stride_tricks.as_strided(
        feat.copy(),    # must copy for safe striding
        shape=(B, _W, _K),
        strides=(s0, s0, s1),
    )

    ones = np.ones(_W)
    m1   = np.empty(B)
    m5   = np.empty(B)
    ent  = np.empty(B)

    eye_r = _R * np.eye(_W)

    for i in range(B):
        pts = wins[i]                                   # (W, K)
        diff = pts[:, None, :] - pts[None, :, :]        # (W, W, K)
        D = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))  # (W, W)

        # ---- t = 1 ---
        Z1 = np.exp(-D) + eye_r
        try:
            z1 = la.solve(Z1, ones, assume_a='pos', check_finite=False)
            m1[i] = z1.sum()
        except la.LinAlgError:
            m1[i] = float(_W)

        # ---- t = 5 ---
        Z5 = np.exp(-5.0 * D) + eye_r
        try:
            z5 = la.solve(Z5, ones, assume_a='pos', check_finite=False)
            m5[i] = z5.sum()
        except la.LinAlgError:
            m5[i] = float(_W)

        # ---- entropy at t = 2 ---
        Z2 = np.exp(-2.0 * D) + eye_r
        try:
            z2 = la.solve(Z2, ones, assume_a='pos', check_finite=False)
            pos = np.clip(z2, 1e-12, None)
            p = pos / pos.sum()
            ent[i] = -np.sum(p * np.log(p + 1e-12))
        except la.LinAlgError:
            ent[i] = np.log(_W)

    return m1, m5, ent


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret  = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
        hl_range = (high - low) / np.where(close > 0, close, np.nan)

    feat_raw = np.column_stack([log_ret, hl_range])
    feat_raw = np.where(np.isfinite(feat_raw), feat_raw, 0.0)

    # Normalise by global std so distance scale is meaningful
    gscale = np.std(feat_raw, axis=0)
    gscale = np.where(gscale < 1e-9, 1.0, gscale)
    feat = feat_raw / gscale      # (n, 2)

    # ---- Compute batch magnitude --------------------------------------------
    m1_raw, m5_raw, ent_raw = _batch_magnitude_and_entropy(feat)

    # Map back to length-n arrays (first W-1 = NaN)
    def _pad(arr):
        out = np.full(n, np.nan)
        out[_W - 1:] = arr
        return out

    mag1 = _pad(m1_raw)
    mag5 = _pad(m5_raw)
    ent  = _pad(ent_raw)

    df["mag_scale1"] = mag1
    df["mag_scale5"] = mag5
    df["mag_ratio"]  = np.where(
        np.isfinite(mag1) & (mag1 > 1e-6), mag5 / mag1, np.nan
    )

    # 5-day delta of mag_scale5
    s5 = pd.Series(mag5, index=df.index)
    df["mag_delta"] = s5 - s5.shift(5)

    # Vectorised percentile rank in trailing 60 rows
    mag5_arr = mag5
    pct = np.full(n, np.nan)
    for i in range(_W, n):
        if not np.isfinite(mag5_arr[i]):
            continue
        start = max(0, i - 60 + 1)
        hist = mag5_arr[start:i]
        fin  = hist[np.isfinite(hist)]
        if len(fin) < 5:
            continue
        pct[i] = float(np.sum(fin < mag5_arr[i])) / len(fin)
    df["mag_pctrank"] = pct

    df["mag_entropy"] = ent

    return df
