"""
Pattern Originality Features  —  arxiv:2606.12260
"Market Design for AI: Beyond the Copyright Binary"

The paper studies "originality penalty": innovative creators are harder to
model and get undercompensated. Translated to OHLCV: how "original" or
distinctive is the current price-action pattern vs the stock's own recent
history? Highly novel patterns may signal regime breaks or unusual events.

Method:
- Represent each bar as a normalized OHLCV feature vector
- Compute the self-distance (L2 or cosine) between the current window
  snapshot and the rolling historical distribution of snapshots
- High distance = "original" / anomalous bar; low = homogenized / typical

Concretely:
  - At each time t, extract a 5-element standardized primitive vector:
    [body_ratio, upper_wick, lower_wick, vol_zscore, range_pct]
  - Compare vs the mean and covariance of the last W snapshots
  - Mahalanobis-like distance (diagonal covariance for speed) = "originality"
  - Also: cosine similarity to rolling centroid, and rolling novelty z-score

Produces 8 columns prefixed "orig_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12260_pattern_originality",
    "description": (
        "Pattern originality / self-distance features inspired by arxiv:2606.12260 — "
        "measures how distinctive the current OHLCV bar pattern is vs the stock's "
        "own recent history using rolling normalized distance metrics."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "orig_diag_mahal_21d",
        "orig_diag_mahal_42d",
        "orig_diag_mahal_63d",
        "orig_cosine_dist_21d",
        "orig_cosine_dist_63d",
        "orig_novelty_z21",
        "orig_novelty_z63",
        "orig_composite",
    ],
    "tags": ["experimental", "market_regime", "volatility"],
    "version": "1.0",
    "author": "paper:2606.12260",
}


def _build_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """5-dimensional OHLCV primitive vector per bar (standardized)."""
    C = df["Close"].astype(np.float64)
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    V = df["Volume"].astype(np.float64)

    hl = (H - L).replace(0, np.nan)
    body_ratio = (C - O) / hl
    upper_wick = (H - np.maximum(O, C)) / hl
    lower_wick = (np.minimum(O, C) - L) / hl
    range_pct = hl / C
    vol_log = np.log(V.replace(0, np.nan))

    # Build raw feature frame
    prims = pd.DataFrame({
        "body": body_ratio,
        "upper": upper_wick,
        "lower": lower_wick,
        "range": range_pct,
        "vol": vol_log,
    }, index=df.index)
    return prims


def compute(df: pd.DataFrame) -> pd.DataFrame:
    prims = _build_primitives(df)
    n_features = prims.shape[1]
    P = prims.to_numpy()  # (N, 5)
    N = len(P)

    # Initialise output arrays
    mahal_21 = np.full(N, np.nan)
    mahal_42 = np.full(N, np.nan)
    mahal_63 = np.full(N, np.nan)
    cos_21 = np.full(N, np.nan)
    cos_63 = np.full(N, np.nan)

    # Pre-compute rolling mean and std per feature dimension using pandas
    for w, mahal_arr, cos_arr in [
        (21, mahal_21, cos_21),
        (42, mahal_42, None),
        (63, mahal_63, cos_63),
    ]:
        for feat_idx in range(n_features):
            pass  # handled below via vectorized rolling

    # Vectorized: use pandas rolling on each column, then compute distance
    # to the rolling mean — diagonal Mahalanobis = sum((x-mu)^2 / sigma^2)
    for w, mahal_arr, cos_arr in [
        (21, mahal_21, cos_21),
        (42, mahal_42, None),
        (63, mahal_63, cos_63),
    ]:
        roll_mean = prims.rolling(w, min_periods=w).mean()  # (N, 5)
        roll_std = prims.rolling(w, min_periods=w).std().clip(lower=1e-9)  # (N, 5)

        mu = roll_mean.to_numpy()
        sigma = roll_std.to_numpy()
        p = P  # current bar (already in P)

        # Use the current bar vs the rolling window mean (ending at t)
        # Shift by 1 so we compare current x[t] vs history ending at t-1
        mu_lag = np.roll(mu, 1, axis=0)
        mu_lag[0] = np.nan
        sigma_lag = np.roll(sigma, 1, axis=0)
        sigma_lag[0] = np.nan

        diff = p - mu_lag  # (N, 5)
        # Diagonal Mahalanobis^2
        mah_sq = np.nansum((diff / sigma_lag) ** 2, axis=1)  # (N,)
        valid = np.sum(np.isfinite(diff / sigma_lag), axis=1) >= 3
        mah_sq = np.where(valid, mah_sq, np.nan)
        mahal_arr[:] = np.sqrt(np.where(mah_sq >= 0, mah_sq, np.nan))

        # Cosine distance: 1 - cosine_similarity(p, mu)
        if cos_arr is not None:
            p_norm = np.sqrt(np.nansum(p ** 2, axis=1)) + 1e-9
            mu_norm = np.sqrt(np.nansum(mu_lag ** 2, axis=1)) + 1e-9
            dot = np.nansum(p * mu_lag, axis=1)
            cos_sim = dot / (p_norm * mu_norm)
            valid_cos = np.isfinite(cos_sim)
            cos_arr[:] = np.where(valid_cos, 1.0 - cos_sim, np.nan)

    df["orig_diag_mahal_21d"] = mahal_21
    df["orig_diag_mahal_42d"] = mahal_42
    df["orig_diag_mahal_63d"] = mahal_63
    df["orig_cosine_dist_21d"] = cos_21
    df["orig_cosine_dist_63d"] = cos_63

    # Novelty z-scores: z-score the mahal series over a longer window
    m21 = pd.Series(mahal_21, index=df.index)
    m63 = pd.Series(mahal_63, index=df.index)

    df["orig_novelty_z21"] = (m21 - m21.rolling(63, min_periods=21).mean()) / (
        m21.rolling(63, min_periods=21).std() + 1e-9
    )
    df["orig_novelty_z63"] = (m63 - m63.rolling(126, min_periods=42).mean()) / (
        m63.rolling(126, min_periods=42).std() + 1e-9
    )

    # Composite: average of the two novelty z-scores
    df["orig_composite"] = (df["orig_novelty_z21"] + df["orig_novelty_z63"]) / 2.0

    return df
