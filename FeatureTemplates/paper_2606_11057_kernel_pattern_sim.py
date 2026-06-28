"""
Kernel Pattern Similarity Features  —  arxiv:2606.11057
"Flexible Kernels for Protein Property Prediction"

The paper uses RBF-style kernels with evolutionary substitution matrices to
capture sequence similarity in protein property space. The key insight:
structure-aware importance weighting makes kernels more discriminative than
raw Euclidean distance.

Applied to OHLCV: use kernel functions to measure similarity between the
current bar's feature vector and the rolling historical centroid.
The "substitution matrix" analog = inverse-variance weighting of feature
dimensions (dimensions with low historical variance = conserved = higher weight).

Implementation (fully vectorized via pandas rolling):
  1. Construct per-bar 6D feature vector [log_ret, range_pct, body, wicks×2, vol_z]
  2. RBF kernel vs rolling centroid: exp(-gamma * ||x - mu_hist||^2)
     with adaptive gamma = 1/(D * rolling_mean_variance)
  3. Variance-weighted RBF (substitution-matrix kernel): down-weight
     high-variance dims (conservative alignment = stronger signal)
  4. Polynomial kernel: (x_norm . mu_norm + c)^2 on L2-normalized vectors
  5. Laplacian kernel: exp(-gamma_L * ||x - mu_hist||_1)
     with gamma_L = 1/(D * rolling_std proxy for L1 scale)
  6. Anomaly score: 1 - RBF similarity

All kernels compare x[t] vs rolling_mean[t-w..t-1] — fully causal.

Produces 8 columns prefixed "kps_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11057_kernel_pattern_sim",
    "description": (
        "Kernel pattern similarity features inspired by arxiv:2606.11057 — "
        "RBF, variance-weighted RBF, polynomial, and Laplacian kernel similarities "
        "between the current OHLCV bar and its rolling historical centroid. "
        "Adaptive bandwidth, substitution-matrix importance weighting. Vectorized."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "kps_rbf_sim_21d",
        "kps_rbf_sim_63d",
        "kps_poly_sim_21d",
        "kps_laplace_sim_21d",
        "kps_vw_rbf_21d",
        "kps_vw_rbf_63d",
        "kps_anomaly_21d",
        "kps_composite",
    ],
    "tags": ["experimental", "market_regime", "volatility"],
    "version": "1.0",
    "author": "paper:2606.11057",
}


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """6D OHLCV feature matrix (returns DataFrame for easy rolling)."""
    C = df["Close"].astype(np.float64)
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    V = df["Volume"].astype(np.float64)

    hl = (H - L).replace(0, np.nan)
    log_ret = np.log(C / C.shift(1))
    range_pct = hl / C
    body = (C - O) / hl
    upper_wick = (H - np.maximum(O, C)) / hl
    lower_wick = (np.minimum(O, C) - L) / hl
    vol_log = np.log(V.replace(0, np.nan))
    vol_z = (vol_log - vol_log.rolling(21, min_periods=5).mean()) / (
        vol_log.rolling(21, min_periods=5).std() + 1e-9
    )

    return pd.DataFrame({
        "lr": log_ret,
        "rp": range_pct,
        "bd": body,
        "uw": upper_wick,
        "lw": lower_wick,
        "vz": vol_z,
    }, index=df.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    feats = _build_features(df)  # (N, 6)
    D = feats.shape[1]

    for w, suffix in [(21, "21d"), (63, "63d")]:
        mp = max(5, w // 3)

        # Rolling statistics (lagged = causal: history ending at t-1)
        roll_mean = feats.rolling(w, min_periods=mp).mean().shift(1)
        roll_var = feats.rolling(w, min_periods=mp).var().shift(1).clip(lower=1e-9)

        diff = feats - roll_mean  # (N, 6)

        # Adaptive gamma: 1 / (D * mean_variance)
        mean_var = roll_var.mean(axis=1).replace(0, 1e-9)
        gamma = 1.0 / (D * mean_var)

        # RBF kernel: exp(-gamma * ||diff||^2)
        sq_dist = (diff ** 2).sum(axis=1)
        rbf = np.exp(-gamma * sq_dist)
        valid = diff.notna().all(axis=1)
        df[f"kps_rbf_sim_{suffix}"] = rbf.where(valid)

        # Variance-weighted RBF (substitution matrix: conserved dims get higher weight)
        # Weight = 1/var, then weighted sum of squared deviations
        inv_var = 1.0 / roll_var
        w_sum = inv_var.sum(axis=1).replace(0, 1e-9)
        # Normalize weights to sum to D (same scale as unweighted)
        inv_var_norm = inv_var.multiply(D / w_sum, axis=0)
        wdiff_sq = ((diff ** 2) * inv_var_norm).sum(axis=1)
        vw_gamma = gamma  # same adaptive gamma scale
        df[f"kps_vw_rbf_{suffix}"] = np.exp(-vw_gamma * wdiff_sq / D).where(valid)

    # Polynomial kernel at w=21: (x_norm . mu_norm + 0.5)^2
    w = 21
    mp = max(5, w // 3)
    roll_mean_21 = feats.rolling(w, min_periods=mp).mean().shift(1)
    diff_21 = feats - roll_mean_21
    valid_21 = diff_21.notna().all(axis=1)

    x_norm = np.sqrt((feats ** 2).sum(axis=1)).clip(lower=1e-9)
    mu_norm = np.sqrt((roll_mean_21 ** 2).sum(axis=1)).clip(lower=1e-9)
    dot = (feats * roll_mean_21).sum(axis=1)
    cos_sim = dot / (x_norm * mu_norm)
    df["kps_poly_sim_21d"] = ((cos_sim + 0.5) ** 2).where(valid_21)

    # Laplacian kernel at w=21: exp(-gamma_L * L1_dist)
    # Use rolling std as L1-scale proxy (simpler than MAD, avoids apply())
    roll_std_21 = feats.rolling(w, min_periods=mp).std().shift(1).clip(lower=1e-9)
    mean_std = roll_std_21.mean(axis=1).replace(0, 1e-9)
    gamma_l = 1.0 / (D * mean_std)
    l1_dist = diff_21.abs().sum(axis=1)
    df["kps_laplace_sim_21d"] = np.exp(-gamma_l * l1_dist).where(valid_21)

    # Anomaly: 1 - RBF similarity (how unlike historical centroid)
    df["kps_anomaly_21d"] = (1.0 - df["kps_rbf_sim_21d"]).clip(lower=0, upper=1)

    # Composite: z-score rbf_sim and vw_rbf, average
    def _z(s, wz=126, mp=42):
        return (s - s.rolling(wz, min_periods=mp).mean()) / (
            s.rolling(wz, min_periods=mp).std() + 1e-9
        )

    vw_z = _z(df["kps_vw_rbf_21d"])
    anom_z = _z(df["kps_anomaly_21d"])
    df["kps_composite"] = (vw_z + anom_z) / 2.0

    return df
