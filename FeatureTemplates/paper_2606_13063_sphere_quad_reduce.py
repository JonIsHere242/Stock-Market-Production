"""
Sphere-Projected Quadratic Order Reduction Features  —  arxiv:2606.13063
"A Quadratic Order Reduction -- Gaussian Process ODE Framework for the
 Inference of Large Continuous Dynamical Systems"

The paper proposes *Quadratic Order Model Reduction (QROM)* combined with
sphere projection to stabilise the learned latent dynamics.  The sphere
projection maps the high-dimensional state onto the unit hypersphere,
preserving angular structure while removing amplitude instability.

Applied to OHLCV:
  1. Embed OHLCV state into a latent vector:
       x = [log_ret, norm_range, vol_z, close_z, hi_lo_ratio]
     (5-dimensional, inspired by the paper's quadratic state augmentation)
  2. Project the state onto the unit sphere: x_hat = x / ||x||
  3. Track angular velocity: angle between consecutive sphere-projected states.
  4. Quadratic features: include second-order terms x_i * x_j (selected pairs)
     that the QROM framework uses to capture nonlinear interactions.
  5. Sphere drift: distance from the rolling mean sphere position.
  6. GP uncertainty proxy: rolling residual variance of the sphere-projected
     state (analogous to GP posterior variance).

All computations strictly causal; fully vectorised.
Produces 8 columns prefixed "sqr_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13063_sphere_quad_reduce",
    "description": (
        "Sphere-projected quadratic order reduction features from arxiv:2606.13063 — "
        "OHLCV latent state projected onto unit sphere, angular velocity, quadratic "
        "cross-terms, sphere drift from rolling mean, and GP-variance proxy."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "sqr_angular_vel_5d",
        "sqr_angular_vel_21d",
        "sqr_sphere_drift_21d",
        "sqr_sphere_drift_63d",
        "sqr_quad_ret_range",
        "sqr_quad_ret_vol",
        "sqr_gp_var_proxy_21d",
        "sqr_latent_norm",
    ],
    "tags": ["market_regime", "volatility", "statistical", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13063",
}


def _build_latent(df: pd.DataFrame) -> np.ndarray:
    """
    Build the 5D latent state vector (each row = one bar).
    Normalise each dimension to zero-mean, unit-std over a 252-day rolling window
    (look-back only — causal).
    """
    C = df["Close"].replace(0, np.nan).astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    V = df["Volume"].replace(0, np.nan).astype(float)
    O = df["Open"].astype(float)

    log_ret    = np.log(C / C.shift(1))
    norm_range = (H - L) / C.clip(lower=1e-9)               # normalised range
    hi_lo_rat  = np.log(H / L.replace(0, np.nan))           # log H/L
    log_vol    = np.log(V)
    gap        = np.log(O / C.shift(1).replace(0, np.nan))  # overnight gap

    features = pd.DataFrame({
        "ret":    log_ret,
        "range":  norm_range,
        "hilorat": hi_lo_rat,
        "logvol": log_vol,
        "gap":    gap,
    })

    # Rolling z-score (strictly causal — shift by 1 before rolling stats)
    W_NORM = 63
    MP_NORM = 20
    normed = pd.DataFrame(index=df.index)
    for col in features.columns:
        s = features[col]
        mu = s.shift(1).rolling(W_NORM, min_periods=MP_NORM).mean()
        sd = s.shift(1).rolling(W_NORM, min_periods=MP_NORM).std()
        normed[col] = (s - mu) / (sd + 1e-9)

    # Replace inf/NaN with 0 for sphere projection
    normed = normed.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return normed.values.astype(np.float64)


def _sphere_project(X: np.ndarray) -> np.ndarray:
    """Project each row onto the unit sphere. Rows with zero norm stay at zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    return X / safe_norms, norms.squeeze()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ── Build latent state ─────────────────────────────────────────────────────
    X = _build_latent(df)                # (n, 5)
    X_hat, X_norm = _sphere_project(X)  # (n, 5), (n,)

    # ── Angular velocity: geodesic angle between consecutive sphere states ──────
    # cos(theta) = x_hat[t] . x_hat[t-1]
    dot_consec = np.einsum("ij,ij->i", X_hat[1:], X_hat[:-1])
    dot_consec = np.clip(dot_consec, -1.0, 1.0)
    ang_step   = np.arccos(dot_consec)   # in [0, pi]
    ang_step   = np.concatenate([[np.nan], ang_step])

    ang_s = pd.Series(ang_step, index=df.index)
    df["sqr_angular_vel_5d"]  = ang_s.rolling(5,  min_periods=2).mean()
    df["sqr_angular_vel_21d"] = ang_s.rolling(21, min_periods=7).mean()

    # ── Sphere drift: distance of X_hat[t] from rolling mean sphere position ──
    X_hat_df = pd.DataFrame(X_hat, index=df.index)

    for w, suffix in [(21, "21d"), (63, "63d")]:
        mp = max(5, w // 4)
        # Rolling column means (strictly causal via shift)
        mu_hat = X_hat_df.shift(1).rolling(w, min_periods=mp).mean().values  # (n,5)
        # Geodesic drift: angle between X_hat[t] and rolling mean direction
        mu_norms = np.linalg.norm(mu_hat, axis=1, keepdims=True)
        safe_mu  = np.where(mu_norms < 1e-12, 1.0, mu_norms)
        mu_hat_n = mu_hat / safe_mu  # normalise mean direction
        dot_drift = np.einsum("ij,ij->i", X_hat, mu_hat_n)
        dot_drift = np.clip(dot_drift, -1.0, 1.0)
        drift_angle = np.arccos(dot_drift)
        # Zero where latent norm is near-zero (no signal)
        drift_angle = np.where(X_norm < 1e-12, np.nan, drift_angle)
        df[f"sqr_sphere_drift_{suffix}"] = drift_angle

    # ── Quadratic cross-terms: interaction between selected dimensions ─────────
    # ret × range  (momentum × volatility interaction) — rolling percentile rank
    # for uniform distribution (better tree signal)
    rr_raw = pd.Series(X[:, 0] * X[:, 1], index=df.index)
    df["sqr_quad_ret_range"] = rr_raw.rolling(63, min_periods=20).rank(pct=True) - 0.5
    # ret × logvol  (momentum × volume interaction)
    rv_raw = pd.Series(X[:, 0] * X[:, 3], index=df.index)
    df["sqr_quad_ret_vol"]   = rv_raw.rolling(63, min_periods=20).rank(pct=True) - 0.5

    # ── GP variance proxy: rolling variance of angular step (uncertainty) ──────
    df["sqr_gp_var_proxy_21d"] = ang_s.rolling(21, min_periods=7).var()

    # ── Latent norm: reflects how "extreme" the current market state is ────────
    # Use RANK within 63d window to make it uniform [0,1] and better correlated
    norm_s = pd.Series(X_norm, index=df.index)
    df["sqr_latent_norm"] = norm_s.rolling(63, min_periods=20).rank(pct=True)

    return df
