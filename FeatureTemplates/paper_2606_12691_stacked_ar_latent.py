"""
Stacked AR latent state features derived from:
  "Two-Layer Linear Auto-Regressive Models Estimate Latent States"
  arXiv 2606.12691

The paper proves that two-layer linear AR models trained on partially observed
linear dynamical systems converge to the Kalman filter's state estimate.
The key insight: the HIDDEN REPRESENTATION of a two-layer AR coincides (up to
rotation) with the optimal latent state estimate.

We implement a MODEL-FREE version: a stacked residual AR structure where each
layer refines the previous layer's prediction, and the residuals at each layer
represent successively deeper latent signals.

  Layer 1: AR(1) on log-returns → residual_1 (fast dynamics, momentum/reversal)
  Layer 2: AR(1) on residual_1  → residual_2 (slow latent dynamics)

From the paper's Theorem 4 (finite-sample prediction error), the layered
structure exposes orthogonal predictive signals.  The AR(1) coefficient itself
is the key learned parameter — it carries directional information about whether
the dynamics are in a momentum vs. mean-reversion regime.

"Benign landscape" metric: the prediction error vs total variance ratio reflects
how close the AR estimate is to the global optimum (Proposition 3 of paper).

Feature family (8 cols):
  sar_l1_resid_z20        layer-1 AR residual z-scored over 20d
  sar_l2_resid_z20        layer-2 AR residual z-scored over 20d
  sar_l1_phi_20           rolling AR(1) coefficient, layer 1, 20d window
  sar_l2_phi_20           rolling AR(1) coefficient on layer-2 residuals, 20d
  sar_prediction_error_20 rolling RMSE of two-layer AR prediction, 20d
  sar_latent_momentum_5   5-day cumsum of layer-2 residual (latent drift)
  sar_landscape_quality   1 - (pred_error^2 / total_variance): benign landscape proxy
  sar_layer_disagreement  sign(l1_resid) != sign(l2_resid): disagreement between layers
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12691_stacked_ar_latent",
    "description": (
        "Stacked two-layer AR residual decomposition: layer-1 fast dynamics, "
        "layer-2 slow latent signal, prediction error, and landscape quality; "
        "from arXiv 2606.12691 (Two-Layer AR Models Estimate Latent States)."
    ),
    "requires": ["Close"],
    "produces": [
        "sar_l1_resid_z20",
        "sar_l2_resid_z20",
        "sar_l1_phi_20",
        "sar_l2_phi_20",
        "sar_prediction_error_20",
        "sar_latent_momentum_5",
        "sar_landscape_quality",
        "sar_layer_disagreement",
    ],
    "tags": ["momentum", "mean_reversion", "statistical", "experimental"],
    "version": "1.1",
    "author": "paper:2606.12691",
}


def _ar1_rolling(y: np.ndarray, window: int) -> tuple:
    """
    Rolling AR(1) fit over a 1-D series using OLS:  y[t] = c + phi * y[t-1].
    Returns (phi_arr, c_arr, resid_arr), all length n.
    Uses incremental vectorised computation per window.
    """
    n = len(y)
    phi_arr   = np.full(n, np.nan)
    c_arr     = np.full(n, np.nan)
    resid_arr = np.full(n, np.nan)

    for i in range(window, n):
        start = i - window + 1
        seg   = y[start: i + 1]
        mask  = np.isfinite(seg)
        if mask.sum() < max(6, window // 3):
            continue
        x   = seg[:-1];  yt = seg[1:]
        mv  = np.isfinite(x) & np.isfinite(yt)
        if mv.sum() < 5:
            continue
        xm  = x[mv].mean(); ym = yt[mv].mean()
        ss  = ((x[mv] - xm) ** 2).sum()
        if ss < 1e-14:
            phi = 0.0; c = ym
        else:
            phi = ((x[mv] - xm) * (yt[mv] - ym)).sum() / ss
            c   = ym - phi * xm
        phi_arr[i] = phi
        c_arr[i]   = c
        # Residual at current bar
        if np.isfinite(y[i]) and np.isfinite(y[i - 1]):
            resid_arr[i] = y[i] - (c + phi * y[i - 1])

    return phi_arr, c_arr, resid_arr


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values.astype(np.float64)
    n = len(df)
    window = 20

    # ---- Layer 1: AR(1) on log returns ----------------------------------------
    l1_phi, l1_c, l1_resid = _ar1_rolling(log_ret, window)

    # ---- Layer 2: AR(1) on layer-1 residuals ----------------------------------
    l2_phi, l2_c, l2_resid = _ar1_rolling(l1_resid, window)

    # ---- Prediction error: RMSE of layer-1 AR forecast over trailing 20d ------
    # resid_arr already contains y[i] - (c + phi * y[i-1]) for each bar
    l1_resid_s = pd.Series(l1_resid, index=df.index)
    pred_err   = np.sqrt((l1_resid_s ** 2).rolling(window, min_periods=window // 3).mean())

    # ---- Build pandas series for rolling normalisation ------------------------
    l2_resid_s = pd.Series(l2_resid, index=df.index)

    def rolling_z(s: pd.Series, w: int = 20) -> pd.Series:
        mu  = s.rolling(w, min_periods=w // 3).mean()
        sig = s.rolling(w, min_periods=w // 3).std()
        return (s - mu) / sig.replace(0.0, np.nan)

    l1_resid_z20 = rolling_z(l1_resid_s)
    l2_resid_z20 = rolling_z(l2_resid_s)

    # ---- Latent momentum: 5-day cumulative layer-2 residual -------------------
    latent_momentum_5 = l2_resid_s.rolling(5, min_periods=2).sum()

    # ---- Landscape quality: 1 - pred_err^2 / total_variance ------------------
    log_ret_s = pd.Series(log_ret, index=df.index)
    total_var = log_ret_s.rolling(window, min_periods=window // 3).var().replace(0.0, np.nan)
    landscape_quality = (1.0 - pred_err ** 2 / total_var).clip(-3.0, 1.0)

    # ---- Layer disagreement: sign(l1_resid) != sign(l2_resid) ----------------
    l1_sign = np.sign(l1_resid)
    l2_sign = np.sign(l2_resid)
    disagree = (l1_sign != l2_sign).astype(float)
    either_nan = ~(np.isfinite(l1_resid) & np.isfinite(l2_resid))
    disagree[either_nan] = np.nan
    disagreement_s = pd.Series(disagree, index=df.index)

    # ---- Assign ---------------------------------------------------------------
    df["sar_l1_resid_z20"]        = l1_resid_z20
    df["sar_l2_resid_z20"]        = l2_resid_z20
    df["sar_l1_phi_20"]           = pd.Series(l1_phi, index=df.index)
    df["sar_l2_phi_20"]           = pd.Series(l2_phi, index=df.index)
    df["sar_prediction_error_20"] = pred_err
    df["sar_latent_momentum_5"]   = latent_momentum_5
    df["sar_landscape_quality"]   = landscape_quality
    df["sar_layer_disagreement"]  = disagreement_s

    return df
