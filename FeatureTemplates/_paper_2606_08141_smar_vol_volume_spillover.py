"""
Structural Matrix Autoregressive (SMAR) per-ticker features derived from:
  "A Structural Matrix Autoregressive Model for the Joint Dynamics of
   Volume, Volatility, and Returns" (arXiv 2606.08141).

The paper models joint dynamics of (return, realized vol, volume) in a VAR
framework, finding that volatility drives trading activity (MDH: Mixture of
Distributions Hypothesis) and that cross-asset spillovers account for >50%
of volume variation at long horizons.

Per-ticker proxy: fit a rolling VAR(1) on the 3-variable system
  z_t = [r_t, rv_t, v_t]  (return, realized vol, normalized volume)
and extract:
  - Coefficient magnitudes (how strongly each variable predicts the others)
  - Forecast error decomposition approximation: vol-driven fraction of volume
  - Innovation covariance: contemporaneous vol-volume coupling
  - VAR residuals as anomaly score (deviation from modeled dynamics)

All rolling, no lookahead.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2606_08141_smar_vol_volume_spillover",
    "description": (
        "Rolling VAR(1) on (return, realized-vol, volume) system; extracts vol->volume "
        "spillover strength, contemporaneous vol-volume coupling, and VAR residual "
        "anomaly; per-ticker proxy for arXiv 2606.08141 SMAR model."
    ),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "smar_vol_to_volume_coef",
        "smar_ret_to_vol_coef",
        "smar_vol_volume_contemp_corr",
        "smar_resid_norm",
    ],
    "tags":        ["volatility", "volume", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.08141",
}

_EPS = 1e-12


def _gk_rv_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  window: int = 5) -> np.ndarray:
    """Short rolling realized vol (Parkinson estimator)."""
    n = len(close)
    rv = np.full(n, np.nan)
    log_hl2 = np.log(np.maximum(high, _EPS) / np.maximum(low, _EPS)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    for i in range(window - 1, n):
        sl = log_hl2[i - window + 1: i + 1]
        rv[i] = np.sqrt(252.0 * factor * np.nanmean(sl))
    return rv


def _rolling_var1(R: np.ndarray, RV: np.ndarray, V: np.ndarray,
                  window: int, min_obs: int):
    """
    Rolling VAR(1): Z_t = A Z_{t-1} + eps
    Z = [R, RV, V] (3-variable system).
    Returns: vol_to_volume_coef, ret_to_vol_coef, contemp_corr, resid_norm
    """
    n = len(R)
    vol_to_vol_arr  = np.full(n, np.nan)
    ret_to_vol_arr  = np.full(n, np.nan)
    contemp_arr     = np.full(n, np.nan)
    resid_norm_arr  = np.full(n, np.nan)

    for i in range(window, n):
        start = i - window
        r  = R[start: i + 1]
        rv = RV[start: i + 1]
        v  = V[start: i + 1]

        # Drop rows with any NaN
        stack = np.column_stack([r, rv, v])
        valid = ~np.isnan(stack).any(axis=1)
        stack_v = stack[valid]
        if len(stack_v) < min_obs + 1:
            continue

        Z_t   = stack_v[1:]   # (T-1, 3) current
        Z_lag = stack_v[:-1]  # (T-1, 3) lagged

        # OLS: Z_t = Z_lag @ A.T + eps  (each row of A is one equation)
        # Use lstsq per equation for simplicity (3 equations)
        T_eff = len(Z_t)
        if T_eff < 6:
            continue

        # Add intercept
        X = np.column_stack([Z_lag, np.ones(T_eff)])  # (T-1, 4)
        try:
            A_full, _, _, _ = np.linalg.lstsq(X, Z_t, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # A_full is (4, 3): rows = [lag_r, lag_rv, lag_v, intercept]
        # columns = [eq_r, eq_rv, eq_v]
        A = A_full[:3, :]  # (3,3) lag coefficients only

        # Vol -> Volume: A[1, 2] (lagged RV predicting current V)
        vol_to_vol_arr[i] = A[1, 2]

        # Ret -> Vol: A[0, 1] (lagged return predicting current RV)
        ret_to_vol_arr[i] = A[0, 1]

        # Residuals
        resids = Z_t - X @ A_full
        resid_norm_arr[i] = np.sqrt((resids[-1] ** 2).sum())

        # Contemporaneous vol-volume correlation in residuals
        if resids.shape[0] >= 4:
            rv_res = resids[:, 1]
            v_res  = resids[:, 2]
            std_rv = rv_res.std()
            std_v  = v_res.std()
            if std_rv > _EPS and std_v > _EPS:
                contemp_arr[i] = np.corrcoef(rv_res, v_res)[0, 1]

    return vol_to_vol_arr, ret_to_vol_arr, contemp_arr, resid_norm_arr


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].values.astype(np.float64)
    high   = df["High"].values.astype(np.float64)
    low    = df["Low"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    # Daily return
    ret = np.full(len(close), np.nan)
    ret[1:] = np.log(np.maximum(close[1:], _EPS) / np.maximum(close[:-1], _EPS))

    # 5-day realized vol
    rv5 = _gk_rv_series(high, low, close, window=5)

    # Volume z-score via expanding mean/std (no lookahead)
    vol_z = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        past = volume[: i + 1]
        past = past[~np.isnan(past)]
        if len(past) < 10:
            continue
        m, s = past.mean(), past.std()
        if s > _EPS:
            vol_z[i] = (volume[i] - m) / s

    window  = 60
    min_obs = 20

    v2v, r2v, contemp, rnorm = _rolling_var1(ret, rv5, vol_z, window, min_obs)

    df["smar_vol_to_volume_coef"]     = v2v
    df["smar_ret_to_vol_coef"]        = r2v
    df["smar_vol_volume_contemp_corr"] = contemp
    df["smar_resid_norm"]             = rnorm

    return df
