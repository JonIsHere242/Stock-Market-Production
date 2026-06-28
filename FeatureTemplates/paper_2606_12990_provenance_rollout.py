"""
Provenance-Aware AR Rollout Features  —  arxiv:2606.12990
"Exposure Bias as Epistemic Underidentification in Recursive Forecasting"

The paper formalises recursive rollout error via *provenance variables* P that
distinguish "teacher-forcing states" (in-distribution) from "rollout-induced
states" (out-of-distribution).  It derives a three-way error decomposition:
  1. Teacher-forcing / rollout mismatch  (distributional shift)
  2. Representation-class approximation error
  3. Provenance information gap  (epistemic underidentification)

Applied to OHLCV:
  * Encode provenance as a binary variable: normal regime (|z| < 1.5) vs
    tail regime (|z| >= 1.5 of rolling return).
  * Per-provenance AR(1) fit — separate intercept+slope for normal vs tail bars.
  * Rollout divergence: k-step recursive prediction using the "wrong" (normal)
    model when current state is in tail provenance → measures the mismatch gap.
  * Epistemic gap: |normal_prediction - tail_prediction| at t+1.
  * Rollout variance: variance of the k-step rollout trajectory (instability).

Produces 8 columns prefixed "prv_".
All windows causal (no lookahead). Vectorised except for one rolling OLS loop.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12990_provenance_rollout",
    "description": (
        "Provenance-aware AR rollout features from arxiv:2606.12990 — binary "
        "provenance encoding (normal vs tail regime), per-provenance AR(1) fits, "
        "epistemic mismatch gap, rollout trajectory variance, and "
        "teacher-forcing/rollout divergence across multiple horizons."
    ),
    "requires": ["Close"],
    "produces": [
        "prv_provenance_z_20d",
        "prv_normal_phi_30d",
        "prv_tail_phi_30d",
        "prv_epistemic_gap_30d",
        "prv_rollout_var_5step",
        "prv_rollout_var_10step",
        "prv_mismatch_div_5d",
        "prv_gap_rank_60d",
    ],
    "tags": ["momentum", "market_regime", "statistical", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12990",
}


def _ar1_fit(x: np.ndarray, y: np.ndarray):
    """OLS AR(1): y = c + phi*x. Returns (c, phi) or (nan, nan)."""
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    xm = x.mean()
    ym = y.mean()
    ss = ((x - xm) ** 2).sum()
    if ss < 1e-14:
        return np.nan, np.nan
    phi = ((x - xm) * (y - ym)).sum() / ss
    c = ym - phi * xm
    return float(c), float(phi)


def _rollout_trajectory(c: float, phi: float, r0: float, k: int) -> np.ndarray:
    """k-step AR(1) rollout trajectory starting at r0."""
    traj = np.empty(k)
    r = r0
    for i in range(k):
        r_next = c + phi * r
        traj[i] = r_next
        r = r_next
    return traj


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan).astype(np.float64)
    log_ret = np.log(close / close.shift(1)).values

    n = len(df)
    WINDOW = 30      # AR fitting window
    Z_THRESH = 1.5   # provenance boundary

    # Output arrays
    prov_z    = np.full(n, np.nan)
    phi_norm  = np.full(n, np.nan)
    phi_tail  = np.full(n, np.nan)
    epist_gap = np.full(n, np.nan)
    rv5       = np.full(n, np.nan)
    rv10      = np.full(n, np.nan)
    mismatch5 = np.full(n, np.nan)

    for i in range(WINDOW, n):
        seg = log_ret[i - WINDOW: i + 1]
        mask = np.isfinite(seg)
        rv_seg = seg[mask]
        if len(rv_seg) < 10:
            continue

        # Provenance z-score of current return
        mu = rv_seg[:-1].mean()
        sigma = rv_seg[:-1].std()
        if sigma < 1e-12:
            continue
        r_cur = rv_seg[-1]
        z = (r_cur - mu) / sigma
        prov_z[i] = z

        # Split window into normal vs tail provenance
        xs_all = rv_seg[:-1]
        ys_all = rv_seg[1:]
        zs = (xs_all - mu) / sigma

        norm_mask = np.abs(zs) < Z_THRESH
        tail_mask = ~norm_mask

        c_n, phi_n = np.nan, np.nan
        c_t, phi_t = np.nan, np.nan

        if norm_mask.sum() >= 4:
            c_n, phi_n = _ar1_fit(xs_all[norm_mask], ys_all[norm_mask])
        if tail_mask.sum() >= 3:
            c_t, phi_t = _ar1_fit(xs_all[tail_mask], ys_all[tail_mask])

        phi_norm[i] = phi_n if np.isfinite(phi_n) else np.nan
        phi_tail[i] = phi_t if np.isfinite(phi_t) else np.nan

        # Epistemic gap: |prediction using normal model - tail model| at t+1
        if np.isfinite(phi_n) and np.isfinite(c_n) and np.isfinite(phi_t) and np.isfinite(c_t):
            pred_n = c_n + phi_n * r_cur
            pred_t = c_t + phi_t * r_cur
            epist_gap[i] = abs(pred_n - pred_t)

            # Rollout variance: variance of k-step trajectory using WRONG model
            # (use normal model even if in tail — teacher/rollout mismatch)
            traj5  = _rollout_trajectory(c_n, phi_n, r_cur, 5)
            traj10 = _rollout_trajectory(c_n, phi_n, r_cur, 10)
            rv5[i]  = traj5.var()
            rv10[i] = traj10.var()

            # Mismatch divergence over 5 steps: normal vs tail rollout
            traj5_t = _rollout_trajectory(c_t, phi_t, r_cur, 5)
            mismatch5[i] = np.abs(traj5 - traj5_t).mean()

    df["prv_provenance_z_20d"]  = prov_z
    df["prv_normal_phi_30d"]    = phi_norm
    df["prv_tail_phi_30d"]      = phi_tail
    df["prv_epistemic_gap_30d"] = epist_gap
    df["prv_rollout_var_5step"] = rv5
    df["prv_rollout_var_10step"] = rv10
    df["prv_mismatch_div_5d"]   = mismatch5

    # Rolling percentile rank of epistemic gap over 60d → gap_rank
    gap_s = pd.Series(epist_gap, index=df.index)
    df["prv_gap_rank_60d"] = gap_s.rolling(60, min_periods=15).rank(pct=True)

    return df
