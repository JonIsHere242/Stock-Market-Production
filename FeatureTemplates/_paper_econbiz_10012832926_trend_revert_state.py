"""
_paper_econbiz_10012832926_trend_revert_state.py
=================================================
Per-ticker trend-vs-revert STATE diagnostic inspired by:
  "To Trend or to Revert — A Portfolio Perspective on Time Series Momentum"
  (econbiz: 10012832926)

DESIGN INTENT
-------------
This block answers the question: "Is this stock currently in a TRENDING or
MEAN-REVERTING regime, and how strongly?"  It is NOT a plain momentum score
(momentum_score.py covers that) and NOT a bare variance-ratio (VR alone cannot
distinguish the cause of persistence).  Instead it synthesises four distinct
statistical lenses into a single continuous score:

  1. Variance-Ratio (Lo-MacKinlay) — VR > 1 ⇒ trending; VR < 1 ⇒ reverting.
  2. Lagged autocorrelation of log-returns (lag-1 and lag-5) — positive ⇒ trend,
     negative ⇒ mean-reversion.
  3. Hurst-exponent-like persistence estimate via the VR slope (how VR(q) grows
     with horizon q). H ≈ 0.5 random; H > 0.5 trending; H < 0.5 reverting.
  4. Multi-horizon sign agreement (1d / 5d / 21d / 63d trailing log-returns):
     four signs all positive ⇒ strong trend; mixed ⇒ transitional.

Everything is computed in causal rolling windows (past-only); the composite
score `tvr_score` is a weighted sum of z-scored components, clipped to [-1, +1].
A discrete `tvr_regime` label encodes {-1 REVERTING, 0 NEUTRAL, +1 TRENDING}.

Honest caveat: all features are per-ticker diagnostics.  Cross-sectional
predictive validity requires the usual in-model marginal-contribution + multi-seed
backtest validation before any promotion to the live feature set.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "trend_revert_state",
    "description": (
        "Per-ticker trend-vs-revert state diagnostic: causal rolling variance-ratio, "
        "lag-1/lag-5 autocorrelation, Hurst-like persistence, and multi-horizon sign "
        "agreement synthesised into a continuous score and discrete regime label "
        "(NOT a plain momentum score or bare variance-ratio)."
    ),
    "requires": ["Close"],
    "produces": [
        "tvr_vr_20",          # Lo-MacKinlay variance-ratio at q=20 (VR>1 trending)
        "tvr_vr_5",           # variance-ratio at q=5
        "tvr_autocorr_lag1",  # rolling lag-1 autocorrelation of 1d log-returns
        "tvr_autocorr_lag5",  # rolling lag-5 autocorrelation of 1d log-returns
        "tvr_hurst",          # Hurst-like exponent from VR slope across horizons
        "tvr_sign_agree",     # fraction of horizons with same-sign trailing return [-1,+1]
        "tvr_score",          # continuous composite score clipped to [-1,+1]
        "tvr_regime",         # discrete: +1 TRENDING / 0 NEUTRAL / -1 REVERTING
    ],
    "tags": ["trend", "mean_reversion", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper: econbiz:10012832926 (To Trend or to Revert)",
}

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_MIN_ROWS = 70          # minimum rows of non-NaN return history to produce values
_VR_WINDOW = 126        # rolling window for VR and autocorrelation estimation
_HURST_WINDOW = 126     # same window for Hurst slope
_HORIZONS = (1, 5, 21, 63)  # multi-horizon sign-agreement look-back periods

# VR horizons used for Hurst slope (must include _q_short and _q_long)
_VR_Q_SHORT = 5
_VR_Q_LONG = 20
_VR_HURST_QS = (2, 5, 10, 20)  # log-spaced horizons for slope fit

# weights for composite score (must sum to 1.0)
_W_VR_LONG = 0.25
_W_VR_SHORT = 0.10
_W_AC1 = 0.20
_W_AC5 = 0.15
_W_HURST = 0.20
_W_SIGN = 0.10


# ---------------------------------------------------------------------------
# Helpers — all causal (use only past data at each row)
# ---------------------------------------------------------------------------

def _log_returns(close: np.ndarray) -> np.ndarray:
    """1d log-returns, length == len(close), first element NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.empty(len(close), dtype=np.float64)
        out[0] = np.nan
        out[1:] = np.log(close[1:] / close[:-1])
    return out


def _rolling_vr(ret: np.ndarray, q: int, window: int) -> np.ndarray:
    """
    Causal rolling Lo-MacKinlay variance ratio at horizon q over `window` past
    1d log-returns.

    VR(q) = Var(q-day return) / (q * Var(1-day return))

    Computed in pure NumPy rolling fashion (no look-ahead).  Each output[t]
    uses ret[t-window+1 .. t] (window of 1d log-returns ending at t).
    """
    n = len(ret)
    out = np.full(n, np.nan, dtype=np.float64)

    # need at least (window + q) observations to form q-day returns inside window
    need = window + q
    if n < need:
        return out

    # Precompute overlapping q-day sums for efficiency (causal prefix-sum trick)
    # q_ret[t] = ret[t] + ret[t-1] + ... + ret[t-q+1]  (NaN if any NaN in window)
    # We compute it as a convolution-like rolling sum, avoiding future data.
    # Strategy: build q_ret via cumulative sum differences.
    # cumsum[t] = sum(ret[0..t]); q_ret[t] = cumsum[t] - cumsum[t-q]
    # This is causal — only past ret values contribute.

    cumsum = np.nancumsum(np.where(np.isnan(ret), 0.0, ret))
    nan_count = np.cumsum(np.isnan(ret).astype(int))

    # q-day log-returns ending at t (inclusive)
    q_ret = np.full(n, np.nan, dtype=np.float64)
    for t in range(q, n):
        n_nan = nan_count[t] - nan_count[t - q]
        if n_nan == 0:
            q_ret[t] = cumsum[t] - cumsum[t - q]

    for t in range(need - 1, n):
        # 1d returns in window: ret[t-window+1 .. t]
        seg1 = ret[t - window + 1: t + 1]
        # q-day returns that END within this window
        # q_ret[t'] for t' in [t-window+1+q-1 .. t]  i.e. the last (window-q+1) q_rets
        seg_q = q_ret[t - window + q: t + 1]

        # count valid observations
        valid1 = seg1[~np.isnan(seg1)]
        valid_q = seg_q[~np.isnan(seg_q)]

        if len(valid1) < max(q + 1, 10) or len(valid_q) < 5:
            continue

        var1 = np.var(valid1, ddof=1)
        var_q = np.var(valid_q, ddof=1)

        if var1 <= 0 or not np.isfinite(var1) or not np.isfinite(var_q):
            continue

        out[t] = var_q / (q * var1)

    return out


def _rolling_autocorr(ret: np.ndarray, lag: int, window: int) -> np.ndarray:
    """
    Causal rolling lag-k autocorrelation of 1d log-returns over `window` rows.
    Positive ⇒ trend persistence; negative ⇒ mean-reversion.
    """
    n = len(ret)
    out = np.full(n, np.nan, dtype=np.float64)
    need = window + lag
    if n < need:
        return out

    for t in range(need - 1, n):
        seg = ret[t - window + 1: t + 1]
        # remove NaN pairs
        x = seg[:-lag]
        y = seg[lag:]
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[mask], y[mask]
        if len(xv) < 15:
            continue
        # Pearson correlation
        xm, ym = xv.mean(), yv.mean()
        num = np.dot(xv - xm, yv - ym)
        denom = np.sqrt(np.dot(xv - xm, xv - xm) * np.dot(yv - ym, yv - ym))
        if denom <= 0 or not np.isfinite(denom):
            continue
        out[t] = num / denom

    return out


def _rolling_hurst(ret: np.ndarray, qs: Tuple[int, ...], window: int) -> np.ndarray:
    """
    Hurst-like exponent estimated from the slope of log VR(q) vs log(q) across
    multiple horizons q in `qs`, using a rolling window of past 1d log-returns.

    Theory: VR(q) ~ q^(2H-1), so  log VR(q) = (2H-1)*log(q) + const
    => slope of OLS(log q, log VR(q)) = 2H - 1
    => H = (slope + 1) / 2
    H = 0.5 ⇒ random walk; H > 0.5 ⇒ trending; H < 0.5 ⇒ mean-reverting.
    """
    n = len(ret)
    out = np.full(n, np.nan, dtype=np.float64)
    max_q = max(qs)
    need = window + max_q
    if n < need:
        return out

    log_q = np.log(np.array(qs, dtype=np.float64))

    # Precompute VR arrays for all q values
    vr_arrays = {q: _rolling_vr(ret, q, window) for q in qs}

    for t in range(need - 1, n):
        log_vr_vals = []
        for q in qs:
            v = vr_arrays[q][t]
            if np.isnan(v) or v <= 0:
                break
            log_vr_vals.append(np.log(v))
        else:
            # OLS slope: slope = cov(log_q, log_vr) / var(log_q)
            if len(log_vr_vals) < 3:
                continue
            lv = np.array(log_vr_vals)
            lq_m = log_q.mean()
            lv_m = lv.mean()
            cov = np.dot(log_q - lq_m, lv - lv_m)
            var_q = np.dot(log_q - lq_m, log_q - lq_m)
            if var_q <= 0:
                continue
            slope = cov / var_q
            out[t] = (slope + 1.0) / 2.0  # H estimate

    return out


def _rolling_sign_agree(
    close: np.ndarray, horizons: Tuple[int, ...], window: int
) -> np.ndarray:
    """
    At each bar t, compute sign of trailing log-return over each horizon h
    (i.e. sign(close[t] / close[t-h] - 1)).  `tvr_sign_agree` is the mean
    of those signs scaled to [-1, +1]:
       +1 all horizons positive (strong trend up)
        0 mixed
       -1 all horizons negative (strong trend down, which is also trending)

    For "trend vs revert" we want |sign_agree| to measure COHERENCE (how much
    all horizons agree), but the raw signed value tells direction too.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    max_h = max(horizons)

    for t in range(max_h, n):
        signs = []
        for h in horizons:
            if t - h < 0 or np.isnan(close[t]) or np.isnan(close[t - h]):
                continue
            if close[t - h] <= 0:
                continue
            lr = np.log(close[t] / close[t - h])
            if lr != 0:
                signs.append(np.sign(lr))
            else:
                signs.append(0.0)
        if len(signs) == len(horizons):
            out[t] = float(np.mean(signs))

    return out


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score of arr over `window` past values."""
    s = pd.Series(arr)
    mn = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = (s - mn) / sd.replace(0.0, np.nan)
    return z.to_numpy(dtype=np.float64)


def _clip_no_inf(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip, replacing any inf/-inf with NaN first."""
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    return np.clip(arr, lo, hi)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker trend-vs-revert state features.

    All features are causal (depend only on rows <= t).  Leading NaN rows are
    expected and left as-is.  No existing column is modified.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    ret = _log_returns(close)
    n = len(close)

    # ---- 1. Variance ratios at q=5 and q=20 --------------------------------
    vr5 = _rolling_vr(ret, _VR_Q_SHORT, _VR_WINDOW)
    vr20 = _rolling_vr(ret, _VR_Q_LONG, _VR_WINDOW)

    # ---- 2. Lagged autocorrelations ----------------------------------------
    ac1 = _rolling_autocorr(ret, lag=1, window=_VR_WINDOW)
    ac5 = _rolling_autocorr(ret, lag=5, window=_VR_WINDOW)

    # ---- 3. Hurst-like exponent --------------------------------------------
    hurst = _rolling_hurst(ret, qs=_VR_HURST_QS, window=_HURST_WINDOW)

    # ---- 4. Multi-horizon sign agreement -----------------------------------
    sign_agree = _rolling_sign_agree(close, horizons=_HORIZONS, window=_VR_WINDOW)

    # ---- 5. Composite score ------------------------------------------------
    # Transform each component into a "trending-positive" signal in [-1, +1]:
    #   VR: trending if VR>1 ⇒ map (VR-1) linearly, clipped
    #   AC: already in [-1,+1] (positive = trend)
    #   Hurst: trending if H>0.5 ⇒ map 2*(H-0.5) to [-1,+1]
    #   Sign agree: already [-1,+1] (|value| = coherence, sign = direction)
    #   For the composite we use the SIGNED value so +score = trending-up,
    #   -score = trending-down (both are "trend" in magnitude).

    # VR component: (VR-1) clipped to [-1,+1] (VR rarely differs from 1 by >1)
    vr5_sig = _clip_no_inf(vr5 - 1.0, -1.0, 1.0)
    vr20_sig = _clip_no_inf(vr20 - 1.0, -1.0, 1.0)

    # AC already in [-1,+1]
    ac1_sig = _clip_no_inf(ac1, -1.0, 1.0)
    ac5_sig = _clip_no_inf(ac5, -1.0, 1.0)

    # Hurst: map [0,1] → [-1,+1] via 2*(H-0.5)
    hurst_sig = _clip_no_inf(2.0 * (hurst - 0.5), -1.0, 1.0)

    # Sign agree: magnitude tells coherence, sign tells direction
    sign_sig = _clip_no_inf(sign_agree, -1.0, 1.0)

    # Weighted composite
    score_raw = (
        _W_VR_LONG  * np.where(np.isnan(vr20_sig),  0.0, vr20_sig)
        + _W_VR_SHORT * np.where(np.isnan(vr5_sig),   0.0, vr5_sig)
        + _W_AC1      * np.where(np.isnan(ac1_sig),   0.0, ac1_sig)
        + _W_AC5      * np.where(np.isnan(ac5_sig),   0.0, ac5_sig)
        + _W_HURST    * np.where(np.isnan(hurst_sig), 0.0, hurst_sig)
        + _W_SIGN     * np.where(np.isnan(sign_sig),  0.0, sign_sig)
    )

    # Mask rows where ALL components are NaN
    all_nan_mask = (
        np.isnan(vr20_sig)
        & np.isnan(vr5_sig)
        & np.isnan(ac1_sig)
        & np.isnan(ac5_sig)
        & np.isnan(hurst_sig)
        & np.isnan(sign_sig)
    )
    score_raw[all_nan_mask] = np.nan

    tvr_score = _clip_no_inf(score_raw, -1.0, 1.0)

    # ---- 6. Discrete regime label ------------------------------------------
    tvr_regime = np.full(n, np.nan, dtype=np.float64)
    valid = ~np.isnan(tvr_score)
    tvr_regime[valid & (tvr_score > 0.15)] = 1.0   # TRENDING
    tvr_regime[valid & (tvr_score < -0.15)] = -1.0  # REVERTING
    tvr_regime[valid & (tvr_score >= -0.15) & (tvr_score <= 0.15)] = 0.0  # NEUTRAL

    # ---- Assign to df ------------------------------------------------------
    idx = df.index
    df["tvr_vr_5"] = pd.Series(vr5, index=idx)
    df["tvr_vr_20"] = pd.Series(vr20, index=idx)
    df["tvr_autocorr_lag1"] = pd.Series(ac1, index=idx)
    df["tvr_autocorr_lag5"] = pd.Series(ac5, index=idx)
    df["tvr_hurst"] = pd.Series(hurst, index=idx)
    df["tvr_sign_agree"] = pd.Series(sign_agree, index=idx)
    df["tvr_score"] = pd.Series(tvr_score, index=idx)
    df["tvr_regime"] = pd.Series(tvr_regime, index=idx)

    return df
