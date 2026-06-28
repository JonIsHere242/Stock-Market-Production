"""
Transformation-based anomaly detection features derived from:
  "ProtoX-AD: Self-Explainable Time Series Anomaly Detection and Characterization"
  arXiv 2606.13277

ProtoX-AD detects anomalies by applying canonical transformations to normal
training samples and training a classifier to distinguish the transformed vs
original patterns.  The core insight: a time-series window that is anomalous
will ALREADY look like a transformed version of a normal window.

We extract this as OHLCV features by applying a family of canonical
transformations to rolling price windows and measuring how "dissimilar" the
current window is from its own rolling-mean archetype (prototype):

Transformations implemented (per paper's self-supervised approach):
  1. Time-reversal     — cosine distance of window vs its own reverse
  2. Amplitude-scaling — current short-window std vs rolling baseline
  3. Trend-subtraction — RMS of linearly-detrended residuals
  4. Noise-injection proxy — variance unexplained by EMA

Feature family (7 columns):
  ptx_reversal_score_20   cosine distance: window vs reversed window, 20-day
  ptx_reversal_score_60   same, 60-day
  ptx_amplitude_score_20  std-ratio anomaly: z-scored current/baseline std
  ptx_trend_resid_20      RMS of linearly-detrended residuals, z-scored
  ptx_trend_resid_60      same, 60-day
  ptx_noise_score_20      pointwise variance unexplained by smoothed signal
  ptx_composite_anom      equal-weighted composite of all 4 raw scores (z-scored)

Implementation: all loops vectorised or replaced with numpy stride tricks.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13277_protox_transform_anomaly",
    "description": (
        "Transformation-based anomaly scores: time-reversal, amplitude, trend, and "
        "noise dissimilarity from rolling archetype; from arXiv 2606.13277 (ProtoX-AD)."
    ),
    "requires": ["Close"],
    "produces": [
        "ptx_reversal_score_20",
        "ptx_reversal_score_60",
        "ptx_amplitude_score_20",
        "ptx_trend_resid_20",
        "ptx_trend_resid_60",
        "ptx_noise_score_20",
        "ptx_composite_anom",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13277",
}


def _rolling_reversal_score(ret_vals: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised reversal score using stride tricks.
    For each bar i, compute cosine distance between
    ret_vals[i-window+1:i+1] and its reverse.

    Cosine distance = 1 - cosine_similarity.
    For a vector v and its reverse v[::-1]:
      dot(v, v[::-1]) = sum(v[k]*v[n-1-k]) for k in 0..n-1
      = autocorrelation at all lags combined.
    If v is monotone, dot is large; if v is symmetric, it equals ||v||^2.

    Returns array of length n; NaN for first (window-1) entries.
    """
    n = len(ret_vals)
    out = np.full(n, np.nan)
    # Replace NaN with 0 for windowing (NaN rows will stay NaN via the finite check)
    vals_filled = np.where(np.isfinite(ret_vals), ret_vals, 0.0)

    # Build stride matrix: shape (n - window + 1, window)
    # Each row is a window of ret_vals
    if n < window:
        return out

    # Use as_strided for O(1) memory view (read-only)
    from numpy.lib.stride_tricks import as_strided
    itemsize = vals_filled.strides[0]
    W = as_strided(
        vals_filled,
        shape=(n - window + 1, window),
        strides=(itemsize, itemsize),
    ).copy()  # copy to avoid mutation hazard

    # norms
    norms = np.linalg.norm(W, axis=1)  # shape (n - window + 1,)
    # dot with reverse: sum(W[i,:] * W[i,::-1])
    dot_rev = np.sum(W * W[:, ::-1], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_sim = np.where(
            norms ** 2 > 1e-14,
            dot_rev / (norms ** 2),
            np.nan,
        )
    cos_dist = 1.0 - cos_sim

    out[window - 1 :] = cos_dist

    # Mask bars where original had NaN
    for i in range(window - 1, n):
        if not np.all(np.isfinite(ret_vals[i - window + 1 : i + 1])):
            out[i] = np.nan

    return out


def _rolling_trend_resid_rms(ret_vals: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling RMS of linearly-detrended residuals.
    Uses the identity: for a window x of length w,
      detrended_rms^2 = var(x) - (cov(x, t))^2 / var(t)
    where t = [0,1,...,w-1].  Fully vectorised via pandas rolling.
    """
    s = pd.Series(ret_vals)
    n = len(ret_vals)

    # Rolling variance (denominator)
    roll_var = s.rolling(window, min_periods=max(5, window // 3)).var()

    # Rolling covariance with linear index (t = 0,1,...,w-1):
    # cov(x, t) = E[x*t] - E[x]*E[t]
    # Precompute t-weighted rolling stats.
    # We build a column of indices and compute rolling cross-term.
    t_arr = np.arange(n, dtype=float)
    st = pd.Series(t_arr)
    # rolling mean of x
    roll_mean_x = s.rolling(window, min_periods=max(5, window // 3)).mean()
    # rolling mean of t (just arithmetic mean of window index positions)
    roll_mean_t = st.rolling(window, min_periods=max(5, window // 3)).mean()
    # rolling mean of x*t
    roll_mean_xt = (s * st).rolling(window, min_periods=max(5, window // 3)).mean()

    cov_xt = roll_mean_xt - roll_mean_x * roll_mean_t
    # var of t over window of length w:  var(0..w-1) = (w^2 - 1)/12
    # For rolling, use actual rolling var of t
    var_t = st.rolling(window, min_periods=max(5, window // 3)).var()

    # residual variance = var(x) - cov(x,t)^2 / var(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid_var = roll_var - (cov_xt ** 2) / var_t.replace(0, np.nan)
    resid_var = resid_var.clip(lower=0)
    return np.sqrt(resid_var.values)


def _zscore_series(s: pd.Series, trail: int = 60, min_p: int = 20) -> pd.Series:
    mu  = s.rolling(trail, min_periods=min_p).mean()
    sig = s.rolling(trail, min_periods=min_p).std()
    return (s - mu) / sig.replace(0, np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    lr_vals = log_ret.values.astype(np.float64)

    # ── Transformation 1: time-reversal score ──────────────────────────────────
    rev20 = _rolling_reversal_score(lr_vals, 20)
    rev60 = _rolling_reversal_score(lr_vals, 60)
    df["ptx_reversal_score_20"] = rev20
    df["ptx_reversal_score_60"] = rev60

    # ── Transformation 2: amplitude anomaly (std ratio vs rolling baseline) ────
    std5  = log_ret.rolling(5,  min_periods=3).std()
    std20 = log_ret.rolling(20, min_periods=10).std()
    amp_ratio = std5 / std20.replace(0, np.nan)
    df["ptx_amplitude_score_20"] = _zscore_series(amp_ratio, trail=60)

    # ── Transformation 3: linear trend residual ───────────────────────────────
    trend20 = _rolling_trend_resid_rms(lr_vals, 20)
    trend60 = _rolling_trend_resid_rms(lr_vals, 60)
    t20_s = pd.Series(trend20, index=df.index)
    t60_s = pd.Series(trend60, index=df.index)
    df["ptx_trend_resid_20"] = _zscore_series(t20_s, trail=60)
    df["ptx_trend_resid_60"] = _zscore_series(t60_s, trail=60)

    # ── Transformation 4: noise proxy (unexplained by EMA smoothing) ──────────
    ema5  = log_ret.ewm(span=5, min_periods=3).mean()
    noise = (log_ret - ema5) ** 2
    noise_roll = noise.rolling(20, min_periods=10).mean()
    df["ptx_noise_score_20"] = _zscore_series(noise_roll, trail=60)

    # ── Composite anomaly score ────────────────────────────────────────────────
    r20_s = pd.Series(rev20, index=df.index)
    r60_s = pd.Series(rev60, index=df.index)
    z_rev20 = _zscore_series(r20_s)
    z_rev60 = _zscore_series(r60_s)

    composite = (
        z_rev20.fillna(0)
        + z_rev60.fillna(0)
        + df["ptx_amplitude_score_20"].fillna(0)
        + df["ptx_trend_resid_20"].fillna(0)
        + df["ptx_noise_score_20"].fillna(0)
    ) / 5.0

    n_valid = (
        z_rev20.notna().astype(int)
        + z_rev60.notna().astype(int)
        + df["ptx_amplitude_score_20"].notna().astype(int)
        + df["ptx_trend_resid_20"].notna().astype(int)
        + df["ptx_noise_score_20"].notna().astype(int)
    )
    composite = composite.where(n_valid >= 3)
    df["ptx_composite_anom"] = composite

    return df
