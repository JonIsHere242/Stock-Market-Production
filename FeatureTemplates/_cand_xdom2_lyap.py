"""
Largest Lyapunov exponent proxy (Rosenstein 1993) -- per-ticker OHLCV implementation.

Rosenstein et al. (1993) "A practical method for calculating largest Lyapunov exponents
from small data sets", Physica D 65, 117-134.

Per-ticker proxy: rolling 120-day window of log-return series, delay-embedded with
m=3 dimensions and lag=1.  For each point in the embedding, the nearest neighbour
(excluding temporally adjacent points) is found, and the average log-divergence of
those pairs over `n_steps=5` future steps is computed.  The slope of that divergence
curve (linear fit) is the Lyapunov estimate for the window.

Positive & rising = sensitive dependence / chaotic instability, historically
associated with impending trend breaks and volatility clustering.

This is inherently a per-ticker rolling estimate; no cross-sectional information used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings

METADATA = {
    "name": "xdom2_lyap",
    "description": (
        "Rolling 120-day Rosenstein (1993) largest Lyapunov exponent proxy on "
        "delay-embedded log-returns (m=3, lag=1, n_steps=5). "
        "xdom2_lyap_120: level estimate; xdom2_lyap_slope: 20-day rolling slope "
        "of level (trend/acceleration of chaotic instability). "
        "Per-ticker proxy -- no cross-sectional data needed. "
        "Positive & rising signals sensitive dependence ahead of breakdowns."
    ),
    "requires": ["Close"],
    "produces": ["xdom2_lyap_120", "xdom2_lyap_slope"],
    "tags": ["lyapunov", "chaos", "nonlinear", "cross-domain", "returns"],
    "version": "1.0",
    "author": "Rosenstein et al. (1993) 'A practical method for calculating largest Lyapunov exponents from small data sets', Physica D 65, 117-134. Impl: cross-domain-method batch 2 (xdom2_lyap).",
}

# ── tuning constants ────────────────────────────────────────────────────────
_WINDOW    = 120   # rolling window length in trading days
_EMBED_DIM = 3     # embedding dimension m
_LAG       = 1     # delay lag (in samples)
_N_STEPS   = 5     # number of divergence steps to track
_MIN_OBS   = _LAG * (_EMBED_DIM - 1) + _N_STEPS + 5  # minimum non-NaN rows needed
_SLOPE_WIN = 20    # window for slope of lyap level


def _rosenstein_lyap(returns: np.ndarray, m: int, lag: int, n_steps: int) -> float:
    """
    Compute the largest Lyapunov exponent estimate via Rosenstein (1993) on a
    1-D array of returns.

    Returns the slope of the mean-log-divergence vs step index (linear fit).
    Returns np.nan if the window is too short or degenerate.
    """
    n = len(returns)
    # Build delay-embedded matrix: shape (n_pts, m)
    # Each row i: [r[i], r[i+lag], r[i+2*lag], ..., r[i+(m-1)*lag]]
    n_pts = n - (m - 1) * lag
    if n_pts < _N_STEPS + 2:
        return np.nan

    # Vectorised embedding
    idx = np.arange(n_pts)[:, None] + np.arange(m) * lag  # (n_pts, m)
    embedded = returns[idx]  # (n_pts, m)

    # --- nearest-neighbour search (O(n_pts^2), but n_pts <= 120 so < 14400 ops) ---
    # Euclidean distance matrix
    # To avoid lookahead: nearest neighbour must differ by >= m*lag+1 in time index
    min_sep = m * lag + 1

    # Pairwise squared distances
    diff = embedded[:, None, :] - embedded[None, :, :]  # (n, n, m)
    dist2 = (diff ** 2).sum(axis=-1)  # (n, n)

    # Mask out diagonal and temporally adjacent pairs
    n_e = n_pts
    mask = np.abs(np.arange(n_e)[:, None] - np.arange(n_e)[None, :]) < min_sep
    dist2[mask] = np.inf

    nn_idx = np.argmin(dist2, axis=1)  # nearest neighbour index for each point

    # --- compute divergence over n_steps ---
    # Only use pairs where both point and its NN can advance n_steps
    valid = (np.arange(n_e) + n_steps < n_e) & (nn_idx + n_steps < n_e)
    if valid.sum() < 2:
        return np.nan

    pts_v  = np.where(valid)[0]
    nn_v   = nn_idx[pts_v]

    log_div = np.zeros((len(pts_v), n_steps))
    for s in range(1, n_steps + 1):
        future_pts = embedded[pts_v + s]
        future_nn  = embedded[nn_v  + s]
        d = np.sqrt(((future_pts - future_nn) ** 2).sum(axis=1))
        # initial distance
        d0 = np.sqrt(dist2[pts_v, nn_v])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ratio = np.where(d0 > 0, d / (d0 + 1e-12), np.nan)
            log_div[:, s - 1] = np.where(ratio > 0, np.log(ratio), np.nan)

    # Mean log divergence across pairs for each step
    mean_log_div = np.nanmean(log_div, axis=0)

    if np.all(np.isnan(mean_log_div)):
        return np.nan

    steps = np.arange(1, n_steps + 1, dtype=float)
    valid_steps = ~np.isnan(mean_log_div)
    if valid_steps.sum() < 2:
        return np.nan

    # Linear fit: slope = Lyapunov exponent estimate
    slope = np.polyfit(steps[valid_steps], mean_log_div[valid_steps], 1)[0]
    return float(slope)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Log returns (shift(1) is safe -- uses only past Close)
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values  # NaN at index 0

    lyap_vals = np.full(n, np.nan)

    if n >= _WINDOW:
        for end in range(_WINDOW - 1, n):
            start = end - _WINDOW + 1
            window_ret = log_ret[start: end + 1]
            # Drop leading NaN (only position 0 in the full series is NaN)
            valid = window_ret[~np.isnan(window_ret)]
            if len(valid) < _MIN_OBS:
                continue
            lyap_vals[end] = _rosenstein_lyap(valid, _EMBED_DIM, _LAG, _N_STEPS)

    df["xdom2_lyap_120"] = lyap_vals

    # 20-day rolling slope of the Lyapunov level
    lyap_series = pd.Series(lyap_vals, index=df.index)
    if n >= _SLOPE_WIN:
        slope_vals = np.full(n, np.nan)
        x = np.arange(_SLOPE_WIN, dtype=float)
        x -= x.mean()
        x_ss = (x ** 2).sum()
        for i in range(_SLOPE_WIN - 1, n):
            chunk = lyap_series.iloc[i - _SLOPE_WIN + 1: i + 1].values
            if np.isnan(chunk).sum() > _SLOPE_WIN // 2:
                continue
            # Use only non-NaN for linear slope estimate
            mask = ~np.isnan(chunk)
            if mask.sum() < 3:
                continue
            xi = x[mask]
            yi = chunk[mask]
            xi_c = xi - xi.mean()
            denom = (xi_c ** 2).sum()
            if denom == 0:
                continue
            slope_vals[i] = (xi_c * yi).sum() / denom
        df["xdom2_lyap_slope"] = slope_vals
    else:
        df["xdom2_lyap_slope"] = np.nan

    return df
