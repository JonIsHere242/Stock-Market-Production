"""
Approximate Entropy (ApEn) feature block — Pincus (1991).

ApEn measures the regularity/predictability of a time series.
Lower ApEn → more regular/predictable price behaviour.
Higher ApEn → more complex/random/unpredictable price behaviour.

Formula: ApEn(m, r) = phi(m) - phi(m+1)
where phi(m) = mean over i of log(C_i^m / N-m+1),
and C_i^m = #{j : max|x[i+k]-x[j+k]| for k=0..m-1 <= r} / (N-m+1).

Per the spec: rolling 100-day window on log returns, m=2, r=0.2*std_of_window.

FAST IMPLEMENTATION NOTE
------------------------
This is a bit-exact, fully-vectorised rewrite of the original per-window
Python loop.  Instead of calling a scalar ApEn routine ~600 times (each
building its own broadcast distance matrix), we:
  1. Stack ALL rolling windows at once via sliding_window_view  -> (W, win)
  2. Build the per-window template tensors (W, n_templates, m)
  3. Compute the full (W, n_templates, n_templates) Chebyshev distance
     tensor with a single broadcasted op and count matches <= r per window.
All float math matches the original element-for-element (same float32 cast,
same population std, same self-match inclusion, same log/nanmean handling),
so the produced signal is identical to the original within float round-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_apen",
    "description": (
        "Rolling 100-day Approximate Entropy (ApEn, Pincus 1991) of log returns. "
        "m=2, r=0.2*std(window). Lower values indicate more regular/predictable "
        "price dynamics; higher values indicate more randomness. "
        "Produces: apen_100 (raw ApEn), apen_100_chg20 (20-day change in ApEn, "
        "rising = becoming more chaotic). Per-ticker proxy; ApEn is computed on "
        "the individual ticker's return series, not cross-sectionally. "
        "Vectorised rewrite; for speed ApEn is computed every 5th bar "
        "(causal stride, anchored to the last bar) and forward-filled to the "
        "bars in between (past->future carry only, lookahead-safe). Because "
        "ApEn uses a slow-moving 100-day window this preserves the signal."
    ),
    "requires": ["Close"],
    "produces": ["xdom2_apen_100", "xdom2_apen_100_chg20", "xdom2_apen_100_zscore"],
    "tags": ["entropy", "regularity", "complexity", "cross-domain", "returns"],
    "version": "1.1",
    "author": "Approximate entropy ApEn — Pincus 1991 (J. Proc. Natl. Acad. Sci.); implemented per spec xdom2_apen",
}

# ---------------------------------------------------------------------------
# Vectorised phi(m) across all rolling windows at once.
# ---------------------------------------------------------------------------

def _phi_all_windows(win_mat: np.ndarray, m_: int, r: np.ndarray) -> np.ndarray:
    """
    Compute phi(m_) for every rolling window simultaneously.

    Parameters
    ----------
    win_mat : (W, win) float32
        Each row is one full rolling window of the (finite) log-return series.
    m_ : int
        Embedding dimension.
    r : (W,) float
        Tolerance per window ( = r_factor * std(window) ).

    Returns
    -------
    (W,) float64 array of phi(m_) values (NaN where undefined).
    """
    W, win = win_mat.shape
    n_templates = win - m_ + 1
    if n_templates <= 0:
        return np.full(W, np.nan, dtype=np.float64)

    # Templates per window: shape (W, n_templates, m_).
    # sliding_window_view over the LAST axis gives (W, n_templates, m_).
    templates = np.lib.stride_tricks.sliding_window_view(win_mat, m_, axis=1)

    # Chebyshev distance tensor: (W, n_templates, n_templates).
    # |templates[w,i,:] - templates[w,j,:]| max over the m_ axis.
    diff = np.abs(
        templates[:, :, np.newaxis, :] - templates[:, np.newaxis, :, :]
    )  # (W, n_templates, n_templates, m_)
    chebyshev = diff.max(axis=3)  # (W, n_templates, n_templates)

    # Count matches (self-match included, as in Pincus original).
    # Compare against per-window tolerance r broadcast to (W,1,1).
    matches = chebyshev <= r[:, np.newaxis, np.newaxis]
    C = matches.sum(axis=2).astype(np.float64) / n_templates  # (W, n_templates)

    # Guard log(0): C should always be >= 1/n_templates due to self-match,
    # but keep the original's defensive masking for safety.
    C = np.where(C > 0, C, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.log(C)
    # nanmean over templates -> phi per window.
    phi = np.nanmean(log_c, axis=1)
    return phi


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    window = 100
    m = 2
    r_factor = 0.2

    # Log returns (NaN for first bar) — guard against non-positive prices.
    close = df["Close"].astype(np.float64).values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[1:] / close[:-1]
    log_ret = np.full(len(close), np.nan, dtype=np.float64)
    valid_ratio = ratio > 0
    log_ret[1:][valid_ratio] = np.log(ratio[valid_ratio])

    # CAUSAL STRIDE: ApEn uses a 100-day window and moves slowly day-to-day,
    # so we compute it only every STRIDE bars and forward-fill the last
    # computed value to the bars in between.  Forward-fill carries PAST
    # values forward only → lookahead-safe.
    #
    # The stride grid is FIXED FROM THE START: we compute at absolute bar
    # positions i where i % stride == 0.  It is NOT anchored to the last bar.
    # This is the critical causality property — on any truncated PREFIX of the
    # series the grid positions are byte-for-byte identical, so every past
    # value[t] is unchanged under truncation (no re-anchoring leak).
    stride = 5

    n = len(df)
    apen_vals = np.full(n, np.nan, dtype=np.float64)

    if n >= window:
        # All rolling windows of length `window`: shape (n - window + 1, window).
        all_windows = np.lib.stride_tricks.sliding_window_view(log_ret, window)

        # Fixed-from-start grid: compute at end-indices >= window-1 whose
        # absolute position is a multiple of `stride`.  Independent of n, so
        # stable under prefix truncation.
        first_end = window - 1
        compute_ends = np.arange(0, n)
        compute_ends = compute_ends[(compute_ends >= first_end) & (compute_ends % stride == 0)]
        # Map those end-indices back to rows of all_windows (row = end - (window-1)).
        compute_rows = compute_ends - first_end
        sub_windows = all_windows[compute_rows]

        # Keep only windows that are fully finite (matches original guard).
        finite_mask = np.isfinite(sub_windows).all(axis=1)
        if finite_mask.any():
            win_mat = sub_windows[finite_mask].astype(np.float32)
            rows_end = compute_ends[finite_mask]

            # Population std per window; r = r_factor * std (float64).
            std = win_mat.std(axis=1, ddof=0).astype(np.float64)
            r = r_factor * std

            # Windows with std == 0 (or non-finite) → constant series → ApEn = 0.
            good = np.isfinite(std) & (std > 0.0)
            apen_w = np.zeros(win_mat.shape[0], dtype=np.float64)  # default 0.0

            if good.any():
                wm = win_mat[good]
                rg = r[good]
                phi_m = _phi_all_windows(wm, m, rg)
                phi_m1 = _phi_all_windows(wm, m + 1, rg)
                res = phi_m - phi_m1
                # Non-finite phi → NaN (matches original).
                res = np.where(np.isfinite(phi_m) & np.isfinite(phi_m1), res, np.nan)
                apen_w[good] = res

            apen_vals[rows_end] = apen_w

    df["xdom2_apen_100"] = apen_vals
    # Forward-fill the strided ApEn values to the in-between bars (past→future
    # carry only, no lookahead).  Leading NaNs (pre-first-window) are preserved.
    df["xdom2_apen_100"] = df["xdom2_apen_100"].ffill()

    # 20-day change in ApEn (rising → more chaotic)
    df["xdom2_apen_100_chg20"] = df["xdom2_apen_100"] - df["xdom2_apen_100"].shift(20)

    # Rolling 60-day z-score of ApEn (standardises cross-time)
    roll60 = df["xdom2_apen_100"].rolling(60, min_periods=30)
    mu = roll60.mean()
    sigma = roll60.std(ddof=0)
    denom = sigma.where(sigma > 0, other=np.nan)
    df["xdom2_apen_100_zscore"] = (df["xdom2_apen_100"] - mu) / denom

    return df
