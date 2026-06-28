"""
Recurrence Quantification Analysis (RQA) — Determinism & Laminarity (FAST)
Per-ticker proxy using a rolling 80-day recurrence matrix on delay-embedded
log-returns (embedding dimension m=2, lag=1).

High determinism -> price returns follow more deterministic/trending dynamics.
High laminarity  -> returns are "sticky" / trapped in narrow states.

Per-ticker implementation: faithful to the Webber-Zbilut RQA definitions;
inherently per-ticker (no cross-sectional ranking needed).

FAST rewrite (bit-exact vs. the reference loop implementation):
  * Diagonal/vertical line counting is fully vectorized with numpy shifts.
    A recurrent point belongs to a line of length >= 2 iff it has a True
    neighbour along the same line direction (vertical: up/down; diagonal:
    up-left/down-right). So points-on-lines>=2 == (RP & (shift_pos | shift_neg)).sum()
    This replaces the per-diagonal / per-column Python run-counting loops and
    is mathematically identical to the run-length-based counting.
  * The per-window NxN distance matrix is built with broadcasting (N<=79).
  * Self-recurrence (main diagonal) is excluded exactly as in the reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_rqa",
    "description": (
        "Rolling 80-day Recurrence Quantification Analysis on delay-embedded log-returns "
        "(m=2, lag=1, fixed recurrence rate threshold 0.10). "
        "xdom2_rqa_det_80: fraction of recurrent points on diagonal lines>=2 (Determinism). "
        "xdom2_rqa_lam_80: fraction of recurrent points on vertical lines>=2 (Laminarity). "
        "Per-ticker per-day rolling window; no cross-sectional data needed. "
        "High determinism=trending/predictable dynamics; high laminarity=sticky/trapped states. "
        "FAST vectorized rewrite: line counts computed via numpy neighbour-shift masks "
        "(bit-exact with the Webber-Zbilut run-length definitions)."
    ),
    "requires": ["Close"],
    "produces": ["xdom2_rqa_det_80", "xdom2_rqa_lam_80"],
    "tags": ["recurrence", "nonlinear", "dynamics", "rqa", "cross-domain"],
    "version": "1.1",
    "author": "Recurrence quantification: determinism & laminarity (Webber-Zbilut)",
}


def _rqa_det_lam(returns_window: np.ndarray, rr_target: float = 0.10) -> tuple[float, float]:
    """
    Compute RQA Determinism and Laminarity for a single window of returns.

    Vectorized, bit-exact with the reference run-length implementation:
      - A recurrent point is on a diagonal line of length >= 2 iff it has a
        True neighbour at (i-1, j-1) or (i+1, j+1).
      - A recurrent point is on a vertical line of length >= 2 iff it has a
        True neighbour at (i-1, j) or (i+1, j).
    Summing these masks counts exactly the points that lie in runs of length
    >= 2, identical to enumerating maximal runs and summing those with len>=2.

    Returns
    -------
    (determinism, laminarity) in [0, 1], or (nan, nan) if insufficient data.
    """
    W = len(returns_window)
    m = 2    # embedding dimension
    lag = 1  # time lag

    # Delay-embedded phase-space vectors: state[t] = [r[t], r[t+lag]]
    n_vecs = W - (m - 1) * lag
    if n_vecs < 4:
        return np.nan, np.nan

    # Shape: (n_vecs, m)
    X = np.column_stack([returns_window[i: i + n_vecs] for i in range(0, m * lag, lag)])
    N = n_vecs

    # Pairwise Euclidean distances (N x N) via broadcasting; N <= 79 so this is small.
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (N, N, m)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    # Threshold for target recurrence rate, from strictly-upper-triangle distances.
    upper_idx = np.triu_indices(N, k=1)
    upper_dists = dist[upper_idx]
    if upper_dists.size == 0:
        return np.nan, np.nan

    threshold = np.quantile(upper_dists, rr_target)

    # Boolean recurrence matrix; exclude main diagonal (self-recurrence).
    RP = dist <= threshold
    np.fill_diagonal(RP, False)

    total_recurrent = int(RP.sum())
    if total_recurrent == 0:
        return 0.0, 0.0

    # ---- Determinism: points on diagonal lines (offset != 0) of length >= 2 ----
    # Diagonal neighbours: (i-1, j-1) and (i+1, j+1).
    # Build padded shifts so border points have False neighbours.
    up_left = np.zeros_like(RP)
    up_left[1:, 1:] = RP[:-1, :-1]        # neighbour at (i-1, j-1)
    down_right = np.zeros_like(RP)
    down_right[:-1, :-1] = RP[1:, 1:]     # neighbour at (i+1, j+1)
    diag_in_line = RP & (up_left | down_right)
    det_points = int(diag_in_line.sum())
    determinism = det_points / total_recurrent

    # ---- Laminarity: points on vertical lines of length >= 2 ----
    # Vertical neighbours: (i-1, j) and (i+1, j).
    up = np.zeros_like(RP)
    up[1:, :] = RP[:-1, :]                # neighbour at (i-1, j)
    down = np.zeros_like(RP)
    down[:-1, :] = RP[1:, :]             # neighbour at (i+1, j)
    vert_in_line = RP & (up | down)
    lam_points = int(vert_in_line.sum())
    laminarity = lam_points / total_recurrent

    return float(determinism), float(laminarity)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    W = 80  # rolling window size (days)

    n = len(df)
    det_vals = np.full(n, np.nan)
    lam_vals = np.full(n, np.nan)

    close = df["Close"].values.astype(np.float64)

    # Log-returns: r[t] = log(close[t] / close[t-1]); r[0] = NaN (uses only past data).
    log_ret = np.empty(n)
    if n > 0:
        log_ret[0] = np.nan
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
            log_ret[1:] = np.log(ratio)

    # For each day t use log_ret[t-W+1 .. t] (W points, all past-or-current).
    for t in range(W - 1, n):
        window = log_ret[t - W + 1: t + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        if np.isnan(window).any():
            window = valid
        det_v, lam_v = _rqa_det_lam(window, rr_target=0.10)
        det_vals[t] = det_v
        lam_vals[t] = lam_v

    df["xdom2_rqa_det_80"] = det_vals
    df["xdom2_rqa_lam_80"] = lam_vals

    return df
