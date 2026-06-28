"""
_paper_2604_19107_rmt_complexity_gap.py
========================================
Per-ticker proxy for the Random Matrix Theory (RMT) Complexity Gap framework
described in:

  "Structural Dynamics of Global Financial Markets: A Random Matrix Theory-Based
   Complexity Gap Approach"  —  arXiv:2604.19107

PER-TICKER PROXY NOTE
---------------------
The paper is cross-sectional: it builds a correlation matrix across many asset
return series.  Here we have ONE series per call.  The closest per-ticker
analogue is a MULTI-HORIZON RETURN MATRIX:

  - At each date t, take the trailing W rows of log-returns.
  - Build a matrix M of shape (W, H) where columns are returns over fixed
    horizons {1, 2, 3, 5, 10, 20} days — each horizon h column is the
    h-day compounded log-return ending at that row.
  - Compute the (H × H) correlation matrix of this (W, H) matrix.
  - Apply RMT metrics to the H-eigenvalues.

This embeds "how correlated are short-horizon vs long-horizon return signals
for this ticker?" — structurally identical to the cross-sectional question but
within a single ticker's own multi-scale dynamics.

COMPLEXITY GAP (headline metric from paper):
  gap = (lambda_1 / mean(lambda)) - avg_off_diag_corr

A positive gap indicates rich multi-factor structure across time horizons; gap
collapsing toward zero signals strong single-mode synchronization (stress /
trending regime). Lower gap → higher realized future volatility.

FEATURES EMITTED (6)
--------------------
  rmt_lambda1_norm         : normalized largest eigenvalue  (lambda_1 / mean)
  rmt_avg_corr             : average off-diagonal correlation of H×H corr matrix
  rmt_complexity_gap       : lambda1_norm - avg_corr  (headline paper metric)
  rmt_participation_ratio  : eigenvalue participation ratio = (sum lambda)^2 / (H * sum lambda^2)
                             close to 1 = spread eigenvalues; near 1/H = single-factor
  rmt_eigval_dispersion    : std(eigenvalues) / mean(eigenvalues)
  rmt_perm_entropy_20      : permutation (ordinal) entropy of 1-day log-returns
                             over a 20-bar rolling window, order m=3
                             (paper section on "directional diversity")

WINDOW / PERFORMANCE NOTES
---------------------------
Rolling window W=60, horizons H=6. Correlation matrix is 6×6 → eigvalsh is
O(H^3) = negligible.  Total cost is dominated by the rolling window loop.
We implement the rolling via stride tricks (as_strided) — avoids an explicit
Python loop over rows.  Target: well under 100 ms on 700 rows.

CAUSALITY
---------
At row t, the multi-horizon matrix uses ONLY rows [t-W+1 .. t] of log-returns.
Horizon h column at row i uses returns ending at i (log(Close_i / Close_{i-h}));
all of that history is strictly <= t.  No forward-looking shift.
"""

import warnings
from itertools import permutations
from math import factorial
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "rmt_complexity_gap",
    "description": (
        "Per-ticker RMT complexity gap proxy: builds a multi-horizon log-return "
        "matrix, extracts eigenvalue structure (normalized lambda1, avg corr, "
        "complexity gap, participation ratio, dispersion) and ordinal entropy "
        "— a rolling causal single-series analogue of arXiv:2604.19107."
    ),
    "requires": ["Close"],
    "produces": [
        "rmt_lambda1_norm",
        "rmt_avg_corr",
        "rmt_complexity_gap",
        "rmt_participation_ratio",
        "rmt_eigval_dispersion",
        "rmt_perm_entropy_20",
    ],
    "tags": ["market_regime", "volatility", "experimental", "rmt"],
    "version": "1.0",
    "author": "paper arXiv:2604.19107 — per-ticker proxy",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HORIZONS: List[int] = [1, 2, 3, 5, 10, 20]   # H = 6 columns
_H = len(_HORIZONS)
_WINDOW: int = 60          # trailing rows fed into the correlation matrix
_MIN_OBS: int = 30         # minimum non-NaN rows required before emitting a value
_PERM_WIN: int = 20        # permutation entropy rolling window
_PERM_ORDER: int = 3       # ordinal pattern order m (m! = 6 patterns)
_EPS: float = 1e-12        # guard for division by near-zero


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perm_entropy_series(log_rets: np.ndarray, win: int, order: int) -> np.ndarray:
    """
    Rolling permutation (ordinal) entropy of a 1-D log-return array — vectorised.

    At each t we count the `win` ordinal (argsort) patterns of the order-length
    subsequences ending by t-1, and take the Shannon entropy of that pattern
    distribution, normalised to [0,1].  Entropy is invariant to how patterns are
    labelled, so this matches the old per-window dict-counting implementation while
    using no Python loops.  First (win + order - 2) values are NaN.
    """
    n = len(log_rets)
    out = np.full(n, np.nan)
    n_patterns = factorial(order)
    log_n = np.log(float(n_patterns))
    if log_n == 0 or n < order:
        return out

    # Ordinal pattern (argsort) of every order-length window; encode each permutation
    # to a FIXED category id so the rolling counts line up across windows.
    sw = np.lib.stride_tricks.sliding_window_view(log_rets, order)   # (n_pat, order)
    n_pat = sw.shape[0]
    finite = np.all(np.isfinite(sw), axis=1)
    args = np.argsort(sw, axis=1, kind="stable")
    weights = order ** np.arange(order - 1, -1, -1)
    codes = args @ weights
    code_to_cat = {sum(int(p[k]) * int(weights[k]) for k in range(order)): ci
                   for ci, p in enumerate(permutations(range(order)))}
    cat = np.array([code_to_cat.get(int(c), -1) for c in codes])

    onehot = np.zeros((n_pat, n_patterns))
    good = finite & (cat >= 0)
    onehot[np.flatnonzero(good), cat[good]] = 1.0

    if n_pat < win:
        return out

    # Rolling count of each pattern over a width-`win` window (exact integer counts).
    cs = np.cumsum(onehot, axis=0)
    counts = np.full((n_pat, n_patterns), np.nan)
    counts[win - 1] = cs[win - 1]
    if n_pat > win:
        counts[win:] = cs[win:] - cs[:-win]

    n_valid = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / n_valid[:, None]
        plogp = np.where(p > 0, p * np.log(p), 0.0)
        ent = -plogp.sum(axis=1) / log_n      # normalised to [0,1]
    ent[n_valid < 2] = np.nan                 # matches old `if valid < 2: skip`

    # Window ending at pattern-index e maps to output time t = e + (order - 1).
    e = np.arange(win - 1, n_pat)
    out[e + (order - 1)] = ent[e]
    return out


def _rmt_metrics_from_corr(corr: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Given an (H, H) correlation matrix, return:
      (lambda1_norm, avg_corr, complexity_gap, participation_ratio, eigval_dispersion)
    Returns (nan, nan, nan, nan, nan) if computation fails.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # eigvalsh returns eigenvalues in ascending order; all real for symmetric.
            eigvals = np.linalg.eigvalsh(corr)
        except np.linalg.LinAlgError:
            return (np.nan,) * 5

    # Guard: eigenvalues of a valid corr matrix should be >= 0 in theory;
    # numerical noise can push tiny ones slightly negative — clip.
    eigvals = np.maximum(eigvals, 0.0)

    mean_lambda = eigvals.mean()
    if mean_lambda < _EPS:
        return (np.nan,) * 5

    lambda1 = eigvals[-1]                         # largest eigenvalue
    lambda1_norm = lambda1 / mean_lambda           # normalized (Marchenko-Pastur ref = 1)

    # Average off-diagonal correlation (H*(H-1) elements)
    H = corr.shape[0]
    off_diag_sum = corr.sum() - H                 # subtract diagonal (all 1.0)
    n_off = H * (H - 1)
    avg_corr = off_diag_sum / n_off if n_off > 0 else np.nan

    # Complexity gap (paper headline)
    complexity_gap = lambda1_norm - float(avg_corr) if np.isfinite(avg_corr) else np.nan

    # Participation ratio = (sum lambda)^2 / (H * sum(lambda^2))
    sum_l = eigvals.sum()
    sum_l2 = (eigvals ** 2).sum()
    participation_ratio = (sum_l ** 2) / (H * sum_l2) if sum_l2 > _EPS else np.nan

    # Eigenvalue dispersion = std / mean
    eigval_dispersion = eigvals.std() / mean_lambda

    return lambda1_norm, avg_corr, complexity_gap, participation_ratio, eigval_dispersion


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling per-ticker RMT complexity gap features.

    For each date t (starting once we have enough history):
      1. Extract trailing _WINDOW rows of multi-horizon log-returns.
      2. Compute correlation matrix across the H horizons.
      3. Extract eigenvalue-based metrics (lambda1_norm, avg_corr, gap, PR, disp).

    Also compute rolling permutation entropy of raw 1-day log-returns.
    """
    n = len(df)

    # Initialise output arrays as NaN
    lambda1_norm_arr = np.full(n, np.nan)
    avg_corr_arr = np.full(n, np.nan)
    gap_arr = np.full(n, np.nan)
    pr_arr = np.full(n, np.nan)
    disp_arr = np.full(n, np.nan)

    # 1-day log returns (first element is NaN)
    close = df["Close"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.empty(n, dtype=np.float64)
        log_rets[0] = np.nan
        log_rets[1:] = np.log(close[1:] / close[:-1])
        # Guard for zero or negative closes
        log_rets = np.where(np.isfinite(log_rets), log_rets, np.nan)

    max_h = max(_HORIZONS)    # = 20
    # We need at least _WINDOW + max_h - 1 rows of close prices to fill
    # the very first valid window.
    first_valid = _WINDOW + max_h - 1   # = 79

    # Precompute the full multi-horizon return matrix ONCE (vectorized).
    # C[i, c] = sum(log_rets[i-h+1 .. i]) for horizon h=_HORIZONS[c]; NaN when i < h-1
    # or any of those h returns is non-finite.  sliding_window_view(...).sum(axis=1)
    # sums each width-h window with the SAME np.sum, in the same order, as the old
    # per-row _build_horizon_matrix (so it is bit-identical) and NaN propagates exactly
    # like the old "all finite else NaN" rule -- but with no Python loop over rows.
    # The window matrix at row t is then just the trailing slice C[t-W+1 : t+1].
    C = np.full((n, _H), np.nan)
    for c, h in enumerate(_HORIZONS):
        if n >= h:
            sw = np.lib.stride_tricks.sliding_window_view(log_rets, h)  # (n-h+1, h)
            C[h - 1:, c] = sw.sum(axis=1)

    for t in range(first_valid, n):
        M = C[t - _WINDOW + 1: t + 1]            # (_WINDOW, H) trailing-window slice

        # Filter rows whose every horizon is finite (matches the old valid_mask).
        valid_mask = np.isfinite(M).all(axis=1)
        n_valid = valid_mask.sum()
        if n_valid < _MIN_OBS:
            continue

        M_valid = M[valid_mask]  # shape (n_valid, H)

        # Compute correlation matrix across horizons (H x H)
        # Each column is a horizon; we want pairwise corr across H columns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # np.corrcoef(rowvar=False) treats columns as variables.
            try:
                corr = np.corrcoef(M_valid, rowvar=False)
            except Exception:
                continue

        if not np.all(np.isfinite(corr)):
            continue

        l1n, ac, gap, pr, disp = _rmt_metrics_from_corr(corr)
        lambda1_norm_arr[t] = l1n
        avg_corr_arr[t] = ac
        gap_arr[t] = gap
        pr_arr[t] = pr
        disp_arr[t] = disp

    # Replace any inadvertent inf with NaN
    def _clean(arr: np.ndarray) -> np.ndarray:
        arr[~np.isfinite(arr)] = np.nan
        return arr

    # Permutation entropy (vectorized helper — separate pass)
    perm_ent = _perm_entropy_series(log_rets, _PERM_WIN, _PERM_ORDER)
    perm_ent = _clean(perm_ent)

    idx = df.index
    df["rmt_lambda1_norm"] = pd.Series(_clean(lambda1_norm_arr), index=idx)
    df["rmt_avg_corr"] = pd.Series(_clean(avg_corr_arr), index=idx)
    df["rmt_complexity_gap"] = pd.Series(_clean(gap_arr), index=idx)
    df["rmt_participation_ratio"] = pd.Series(_clean(pr_arr), index=idx)
    df["rmt_eigval_dispersion"] = pd.Series(_clean(disp_arr), index=idx)
    df["rmt_perm_entropy_20"] = pd.Series(perm_ent, index=idx)

    return df
