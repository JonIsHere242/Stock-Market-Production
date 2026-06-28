"""
_paper_doaj_0f39ae_gaf_mtf_image_stats.py
------------------------------------------
Per-ticker GAF/MTF image-statistic proxy (no CNN).

Inspired by:
  "Hierarchical Attention Fused CNN-LSTM Using Structured 2D Indicator Matrices
   for Stock Trading Action Detection"  (DOAJ ref: 0f39ae...)

The paper encodes a sliding OHLCV window into a 2-D structured matrix image
(Gramian Angular Field / Markov Transition Field) then feeds it into a CNN.
This block implements the ENCODING step only — no neural net — and emits
lightweight summary statistics of those matrices as numeric features.

APPROACH
--------
For every row t the block builds two N×N matrices from the N-day trailing
window of log-returns (window = GAF_W = 20).  NaN is emitted for the first
GAF_W − 1 rows.

GAF (Gramian Angular Field — Summation variant GASF and Difference variant GADF):
  1.  Clip the window of log-returns to [min, max] (window-local, causal).
  2.  Rescale to [-1, 1]:  x_hat = 2*(x − min)/(max − min) − 1
      (if max == min, treat as all-zeros, which yields all-1 GASF).
  3.  phi_i = arccos(x_hat_i)
  4.  GASF[i,j] = cos(phi_i + phi_j) = x_hat_i * x_hat_j − sqrt(1−x_hat_i²)*sqrt(1−x_hat_j²)
      GADF[i,j] = sin(phi_i − phi_j) = sqrt(1−x_hat_i²)*x_hat_j − x_hat_i*sqrt(1−x_hat_j²)
  5.  Summary statistics emitted:
        gmt_gasf_mean            – overall mean (correlation structure level)
        gmt_gasf_diag_energy     – mean of main diagonal (self-correlation)
        gmt_gasf_tri_asym        – GADF upper-tri minus lower-tri mean (directional asymmetry; GADF is antisymmetric so this is non-trivial)
        gmt_gasf_antidiag_mean   – mean of the GASF anti-diagonal (oldest↔newest coupling)
        gmt_gadf_std             – std of GADF (captures angular spread / volatility)

MTF (Markov Transition Field):
  1.  Assign each value in the window to one of Q=4 quantile bins (window-local,
      causal — bin edges from the window itself).
  2.  Build Q×Q Markov transition matrix M from consecutive-bin transitions.
  3.  Assign MTF[i,j] = M[bin_i, bin_j] (transition probability at that time pair).
  4.  Summary statistics emitted:
        gmt_mtf_diag_mass        – sum of diagonal / total (self-transition weight)
        gmt_mtf_entropy          – Shannon entropy of the transition matrix (row-normalised)
        gmt_mtf_offdiag_std      – std of off-diagonal elements (spread of jumps)
        gmt_mtf_max_transition   – maximum single off-diagonal entry (dominant jump)

All windows are strictly causal (rows ≤ t).  No global normalisation.
Leading NaN rows: first GAF_W − 1 rows.
"""

import math
import warnings
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_doaj_0f39ae_gaf_mtf_image_stats",
    "description": (
        "Per-ticker GAF/MTF image-statistic proxy (no CNN): encodes a causal "
        "trailing window of log-returns into Gramian Angular Field and Markov "
        "Transition Field matrices and emits ~9 summary statistics as features."
    ),
    "requires":    ["Close"],
    "produces":    [
        "gmt_gasf_mean",
        "gmt_gasf_diag_energy",
        "gmt_gasf_tri_asym",
        "gmt_gasf_antidiag_mean",
        "gmt_gadf_std",
        "gmt_mtf_diag_mass",
        "gmt_mtf_entropy",
        "gmt_mtf_offdiag_std",
        "gmt_mtf_max_transition",
    ],
    "tags":        ["image_encoding", "structure", "experimental", "paper"],
    "version":     "1.0",
    "author":      "paper: doaj:0f39ae - Hierarchical Attention Fused CNN-LSTM (2D Indicator Matrices)",
}

# ---------------------------------------------------------------------------
# Constants — keep window small for speed
# ---------------------------------------------------------------------------
GAF_W: int = 20   # trailing window length for matrix encoding
MTF_Q: int = 4    # number of quantile bins for MTF


# ---------------------------------------------------------------------------
# Internal helpers (pure numpy, no IO, no randomness)
# ---------------------------------------------------------------------------

def _gasf_gadf_stats(window: np.ndarray):
    """
    Given a 1-D array of length W (log-returns, already validated finite),
    compute GASF and GADF summary statistics.

    Returns tuple of 5 floats:
        (gasf_mean, gasf_diag_energy, gasf_tri_asym, gasf_antidiag_mean, gadf_std)
    or (nan, nan, nan, nan, nan) on degenerate input.
    """
    w = len(window)

    # ---- 1.  Causal window-local min/max scaling to [-1, 1] ----------------
    lo = window.min()
    hi = window.max()
    span = hi - lo
    if span < 1e-12:
        # constant window → all phi = 0 → GASF = all 1, GADF = all 0
        x_hat = np.zeros(w, dtype=np.float64)
    else:
        x_hat = 2.0 * (window - lo) / span - 1.0

    # Clamp to [-1, 1] to guard floating-point edge cases before arccos
    x_hat = np.clip(x_hat, -1.0, 1.0)
    phi = np.arccos(x_hat)                       # shape (W,)

    # sin_phi is needed for GADF
    sin_phi = np.sqrt(np.maximum(1.0 - x_hat ** 2, 0.0))  # stable sqrt

    # ---- 2.  GASF[i,j] = cos(phi_i + phi_j) = x_i*x_j - sin_i*sin_j ------
    # outer products
    xx = np.outer(x_hat, x_hat)          # (W, W)
    ss = np.outer(sin_phi, sin_phi)      # (W, W)
    gasf = xx - ss                       # (W, W), values in [-1, 1]

    # ---- 3.  GADF[i,j] = sin(phi_i - phi_j) = sin_i*x_j - x_i*sin_j ------
    gadf = np.outer(sin_phi, x_hat) - np.outer(x_hat, sin_phi)   # (W, W)

    # ---- 4.  Summary statistics for GASF -----------------------------------
    gasf_mean = float(np.mean(gasf))

    # Diagonal energy: mean of the main diagonal cos(2*phi_i) — reflects how
    # far individual returns are from the boundary of the angular mapping.
    gasf_diag_energy = float(np.mean(np.diag(gasf)))

    # GADF is antisymmetric: GADF[i,j] = -GADF[j,i].  Its upper triangle
    # captures the angular-difference structure (non-trivial signal).
    idx_upper = np.triu_indices(w, k=1)
    idx_lower = np.tril_indices(w, k=-1)
    # Upper-tri mean of GADF gives the net directional asymmetry in return pairs
    gadf_tri_asym = float(np.mean(gadf[idx_upper]) - np.mean(gadf[idx_lower]))

    # Anti-diagonal of GASF: entries (i, w-1-i) — oldest value paired with
    # newest and vice versa.  Captures how the most-recent bar couples to history.
    anti_idx = (np.arange(w), np.arange(w - 1, -1, -1))
    gasf_antidiag_mean = float(np.mean(gasf[anti_idx]))

    # ---- 5.  Summary statistic for GADF ------------------------------------
    gadf_std = float(np.std(gadf))

    return (gasf_mean, gasf_diag_energy, gadf_tri_asym, gasf_antidiag_mean, gadf_std)


def _mtf_stats(window: np.ndarray, q: int = MTF_Q):
    """
    Given a 1-D array of length W, compute MTF summary statistics.

    Returns tuple of 4 floats:
        (diag_mass, entropy, offdiag_std, max_offdiag_transition)
    or (nan, nan, nan, nan) on degenerate input.
    """
    w = len(window)

    # ---- 1.  Quantile bin assignment (window-local, causal) ----------------
    # np.quantile over the window itself — purely causal (no future data).
    # Use q equal-width quantile edges.
    edges = np.quantile(window, np.linspace(0.0, 1.0, q + 1))
    # digitize: bin index 0..q-1
    # Use left-closed intervals; clip to [0, q-1].
    bins = np.digitize(window, edges[1:-1])  # 0-indexed, 0..q-1

    # ---- 2.  Q×Q Markov transition count matrix ----------------------------
    # Vectorised: encode (from, to) pairs as single integers and use bincount
    from_bins = bins[:-1].astype(np.intp)
    to_bins   = bins[1:].astype(np.intp)
    flat_idx  = from_bins * q + to_bins
    counts    = np.bincount(flat_idx, minlength=q * q).astype(np.float64)
    trans     = counts.reshape(q, q)

    # Row-normalise to get probability matrix
    row_sums = trans.sum(axis=1, keepdims=True)
    # Avoid divide by zero for rows with no transitions (shouldn't happen for
    # large enough window, but guard for very short windows or constant series).
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    prob = trans / row_sums_safe   # (Q, Q) probability matrix

    # ---- 3.  Summary statistics --------------------------------------------
    # Self-transition concentration: diagonal mass (sum of diagonal / total mass)
    total_prob = prob.sum()
    diag_mass = float(np.trace(prob) / total_prob) if total_prob > 0 else float("nan")

    # Transition entropy: Shannon entropy over the flattened prob matrix
    # (ignoring zeros).
    flat = prob.ravel()
    nonzero = flat[flat > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero + 1e-15)))

    # Off-diagonal spread
    offdiag_mask = ~np.eye(q, dtype=bool)
    offdiag_vals = prob[offdiag_mask]
    offdiag_std = float(np.std(offdiag_vals))
    max_transition = float(np.max(offdiag_vals)) if offdiag_vals.size > 0 else float("nan")

    return (diag_mass, entropy, offdiag_std, max_transition)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GAF and MTF image-summary features on a trailing window of
    log-returns.  One set of features per row.  First GAF_W-1 rows → NaN.
    """
    n = len(df)

    # Pre-allocate output arrays as NaN
    gasf_mean       = np.full(n, np.nan, dtype=np.float64)
    gasf_diag_en    = np.full(n, np.nan, dtype=np.float64)
    gasf_tri_asym   = np.full(n, np.nan, dtype=np.float64)
    gasf_antidiag   = np.full(n, np.nan, dtype=np.float64)
    gadf_std        = np.full(n, np.nan, dtype=np.float64)
    mtf_diag_mass   = np.full(n, np.nan, dtype=np.float64)
    mtf_entropy     = np.full(n, np.nan, dtype=np.float64)
    mtf_offdiag_std = np.full(n, np.nan, dtype=np.float64)
    mtf_max_trans   = np.full(n, np.nan, dtype=np.float64)

    # Log-returns; first element is NaN (diff at position 0)
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret = np.log(close_arr[1:] / close_arr[:-1])  # length n-1
        log_ret = np.concatenate([[np.nan], log_ret])     # length n, aligns with rows

    # We need GAF_W consecutive valid log-returns to fill row t.
    # Row t uses window log_ret[t-GAF_W+1 : t+1].
    # The very first log_ret (index 0) is NaN; so effective start is row GAF_W
    # (i.e., 0-based index GAF_W — the (GAF_W+1)-th row).
    first_valid = GAF_W   # index of the first row with a full window of valid log_ret

    for t in range(first_valid, n):
        window = log_ret[t - GAF_W + 1 : t + 1]  # shape (GAF_W,)

        # Skip if any NaN in the window (e.g. stock had a gap)
        if not np.all(np.isfinite(window)):
            continue

        # -- GAF stats --
        (gm, gde, gta, gad, gds) = _gasf_gadf_stats(window)
        gasf_mean[t]     = gm
        gasf_diag_en[t]  = gde
        gasf_tri_asym[t] = gta
        gasf_antidiag[t] = gad
        gadf_std[t]      = gds

        # -- MTF stats --
        (dm, ent, ods, mxt) = _mtf_stats(window, q=MTF_Q)
        mtf_diag_mass[t]   = dm
        mtf_entropy[t]     = ent
        mtf_offdiag_std[t] = ods
        mtf_max_trans[t]   = mxt

    # Assign to df — new columns only
    idx = df.index
    df["gmt_gasf_mean"]          = pd.array(gasf_mean,       dtype="Float64")
    df["gmt_gasf_diag_energy"]   = pd.array(gasf_diag_en,    dtype="Float64")
    df["gmt_gasf_tri_asym"]      = pd.array(gasf_tri_asym,   dtype="Float64")
    df["gmt_gasf_antidiag_mean"] = pd.array(gasf_antidiag,   dtype="Float64")
    df["gmt_gadf_std"]           = pd.array(gadf_std,         dtype="Float64")
    df["gmt_mtf_diag_mass"]      = pd.array(mtf_diag_mass,   dtype="Float64")
    df["gmt_mtf_entropy"]        = pd.array(mtf_entropy,      dtype="Float64")
    df["gmt_mtf_offdiag_std"]    = pd.array(mtf_offdiag_std, dtype="Float64")
    df["gmt_mtf_max_transition"] = pd.array(mtf_max_trans,   dtype="Float64")

    return df
