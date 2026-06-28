"""
Singular Spectrum Analysis (SSA) leading-component energy feature block.

Broomhead-King (1986) SSA: embed a return series in a trajectory (Hankel) matrix
via a lag-embedding window, decompose with SVD, and measure how much variance lives
in the top eigentriple(s) vs. the noise floor.

Per-ticker proxy: computes on rolling 60-day log-return windows.
  - xdom_ssa_energy_lead   : share of total variance in the top singular value
                              squared (leading eigentriple). High = dominant trend/cycle.
  - xdom_ssa_energy_top3   : share in the top-3 eigentriples combined.
  - xdom_ssa_energy_noise  : share in components beyond top-3 (noise floor).
                              Low = signal-rich; High = noise-like.

These are computed at every bar by re-doing the SVD on the trailing 60-day
return window (lag L=10 => 51x10 trajectory matrix).  numpy linalg.svd operates
on small matrices so it stays well under the 100ms budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

METADATA = {
    "name": "xdom_ssa_energy",
    "description": (
        "Singular Spectrum Analysis (SSA) leading-component energy (Broomhead-King 1986). "
        "Per-ticker proxy: on each rolling 60-day log-return window, embeds returns into a "
        "Hankel trajectory matrix (lag L=10) and decomposes via SVD. Produces the variance "
        "share of the top eigentriple (lead energy), the top-3 combined share, and the "
        "noise-floor share (components 4+). High lead-energy = dominant low-rank trend/cycle; "
        "low lead-energy / high noise = noisy/mean-reverting. Inherently per-ticker; "
        "no cross-sectional component."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_ssa_energy_lead",
        "xdom_ssa_energy_top3",
        "xdom_ssa_energy_noise",
    ],
    "tags": ["cross-domain", "signal-processing", "ssa", "svd", "econophysics", "trend-quality"],
    "version": "1.0",
    "author": "Broomhead & King (1986) — Singular Spectrum Analysis, Physica D; impl per-ticker proxy",
}

# Rolling window and lag parameters
_WINDOW = 60   # rolling return window (days)
_LAG = 10      # SSA embedding lag L; trajectory matrix is (N-L+1) x L
_TOP_K = 3     # number of leading eigentriples for top3 and noise split


def _ssa_energies(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Given a 1-D array of returns of length >= LAG+1, build the Hankel trajectory
    matrix and compute SSA variance shares.

    Returns (lead_share, top3_share, noise_share) or (nan, nan, nan) on failure.
    """
    n = len(returns)
    L = _LAG
    K = n - L + 1  # number of columns in trajectory matrix
    if K < 2 or L < 2:
        return np.nan, np.nan, np.nan

    # Build Hankel (trajectory) matrix: shape K x L
    # Row i = [returns[i], returns[i+1], ..., returns[i+L-1]]
    # Use stride tricks for efficiency (read-only view, no copy until svd)
    shape = (K, L)
    strides = (returns.strides[0], returns.strides[0])
    X = np.lib.stride_tricks.as_strided(returns, shape=shape, strides=strides)

    # SVD — we only need singular values (not full matrices)
    try:
        sv = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # SSA variance is proportional to sv^2
    sv2 = sv ** 2
    total = sv2.sum()
    if total <= 0.0:
        return np.nan, np.nan, np.nan

    lead_share = float(sv2[0] / total)
    top3_share = float(sv2[:_TOP_K].sum() / total)
    noise_share = float(sv2[_TOP_K:].sum() / total)

    return lead_share, top3_share, noise_share


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)

    # Pre-allocate output arrays with NaN
    lead_arr = np.full(n_rows, np.nan, dtype=np.float32)
    top3_arr = np.full(n_rows, np.nan, dtype=np.float32)
    noise_arr = np.full(n_rows, np.nan, dtype=np.float32)

    # Log returns (length n_rows; first element is NaN)
    close = df["Close"].to_numpy(dtype=np.float64)

    # Guard against zeros in Close to avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # log_ret has length n_rows-1; index i corresponds to df row i+1

    min_needed = _WINDOW  # need WINDOW returns => WINDOW+1 price bars

    for i in range(min_needed, n_rows):
        # Returns for the 60-day window ending at bar i (inclusive)
        # log_ret[i-1] is return from bar i-1 to bar i => last return ending at i
        ret_slice = log_ret[i - _WINDOW: i]  # length = _WINDOW
        if np.isnan(ret_slice).any():
            continue
        lead, top3, noise = _ssa_energies(ret_slice)
        lead_arr[i] = lead
        top3_arr[i] = top3
        noise_arr[i] = noise

    df["xdom_ssa_energy_lead"] = lead_arr
    df["xdom_ssa_energy_top3"] = top3_arr
    df["xdom_ssa_energy_noise"] = noise_arr

    return df
