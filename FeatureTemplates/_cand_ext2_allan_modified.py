"""
Modified Allan variance (white vs flicker noise discrimination) on log-returns.

Like the plain Allan variance but averages the phase (cumulative sum) over tau
before differencing, which discriminates white-PM from flicker-PM noise that the
plain Allan variance conflates. Produces the modified Allan deviation and the
modified/plain Allan deviation ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext2_allan_modified",
    "description": (
        "Modified Allan variance (MVAR) on log-returns using tau=5 and an 80-day "
        "rolling window. The modified Allan variance averages the phase (running "
        "sum of log-returns) over tau samples before differencing, which "
        "discriminates white-PM from flicker-PM noise that the plain Allan variance "
        "conflates. Produces: (1) ext2_allan_modified_mdev -- modified Allan "
        "deviation (sqrt of MVAR); (2) ext2_allan_modified_ratio -- ratio of "
        "modified to plain Allan deviation (>1 => flicker-dominant, <1 => "
        "white-dominant); (3) ext2_allan_modified_slope -- 5-day difference of "
        "mdev to capture direction. Per-ticker OHLCV proxy; no cross-sectional "
        "averaging needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_allan_modified_mdev",
        "ext2_allan_modified_ratio",
        "ext2_allan_modified_slope",
    ],
    "tags": ["volatility", "noise", "allan", "modified_allan", "frequency_stability"],
    "version": "1.0",
    "author": "Spec: Round-3 deep exploration of a rich winner vein (xdom_allan_variance)",
}

# ---------------------------------------------------------------------------
# Helper: compute modified Allan variance for a 1-D array of phase values
# ---------------------------------------------------------------------------

def _mvar_from_phase(phase: np.ndarray, tau: int) -> float:
    """
    Modified Allan Variance for a single window of phase samples.

    Phase x[i] = cumulative sum of log-returns (i.e. x[i] = sum_{k=0}^{i} r[k]).
    The N-observation phase array maps to N-1 frequency/return samples.

    MVAR(tau) = 1 / (2 * tau^2 * N_terms) * sum_{i=0}^{N_terms-1}
                    (sum_{k=i}^{i+tau-1} x[k+tau] - x[k])^2
    where N_terms = len(phase) - 2*tau.

    Equivalently using the sliding sum of phase differences of length tau:
        inner_k = x[k+tau] - x[k]           (plain Allan sub-difference)
        MVAR(tau) = 1/(2*tau^2*M) * sum_i ( sum_{k=i}^{i+tau-1} inner_k )^2
    """
    n = len(phase)
    if n < 2 * tau + 1:
        return np.nan

    # plain phase differences of stride tau: inner[k] = phase[k+tau] - phase[k]
    inner = phase[tau:] - phase[:-tau]  # length = n - tau

    # sliding window sum of 'inner' of length tau
    # averaged_diff[i] = sum_{k=i}^{i+tau-1} inner[k]
    # We need i from 0 to (n - 2*tau - 1), giving M terms
    m_terms = n - 2 * tau
    if m_terms <= 0:
        return np.nan

    # cumsum trick for rolling sum of inner
    cs = np.concatenate(([0.0], np.cumsum(inner)))
    # cs[i+tau] - cs[i] = sum_{k=i}^{i+tau-1} inner[k]
    if len(cs) < m_terms + tau:
        return np.nan
    averaged_diff = cs[tau: tau + m_terms] - cs[:m_terms]

    mvar = np.sum(averaged_diff ** 2) / (2.0 * tau * tau * m_terms)
    return float(mvar)


def _avar_from_phase(phase: np.ndarray, tau: int) -> float:
    """
    Plain Allan Variance for a single window of phase samples.
    AVAR(tau) = 1 / (2 * tau^2 * M) * sum_{i=0}^{M-1} (x[i+2*tau] - 2*x[i+tau] + x[i])^2
    where M = n - 2*tau.
    """
    n = len(phase)
    m_terms = n - 2 * tau
    if m_terms <= 0:
        return np.nan
    diff2 = phase[2 * tau:] - 2.0 * phase[tau: tau + m_terms] - (-phase[:m_terms])
    # correct sign: (x[i+2tau] - 2*x[i+tau] + x[i])
    d = phase[2 * tau:n] - 2.0 * phase[tau: n - tau] + phase[:n - 2 * tau]
    avar = np.sum(d ** 2) / (2.0 * tau * tau * m_terms)
    return float(avar)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    TAU = 5
    WINDOW = 80  # rolling window length in trading days
    SLOPE_LAG = 5

    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # log-returns (length n; first element NaN)
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    mdev_arr = np.full(n, np.nan, dtype=np.float64)
    ratio_arr = np.full(n, np.nan, dtype=np.float64)

    # phase = cumulative sum of log-returns starting from 0
    # We compute a running phase then slice per window
    # phase[i] relative to window start = cumsum within the window slice

    for i in range(WINDOW - 1, n):
        window_ret = log_ret[i - WINDOW + 1: i + 1]
        if np.any(np.isnan(window_ret)):
            continue

        # phase within this window: phase[0] = 0, phase[k] = sum(ret[0..k-1])
        phase = np.concatenate(([0.0], np.cumsum(window_ret)))  # length WINDOW+1

        mvar = _mvar_from_phase(phase, TAU)
        avar = _avar_from_phase(phase, TAU)

        if np.isnan(mvar) or mvar < 0:
            mdev = np.nan
        else:
            mdev = np.sqrt(mvar)

        mdev_arr[i] = mdev

        if not np.isnan(avar) and avar > 0 and not np.isnan(mvar) and mvar >= 0:
            adev = np.sqrt(max(avar, 0.0))
            if adev > 0:
                ratio_arr[i] = np.sqrt(mvar) / adev if mvar >= 0 else np.nan
            else:
                ratio_arr[i] = np.nan
        else:
            ratio_arr[i] = np.nan

    # slope: 5-day difference of mdev
    slope_arr = np.full(n, np.nan, dtype=np.float64)
    valid = ~np.isnan(mdev_arr)
    idx = np.where(valid)[0]
    # vectorised: slope[i] = mdev[i] - mdev[i - SLOPE_LAG] when both valid
    mdev_series = pd.Series(mdev_arr)
    slope_series = mdev_series.diff(SLOPE_LAG)
    slope_arr = slope_series.to_numpy(dtype=np.float64)

    # guard inf
    def _clean(arr: np.ndarray) -> np.ndarray:
        arr = np.where(np.isinf(arr), np.nan, arr)
        return arr

    df["ext2_allan_modified_mdev"] = _clean(mdev_arr)
    df["ext2_allan_modified_ratio"] = _clean(ratio_arr)
    df["ext2_allan_modified_slope"] = _clean(slope_arr)

    return df
