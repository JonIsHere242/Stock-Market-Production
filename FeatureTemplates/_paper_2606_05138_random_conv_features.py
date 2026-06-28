"""
_paper_2606_05138_random_conv_features.py  --  Per-ticker causal random-convolution
(ROCKET-style) features, fixed seed.

Inspired by:
    "Generating Financial Time Series by Matching Random Convolutional Features"
    arXiv 2606.05138.

ROCKET (Random Convolutional Kernel Transform, Dempster et al. 2020) shows that a
small bank of random 1-D kernels applied to a time series, followed by simple
pooling (PPV = proportion of positive values, and max activation), provides a
surprisingly powerful and cheap feature representation.

This block applies that idea to daily log-return series, causally (each output[t]
uses only samples <= t).  Kernels are FIXED at module load time via a hard-coded
seed so every run is bit-identical.

Implementation notes
--------------------
- 8 kernels with lengths drawn from {3, 5, 7, 9} (fixed seed 12345).
- Each kernel is applied to the z-scored trailing-252-day log-return series via
  a rolling dot-product (causal — no future data ever touched).
- Two pools per kernel: PPV over a 63-day trailing window, and max activation
  over the same 63-day trailing window.
- Naming: rcf_ppv_k{i}, rcf_max_k{i}  (i = 0..7 → 16 columns total).
- Leading NaNs: rows < max(kernel_length, zscore_window, pool_window) are NaN.
  Inf is impossible because the z-score denominator has an eps guard.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# KERNEL BANK  — generated ONCE at module load with a hard-coded seed.
# Everything here is a plain Python/NumPy constant; no runtime randomness.
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(12345)

_N_KERNELS: int = 8
_KERNEL_LENGTHS: List[int] = [3, 5, 7, 9, 3, 5, 7, 9]   # one per kernel

# Random weights (each row = one kernel, zero-padded to max length for storage).
_MAX_KERNEL_LEN: int = max(_KERNEL_LENGTHS)
_KERNEL_WEIGHTS: List[np.ndarray] = []
for _klen in _KERNEL_LENGTHS:
    _w = _RNG.standard_normal(_klen).astype(np.float64)
    # mean-centre the kernel (standard ROCKET practice)
    _w -= _w.mean()
    _KERNEL_WEIGHTS.append(_w)

# Random bias terms (one per kernel)
_KERNEL_BIASES: np.ndarray = _RNG.uniform(-1.0, 1.0, size=_N_KERNELS).astype(np.float64)

# Pool window (rolling horizon used for PPV and max computation)
_POOL_WINDOW: int = 63        # ~one quarter
_ZSCORE_WINDOW: int = 252     # trailing window for return standardisation

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_2606_05138_random_conv_features",
    "description": (
        "Per-ticker causal random-convolution (ROCKET-style) features from fixed-seed "
        "1-D kernels on log-returns: PPV and max-activation pooled over 63-day window "
        "(inspired by arXiv 2606.05138)."
    ),
    "requires":    ["Close"],
    "produces":    [
        "rcf_ppv_k0", "rcf_max_k0",
        "rcf_ppv_k1", "rcf_max_k1",
        "rcf_ppv_k2", "rcf_max_k2",
        "rcf_ppv_k3", "rcf_max_k3",
        "rcf_ppv_k4", "rcf_max_k4",
        "rcf_ppv_k5", "rcf_max_k5",
        "rcf_ppv_k6", "rcf_max_k6",
        "rcf_ppv_k7", "rcf_max_k7",
    ],
    "tags":        ["momentum", "experimental", "paper"],
    "version":     "1.0",
    "author":      "paper arXiv 2606.05138 — ROCKET-style causal random-kernel features",
}

# ---------------------------------------------------------------------------
# Helper: causal rolling dot-product between a 1-D signal and a fixed kernel
# ---------------------------------------------------------------------------

def _causal_conv_activation(signal: np.ndarray, kernel: np.ndarray, bias: float) -> np.ndarray:
    """
    Apply a fixed 1-D kernel causally to `signal`.

    output[t] = dot(signal[t-k+1 : t+1], kernel) + bias
              = sum_{j=0}^{k-1}  signal[t-j] * kernel[k-1-j]  + bias

    The kernel is reversed so index 0 is the most-recent tap (standard conv).
    Rows where t < k-1 are NaN (insufficient history).

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        Already cleaned / standardised 1-D signal (may contain NaN).
    kernel : np.ndarray, shape (k,)
        Kernel weights (NOT reversed yet — we reverse inside this function).
    bias : float
        Scalar bias added to each activation.

    Returns
    -------
    np.ndarray, shape (n,)  — causal activations, NaN where history insufficient.
    """
    n = len(signal)
    k = len(kernel)
    # Reverse the kernel so kernel[0] is paired with signal[t] (most recent).
    kr = kernel[::-1]   # shape (k,)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < k:
        return out

    # Vectorised sliding window via stride tricks — safe because we never look forward.
    # Build a (n-k+1, k) view: row i contains signal[i : i+k].
    # Then out[i + k - 1] = dot(row_i, kr) + bias,  i.e., the output at time t = i+k-1
    # uses only signal[i..t], fully causal.
    shape   = (n - k + 1, k)
    strides = (signal.strides[0], signal.strides[0])
    try:
        windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    except Exception:
        # Fallback: explicit loop (slower but always correct)
        for t in range(k - 1, n):
            chunk = signal[t - k + 1 : t + 1]
            if not np.any(np.isnan(chunk)):
                out[t] = float(np.dot(chunk, kr)) + bias
        return out

    # Dot product with kernel for all valid windows
    dots = windows @ kr  # shape (n-k+1,)

    # Any window containing NaN → NaN output
    has_nan = np.any(np.isnan(windows), axis=1)
    dots[has_nan] = np.nan

    out[k - 1 :] = dots + bias
    return out


# ---------------------------------------------------------------------------
# Helper: causal rolling PPV and max over a trailing window
# ---------------------------------------------------------------------------

def _rolling_ppv(activations: np.ndarray, window: int) -> np.ndarray:
    """Proportion of positive activations over the trailing `window` bars (causal)."""
    n = len(activations)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    # Use pandas rolling for simplicity and correctness; min_periods=window forces
    # full window (leading NaN where history < window).
    s = pd.Series(activations)
    # Count positives strictly greater than 0 (ROCKET definition of PPV).
    out_s = s.rolling(window, min_periods=window).apply(
        lambda x: float(np.sum(x > 0.0)) / window, raw=True
    )
    return out_s.to_numpy(dtype=np.float64)


def _rolling_max(activations: np.ndarray, window: int) -> np.ndarray:
    """Maximum activation value over the trailing `window` bars (causal)."""
    n = len(activations)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    s = pd.Series(activations)
    out_s = s.rolling(window, min_periods=window).max()
    return out_s.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 random-kernel causal convolution features (PPV + max each) on
    the per-ticker log-return series.

    All output columns are prefixed `rcf_`.  Leading NaNs are expected; no
    inf values are emitted.  The computation is fully causal: output[t] depends
    only on df rows with index <= t.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # --- 1. Compute log returns (first row is NaN by construction) -----------
    log_ret = np.full(n, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # --- 2. Causal z-score of log returns (trailing _ZSCORE_WINDOW) ----------
    # z_ret[t] = (log_ret[t] - mean(log_ret[t-W+1:t+1])) / std(...) where the
    # window is applied CAUSALLY (rolling, past only).
    lr_series = pd.Series(log_ret)
    roll_mean = lr_series.rolling(_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW).mean()
    roll_std  = lr_series.rolling(_ZSCORE_WINDOW, min_periods=_ZSCORE_WINDOW).std()
    # Guard against zero std
    std_safe  = roll_std.where(roll_std > 1e-10, other=np.nan)
    z_ret     = ((lr_series - roll_mean) / std_safe).to_numpy(dtype=np.float64)

    # --- 3. Apply each kernel and pool ---------------------------------------
    new_cols: dict = {}
    for i, (kernel, bias) in enumerate(zip(_KERNEL_WEIGHTS, _KERNEL_BIASES)):
        # Causal convolution activation
        act = _causal_conv_activation(z_ret, kernel, float(bias))

        # Pool: PPV and max over trailing _POOL_WINDOW bars
        ppv_col = f"rcf_ppv_k{i}"
        max_col = f"rcf_max_k{i}"

        new_cols[ppv_col] = _rolling_ppv(act, _POOL_WINDOW)
        new_cols[max_col] = _rolling_max(act, _POOL_WINDOW)

    # --- 4. Assign to df (no existing column is modified) --------------------
    for col_name, values in new_cols.items():
        df[col_name] = values

    return df
