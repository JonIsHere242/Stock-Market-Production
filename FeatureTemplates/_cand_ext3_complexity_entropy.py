"""
Complexity-Entropy (Jensen-Shannon) plane via Bandt-Pompe ordinal patterns.

Permutation entropy H and Jensen-Shannon statistical complexity C are computed
over a rolling window of Close returns, using ordinal patterns of embedding
dimension d=4. This captures the degree of structure/determinism in price
dynamics (low H = more ordered; high C = structured complexity away from
both fully random and fully ordered processes).

Reference: Bandt & Pompe (2002); Rosso et al. (2007) PRL "Distinguishing
Noise from Chaos".
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from itertools import permutations as _permutations

METADATA = {
    "name": "ext3_complexity_entropy",
    "description": (
        "Ordinal-pattern permutation entropy H (Bandt-Pompe d=4) and Jensen-Shannon "
        "statistical complexity C, computed over a rolling 120-day window of log-returns. "
        "H near 1 = high entropy (random); C quantifies distance of pattern distribution "
        "from uniform (structured complexity). Also emits the Euclidean distance from the "
        "fully-random corner (H=1, C=0) in the H-C plane. Pure OHLCV proxy; per-ticker."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_complexity_entropy_H",   # permutation entropy in [0,1]
        "ext3_complexity_entropy_C",   # Jensen-Shannon complexity in [0,1]
        "ext3_complexity_entropy_dist" # distance from (H=1, C=0) random corner
    ],
    "tags": ["entropy", "complexity", "ordinal-patterns", "bandt-pompe", "information-theory"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom_allan_variance); Bandt & Pompe (2002); Rosso et al. (2007)",
}

# --- pre-compute permutation index map for d=4 ---
_D = 4
_PERMS = list(_permutations(range(_D)))
_NPAT = len(_PERMS)  # 4! = 24
_PERM_TO_IDX = {p: i for i, p in enumerate(_PERMS)}

# Normalisation constants
_LN_NPAT = math.log(_NPAT)  # log(24) for H normalisation

# Maximum complexity C_max depends on pattern count n.
# For uniform distribution p_u = 1/n, H_u = 1 (by def after normalisation).
# Jensen-Shannon divergence of p from p_u:
#   JS(p, p_u) = H((p + p_u)/2) - H(p)/2 - H(p_u)/2
# Q constant (normalisation of JS to [0,1]):
#   Q_0 = -2 * [(n+1)/n * log(n+1) - log(2n) + log(n)] / ln_n
# C = Q_0 * JS(p, p_u) * H_norm
# We use the standard Q0 normalisation.

def _q0_norm(n: int) -> float:
    """Normalisation constant Q0 so that JS divergence from uniform is in [0,1]."""
    # Q0 = 1 / (-2 * [(n+1)/n * ln(n+1) - ln(2n) + ln(n)])
    # = 1 / (-2 * [(n+1)/n * ln(n+1) - ln(2) - ln(n) + ln(n)])
    # = 1 / (-2 * [(n+1)/n * ln(n+1) - ln(2)])
    # After simplification this equals:
    # -2 * ( (n+1)/n * ln(n+1) - ln(2*n) + ln(n) ) but let's derive carefully:
    # JS(p_u, p_u) = 0 (trivially); maximum JS is at a Dirac on one symbol.
    # Standard formula from Lamberti et al. 2004:
    # Q0^{-1} = -2 [ (n+1)/n * ln(n+1) - 2*ln(2n) + ln(n) ]
    inv_q0 = -2.0 * (
        (n + 1) / n * math.log(n + 1)
        - 2.0 * math.log(2 * n)
        + math.log(n)
    )
    if inv_q0 == 0.0:
        return 0.0
    return 1.0 / inv_q0


_Q0 = _q0_norm(_NPAT)


def _ordinal_probs(arr: np.ndarray) -> np.ndarray:
    """
    Given a 1-D array of length >= d, return the probability vector
    of ordinal patterns (length _NPAT = 24).
    Uses rank-based tie-breaking (np.argsort stable).
    """
    n = len(arr)
    counts = np.zeros(_NPAT, dtype=np.float64)
    patterns_n = n - _D + 1
    if patterns_n <= 0:
        return counts
    # Build pattern matrix: shape (patterns_n, _D)
    # Use stride trick for efficiency
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, window_shape=_D)  # (patterns_n, _D)
    # Ordinal pattern = argsort of each window
    ranks = np.argsort(windows, axis=1, kind='stable')  # (patterns_n, _D)
    for r in ranks:
        idx = _PERM_TO_IDX.get(tuple(r))
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total == 0.0:
        return counts
    return counts / total


def _pe_and_c(probs: np.ndarray) -> tuple[float, float]:
    """
    Given normalised probability vector over ordinal patterns,
    return (H_norm, C) where:
      H_norm = -sum(p * log(p)) / log(n)   in [0,1]
      C = Q0 * JS(p, p_u) * H_norm          in [0,1]
    """
    n = _NPAT
    mask = probs > 0.0
    if not mask.any():
        return (float('nan'), float('nan'))

    # Shannon entropy (nats)
    h_raw = -np.sum(probs[mask] * np.log(probs[mask]))
    h_norm = h_raw / _LN_NPAT  # in [0, 1]

    # Jensen-Shannon divergence of probs from uniform
    p_u = 1.0 / n
    mix = 0.5 * (probs + p_u)  # (p + p_u)/2
    # H_mix
    mix_mask = mix > 0.0
    h_mix = -np.sum(mix[mix_mask] * np.log(mix[mix_mask]))
    # H_u = log(n)
    h_u = _LN_NPAT
    # JS = H_mix - 0.5*H_raw - 0.5*H_u
    js = h_mix - 0.5 * h_raw - 0.5 * h_u

    # Statistical complexity C
    c = _Q0 * js * h_norm

    return (h_norm, c)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute permutation entropy (H), Jensen-Shannon complexity (C),
    and distance from random corner over rolling 120-day windows.
    """
    WINDOW = 120

    close = df["Close"].values.astype(np.float64)
    n = len(close)

    H_out = np.full(n, np.nan)
    C_out = np.full(n, np.nan)

    # Log returns for ordinal pattern encoding
    # Using returns avoids scale effects; d=4 patterns are scale-invariant anyway
    # but we use log-Close directly for the ordinal ranking (equivalent for patterns)
    log_close = np.where(close > 0.0, np.log(close), np.nan)

    for i in range(WINDOW - 1, n):
        window_data = log_close[i - WINDOW + 1: i + 1]
        # Skip if too many NaNs
        if np.isnan(window_data).sum() > WINDOW * 0.2:
            continue
        # Fill interior NaNs by forward-fill (minimal)
        if np.any(np.isnan(window_data)):
            mask_nan = np.isnan(window_data)
            window_data = window_data.copy()
            # simple interpolation: fill with preceding value
            for k in range(1, len(window_data)):
                if mask_nan[k] and not mask_nan[k - 1]:
                    window_data[k] = window_data[k - 1]
            # drop if first element is nan
            if np.isnan(window_data[0]):
                continue
        probs = _ordinal_probs(window_data)
        h, c = _pe_and_c(probs)
        H_out[i] = h
        C_out[i] = c

    # Distance from random corner (H=1, C=0)
    dist_out = np.sqrt((H_out - 1.0) ** 2 + C_out ** 2)

    df["ext3_complexity_entropy_H"] = H_out
    df["ext3_complexity_entropy_C"] = C_out
    df["ext3_complexity_entropy_dist"] = dist_out

    return df
