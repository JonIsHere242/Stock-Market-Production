"""
Ordinal Pattern Transition Entropy (Bandt-Pompe transition network) — per-ticker proxy.

For each 3-day window of close returns we assign one of 6 permutation symbols
(ordinal patterns). On a rolling 120-day history we build the 6×6 symbol→next-symbol
transition matrix and compute the mean Shannon entropy across outgoing rows.
High entropy → transitions between patterns are nearly uniform (structurally random /
regime-less); low entropy → the sequence has strong serial structure (e.g. trending
or mean-reverting patches follow each other predictably).

Per-ticker proxy: Bandt-Pompe ordinal patterns are inherently per-series, so no
cross-sectional approximation is needed. The only missing ingredient vs the original
method is cross-stock comparison of transition matrices, which is a meta-level ranking;
the within-ticker signal is implemented exactly.

SOURCE / AUTHORS: Ordinal pattern transition entropy (Bandt-Pompe transition network);
cross-domain method transfer from signal processing / econophysics / HRV / DSP.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import permutations

METADATA = {
    "name": "xdom_ordinal_transition",
    "description": (
        "Rolling 120-day ordinal-pattern transition entropy (Bandt-Pompe / econophysics). "
        "Assigns each 3-day return window one of 6 permutation symbols, builds the 6x6 "
        "symbol→next-symbol transition matrix on a rolling 120-day history, and returns "
        "the mean Shannon entropy of outgoing rows. "
        "High value = structurally random transitions; low value = strong sequential "
        "pattern regularity. Also emits a 20-day short-window variant and a "
        "contrast slope (diff of the two) as a dynamics signal. "
        "Per-ticker (no cross-sectional ranks needed — ordinal patterns are within-series). "
        "SOURCE: Bandt-Pompe transition network; cross-domain from signal processing / "
        "econophysics / HRV / DSP."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_ordinal_transition_ent120",   # 120-day rolling transition entropy (main)
        "xdom_ordinal_transition_ent20",    # 20-day short-window variant
        "xdom_ordinal_transition_contrast", # long-entropy minus short-entropy (regime shift proxy)
    ],
    "tags": ["entropy", "ordinal-patterns", "transition", "structure", "cross-domain", "econophysics"],
    "version": "1.0",
    "author": (
        "Ordinal pattern transition entropy (Bandt-Pompe transition network); "
        "cross-domain method transfer from signal processing / econophysics / HRV / DSP."
    ),
}

# --- Build the fixed ordinal-pattern lookup once at module load ---
# 3-element ordinal patterns → integer symbol 0..5
_ORDER = 3
_PERMS = {p: i for i, p in enumerate(permutations(range(_ORDER)))}  # 6 symbols


def _rank_pattern(x: np.ndarray) -> int:
    """Return the ordinal-pattern index of a length-3 array."""
    # argsort gives the rank permutation
    return _PERMS[tuple(np.argsort(x, kind="stable"))]


def _transition_entropy(symbols: np.ndarray) -> float:
    """
    Given a 1-D array of symbol indices (0..5), build the 6×6 transition
    count matrix from consecutive pairs and return the mean row Shannon entropy
    over rows that have at least one transition.
    """
    n_sym = len(_PERMS)
    trans = np.zeros((n_sym, n_sym), dtype=np.float64)
    for i in range(len(symbols) - 1):
        trans[symbols[i], symbols[i + 1]] += 1.0

    row_sums = trans.sum(axis=1, keepdims=True)
    # avoid div-by-zero; rows with zero outgoing count contribute 0 entropy
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    probs = trans / row_sums_safe  # (6,6)

    # Shannon entropy per row: -sum(p * log2(p)), ignore zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log2(probs), 0.0)
    row_entropies = -(probs * log_p).sum(axis=1)  # (6,)

    # only average over rows that had outgoing transitions
    active = (row_sums.ravel() > 0)
    if active.sum() == 0:
        return np.nan
    return float(row_entropies[active].mean())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # 1-day log returns (need _ORDER-1 = 2 consecutive returns for a 3-element window)
    close = df["Close"].to_numpy(dtype=np.float64)

    # log-return series; length = n-1
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(close[:-1] > 0, np.log(close[1:] / close[:-1]), np.nan)

    # ordinal patterns: each window of _ORDER=3 consecutive returns → one symbol
    # first valid symbol at index 2 of log_ret (return index 0,1,2) → original row index 3
    # symbol[k] corresponds to df row k + _ORDER (using 0-based log_ret index k)
    n_ret = len(log_ret)                     # n-1
    n_sym = n_ret - _ORDER + 1              # number of valid symbols = n - _ORDER

    symbols = np.full(n_sym, -1, dtype=np.int8)
    for k in range(n_sym):
        window = log_ret[k: k + _ORDER]
        if np.any(np.isnan(window)):
            symbols[k] = -1
        else:
            symbols[k] = _rank_pattern(window)

    # result arrays (aligned to df rows, length n)
    ent120 = np.full(n, np.nan, dtype=np.float64)
    ent20 = np.full(n, np.nan, dtype=np.float64)

    # symbol[k] → df row offset:  row = k + _ORDER   (0-based)
    # For a rolling window ending at df row r, we need symbols up through index r - _ORDER.
    # A 120-day window of transitions needs at least 2 symbols → 121 rows minimum.
    # Symbol index = row - _ORDER.

    WIN_LONG = 120
    WIN_SHORT = 20

    for r in range(_ORDER, n):
        sym_end = r - _ORDER          # inclusive last symbol index for row r
        # short window
        sym_start_short = max(0, sym_end - WIN_SHORT + 1)
        seg_short = symbols[sym_start_short: sym_end + 1]
        valid_short = seg_short[seg_short >= 0]
        if len(valid_short) >= 4:     # need at least 3 symbols for 2 transitions
            ent20[r] = _transition_entropy(valid_short)

        # long window
        sym_start_long = max(0, sym_end - WIN_LONG + 1)
        seg_long = symbols[sym_start_long: sym_end + 1]
        valid_long = seg_long[seg_long >= 0]
        if len(valid_long) >= 4:
            ent120[r] = _transition_entropy(valid_long)

    df["xdom_ordinal_transition_ent120"] = ent120
    df["xdom_ordinal_transition_ent20"] = ent20

    # contrast: long-window entropy minus short-window (positive = long-term more chaotic than recent)
    with np.errstate(invalid="ignore"):
        contrast = np.where(
            np.isnan(ent120) | np.isnan(ent20),
            np.nan,
            ent120 - ent20,
        )
    df["xdom_ordinal_transition_contrast"] = contrast

    return df
