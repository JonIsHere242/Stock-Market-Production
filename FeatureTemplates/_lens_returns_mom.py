"""
_lens_returns_mom.py — CANDIDATE (hidden, single-underscore staged).

Contextual rank-normalization lens for the "returns_mom" cluster.

A global monotone transform of a single feature is a no-op for a gradient-boosted
tree (trees split on rank order). New information is injected only by a CONDITIONAL
rank the tree cannot reconstruct from the raw column. Here that is the TRAILING
TIME-SERIES PERCENTILE RANK (WorldQuant ts_rank): rank of today's value among the
SAME ticker's own trailing window. The comparison set slides each day, so it is not a
global monotone map; it is leak-free (past+present only) and distribution-free (robust
to the heavy tails / regime drift in raw returns and idiosyncratic-momentum series).

Blessed transform (identical to price_differential_signal_pack.py):
    df[out] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
-> output in (0, 1]; NaN only for the leading max(20, win//2)-1 rows (left unfilled).

Source selection (surgical — only raw level / return / ratio / clip-only columns;
already-z-scored / already-bounded-fraction columns were deliberately skipped):
  * logret_20d        (returns.py)              cumulative ~month log-return, heavy-tailed level
  * return_5d_abs     (price_momentum_features) short-horizon move MAGNITUDE, fast/behavioral
  * csa_resmom_252    (residual_momentum.py)    cumulative idiosyncratic (CAPM-residual) move
  * asy_gainpain_63   (asym_updown_moves.py)    gain/pain payoff-asymmetry RATIO (clipped 0..50)
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with sibling blocks; transform is pandas-only)
import pandas as pd

# (source_col, window) pairs. tsr = trailing series rank.
_PAIRS = [
    ("logret_20d", 63),
    ("logret_20d", 252),
    ("return_5d_abs", 21),
    ("return_5d_abs", 63),
    ("csa_resmom_252", 252),
    ("asy_gainpain_63", 252),
]

METADATA = {
    "name":        "lens_returns_mom",
    "description": (
        "Trailing time-series percentile rank (ts_rank, rolling().rank(pct=True)) of "
        "raw return / idiosyncratic-momentum / payoff-asymmetry columns in the returns_mom "
        "cluster. Contextual per-ticker rank that is leak-free and distribution-free; a "
        "no-op-resistant re-expression of heavy-tailed level-like sources."
    ),
    "requires":    ["logret_20d", "return_5d_abs", "csa_resmom_252", "asy_gainpain_63"],
    "produces":    [f"{src}_tsr{win}" for src, win in _PAIRS],
    "tags":        ["experimental", "lens", "rank", "momentum", "returns"],
    "version":     "1.0",
    "author":      "lens batch (returns_mom cluster) — contextual rank normalization",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _PAIRS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
