"""
_lens_realized_moments.py — CANDIDATE (staged, hidden) lens block.

Contextual rank-normalization (WorldQuant ts_rank) of the most heavy-tailed,
level-like realized-moment columns in the "realized_moments" cluster. A
gradient-boosted tree splits on rank order, so a GLOBAL monotone transform of a
single column is a no-op; a TRAILING per-ticker percentile rank injects new
information because the comparison set (this ticker's own trailing window)
changes each day and the tree cannot reconstruct it from the raw column alone.

EXACT leak-free transform (blessed pattern from price_differential_signal_pack):
    df[c] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
-> output in (0,1], NaN only for the leading min_periods-1 rows (do NOT fill).

Source columns chosen are realized higher moments / idiosyncratic skew / systematic
coskew — heavy-tailed shape statistics (clip-only, not already z-scored/ranked).
We re-express where today's moment sits in the name's OWN recent history of that
moment (i.e. "is this name unusually lottery-like vs its own past?"), which is the
behaviorally-relevant conditioning the cross-section cannot see from the raw value.

Self-contained: imports only pandas/numpy. Single 63d (~quarter) trailing window.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for contract symmetry; not required)
import pandas as pd

_WIN = 63
_MP = max(20, _WIN // 2)

# (source column, trailing-rank window). Source columns verified to exist in the
# assigned blocks; all are heavy-tailed level/ratio shape statistics (clip-only).
_SOURCES = [
    ("rskew_63", _WIN),              # realized skewness (ACJV 2015 lottery signal)
    ("rkurt_63", _WIN),             # realized kurtosis (very heavy-tailed)
    ("lot_idioskew_resid_60", _WIN),  # idiosyncratic (firm-specific) skew (BMV 2010)
    ("lot_rawskew_60", _WIN),        # raw rolling skew reference
    ("scm_coskew_63", _WIN),         # systematic coskewness vs SPY (Harvey-Siddique)
    ("scm_xcomom3_63", _WIN),        # raw 3rd cross-comoment numerator (heavy-tailed)
]

METADATA = {
    "name":        "lens_realized_moments",
    "description": (
        "Trailing per-ticker time-series percentile rank (ts_rank, pct) of heavy-tailed "
        "realized-moment columns (realized skew/kurt, idio skew, raw skew, systematic "
        "coskew, raw cross-comoment) over a 63d window. Contextual rank-normalization: "
        "ranks each name's current moment within its OWN recent history, leak-free."
    ),
    "requires":    [src for src, _ in _SOURCES],
    "produces":    [f"{src}_tsr{win}" for src, win in _SOURCES],
    "tags":        ["experimental", "lens", "rank", "tail", "higher_moments"],
    "version":     "1.0",
    "author":      "lens batch (realized_moments cluster)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SOURCES:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = (
            df[src].rolling(win, min_periods=mp).rank(pct=True)
        )
    return df
