"""
_lens_tail_risk.py - CONTEXTUAL RANK-NORMALIZATION lens over the tail_risk cluster (CANDIDATE).

Hidden candidate block (single-underscore filename -> excluded from production builds,
included only under --include_candidates). It adds NOTHING new physically: it re-expresses a
handful of already-built tail-risk columns as their TRAILING TIME-SERIES PERCENTILE RANK
(WorldQuant ts_rank) over each ticker's own sliding window.

Why this is not a no-op for the tree: a GLOBAL monotone transform (log / plain z / global
percentile) is invisible to a gradient-boosted tree because it preserves rank order. The
trailing rolling rank is NOT global-monotone -- the comparison set (the trailing window)
slides each day, so the same raw value maps to different ranks under different local regimes.
It injects context the tree cannot reconstruct from the single raw column, is leak-free
(uses only past+present), and is distribution-free / robust to the heavy tails and clip
saturation of these tail-risk ratios and conditional betas.

Blessed transform (identical to price_differential_signal_pack.py rolling rank):
    df[c] = df[src].rolling(win, min_periods=max(20, win // 2)).rank(pct=True)
-> output in (0, 1], NaN for the leading min_periods-1 rows (expected; do NOT fill).

Source columns chosen are all clip-only ratios / conditional betas / return-unit differences
(level-like, heavy-tailed) -- NONE are already within-window z-scored or ranked, so the rank
adds genuine local-context ordering rather than re-ranking an existing rank.
"""
from __future__ import annotations

import pandas as pd

# (source_col, window). 63d = the native quarter horizon every source block uses.
_TARGETS = [
    ("htail_tail_asym_63", 63),    # fat-right vs fat-left tail shape (marquee, signed level)
    ("htail_tail_thick_63", 63),   # overall fat-tailedness (extreme/moderate |ret| ratio)
    ("dtb_beta_asym_63", 63),      # Ang-Chen-Xing downside-minus-upside beta spread
    ("dtb_beta_minus_63", 63),     # downside beta (the priced leg)
    ("htail_crash_lift_63", 63),   # excess co-crash lift vs market base rate
    ("xdm_creskew_raw", 63),       # return-unit state-conditioned drift asymmetry
]


METADATA = {
    "name":        "lens_tail_risk",
    "description": "Contextual rank-normalization lens: trailing per-ticker time-series percentile rank (ts_rank, pct) of clip-only tail-risk columns (Hill tail thickness/asymmetry, Ang-Chen-Xing downside/asym beta, market co-crash lift, conditional reversion skew) over a 63d sliding window. Leak-free, distribution-free re-expression of existing columns; no new base quantities.",
    "requires":    [src for src, _ in _TARGETS],
    "produces":    [f"{src}_tsr{w}" for src, w in _TARGETS],
    "tags":        ["experimental", "lens", "rank", "tail", "market_regime"],
    "version":     "1.0",
    "author":      "lens batch (tail_risk cluster, ts_rank)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _TARGETS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
