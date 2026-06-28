"""
_lens_regime_info.py  (HIDDEN CANDIDATE — single-underscore staging)

Contextual rank normalization (WorldQuant ts_rank) for the regime_info cluster.

A GLOBAL monotone transform is a no-op for a tree. The trailing time-series
percentile rank (rolling().rank(pct=True)) is NOT a no-op: the comparison window
slides, so it injects the ticker's own recent-history context the tree cannot
reconstruct from the single raw column. Leak-free (past+present only), distribution-
free (robust to the heavy tails / regime drift that break plain z-scores).

We target ONLY raw level / signed-ratio / return-driven flow columns from the
assigned blocks. We deliberately SKIP columns that are already a within-window
z-score (informationratio), already a bounded correlation (autocorr_*, infodecay_*,
csa_gap_cont_60), already a binary/sign (trend_direction_*), or already a centered
re-expression of a column we already rank (csa_vr_signal_126) — re-ranking those is
near-redundant noise.

Chosen sources (heavy-tailed / level-like, where a trailing percentile best
linearizes the signal):
  distance_from_sma_50d  (market_regime_features) — signed (close-sma)/sma ratio.
  csa_vr_5_126           (variance_ratio)         — Lo-MacKinlay VR(5) ratio ~1, fat-tailed.
  cont_mom_120           (info_discreteness)      — continuity-weighted cumulative return.
  csa_pead_score         (informed_drift)         — EWM informed-flow drift score (fast).

Windows: 63 (~quarter) and 252 (~year) for the slow level/ratio/return signals;
21 / 63 for the fast behavioral flow score (csa_pead_score).
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for contract symmetry; transform is pure pandas)
import pandas as pd

# (source_col, window) — at most two windows per source, <=8 columns total.
_SPECS = [
    ("distance_from_sma_50d", 63),
    ("distance_from_sma_50d", 252),
    ("csa_vr_5_126", 63),
    ("csa_vr_5_126", 252),
    ("cont_mom_120", 63),
    ("cont_mom_120", 252),
    ("csa_pead_score", 21),
    ("csa_pead_score", 63),
]

METADATA = {
    "name":        "lens_regime_info",
    "description": "Trailing time-series percentile rank (ts_rank) of selected raw level / ratio / return-driven flow columns from the regime_info cluster (distance_from_sma_50d, csa_vr_5_126, cont_mom_120, csa_pead_score). Contextual rank normalization: per-ticker rolling rank(pct=True), leak-free and distribution-free.",
    "requires":    ["distance_from_sma_50d", "csa_vr_5_126", "cont_mom_120", "csa_pead_score"],
    "produces":    [f"{col}_tsr{win}" for col, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "market_regime"],
    "version":     "1.0",
    "author":      "lens batch (regime_info ts_rank)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{col}_tsr{win}"] = df[col].rolling(win, min_periods=mp).rank(pct=True)
    return df
