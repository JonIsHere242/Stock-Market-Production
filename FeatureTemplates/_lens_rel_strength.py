"""
_lens_rel_strength.py — LENS candidate (HIDDEN: single-underscore filename).

Trailing time-series percentile rank (ts_rank) of the most LEVEL-like / heavy-tailed
relative-strength columns from the rel_strength cluster. For a gradient-boosted tree a
GLOBAL monotone transform is a no-op; a TRAILING rolling rank is not — the comparison
window slides, so it injects context the tree cannot reconstruct from the raw column, and
it is distribution-free / robust to the heavy tails and regime drift of cumulative RS
series. Leak-free: rolling().rank(pct=True) uses only past+present.

Source columns chosen (raw level / cumulative / return — NOT already z-scored or ranked):
  - xas_rs_line_spy   (relative_strength.py) : cumulative (stock-market) log-return LINE;
                                               a path/regime-dependent LEVEL -> tsr63, tsr252
  - xas_resid_drift_60(relative_strength.py) : 60d cumulative beta-adjusted residual drift;
                                               heavy-tailed LEVEL -> tsr63, tsr252
  - xas_excess_ret_60 (relative_strength.py) : 60d summed excess log return -> tsr252
  - xas_rs_qqq_20     (style_tilt.py)        : 20d relative strength vs the growth basket;
                                               fast/behavioral -> tsr21

SKIPPED in the cluster (per DISCIPLINE): beta_*/corr_* (bounded, clipped, not level-like),
alpha_* (per-day mean, not heavy-tailed level), csa_beta_trend/accel/vol (already-derived
beta dynamics: slope/diff/std), xas_*slope*/style_beta/size_beta/growth_value_tilt (already
rate/beta derivatives). re-ranking those is near-redundant noise.
"""

from __future__ import annotations

import pandas as pd

METADATA = {
    "name":        "lens_rel_strength",
    "description": (
        "Trailing time-series percentile rank (ts_rank, rolling().rank(pct=True)) of the "
        "level-like relative-strength columns: cumulative RS line vs SPY (63/252), 60d "
        "cumulative residual drift (63/252), 60d excess return (252), and 20d relative "
        "strength vs QQQ (21). Contextual rank that linearizes heavy-tailed, regime-drifting "
        "RS levels for the cross-sectional tree."
    ),
    "requires":    [
        "xas_rs_line_spy",
        "xas_resid_drift_60",
        "xas_excess_ret_60",
        "xas_rs_qqq_20",
    ],
    "produces":    [
        "xas_rs_line_spy_tsr63",
        "xas_rs_line_spy_tsr252",
        "xas_resid_drift_60_tsr63",
        "xas_resid_drift_60_tsr252",
        "xas_excess_ret_60_tsr252",
        "xas_rs_qqq_20_tsr21",
    ],
    "tags":    ["experimental", "lens", "rank", "relative_strength", "market_regime"],
    "version": "1.0",
    "author":  "lens batch (rel_strength)",
}

# (source_col, window) pairs.
_SPECS = [
    ("xas_rs_line_spy",    63),
    ("xas_rs_line_spy",    252),
    ("xas_resid_drift_60", 63),
    ("xas_resid_drift_60", 252),
    ("xas_excess_ret_60",  252),
    ("xas_rs_qqq_20",      21),
]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = (
            df[src].rolling(win, min_periods=mp).rank(pct=True)
        )
    return df
