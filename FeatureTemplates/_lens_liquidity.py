"""
_lens_liquidity.py  --  CONTEXTUAL RANK NORMALIZATION (ts_rank lens) for the liquidity cluster.

Trailing time-series percentile rank (WorldQuant ts_rank): rank of today's value among the
SAME ticker's trailing window of its own history. This is NOT a global monotone transform (the
comparison set slides every day), so it is NOT a tree no-op; it is leak-free (past+present only)
and distribution-free, which matters because the liquidity level columns below are extremely
heavy-tailed (Amihud illiquidity, turnover, volume-spike, liquidity-stress) and break plain
z-scores / regime drift.

We rank ONLY raw level / ratio / clip-only source columns where a trailing percentile best
linearizes the signal. We deliberately SKIP source columns that are already within-window
z-scores, ranks, percentiles, logs, or asymmetries (re-ranking those is near-redundant):
  - micro_amihud_asym, micro_amihud_ratio (bounded / regime-ratio)
  - all szc_amihud_* (log1p / log / self-relative-median / trend / asym)
  - dollar_volume_zscore, dollar_volume_percentile (already z / already a 252d percentile)
  - szc_turnover_z_60, szc_turnover_trend_60, szc_turnover_accel,
    szc_capadj_vol_shock_20 (already z / log-diffs)

Chosen sources (heavy-tailed, level-like, one window-level per base to avoid same-base overlap):
  - micro_amihud_63        Amihud illiquidity level (price impact per dollar)      -> tsr63, tsr252
  - szc_turnover_60        cap-relative turnover level (Datar-Naik turnover)        -> tsr63, tsr252
  - volume_spike_ratio     fast volume-burst ratio (behavioral flow)                -> tsr21, tsr63
  - liquidity_stress_3d    fast liquidity-stress ratio (3d vs 252d median)          -> tsr21

Transform is the blessed pattern from price_differential_signal_pack.py:
    df[c] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
Output in (0, 1]; leading NaN for the first mp-1 rows is expected -- do NOT fill.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for contract; transform uses pandas only)
import pandas as pd

# (source_col, window) pairs. tsr = trailing series rank.
_SPECS = [
    ("micro_amihud_63", 63),
    ("micro_amihud_63", 252),
    ("szc_turnover_60", 63),
    ("szc_turnover_60", 252),
    ("volume_spike_ratio", 21),
    ("volume_spike_ratio", 63),
    ("liquidity_stress_3d", 21),
]

_REQUIRES = sorted({src for src, _ in _SPECS})
_PRODUCES = [f"{src}_tsr{win}" for src, win in _SPECS]

METADATA = {
    "name":        "lens_liquidity",
    "description": (
        "Trailing time-series percentile rank (ts_rank lens) of heavy-tailed liquidity level/ratio "
        "columns: Amihud illiquidity (63d), cap-relative turnover (60d), volume-spike ratio, and "
        "3d liquidity-stress -- each ranked within its own ticker's trailing window (63/252/21)."
    ),
    "requires":    _REQUIRES,
    "produces":    _PRODUCES,
    "tags":        ["experimental", "lens", "rank", "liquidity"],
    "version":     "1.0",
    "author":      "lens batch (liquidity cluster)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
