"""
_lens_vol_extra.py — HIDDEN CANDIDATE (lens: contextual rank normalization).

Trailing per-ticker time-series percentile rank (WorldQuant ts_rank) of a few
heavy-tailed, level-like volatility / volume-flow columns from the "vol_extra"
cluster. A trailing rolling().rank(pct=True) is NOT a global monotone transform
(window slides, comparison set changes each day), is leak-free (past+present
only), and is distribution-free — exactly what linearizes the heavy-tailed vol
level / gap magnitude / volume-weighted momentum that plain z-scores mangle.

Source-column selection (read each producer block's METADATA + compute body):
  - atr_percentage  (volatility_atr_features) : ATR/Close ratio, heavy-tailed vol
        LEVEL, raw ratio. PRIME target. (atr_percentile_rank already exists but is
        a 252-only fixed rank; we add 63 + 252 with the blessed pattern.)
  - gap_absolute    (volatility_atr_features) : |open gap|, fast behavioral flow,
        heavy-tailed magnitude -> windows 21 + 63 (fast signal => 21 not 252).
  - vol_vwmom_20    (vol_orderflow)           : volume-weighted momentum
        sum(ret*vol)/sum(vol), return-like / heavy-tailed.
  - std_14          (moving_average_metrics)  : raw price-scale rolling std of
        Close, a level-like vol magnitude (NOT normalized by price).

Skipped (per discipline): atr_percentile_rank (already a rank), vol_signret_corr*
& vol_pv_divergence (already bounded/normalized), rsi_* (already [0,100]
oscillators), atr_regime_* (binary), parabolic_sar / ma_14 (raw price levels =
trend position the tree already reconstructs), ma_14_pct* / *_change / *_count
(derivative/count), gap_in_atr_terms (already ATR-normalized).
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with blessed pattern; not required)
import pandas as pd

# (source_col, window) pairs -> at most TWO windows per source column.
_SPECS = [
    ("atr_percentage", 63),
    ("atr_percentage", 252),
    ("gap_absolute", 21),
    ("gap_absolute", 63),
    ("vol_vwmom_20", 63),
    ("vol_vwmom_20", 252),
    ("std_14", 63),
    ("std_14", 252),
]

METADATA = {
    "name":        "lens_vol_extra",
    "description": "Trailing per-ticker time-series percentile rank (ts_rank) of heavy-tailed vol/flow levels: atr_percentage, gap_absolute, vol_vwmom_20, std_14.",
    "requires":    ["atr_percentage", "gap_absolute", "vol_vwmom_20", "std_14"],
    "produces":    [f"{col}_tsr{win}" for col, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "volatility", "volume"],
    "version":     "1.0",
    "author":      "lens vol_extra (contextual rank normalization)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{col}_tsr{win}"] = (
            df[col].rolling(win, min_periods=mp).rank(pct=True)
        )
    return df
