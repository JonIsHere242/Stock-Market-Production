"""
_lens_micro_spread.py — Trailing time-series percentile rank (ts_rank) of the
micro_spread cluster's heavy-tailed LEVEL columns.

HIDDEN CANDIDATE block (single-underscore filename, name "lens_micro_spread"):
excluded from production builds, included only under --include_candidates.

THE LENS: a global monotone transform of a single column is a no-op for a tree.
A TRAILING percentile rank (WorldQuant ts_rank) injects new information because
the comparison set is the ticker's OWN trailing window (slides each day) — it is
leak-free (past+present only) and distribution-free, which linearizes the
heavy-tailed, regime-drifting liquidity/spread LEVELS below.

Source columns chosen are all raw clip-only LEVELS (no incumbent within-window
z/rank), one per distinct microstructure estimator family, to stay surgical:
  - mst_amihud_illiq_20d   : Amihud price-impact-per-dollar (canonical heavy tail)
  - mst_corwin_schultz_2d  : rawest 2-day Corwin-Schultz high-low spread level
  - mst_abdi_ranaldo_chl_20d : Abdi-Ranaldo CHL spread level (distinct estimator)
  - mst_roll_spread_20d    : Roll serial-covariance effective spread level

micro_spread_hl.py's micro_cs/ar columns are smoothed rolling means (near-dupes
of the mst spreads) and its *_ratio cols are already contextual ratios -> SKIPPED.
Bounded freq/asym/ratio cols and signed Kyle-lambda -> SKIPPED.

Windows: 63 (~quarter) and 252 (~year) on the two richest level families
(Amihud, Corwin-Schultz); 63 only on the other two estimators.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with sibling blocks)
import pandas as pd

# (source_col, window) pairs -> output "<source_col>_tsr<window>".
_SPECS = [
    ("mst_amihud_illiq_20d", 63),
    ("mst_amihud_illiq_20d", 252),
    ("mst_corwin_schultz_2d", 63),
    ("mst_corwin_schultz_2d", 252),
    ("mst_abdi_ranaldo_chl_20d", 63),
    ("mst_roll_spread_20d", 63),
]

METADATA = {
    "name":        "lens_micro_spread",
    "description": (
        "Trailing time-series percentile rank (ts_rank) of micro_spread cluster "
        "heavy-tailed LEVEL columns (Amihud illiquidity, Corwin-Schultz 2d, "
        "Abdi-Ranaldo CHL, Roll spread) at 63d/252d. Leak-free per-ticker "
        "rolling().rank(pct=True); contextual rank normalization for the tree."
    ),
    "requires":    [
        "mst_amihud_illiq_20d",
        "mst_corwin_schultz_2d",
        "mst_abdi_ranaldo_chl_20d",
        "mst_roll_spread_20d",
    ],
    "produces":    [f"{col}_tsr{win}" for col, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "liquidity", "microstructure"],
    "version":     "1.0",
    "author":      "lens batch (micro_spread cluster)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col, win in _SPECS:
        mp = max(20, win // 2)
        new_cols[f"{col}_tsr{win}"] = (
            df[col].rolling(win, min_periods=mp).rank(pct=True)
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
