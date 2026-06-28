"""
_lens_volume_flow.py — CONTEXTUAL RANK NORMALIZATION (ts_rank lens) for the
volume_flow cluster.

Staged candidate (single-underscore => HIDDEN, excluded from production builds,
included only under --include_candidates).

Lens: a GLOBAL monotone transform of a single column is a no-op for a tree.
The trailing TIME-SERIES PERCENTILE RANK (WorldQuant ts_rank) injects NEW
information because the comparison set slides every day — it ranks today's value
against THIS ticker's own trailing window. Leak-free (past+present only),
distribution-free, robust to heavy tails and regime drift.

We re-express only RAW level / ratio / clip-only source columns (skipping any
column already z-scored or already within-window ranked, which would be
near-redundant):

  - vfd_vol_hhi_60          (raw Herfindahl concentration level, 1/n..1; slow)
  - vfd_cmf_20              (bounded [-1,1] Chaikin Money Flow accumulation level)
  - volume_percent          (raw % deviation of volume from 28d MA; fast, fat-tailed)
  - volume_momentum_ratio   (raw 5d/20d dollar-volume ratio; right-skewed)

Each gets a trailing percentile rank at 63 (~quarter) and/or 252 (~year).
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for self-contained pandas/numpy contract)
import pandas as pd

METADATA = {
    "name":        "lens_volume_flow",
    "description": (
        "Trailing time-series percentile rank (ts_rank) of raw level/ratio "
        "volume-flow columns: HHI concentration, Chaikin Money Flow, "
        "volume %-deviation, and dollar-volume momentum ratio. Contextual rank "
        "normalization — leak-free rolling().rank(pct=True), windows 63/252."
    ),
    "requires": [
        "vfd_vol_hhi_60",
        "vfd_cmf_20",
        "volume_percent",
        "volume_momentum_ratio",
    ],
    "produces": [
        "vfd_vol_hhi_60_tsr63",
        "vfd_vol_hhi_60_tsr252",
        "vfd_cmf_20_tsr63",
        "volume_percent_tsr63",
        "volume_momentum_ratio_tsr63",
        "volume_momentum_ratio_tsr252",
    ],
    "tags":    ["experimental", "lens", "rank", "volume", "flow"],
    "version": "1.0",
    "author":  "lens-volume_flow",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # (source_col, window) -> "<source_col>_tsr<window>"
    plan = [
        ("vfd_vol_hhi_60",        63),
        ("vfd_vol_hhi_60",        252),
        ("vfd_cmf_20",            63),
        ("volume_percent",        63),
        ("volume_momentum_ratio", 63),
        ("volume_momentum_ratio", 252),
    ]

    for src, win in plan:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = (
            df[src].rolling(win, min_periods=mp).rank(pct=True)
        )

    return df
