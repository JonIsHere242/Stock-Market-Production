"""
_lens_price_action.py — CONTEXTUAL RANK NORMALIZATION (trailing time-series percentile
rank) for the price_action cluster.

A global monotone transform (log / plain z / global percentile) of a single column is a
literal NO-OP for a gradient-boosted tree. The trailing time-series percentile rank
(WorldQuant ts_rank) is NOT a global monotone: the comparison window slides each day, so it
injects genuinely new, leak-free, distribution-free context the tree cannot reconstruct
from the single raw column. This is the blessed pattern already used in
price_differential_signal_pack.py:

    df[out] = df[src].rolling(win, min_periods=max(20, win // 2)).rank(pct=True)

Output is in (0, 1]; NaN only for the leading max(20, win//2)-1 rows (expected, NOT filled).

Source columns chosen — each is RAW level / simple ratio (NOT already z-scored or ranked),
heavy-tailed / level-like, where a trailing percentile best linearizes the signal:

  - intraday_range_pct   (price_action_features)  : (High-Low)/Close range ratio, heavy
                                                     right tail; never normalized elsewhere.
  - price_quality_score  (price_action_features)  : log(Close) level; ts_rank => 52-week-style
                                                     trailing price-position (new context).
  - ish_range_expansion_20 (ish_candle_shape)     : TR / 20d-avg-TR expansion ratio, skewed.
  - ish_on_id_vol_ratio_20 (ish_return_decomposition): overnight/intraday vol ratio, heavy
                                                     right tail (clipped 0..20).
  - days_since_high      (price_structure_metrics) : bars stuck below the running high, a
                                                     level-like persistence count.

Deliberately SKIPPED (discipline): ish_gap_z_60 / high_close_ratio_norm (already z-scored),
the *_mean_* / *_freq_* / *_fill_rate_* columns (already rolling means), single-day returns
(ish_overnight_ret/ish_intraday_ret/open_close_ratio — too noisy to rank alone), and
percent_range / percent_from_high (range-ratio redundant with intraday_range_pct / already an
expanding position measure).
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with sibling blocks; not strictly needed)
import pandas as pd

# (source_col, window) pairs -> output "<source_col>_tsr<window>"
_SPECS = [
    ("intraday_range_pct", 63),
    ("intraday_range_pct", 252),
    ("price_quality_score", 252),
    ("ish_range_expansion_20", 63),
    ("ish_on_id_vol_ratio_20", 63),
    ("days_since_high", 252),
]

METADATA = {
    "name":        "lens_price_action",
    "description": (
        "Trailing time-series percentile rank (ts_rank, rolling().rank(pct=True)) of "
        "raw-level / ratio price-action columns: intraday range, log-close position, "
        "true-range expansion, overnight/intraday vol ratio and days-since-high. "
        "Contextual (window-conditional) rank => non-no-op for trees, leak-free, "
        "distribution-free."
    ),
    "requires": [
        "intraday_range_pct",
        "price_quality_score",
        "ish_range_expansion_20",
        "ish_on_id_vol_ratio_20",
        "days_since_high",
    ],
    "produces": [f"{col}_tsr{win}" for col, win in _SPECS],
    "tags":     ["experimental", "lens", "rank", "price_structure"],
    "version":  "1.0",
    "author":   "lens-price_action",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{col}_tsr{win}"] = df[col].rolling(win, min_periods=mp).rank(pct=True)
    return df
