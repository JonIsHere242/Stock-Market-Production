"""
_lens_insider.py  --  HIDDEN CANDIDATE (single-underscore) lens block for the INSIDER cluster.

Contextual rank normalization (WorldQuant ts_rank): trailing per-ticker time-series percentile
rank of an existing insider column among that SAME ticker's own past window. Leak-free (uses only
past+present; the comparison set slides each day), distribution-free, and NOT a global monotone
no-op for a gradient-boosted tree.

Blessed pattern (see price_differential_signal_pack.py):
    df[col] = src.rolling(win, min_periods=max(20, win // 2)).rank(pct=True)

Source columns chosen are the most HEAVY-TAILED / level-like raw quantities produced by the three
assigned inx_insider_* blocks -- dollar-per-buy-ticket, log breadth of distinct buyers, and
seniority-weighted buy intensity. These are per-ticker scale-incomparable levels where a trailing
percentile best linearizes the signal. Bounded ratios ([-1,1] net-dollar, [0,1] role mix), binary
flags, signs, and already-divergenced/diffed columns are deliberately SKIPPED (re-ranking those
adds little and risks diluting the top-1% selection).

Re-expresses EXISTING columns only -- producer blocks run first via requires/produces topo-sort.
Self-contained: imports only pandas/numpy; stateless; adds only METADATA["produces"]; returns df.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with sibling blocks; pandas does the work)
import pandas as pd

METADATA = {
    "name":        "lens_insider",
    "description": "Trailing per-ticker time-series percentile rank (ts_rank, pct=True) of the most "
                   "heavy-tailed insider level columns: average buy-ticket dollars, log buyer "
                   "breadth, and seniority-weighted buy intensity. Contextual rank normalization at "
                   "63d (quarter) and 252d (year). Leak-free (rolling, past+present only).",
    "requires": [
        "inx_ins_avg_buy_ticket_90",
        "inx_ins_buyer_breadth_log_90",
        "inx_ins_seniority_intensity_90",
    ],
    "produces": [
        "inx_ins_avg_buy_ticket_90_tsr63",
        "inx_ins_avg_buy_ticket_90_tsr252",
        "inx_ins_buyer_breadth_log_90_tsr63",
        "inx_ins_buyer_breadth_log_90_tsr252",
        "inx_ins_seniority_intensity_90_tsr63",
        "inx_ins_seniority_intensity_90_tsr252",
    ],
    "tags":        ["experimental", "lens", "rank", "ts_rank", "insider"],
    "version":     "1.0",
    "author":      "feature-gen",
}

# (source_col, window) pairs. tsr = trailing series rank. Windows: 63 = quarter, 252 = year.
_SPEC = [
    ("inx_ins_avg_buy_ticket_90",      63),
    ("inx_ins_avg_buy_ticket_90",      252),
    ("inx_ins_buyer_breadth_log_90",   63),
    ("inx_ins_buyer_breadth_log_90",   252),
    ("inx_ins_seniority_intensity_90", 63),
    ("inx_ins_seniority_intensity_90", 252),
]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPEC:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
