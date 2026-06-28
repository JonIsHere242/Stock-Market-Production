"""
_lens_behavioral_reference.py — CONTEXTUAL RANK NORMALIZATION for the behavioral-reference cluster.

HIDDEN CANDIDATE block (single-underscore filename): excluded from production builds,
included only under --include_candidates.

Lens: a GLOBAL monotone transform of a single column is a no-op for a tree (it splits on
rank order). New information appears ONLY via a CONTEXTUAL rank the tree cannot reconstruct
from the raw column. The in-block, leak-free form is the TRAILING TIME-SERIES PERCENTILE
RANK (WorldQuant ts_rank): rank of today's value among that same ticker's own trailing
window. The window slides (comparison set changes each day) so it is NOT a global monotone
no-op; it uses only past+present so it is leak-free; it is distribution-free, robust to the
heavy tails / regime drift that break plain z-scores.

Blessed pattern (from price_differential_signal_pack.py):
    df[out] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
-> output in (0, 1]; NaN only for the leading min_periods-1 rows (expected, NOT filled).

Source columns chosen (all currently RAW level / return / ratio / clip-only — re-expressed,
never recomputed):
  xdm_cgo       (capital_gains_overhang)      signed (P-RP)/P cost-basis distance, clip-only
  xdm_vwap_dist (capital_gains_overhang)      signed (P-VWAP)/VWAP distance, clip-only
  rdd_ref_disp  (reference_price_dispersion)  cost-basis dispersion / price ratio, clip-only
  lot_max5      (lottery_max_effect)          mean of 5 largest daily returns, clip-only

SKIPPED (already a within-window z-score / rank of the same base, or near-redundant):
  xdm_cgo_tsz        — already a 252d within-ticker z-score.
  rdd_overhang_skew  — already a standardized (skew) moment.
  lot_max1/min1/...  — near-redundant with lot_max5.
  anchoring_52w.py   — assigned block DOES NOT EXIST in FeatureTemplates/ (only
                       anchoring_52w_gh.py, whose outputs are already bounded (0,1] position
                       ratios); SKIPPED to stay surgical and on assigned names.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with sibling blocks; not required)
import pandas as pd

METADATA = {
    "name":        "lens_behavioral_reference",
    "description": (
        "Trailing time-series percentile ranks (ts_rank, rolling().rank(pct=True)) of "
        "raw/clip-only behavioral-reference columns: capital-gains overhang distance "
        "(xdm_cgo), VWAP distance (xdm_vwap_dist), cost-basis dispersion (rdd_ref_disp), "
        "and lottery MAX5 (lot_max5). Contextual per-ticker rank, leak-free, "
        "distribution-free; new info a tree cannot reconstruct from the raw column."
    ),
    "requires":    ["xdm_cgo", "xdm_vwap_dist", "rdd_ref_disp", "lot_max5"],
    "produces":    [
        "xdm_cgo_tsr63", "xdm_cgo_tsr252",
        "xdm_vwap_dist_tsr63",
        "rdd_ref_disp_tsr63", "rdd_ref_disp_tsr252",
        "lot_max5_tsr21", "lot_max5_tsr63",
    ],
    "tags":        ["experimental", "lens", "rank", "behavioral", "mean_reversion"],
    "version":     "1.0",
    "author":      "Lens batch (behavioral_reference cluster) — contextual rank normalization",
}

# (source_col, window) pairs -> output "<source_col>_tsr<window>"
_PAIRS = [
    ("xdm_cgo", 63),
    ("xdm_cgo", 252),
    ("xdm_vwap_dist", 63),
    ("rdd_ref_disp", 63),
    ("rdd_ref_disp", 252),
    ("lot_max5", 21),
    ("lot_max5", 63),
]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _PAIRS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
