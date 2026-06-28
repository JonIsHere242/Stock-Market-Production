"""
_lens_dist_dynamics.py - Trailing time-series percentile-rank lens for the
"dist_dynamics" cluster (HIDDEN CANDIDATE, single-underscore => excluded from
production builds; included only under --include_candidates).

CONTEXTUAL RANK NORMALIZATION (WorldQuant ts_rank). A global monotone transform
of a single raw column is a no-op for a gradient-boosted tree. The TRAILING
time-series percentile rank injects new information: rank of today's value among
the SAME ticker's own trailing window (the comparison set slides each day),
leak-free (past+present only), distribution-free / robust to heavy tails and
regime drift.

EXACT, BLESSED TRANSFORM (as used in price_differential_signal_pack.py):
    df[new] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
Output in (0, 1]; NaN only for the leading min_periods-1 rows (expected; NOT filled).

Source columns chosen are raw level / ratio quantities whose CURRENT transform is
level/ratio/clip-only and which are heavy-tailed or skewed (where a trailing
percentile best linearizes the signal). Already-bounded shape metrics
(bowleyskew, hhi, top*share, the drift slopes and conditional-drift differences)
are SKIPPED: they are effectively normalized already, so re-ranking them just
adds redundant noise.

Chosen sources (all produced by the assigned cluster blocks):
  dsh_tailratio_63   (dist_quantile_geometry) - q95/|q05| ratio, very heavy-tailed -> tsr63, tsr252
  dsh_idr80_63       (dist_quantile_geometry) - inter-decile return dispersion level -> tsr63
  dsh_iqr50_63       (dist_quantile_geometry) - inter-quartile return dispersion level -> tsr63
  rda_dn_disagree    (rda_resolution_asymmetry) - down-day abnormal-volume intensity,
                       fast behavioral/flow -> tsr21, tsr63
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity; transform uses pandas only)
import pandas as pd

# (source_col, window) -> output "<source_col>_tsr<window>"
_SPECS = [
    ("dsh_tailratio_63", 63),
    ("dsh_tailratio_63", 252),
    ("dsh_idr80_63", 63),
    ("dsh_iqr50_63", 63),
    ("rda_dn_disagree", 21),
    ("rda_dn_disagree", 63),
]

METADATA = {
    "name":        "lens_dist_dynamics",
    "description": "Trailing time-series percentile-rank (ts_rank) lens over heavy-tailed level/ratio columns from the dist_dynamics cluster: tail ratio, inter-decile/inter-quartile return dispersion, and down-day abnormal-volume intensity. Contextual rank normalization (rolling rank(pct=True)) that a tree cannot reconstruct from the raw column; leak-free, distribution-free.",
    "requires":    ["dsh_tailratio_63", "dsh_idr80_63", "dsh_iqr50_63", "rda_dn_disagree"],
    "produces":    [f"{src}_tsr{win}" for src, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "distributional"],
    "version":     "1.0",
    "author":      "lens dist_dynamics (trailing ts_rank)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
