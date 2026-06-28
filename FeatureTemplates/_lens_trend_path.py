"""
_lens_trend_path.py — CONTEXTUAL TRAILING-SERIES-RANK lens over the trend_path cluster.

HIDDEN CANDIDATE block (leading single underscore): excluded from production builds,
included only under --include_candidates.

The "trend_path" cluster (path_acceleration / path_drawdown / drift_stability /
trend_quality / mann_kendall_trend) emits mostly raw clip-only levels, return spreads
and bounded ratios. For a gradient-boosted tree a GLOBAL monotone transform of any single
one of these is a no-op — trees already split on rank order. New information enters ONLY
via a rank CONDITIONAL on a context the tree cannot reconstruct from the single raw column.

The in-block form of that is the TRAILING TIME-SERIES PERCENTILE RANK (WorldQuant ts_rank):
rank of today's value among the trailing window of that SAME ticker's own history. It is
leak-free (past+present only, sliding window), distribution-free (robust to the heavy tails
and regime drift that break plain z-scores), and answers "how extreme is today's value
RELATIVE TO THIS NAME's own recent regime" — context the raw column does not carry.

Source columns chosen (raw-level / return-spread / bounded-ratio / clip-only ONLY):
  path_drawdown_252   slow heavy-left-tailed drawdown level (path_drawdown.py)
  path_max_dd_63      63d rolling max drawdown level, heavy left tail (path_drawdown.py)
  path_mom_ratio_5_20 clipped 5d/20d momentum-rate ratio, heavy-tailed (path_acceleration.py)
  path_curvature_20   skewed chord-deviation convexity level (path_acceleration.py)
  trq_kaufman_er_63   bounded [0,1] trend-efficiency level (trend_quality.py)

DELIBERATELY SKIPPED (already a rank / z-score / monotone-compressed of the same base —
re-ranking is near-redundant noise):
  - xdm_mk_tau_10/20/40/accel  : already Kendall-tau rank concordance (a rank statistic).
  - trq_sharpe_21/63           : already a mean/std (z-score-like) ratio.
  - path_time_underwater       : log1p(count) — monotone; rank == rank of raw count.
  - path_up_streak             : tanh(streak) — monotone; rank == rank of raw streak.
  - trq_r2_*, dst_*, path_accel_2d, path_jerk_10 : weaker / noisier / already-statistic.

Transform is the blessed pattern from price_differential_signal_pack.py:
    df[col].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
Output in (0, 1]; NaN only for the leading min_periods-1 rows (expected; NOT filled).
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for contract/self-containment; not strictly used)
import pandas as pd


# (source_column, trailing window in bars)
_SPECS = [
    ("path_drawdown_252",   252),  # depth of today's drawdown vs own trailing year
    ("path_max_dd_63",      252),  # severity of recent worst drawdown vs own history
    ("path_mom_ratio_5_20",  63),  # is momentum-rate ratio extreme vs own quarter
    ("path_mom_ratio_5_20", 252),  # ... and vs own year
    ("path_curvature_20",    63),  # path convexity extremity vs own quarter
    ("trq_kaufman_er_63",   252),  # trend cleanliness vs own trailing year
]

_REQUIRES = sorted({src for src, _ in _SPECS})
_PRODUCES = [f"{src}_tsr{win}" for src, win in _SPECS]


METADATA = {
    "name":        "lens_trend_path",
    "description": (
        "Contextual trailing-series-rank (ts_rank) lens over the trend_path cluster: "
        "per-ticker rolling percentile rank of raw drawdown levels, the momentum-rate "
        "ratio, path curvature, and Kaufman trend-efficiency at 63d/252d. Injects "
        "own-history regime context a tree cannot reconstruct from the raw column; "
        "leak-free, distribution-free. Re-ranks only clip-only level/ratio columns, "
        "skipping already-rank/z-scored sources."
    ),
    "requires":    _REQUIRES,
    "produces":    _PRODUCES,
    "tags":        ["experimental", "lens", "rank", "trend", "momentum"],
    "version":     "1.0",
    "author":      "lens batch (trend_path cluster, trailing ts_rank)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
