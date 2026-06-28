"""
_lens_behavioral_value.py - Trailing time-series percentile-rank (ts_rank) lens over the
behavioral_value cluster (HIDDEN CANDIDATE; single-underscore => excluded from production
builds, included only under --include_candidates).

THE LENS: a GLOBAL monotone transform of one column is a no-op for a gradient-boosted tree.
A trailing rolling percentile rank (WorldQuant ts_rank) injects NEW information because the
comparison set slides each day -- it is conditional on the ticker's OWN recent regime, is
leak-free (past+present only), and is distribution-free / robust to the heavy tails and regime
drift that these clipped behavioral value signals exhibit.

Sources are CPT / loss-aversion / salience VALUE levels (signed, heavy-tailed, clip-only) --
the exact case where a trailing percentile best linearizes the signal. We SKIP source columns
that are already within-window z-scores or ranks (e.g. xdm_salience_tsz) since re-ranking those
is near-redundant. Each chosen source gets at most two windows.

Blessed transform (per price_differential_signal_pack.py):
    df[new] = df[src].rolling(win, min_periods=max(20, win//2)).rank(pct=True)
-> output in (0, 1], NaN only for the leading min_periods-1 rows (expected; not filled).
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for self-contained contract; not strictly used)
import pandas as pd

# (source_col, window) pairs. Slow structural behavioral signals -> 63 (quarter) / 252 (year).
_SPECS = [
    ("rdd_pt_loss_60", 63),    # MEMORY: single best screen feature; loss-side CPT dread, heavy-tailed
    ("rdd_pt_loss_60", 252),
    ("rdd_pt_value_60", 252),  # canonical CPT lottery value; level-like, heavy-tailed
    ("phv_pt_loss_5d", 63),    # MEMORY: richest seam; multi-day loss-side CPT
    ("lax_loss_mass", 252),    # weighted unrealized-loss mass; non-negative, very skewed
    ("xdm_salience_21", 63),   # raw salience-value covariance (NOT the tsz z-scored variant)
]

METADATA = {
    "name":        "lens_behavioral_value",
    "description": "Trailing time-series percentile-rank (ts_rank) lens over the behavioral_value cluster: per-ticker rolling rank(pct) of clip-only CPT / loss-aversion / salience VALUE levels at 63d/252d. Distribution-free re-expression of heavy-tailed behavioral signals; conditional on the ticker's own regime so it is not a tree no-op. Skips already-z-scored sources.",
    "requires":    ["rdd_pt_loss_60", "rdd_pt_value_60", "phv_pt_loss_5d", "lax_loss_mass", "xdm_salience_21"],
    "produces":    [f"{src}_tsr{win}" for src, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "behavioral"],
    "version":     "1.0",
    "author":      "behavioral_value ts_rank lens",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
