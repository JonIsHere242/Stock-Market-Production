"""
_lens_range_vol.py — CANDIDATE (staged, hidden) trailing time-series percentile
rank (ts_rank) lens over the range_vol cluster.

Core lens: a GLOBAL monotone transform of a single column is a no-op for a tree.
A TRAILING per-ticker percentile rank (rolling().rank(pct=True)) injects new
information because the comparison set slides each day and is conditional on the
ticker's own recent regime — leak-free (past+present only), distribution-free,
robust to the heavy tails / regime drift that wreck plain z-scores.

Targets chosen are raw clip-only LEVELS / heavy-tailed RATIOS from the range_vol
cluster (NOT already z-scored or ranked):
  * rng_yangzhang_63       — minimum-variance OHLC vol level (clip-only).
  * rng_parkinson_63       — pure High-Low range vol level (clip-only).
  * rng_hl_close_ratio_63  — (High-Close)/(Close-Low) demand-asymmetry ratio (clipped).
  * rng_updown_vol_ratio_63— up-day / down-day range-vol ratio (clipped, heavy-tailed).

SKIPPED (re-ranking would add noise / hurt):
  * rng_close_loc_*, rng_rs_up_share_* — already bounded [0,1] share/location means.
  * rvg_wl_* — cosine kernel similarities already in [0,1].
  * vvg_*, vaq_* — graph/spectral families flagged as hurting + vaq_ret_qrank is
    already a quantile rank.

Pure pandas/numpy, self-contained, stateless.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for parity with repo blocks; not strictly needed)
import pandas as pd

# (source_col, window) pairs. tsr = trailing series rank.
_SPECS = [
    ("rng_yangzhang_63", 63),
    ("rng_yangzhang_63", 252),
    ("rng_parkinson_63", 252),
    ("rng_hl_close_ratio_63", 63),
    ("rng_updown_vol_ratio_63", 63),
]

METADATA = {
    "name":        "lens_range_vol",
    "description": "Trailing per-ticker time-series percentile rank (ts_rank) of "
                   "raw range/vol LEVELS and heavy-tailed ratios from the range_vol "
                   "cluster (Yang-Zhang, Parkinson, HL-close ratio, up/down vol "
                   "ratio) at 63d/252d. Leak-free rolling().rank(pct=True).",
    "requires":    sorted({src for src, _ in _SPECS}),
    "produces":    [f"{src}_tsr{win}" for src, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "volatility", "range"],
    "version":     "1.0",
    "author":      "range_vol cluster ts_rank lens",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = (
            df[src].rolling(win, min_periods=mp).rank(pct=True)
        )
    return df
