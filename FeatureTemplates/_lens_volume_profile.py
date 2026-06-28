"""
_lens_volume_profile.py — CONTEXTUAL RANK NORMALIZATION (trailing time-series percentile)
for the volume_profile cluster. HIDDEN CANDIDATE (single-underscore filename, METADATA name
"lens_volume_profile") — excluded from production builds, included only under
--include_candidates.

Lens: a GLOBAL monotone transform of a single column is a no-op for a gradient-boosted tree.
The trailing per-ticker percentile rank (WorldQuant ts_rank) injects NEW information because
the comparison set slides each day; it is leak-free (past+present only) and distribution-free,
which linearizes the heavy-tailed, reference-relative volume-profile distances we re-express
here. Blessed pattern (from price_differential_signal_pack.py):
    df[col] = df[src].rolling(win, min_periods=max(20, win // 2)).rank(pct=True)

Chosen source columns (all raw level / ratio / return — NOT already z-scored or ranked):
  - vpa_poc_dist     (signed distance to point-of-control; overhead-supply level)  -> 63, 252
  - vpa_vwmed_dist   (distance to volume-weighted median / fair value)             -> 63, 252
  - vwap_percent     (percent distance of Close from lagged VWAP-14; heavy-tailed) -> 63, 252
  - ovn_ret_last     (latest realised overnight gap; fast behavioral flow)         -> 21, 63

Skipped within this cluster:
  - vwap_14 (raw price level; ranking it just proxies price momentum)
  - vwap_std14/std20 (already rolling-volatility magnitudes)
  - volume_spectral_splatter (already a normalized 0-1 entropy statistic)
  - ovn_*_mean_* / ovn_minus_intraday_* (already rolling-window means)
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for self-containment / parity with sibling blocks)
import pandas as pd

_SPECS = [
    ("vpa_poc_dist", 63),
    ("vpa_poc_dist", 252),
    ("vpa_vwmed_dist", 63),
    ("vpa_vwmed_dist", 252),
    ("vwap_percent", 63),
    ("vwap_percent", 252),
    ("ovn_ret_last", 21),
    ("ovn_ret_last", 63),
]

METADATA = {
    "name":        "lens_volume_profile",
    "description": (
        "Trailing time-series percentile rank (ts_rank, rolling().rank(pct=True)) of "
        "heavy-tailed level/ratio columns from the volume_profile cluster: point-of-control "
        "distance, volume-weighted-median distance, Close-vs-VWAP percent, and the latest "
        "overnight gap. Distribution-free contextual rank normalization; leak-free."
    ),
    "requires":    ["vpa_poc_dist", "vpa_vwmed_dist", "vwap_percent", "ovn_ret_last"],
    "produces":    [f"{src}_tsr{win}" for src, win in _SPECS],
    "tags":        ["experimental", "lens", "rank", "volume", "behavioral"],
    "version":     "1.0",
    "author":      "lens batch (volume_profile cluster — trailing series rank)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for src, win in _SPECS:
        mp = max(20, win // 2)
        df[f"{src}_tsr{win}"] = df[src].rolling(win, min_periods=mp).rank(pct=True)
    return df
