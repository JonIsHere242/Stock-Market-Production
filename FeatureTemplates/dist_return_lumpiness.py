"""
dist_return_lumpiness.py — Concentration / lumpiness of recent returns (Tier-2).

How "lumpy" is a name's recent cumulative move? A move built from one giant day
is information-discrete (jump-driven, prone to fade — Da-Gurun-Warachka 2014),
whereas a move spread evenly across many days is continuous (persists). We
measure concentration of the recent |return| mass directly with a
Herfindahl-Hirschman style index (HHI; Hirschman 1945) and the top-1/top-2 gap:

Over a rolling window W of daily |return| a = |r|, with S = Σa:
    hhi          = Σ(a/S)²  in [1/W, 1]           full concentration index
    top1_share   = max(a)/S in [0,1]              share from the single largest day
    top2_gap     = (a(1) - a(2))/S in [0,1]       dominance of largest over 2nd
    eff_days     = 1/hhi                           effective # of "active" days

High top1_share / hhi == lumpy/jump-driven; low == smooth/continuous. This sorts
the cross-section by the geometry of HOW a move was assembled, independent of its
size — a Tier-2 separator. The top-2 are extracted with vectorised rolling max
(largest) and a masked rolling max (second largest); no Python per-bar loop.

Pure per-ticker, vectorised. Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "dist_return_lumpiness",
    "description": "Concentration/lumpiness of rolling |returns|: Herfindahl index, single-largest-day share, top-1-vs-top-2 gap, and effective active days at 21d/63d (Hirschman HHI; jump vs continuous move geometry).",
    "requires":    ["Close"],
    "produces":    [
        f"dsh_{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("hhi", "top1share", "top2gap", "effdays")
    ],
    "tags":        ["volatility", "tail", "distributional", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 dist-shape build (Hirschman HHI / lumpiness)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    a = df["Close"].pct_change().abs()

    for w, mp in _WINDOWS:
        roll = a.rolling(w, min_periods=mp)
        s = roll.sum()
        s_safe = s.replace(0, np.nan)
        s2 = (a ** 2).rolling(w, min_periods=mp).sum()

        top1 = roll.max()
        # second largest |return| in each window (vectorised via strided
        # windows + np.partition — see helper). Used for the top-1-vs-top-2 gap.
        top2 = _rolling_second_max(a, w, mp)

        df[f"dsh_hhi_{w}"] = (s2 / (s_safe ** 2)).clip(0, 1).values
        df[f"dsh_top1share_{w}"] = (top1 / s_safe).clip(0, 1).values
        df[f"dsh_top2gap_{w}"] = ((top1 - top2) / s_safe).clip(0, 1).values
        df[f"dsh_effdays_{w}"] = ((s_safe ** 2) / s2.replace(0, np.nan)).clip(1, w).values

    return df


def _rolling_second_max(a: pd.Series, w: int, mp: int) -> pd.Series:
    """Vectorised rolling second-largest value (per window).

    The full-window region is computed in one shot with a strided 2D view of all
    length-w windows (sliding_window_view) and np.partition along axis=1 — an
    O(N*w) vectorised partition, NOT a per-bar python callback. The short
    [mp-1, w-1) warm-up region uses an expanding partition over at most (w-mp)
    rows (≤6 here); these are NaN-tolerant warm-up bars.
    """
    arr = a.to_numpy(dtype="float64")
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if n < mp:
        return pd.Series(out, index=a.index)

    if n >= w:
        windows = np.lib.stride_tricks.sliding_window_view(arr, w)  # (n-w+1, w)
        part = np.partition(windows, -2, axis=1)
        second = part[:, -2].copy()
        # a NaN anywhere in a window -> partition floats it to the top; void it.
        bad = np.isnan(windows).any(axis=1)
        second[bad] = np.nan
        out[w - 1:] = second

    # warm-up region: expanding second-max over the first up-to-w bars
    for i in range(mp - 1, min(w - 1, n)):
        seg = arr[: i + 1]
        if seg.size >= 2 and not np.isnan(seg).any():
            out[i] = np.partition(seg, -2)[-2]

    return pd.Series(out, index=a.index)
