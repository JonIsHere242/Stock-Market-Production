"""
volume_profile_anchor.py - Volume-at-price anchors / overhead supply (Tier-2).

A genuinely fresh angle for this library: where shares actually CHANGED HANDS. The trailing
volume-by-price profile marks the levels that act as behavioral support/resistance -- the
point-of-control (price with the most traded volume) is the strongest anchor, and the balance
of volume above vs below the current price measures overhead supply (trapped longs) vs
underlying support. These are reference-relative, magnitude-free participation statistics, so
volatility-orthogonal and unrelated to any price-move size.

  vpa_vwmed_dist: distance from current price to the volume-weighted median price (fair value)
  vpa_poc_dist  : signed distance to the point-of-control (heaviest-volume price level);
                  >0 = the volume magnet sits ABOVE (overhead supply), <0 = below (support)

Per-ticker, vectorised (strided windows + a bincount volume histogram). Window = 120d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_W = 120
_NB = 25
_LO, _HI = -0.5, 0.5

METADATA = {
    "name":        "volume_profile_anchor",
    "description": "Volume-at-price behavioral anchors over 120d: distance to the volume-weighted median price (fair value), and signed distance to the point-of-control (heaviest-volume level; >0 overhead supply, <0 support). Reference-relative participation stats, volatility-orthogonal.",
    "requires":    ["Close", "Volume"],
    "produces":    ["vpa_vwmed_dist", "vpa_poc_dist"],
    "tags":        ["volume", "behavioral", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 build (volume-profile anchors / overhead supply)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    vm = np.full(n, np.nan); pc = np.full(n, np.nan)

    if n >= _W:
        pw = sliding_window_view(df["Close"].astype(float).values, _W)
        vw = sliding_window_view(df["Volume"].astype(float).values, _W)
        vw = np.nan_to_num(vw, nan=0.0)
        cur = pw[:, -1]                                       # window ends at output day t
        tot = vw.sum(1)

        order = np.argsort(pw, axis=1)
        ps = np.take_along_axis(pw, order, axis=1)
        vs = np.take_along_axis(vw, order, axis=1)
        cum = np.cumsum(vs, axis=1)
        idx = (cum >= 0.5 * tot[:, None]).argmax(1)
        vwmed = ps[np.arange(ps.shape[0]), idx]
        vwmed_dist = (cur - vwmed) / np.where(cur != 0, cur, np.nan)

        rel = np.nan_to_num(pw / cur[:, None] - 1.0, nan=0.0)
        binidx = np.clip(((rel - _LO) / (_HI - _LO) * _NB).astype(int), 0, _NB - 1)
        m = pw.shape[0]
        rowidx = np.repeat(np.arange(m), _W)
        flat = rowidx * _NB + binidx.ravel()
        hist = np.bincount(flat, weights=vw.ravel(), minlength=m * _NB).reshape(m, _NB)
        poc_bin = hist.argmax(1)
        poc_rel = _LO + (poc_bin + 0.5) / _NB * (_HI - _LO)

        end = n - _W + 1
        vm[_W - 1:] = vwmed_dist[:end]
        pc[_W - 1:] = poc_rel[:end]

    df["vpa_vwmed_dist"] = np.clip(vm, -0.5, 0.5)
    df["vpa_poc_dist"] = np.clip(pc, -0.5, 0.5)
    return df
