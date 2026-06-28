"""
mann_kendall_trend.py - Mann-Kendall rank-trend concordance (Tier-2).

Mann (1945) / Kendall (1975) non-parametric trend test, the workhorse monotonic-trend
detector in climatology and hydrology. Kendall's tau over a window is

    S   = sum_{i<j} sign(x_j - x_i),     tau = S / (n(n-1)/2)

i.e. the net fraction of time-ordered pairs that are concordant (later > earlier). It uses
ONLY the sign of pairwise differences, so it is completely scale-invariant and ignores the
size of moves entirely -- unlike trend-R^2 / Kaufman efficiency ratio (already built), which
load on dispersion. It measures pure monotone-trend cleanliness: a name grinding steadily up
scores near +1 regardless of how small the steps are; a choppy name nets ~0. Volatility-
orthogonal by construction. Emitted at 10/20/40d plus a short-vs-long concordance acceleration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_WINS = [10, 20, 40]

METADATA = {
    "name":        "mann_kendall_trend",
    "description": "Mann-Kendall / Kendall-tau rank-trend concordance (net sign of time-ordered pairwise differences) at 10/20/40d plus a short-minus-long acceleration. Pure pairwise-sign -> scale/volatility-invariant, distinct from dispersion-loading trend-quality blocks.",
    "requires":    ["Close"],
    "produces":    ["xdm_mk_tau_10", "xdm_mk_tau_20", "xdm_mk_tau_40", "xdm_mk_tau_accel"],
    "tags":        ["trend", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (Mann 1945 / Kendall 1975)",
}


def _kendall_tau(c: np.ndarray, w: int) -> np.ndarray:
    n = c.shape[0]
    out = np.full(n, np.nan)
    if n < w:
        return out
    cw = sliding_window_view(c, w)                          # (m, w)
    # diff[:, i, j] = x_j - x_i ; sign summed over i<j
    diff = cw[:, None, :] - cw[:, :, None]
    sgn = np.sign(diff).astype(np.int8)
    iu = np.triu_indices(w, k=1)
    S = sgn[:, iu[0], iu[1]].sum(axis=1).astype(float)
    tau = S / (w * (w - 1) / 2.0)
    out[w - 1:] = tau[: n - w + 1]                          # window ending at t -> row t-w+1
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].values.astype(float)
    taus = {}
    for w in _WINS:
        t = _kendall_tau(c, w)
        taus[w] = t
        df[f"xdm_mk_tau_{w}"] = np.clip(t, -1, 1)

    df["xdm_mk_tau_accel"] = np.clip(taus[10] - taus[40], -2, 2)
    return df
