"""
pettitt_changepoint.py - Pettitt rank-based change-point recency (Tier-2).

Pettitt (1979) "A non-parametric approach to the change-point problem"; standard in
hydrology/climatology (Pohlert 'trend' package). It locates a shift in the DISTRIBUTION of a
series using only ranks, via the Mann-Whitney statistic

    U_t = 2 * cumsum(rank(x))_t  -  t*(N+1),      tau* = argmax_t |U_t|

so the change point and its direction depend on the ordering of values, NOT their magnitude
-> scale- and volatility-invariant, and distinct from the magnitude-based CUSUM/BOCD blocks.
We emit, over a 60d window:

  - recency  : signed recency of the most significant median shift (dir * how recently it broke),
  - effect   : normalized strength |U_max| / (N^2/4) of that shift (a regime indicator),
  - signal   : dir * recency * effect (a directional, magnitude-gated break signal).

Per the verifier's refinement, the change point is NOT allowed to sit in the most-recent 2 bars
(fresh breaks are reversal-prone at a 1-day horizon). Fully vectorized via strided windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_W = 60
_EXCL = 2          # exclude the most-recent 2 bars from being the change point

METADATA = {
    "name":        "pettitt_changepoint",
    "description": "Pettitt (1979) rank-based change-point over a 60d window: signed recency of the most significant median shift, its normalized effect size, and a direction*recency*effect signal. Rank-only -> scale/vol-invariant; excludes the most-recent 2 bars.",
    "requires":    ["Close"],
    "produces":    ["xdm_pettitt_signal", "xdm_pettitt_recency", "xdm_pettitt_effect"],
    "tags":        ["trend", "market_regime", "change_point", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (Pettitt 1979)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    c = df["Close"].values.astype(float)

    sig = np.full(n, np.nan)
    rec = np.full(n, np.nan)
    eff = np.full(n, np.nan)

    if n >= _W:
        cw = sliding_window_view(c, _W)                    # (m, W): row k covers days k..k+W-1
        ranks = np.argsort(np.argsort(cw, axis=1), axis=1) + 1   # 1..W ranks within each window
        cum = np.cumsum(ranks, axis=1)
        t_arr = np.arange(1, _W + 1)
        U = 2 * cum - t_arr * (_W + 1)                     # (m, W): U_t for t=1..W
        absU = np.abs(U).astype(float)
        absU[:, : 1] = -np.inf                             # t=1 is degenerate
        absU[:, _W - _EXCL:] = -np.inf                     # exclude freshest bars
        tau = absU.argmax(1)                               # change-point position 0..W-1
        K = np.take_along_axis(np.abs(U), tau[:, None], axis=1).ravel()
        Uat = np.take_along_axis(U, tau[:, None], axis=1).ravel()
        direction = np.sign(Uat)                           # >0 => upward median shift after tau
        recency = (_W - 1 - tau) / float(_W - 1)           # 1 = most recent allowed, ~0 = oldest
        effect = K / (_W * _W / 4.0)

        m = cw.shape[0]
        s_sig = direction * recency * effect
        s_rec = direction * recency
        sig[_W - 1:] = s_sig[: n - _W + 1]                 # window ending at t -> row t-W+1
        rec[_W - 1:] = s_rec[: n - _W + 1]
        eff[_W - 1:] = effect[: n - _W + 1]

    df["xdm_pettitt_signal"] = np.clip(sig, -1, 1)
    df["xdm_pettitt_recency"] = np.clip(rec, -1, 1)
    df["xdm_pettitt_effect"] = np.clip(eff, 0, 1)

    return df
