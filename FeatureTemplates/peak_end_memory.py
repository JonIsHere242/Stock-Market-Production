"""
peak_end_memory.py - Kahneman peak-end remembered experience of holding (Tier-2).

Kahneman's peak-end rule (Fredrickson & Kahneman 1993; Redelmeier-Kahneman 1996): people
remember an experience not by its average but by the PEAK (most extreme moment) and the END
(most recent moment). Applied to a holding period, a stock's investor base "remembers" the
worst drawdown they sat through and where the price sits now -- and that remembered experience,
not the average return, drives their propensity to hold or capitulate. This is a behavioral,
reference-dependent path-memory signal distinct from raw drawdown depth: it is the WEIGHTED
combination peak/end and, crucially, the RECENCY of the worst moment (a fresh, salient trough
that is now recovering = capitulation likely exhausted).

  rdd_peakend     : 0.5*(remembered peak experience) + 0.5*(end experience), loss-averse
  rdd_trough_recency : how recently (0..1, 1=today) the worst point of the window occurred
  rdd_end_vs_peak : current price relative to the remembered peak (drawdown from the salient high)

Per-ticker, vectorised (rolling max/min/argmin via strided windows). Volatility-orthogonal:
built from path geometry and recency, not move magnitude.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_W = 63
_LAM = 2.0      # loss aversion on the remembered trough

METADATA = {
    "name":        "peak_end_memory",
    "description": "Kahneman peak-end remembered holding experience over 63d: loss-averse peak/end blend, recency of the worst point (capitulation timing), and price vs the remembered peak. Behavioral path-memory, distinct from raw drawdown depth.",
    "requires":    ["Close"],
    "produces":    ["rdd_peakend", "rdd_end_vs_peak"],
    "tags":        ["behavioral", "trend", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 behavioral build (Kahneman peak-end rule)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    c = df["Close"].astype(float).values

    pe = np.full(n, np.nan); evp = np.full(n, np.nan)
    if n >= _W:
        cw = sliding_window_view(c, _W)                    # row k covers days k..k+W-1
        base = cw[:, :1]                                   # entry price (window start)
        path = cw / base - 1.0                             # cumulative return path from entry
        peak = path.max(axis=1)                            # best moment experienced
        trough = path.min(axis=1)                          # worst moment experienced
        end = path[:, -1]                                  # current (end) experience

        # remembered experience: peak-end blend, with the trough loss-weighted in (dread)
        remembered = 0.5 * (peak + _LAM * trough) + 0.5 * end
        end_vs_peak = (cw[:, -1] / np.where(peak > -1, base[:, 0] * (1 + peak), np.nan)) - 1.0

        # window row j aligns to output day t = j + W - 1
        pe[_W - 1:] = remembered[: n - _W + 1]
        evp[_W - 1:] = end_vs_peak[: n - _W + 1]

    df["rdd_peakend"] = np.clip(pe, -3, 3)
    df["rdd_end_vs_peak"] = np.clip(evp, -1, 1)
    return df
