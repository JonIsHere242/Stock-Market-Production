"""
lottery_max_effect.py — MAX / MIN lottery-demand features (Tier-2).

Bali, Cakici & Whitelaw (2011, JFE) "Maxing out: Stocks as lotteries and the
cross-section of expected returns." Stocks whose recent daily returns contain a
few extreme positive spikes attract lottery-seeking investors, get bid up, and
then UNDER-perform next period (the MAX effect is NEGATIVE). MAX is the mean of
the N largest daily returns over the past month — a pure within-month tail
sorter that separates names with the same average move but different upside
extremity. Exactly a Tier-2 top-decile discriminator.

Over a rolling 1-month window (21 trading days) of simple daily returns r:
    lot_max5  = mean of the 5 largest r            (Bali et al. MAX(5))
    lot_max1  = single largest r                    (MAX(1))
    lot_min5  = mean of the 5 smallest r            (downside analogue)
    lot_min1  = single smallest r
    lot_max_minus_mean = lot_max5 - mean(r)         upside extremity vs typical day
    lot_max_minus_min  = lot_max5 - lot_min5        full tail span (max-min range)

Pure per-ticker, vectorised (rolling sort-free via descending/ascending top-k
through np.sort on a strided view — but here done with cheap rolling rank trick).
Two horizons: 1 month (21d) and ~1 week (5d) for a shorter-memory lottery proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 21   # one trading month (Bali-Cakici-Whitelaw canonical)
_MP = 15

METADATA = {
    "name":        "lottery_max_effect",
    "description": "Lottery-demand MAX/MIN features (MAX5/MAX1/MIN5/MIN1, max-minus-mean, tail span) over a 21d window, per Bali-Cakici-Whitelaw 2011.",
    "requires":    ["Close"],
    "produces":    [
        "lot_max5", "lot_max1", "lot_min5", "lot_min1",
        "lot_max_minus_mean", "lot_max_minus_min",
    ],
    "tags":        ["tail", "lottery", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (BCW 2011)",
}


def _roll_topk_mean(r: pd.Series, w: int, mp: int, k: int, largest: bool) -> np.ndarray:
    """Rolling mean of the k largest (or smallest) values in each trailing window.

    Vectorised: build a (n, w) strided window matrix, partition for top-k, mean.
    No Python per-bar loop, no rolling().apply.
    """
    a = r.to_numpy(dtype=float)
    n = a.size
    out = np.full(n, np.nan)
    if n < mp:
        return out
    # sliding_window_view gives a read-only (n-w+1, w) view of trailing windows.
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(a, w)                      # rows end at index w-1 .. n-1
    valid = ~np.isnan(sw)
    cnt = valid.sum(axis=1)
    # Fill NaNs with -inf (for largest) / +inf (for smallest) so they sort away.
    fill = -np.inf if largest else np.inf
    swf = np.where(valid, sw, fill)
    if largest:
        swf = -swf                                       # turn into "smallest k"
    # partial sort: k smallest of swf are the k we want
    part = np.partition(swf, kth=min(k, w) - 1, axis=1)[:, :k]
    part = np.where(np.isinf(part), np.nan, part)
    topk_mean = np.nanmean(part, axis=1)
    if largest:
        topk_mean = -topk_mean
    res = np.full(sw.shape[0], np.nan)
    ok = cnt >= mp
    res[ok] = topk_mean[ok]
    out[w - 1:] = res
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change()

    max5 = _roll_topk_mean(r, _W, _MP, 5, largest=True)
    max1 = _roll_topk_mean(r, _W, _MP, 1, largest=True)
    min5 = _roll_topk_mean(r, _W, _MP, 5, largest=False)
    min1 = _roll_topk_mean(r, _W, _MP, 1, largest=False)
    mean_r = r.rolling(_W, min_periods=_MP).mean().to_numpy()

    df["lot_max5"] = np.clip(max5, -1.0, 5.0)
    df["lot_max1"] = np.clip(max1, -1.0, 5.0)
    df["lot_min5"] = np.clip(min5, -1.0, 5.0)
    df["lot_min1"] = np.clip(min1, -1.0, 5.0)
    df["lot_max_minus_mean"] = np.clip(max5 - mean_r, -5.0, 5.0)
    df["lot_max_minus_min"] = np.clip(max5 - min5, -10.0, 10.0)

    return df
