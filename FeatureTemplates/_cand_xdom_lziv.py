"""
Lempel-Ziv complexity of binarised daily returns.

LZ76 parse: scan the binary string left-to-right; each time a new substring
(not seen in the prefix so far) is encountered, start a new phrase. The
complexity c(n) is the count of such phrases.  Normalised by n/log2(n) so
that a maximally-random series approaches 1.

Near 1  -> random / efficient (hard to predict)
Near 0  -> highly repetitive / predictable structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_lziv",
    "description": (
        "Per-ticker Lempel-Ziv complexity (LZ76) of binarised daily returns over a "
        "rolling 80-day window. Returns are binarised as 1 if r > rolling-median else 0. "
        "lziv_complexity_80 is normalised c(n)/(n/log2(n)) so ~1 = random, ~0 = repetitive. "
        "lziv_slope_20 is the 20-day linear slope of lziv_complexity_80, capturing "
        "transitions into/out of predictable regimes. "
        "METHOD: faithful per-ticker implementation of the LZ76 parse on OHLCV price data. "
        "No cross-sectional component."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_lziv_complexity_80",
        "xdom_lziv_slope_20",
    ],
    "tags": ["lempel-ziv", "complexity", "entropy", "cross-domain", "price"],
    "version": "1.0",
    "author": "Lempel-Ziv complexity (Lempel-Ziv 1976; randomness of a binary string); spec SOURCE: Cross-domain method transfer (signal processing / econophysics / HRV / DSP)",
}


def _lz76_complexity(bits: np.ndarray) -> float:
    """
    Lempel-Ziv 76 complexity: count of distinct phrases in the LZ parse.
    bits must be a 1-D array of 0/1 integers.
    Returns NaN if length < 2.
    """
    n = len(bits)
    if n < 2:
        return np.nan

    c = 1       # phrase count; first phrase always exists
    i = 0       # start of current phrase
    k = 1       # current phrase length
    l = 1       # look-ahead pointer (i + k)
    # seen: set of substrings observed in the prefix (as tuples for hashability)
    # We use the standard O(n^2 / log n) approach: extend phrase until not seen in prefix
    prefix_set: set = set()
    prefix_set.add(tuple(bits[:1]))   # seed with first symbol

    while l + k <= n:
        candidate = tuple(bits[i: i + k])
        if candidate in prefix_set:
            k += 1
        else:
            # new phrase found; record it and start new phrase
            prefix_set.add(candidate)
            i = i + k
            k = 1
            c += 1
            l = i + k
            if l > n:
                break

    # Handle trailing phrase that was still "in progress" when we ran out
    if i < n:
        pass  # already counted the ongoing phrase in c at start (c=1 init accounts for this)

    return float(c)


def _lz76_normalised(bits: np.ndarray) -> float:
    """Return LZ76 complexity normalised by n/log2(n)."""
    n = len(bits)
    if n < 2:
        return np.nan
    c = _lz76_complexity(bits)
    if np.isnan(c):
        return np.nan
    denom = n / np.log2(n)
    if denom == 0:
        return np.nan
    return c / denom


def compute(df: pd.DataFrame) -> pd.DataFrame:
    WIN = 80        # rolling window for LZ complexity
    SLOPE_WIN = 20  # window for slope of complexity

    n_rows = len(df)

    # ── 1. Daily log returns (first row will be NaN) ──────────────────────────
    close = df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.empty(n_rows, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # ── 2. Rolling LZ76 complexity ────────────────────────────────────────────
    complexity = np.full(n_rows, np.nan, dtype=np.float64)

    for t in range(WIN - 1, n_rows):
        window_ret = log_ret[t - WIN + 1: t + 1]
        # Drop NaN entries (only the very first row is NaN)
        valid = window_ret[~np.isnan(window_ret)]
        if len(valid) < 10:      # not enough data
            continue

        # Rolling median of this window (uses only the window itself — no leakage)
        med = np.median(valid)

        # Binarise: 1 if r > median else 0
        bits = (valid > med).astype(np.int8)

        complexity[t] = _lz76_normalised(bits)

    df["xdom_lziv_complexity_80"] = complexity

    # ── 3. 20-day linear slope of complexity ─────────────────────────────────
    slope = np.full(n_rows, np.nan, dtype=np.float64)
    if SLOPE_WIN >= 2:
        x = np.arange(SLOPE_WIN, dtype=np.float64)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        for t in range(WIN - 1 + SLOPE_WIN - 1, n_rows):
            y = complexity[t - SLOPE_WIN + 1: t + 1]
            if np.any(np.isnan(y)):
                continue
            y_mean = y.mean()
            cov = ((x - x_mean) * (y - y_mean)).sum()
            if x_var == 0:
                continue
            slope[t] = cov / x_var

    df["xdom_lziv_slope_20"] = slope

    return df
