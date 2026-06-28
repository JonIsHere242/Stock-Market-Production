"""
Volatility-regime-conditioned short-term reversal.

Extends the osap_streversal family by conditioning the 21-day reversal signal
on the ratio of short-term (20d) to long-term (120d) realised volatility.
Reversal tends to be stronger in high-vol regimes; the vol ratio captures that
conditioning axis independently of the raw reversal level.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

METADATA = {
    "name": "ext_reversal_conditional_vol",
    "description": (
        "Volatility-regime-conditioned short-term reversal. "
        "reversal_sig = negative sum of daily log-returns over 21 days "
        "(sign flip = mean-reversion signal; higher = more oversold). "
        "vol_ratio = rolling 20d realised vol / rolling 120d realised vol: "
        "captures the short-vs-long volatility regime. "
        "vol_cond = reversal_sig * vol_ratio: amplifies the reversal when "
        "short-vol is elevated relative to the long-run baseline and dampens "
        "it in calm regimes. This is a DIFFERENT axis from the raw reversal "
        "level -- it is the regime-conditioned version. Per-ticker, causal, "
        "vectorised via sliding_window_view. "
        "Proxy note: pure OHLCV; parent osap_streversal uses plain 21d "
        "negation; this block adds the vol-regime multiplier as a new axis."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_reversal_conditional_vol_sig",    # 21d reversal signal (raw)
        "ext_reversal_conditional_vol_ratio",  # short/long vol ratio (regime)
        "ext_reversal_conditional_vol_cond",   # conditioned = sig * ratio
    ],
    "tags": ["reversal", "volatility", "regime", "momentum", "mean_reversion"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner (osap_streversal); "
        "spec: project team (2026-06-27)"
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ------------------------------------------------------------------ #
    # 1. Daily log returns  (index t = log(close[t] / close[t-1]))        #
    #    log_ret[0] = NaN (no prior bar)                                  #
    # ------------------------------------------------------------------ #
    log_ret = np.full(n, np.nan)
    prev = close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0, close[1:] / prev, np.nan)
        log_ret[1:] = np.where(ratio > 0, np.log(ratio), np.nan)

    # ------------------------------------------------------------------ #
    # 2. 21-day reversal signal = -sum(log_ret[t-20 .. t])               #
    #    Requires 21 log-return values; first valid index = 21            #
    # ------------------------------------------------------------------ #
    REV_WIN = 21
    reversal_raw = np.full(n, np.nan)
    if n >= REV_WIN:
        windows_rev = sliding_window_view(log_ret, REV_WIN)  # shape (n-REV_WIN+1, REV_WIN)
        nan_mask_rev = np.any(np.isnan(windows_rev), axis=1)
        sums = np.nansum(windows_rev, axis=1)
        sums[nan_mask_rev] = np.nan
        reversal_raw[REV_WIN - 1:] = -sums

    # ------------------------------------------------------------------ #
    # 3. Short-term realised vol: std(log_ret) over 20 days              #
    # ------------------------------------------------------------------ #
    VS_WIN = 20
    vol_short = np.full(n, np.nan)
    if n >= VS_WIN:
        windows_vs = sliding_window_view(log_ret, VS_WIN)
        nan_mask_vs = np.any(np.isnan(windows_vs), axis=1)
        with np.errstate(invalid="ignore"):
            stds_vs = np.std(windows_vs, axis=1, ddof=1)
        stds_vs[nan_mask_vs] = np.nan
        vol_short[VS_WIN - 1:] = stds_vs

    # ------------------------------------------------------------------ #
    # 4. Long-term realised vol: std(log_ret) over 120 days              #
    # ------------------------------------------------------------------ #
    VL_WIN = 120
    vol_long = np.full(n, np.nan)
    if n >= VL_WIN:
        windows_vl = sliding_window_view(log_ret, VL_WIN)
        nan_mask_vl = np.any(np.isnan(windows_vl), axis=1)
        with np.errstate(invalid="ignore"):
            stds_vl = np.std(windows_vl, axis=1, ddof=1)
        stds_vl[nan_mask_vl] = np.nan
        vol_long[VL_WIN - 1:] = stds_vl

    # ------------------------------------------------------------------ #
    # 5. Vol ratio = short / long  (guard zero denominator -> NaN)       #
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(
            (vol_long > 0) & ~np.isnan(vol_long) & ~np.isnan(vol_short),
            vol_short / vol_long,
            np.nan,
        )

    # ------------------------------------------------------------------ #
    # 6. Conditioned reversal = reversal_sig * vol_ratio                 #
    # ------------------------------------------------------------------ #
    conditioned = np.where(
        ~np.isnan(reversal_raw) & ~np.isnan(vol_ratio),
        reversal_raw * vol_ratio,
        np.nan,
    )

    df["ext_reversal_conditional_vol_sig"] = reversal_raw
    df["ext_reversal_conditional_vol_ratio"] = vol_ratio
    df["ext_reversal_conditional_vol_cond"] = conditioned

    return df
