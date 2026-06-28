"""
Campbell-Grossman-Wang (1993) volume-return cross-lag features.

Economic intuition: High volume on down days signals temporary selling pressure
from liquidity/noise traders; subsequent reversal is a measure of illiquidity
premium. The volume-return correlation captures whether current volume predicts
same-day return direction (positive = momentum, negative = contrarian signal).

Per-ticker proxy: implemented fully from OHLCV. Cross-sectional ranking is not
used here; both signals are rolling per-stock time-series estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_autocorr_volume_return",
    "description": (
        "Campbell-Grossman-Wang (1993) volume-return cross-lag proxy. "
        "xdom2_autocorr_volume_return_corr60: rolling 60-day Pearson correlation "
        "between detrended log-volume and same-day close return — captures the "
        "volume-return contemporaneous link (positive = momentum days, negative = "
        "reversal/illiquidity regime). "
        "xdom2_autocorr_volume_return_reversal60: rolling 60-day mean next-day "
        "return conditioned on high-volume (top-tercile) down days within the window "
        "— the CGW illiquidity-premium reversal. "
        "xdom2_autocorr_volume_return_volret_lag1: rolling 60-day lag-1 "
        "cross-correlation (volume[t] vs return[t+1] within the window, shift "
        "applied only to past data) — causal direction of the CGW mechanism. "
        "All computed per-ticker from OHLCV; no cross-sectional ranking. "
        "Leading NaNs expected for the first 60 rows."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "xdom2_autocorr_volume_return_corr60",
        "xdom2_autocorr_volume_return_reversal60",
        "xdom2_autocorr_volume_return_volret_lag1",
    ],
    "tags": ["volume", "return", "cross-lag", "liquidity", "reversal", "cgw"],
    "version": "1.0",
    "author": "Cross-domain / practitioner method transfer (batch 2); Campbell, Grossman & Wang (1993) 'Trading Volume and Serial Correlation in Stock Returns', QJE.",
}

_WINDOW = 60


def _rolling_corr_manual(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Pearson correlation between two 1-D arrays, length-n output.
    Returns NaN for the first (window-1) elements.
    No lookahead: each result[i] uses x[i-window+1 : i+1] and y[i-window+1 : i+1].
    """
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        xi = x[i - window + 1 : i + 1]
        yi = y[i - window + 1 : i + 1]
        # skip if constant or all-nan
        xv = xi[~np.isnan(xi) & ~np.isnan(yi)]
        yv = yi[~np.isnan(xi) & ~np.isnan(yi)]
        if len(xv) < 10:
            continue
        xs, ys = xv - xv.mean(), yv - yv.mean()
        denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        if denom == 0.0:
            continue
        out[i] = (xs * ys).sum() / denom
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # --- base series ---
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)

    # daily log returns (no shift(-1); shift(1) = previous day, no leakage)
    # ret[i] = log(Close[i] / Close[i-1])
    log_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[1:] / close[:-1]
        log_ret[1:] = np.where(ratio > 0, np.log(ratio), np.nan)

    # detrended log-volume: log(vol) minus 5-day rolling mean of log(vol)
    log_vol = np.where(volume > 0, np.log(volume), np.nan)
    # rolling 5-day mean for detrending (uses pandas for convenience, no leakage)
    log_vol_s = pd.Series(log_vol)
    log_vol_trend = log_vol_s.rolling(5, min_periods=3).mean().to_numpy()
    detrended_vol = log_vol - log_vol_trend  # residual volume

    # ---------------------------------------------------------------
    # Feature 1: corr60 — rolling 60d Pearson corr(detrended_vol, ret)
    # Both series are same-day, no shift -> no lookahead.
    # ---------------------------------------------------------------
    corr60 = _rolling_corr_manual(detrended_vol, log_ret, _WINDOW)

    # ---------------------------------------------------------------
    # Feature 2: reversal60 — rolling mean of next-day return after
    # high-volume down days (within the past 60-day window).
    #
    # For each bar i, look at the window [i-W+1 .. i].
    # Within that window find bars where:
    #   (a) same-day return < 0  (down day)
    #   (b) detrended volume is in top-tercile of the window
    # Then average the NEXT-day return for those bars.
    # The "next-day return" for bar j within the window is log_ret[j+1],
    # and j+1 <= i (still in the past), so no leakage.
    # We start the meaningful signal at window+1 bars in to ensure
    # the next-day return is always available.
    # ---------------------------------------------------------------
    reversal60 = np.full(n, np.nan)
    W = _WINDOW
    for i in range(W, n):  # i is current bar; window is [i-W .. i-1] shifted 1
        # use [i-W .. i-1] so that "next-day" return index j+1 <= i (past)
        start = i - W
        end = i  # exclusive; slice [start:end] = [i-W .. i-1]
        dv_w = detrended_vol[start:end]
        ret_w = log_ret[start:end]
        # next-day returns = log_ret[start+1 : end+1], but end+1=i+1 > i -> leakage!
        # Fix: use [start+1 : end] (i.e. ret[j] for j in [start+1..end-1]);
        # this means we pair dv_w[k] -> ret_w[k+1] for k in 0..W-2.
        if end - start < 10:
            continue
        dv_core = dv_w[:-1]  # indices [start .. end-2], shape W-1
        ret_next = ret_w[1:]  # ret[start+1..end-1], shape W-1; all <= i-1 < i ✓
        ret_same = ret_w[:-1]  # same-day ret for the trigger bar

        valid = ~np.isnan(dv_core) & ~np.isnan(ret_next) & ~np.isnan(ret_same)
        if valid.sum() < 5:
            continue

        dv_v = dv_core[valid]
        ret_next_v = ret_next[valid]
        ret_same_v = ret_same[valid]

        # high-volume = top-tercile of detrended vol in this window
        thr = np.nanpercentile(dv_v, 66.7)
        mask_hvol = dv_v >= thr
        mask_down = ret_same_v < 0
        mask = mask_hvol & mask_down

        if mask.sum() < 2:
            continue
        reversal60[i] = ret_next_v[mask].mean()

    # ---------------------------------------------------------------
    # Feature 3: volret_lag1 — rolling 60d lag-1 cross-correlation
    # corr(detrended_vol[t], ret[t+1]) within the window.
    # Equivalent to pairing vol[j] with ret[j+1] for j in window.
    # We use the same W-1 pairs as above (no leakage because ret[j+1]
    # is always strictly before bar i when the window ends at i-1).
    # ---------------------------------------------------------------
    volret_lag1 = np.full(n, np.nan)
    for i in range(W + 1, n):
        start = i - W
        end = i  # [start .. i-1]
        dv_w = detrended_vol[start : end - 1]  # vol[start..i-2]
        ret_w = log_ret[start + 1 : end]        # ret[start+1..i-1]  ✓ no leakage
        valid = ~np.isnan(dv_w) & ~np.isnan(ret_w)
        if valid.sum() < 10:
            continue
        x = dv_w[valid]
        y = ret_w[valid]
        xs, ys = x - x.mean(), y - y.mean()
        denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
        if denom == 0.0:
            continue
        volret_lag1[i] = (xs * ys).sum() / denom

    # guard against any stray inf
    corr60 = np.where(np.isfinite(corr60), corr60, np.nan)
    reversal60 = np.where(np.isfinite(reversal60), reversal60, np.nan)
    volret_lag1 = np.where(np.isfinite(volret_lag1), volret_lag1, np.nan)

    df["xdom2_autocorr_volume_return_corr60"] = corr60
    df["xdom2_autocorr_volume_return_reversal60"] = reversal60
    df["xdom2_autocorr_volume_return_volret_lag1"] = volret_lag1

    return df
