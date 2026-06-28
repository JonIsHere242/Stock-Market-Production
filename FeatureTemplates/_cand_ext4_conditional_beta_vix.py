"""
Regime-conditional beta (high vs low VIX) — per-ticker OHLCV + SPY + VIX proxy.

For each trading day, estimates the stock's rolling beta to SPY separately on
high-VIX vs low-VIX regimes (split at the trailing 250-day VIX median).
Produces: high-VIX beta, low-VIX beta, and their difference (stress-beta spread).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by path, as required)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_conditional_beta_vix",
    "description": (
        "Regime-conditional beta to SPY split by trailing-250d VIX median. "
        "On each bar, classifies the last 250 days as high-VIX or low-VIX relative "
        "to the 250-day rolling VIX median, then estimates OLS beta within each regime "
        "using a 250-day trailing window of (stock_ret, SPY_ret) pairs. "
        "Produces ext4_conditional_beta_vix_hi (high-VIX beta), "
        "ext4_conditional_beta_vix_lo (low-VIX beta), and "
        "ext4_conditional_beta_vix_spread (hi - lo; positive = beta rises in stress). "
        "Per-ticker proxy; no cross-sectional look; lookahead-safe via merge_asof backward."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_conditional_beta_vix_hi",
        "ext4_conditional_beta_vix_lo",
        "ext4_conditional_beta_vix_spread",
    ],
    "tags": ["beta", "regime", "vix", "macro", "risk"],
    "version": "1.0.0",
    "author": "Round-5 expansion (ext3_beta_instability)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 250        # trailing window (days) for both VIX median and beta regression
_MIN_OBS = 30        # minimum regime observations required to compute beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime-conditional beta columns to df (in-place; returns df)."""

    n = len(df)

    # Initialise output columns with NaN
    df["ext4_conditional_beta_vix_hi"] = np.nan
    df["ext4_conditional_beta_vix_lo"] = np.nan
    df["ext4_conditional_beta_vix_spread"] = np.nan

    if n < _MIN_OBS + 2:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close and align to df dates (backward merge = safe)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        return df  # degrade gracefully if index data missing

    # ------------------------------------------------------------------
    # 2. Fetch VIX and align
    # ------------------------------------------------------------------
    try:
        vix_df = _indexes.vix_daily_close()  # DataFrame[Date, vix_close]
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    except Exception:
        return df

    # ------------------------------------------------------------------
    # 3. Merge both into df copy (merge_asof requires sorted Date)
    # ------------------------------------------------------------------
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date").reset_index(drop=False)  # preserve original index

    spy_df = spy_df.sort_values("Date")
    vix_df = vix_df.sort_values("Date")

    work = pd.merge_asof(work, spy_df, on="Date", direction="backward")
    work = pd.merge_asof(work, vix_df, on="Date", direction="backward")

    # ------------------------------------------------------------------
    # 4. Compute daily log returns (shift(1) = previous day, no lookahead)
    # ------------------------------------------------------------------
    stock_ret = np.log(work["Close"] / work["Close"].shift(1))
    spy_ret = np.log(work["spy_close"] / work["spy_close"].shift(1))
    vix_vals = work["vix_close"].values

    stock_ret = stock_ret.values
    spy_ret = spy_ret.values

    hi_beta = np.full(n, np.nan)
    lo_beta = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 5. Rolling regime-conditional OLS beta (vectorised per bar)
    # ------------------------------------------------------------------
    # For each bar t, we use the window [t-WINDOW+1 .. t] (inclusive).
    # VIX median is computed over that same window.
    # OLS beta = cov(x,y) / var(x)  with x=spy_ret, y=stock_ret.

    for t in range(_WINDOW - 1, n):
        sl = slice(t - _WINDOW + 1, t + 1)
        sr = stock_ret[sl]
        mr = spy_ret[sl]
        vix_w = vix_vals[sl]

        # Drop NaN rows (first return row, any missing)
        mask_valid = np.isfinite(sr) & np.isfinite(mr) & np.isfinite(vix_w)
        if mask_valid.sum() < _MIN_OBS:
            continue

        sr_v = sr[mask_valid]
        mr_v = mr[mask_valid]
        vix_v = vix_w[mask_valid]

        vix_med = np.median(vix_v)
        hi_mask = vix_v >= vix_med
        lo_mask = ~hi_mask

        if hi_mask.sum() >= _MIN_OBS:
            x_hi = mr_v[hi_mask]
            y_hi = sr_v[hi_mask]
            var_x = np.var(x_hi)
            if var_x > 0:
                hi_beta[t] = np.cov(x_hi, y_hi, ddof=1)[0, 1] / np.var(x_hi, ddof=1)

        if lo_mask.sum() >= _MIN_OBS:
            x_lo = mr_v[lo_mask]
            y_lo = sr_v[lo_mask]
            var_x = np.var(x_lo, ddof=1)
            if var_x > 0:
                lo_beta[t] = np.cov(x_lo, y_lo, ddof=1)[0, 1] / var_x

    # ------------------------------------------------------------------
    # 6. Map results back to original df index (work may be reordered)
    # ------------------------------------------------------------------
    orig_index = work["index"].values  # original positional index before sort

    out_hi = pd.Series(hi_beta, index=orig_index)
    out_lo = pd.Series(lo_beta, index=orig_index)

    df["ext4_conditional_beta_vix_hi"] = out_hi.reindex(df.index).values
    df["ext4_conditional_beta_vix_lo"] = out_lo.reindex(df.index).values

    spread = out_hi - out_lo
    df["ext4_conditional_beta_vix_spread"] = spread.reindex(df.index).values

    # Guard: replace any inf with nan
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
