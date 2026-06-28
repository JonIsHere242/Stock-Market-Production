"""
ext4_jump_intensity — Jump Intensity (Lee-Mykland-style proxy)

Estimates local volatility via bipower variation (rolling 20-day mean of
|ret_t| * |ret_{t-1}| * pi/2), flags days where |ret| exceeds 4x that
local vol estimate, then computes:
  - rolling 60-day jump count (intensity)
  - rolling 60-day mean absolute jump size (conditioned on jump days)

Pure OHLCV, no cross-sectional dependency, no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_jump_intensity",
    "description": (
        "Lee-Mykland-style jump intensity proxy. Local volatility estimated via "
        "bipower variation (rolling 20d mean of |r_t|*|r_{t-1}|*pi/2). A day is "
        "flagged as a 'jump' when |daily_ret| > 4 * bpv_vol. Produces: rolling 60d "
        "jump count (intensity) and rolling 60d mean absolute jump size. "
        "Per-ticker OHLCV proxy — no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_jump_intensity_count60",   # rolling 60d count of jump days
        "ext4_jump_intensity_size60",    # rolling 60d mean |ret| on jump days (NaN if no jumps)
        "ext4_jump_intensity_bpv_vol",   # rolling 20d bipower-variation vol estimate (annualised)
    ],
    "tags": ["jump", "volatility", "bipower", "microstructure", "risk"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom_allan_variance); Lee & Mykland (2008) style",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Daily log returns (no lookahead: shift(1) = yesterday's close)   #
    # ------------------------------------------------------------------ #
    close = df["Close"].values.astype(np.float64)

    # log return: r_t = log(Close_t / Close_{t-1})
    log_ret = np.empty(len(close), dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(
            close[:-1] > 0,
            np.log(close[1:] / close[:-1]),
            np.nan,
        )

    abs_ret = np.abs(log_ret)

    # ------------------------------------------------------------------ #
    # 2. Bipower variation local vol (rolling 20-day)                     #
    #    BPV_t = (pi/2) * mean_{s in [t-19..t]}(|r_s| * |r_{s-1}|)      #
    #    This estimates sigma^2 robustly, so sqrt gives sigma.            #
    # ------------------------------------------------------------------ #
    # product |r_t| * |r_{t-1}|
    prod_abs = np.empty(len(abs_ret), dtype=np.float64)
    prod_abs[0] = np.nan
    prod_abs[1:] = abs_ret[1:] * abs_ret[:-1]

    s = pd.Series(prod_abs, index=df.index)

    # rolling mean of cross-products, then scale by pi/2 to get variance
    bpv_var = s.rolling(20, min_periods=10).mean() * (np.pi / 2.0)
    # vol in daily terms; guard sqrt of negatives (shouldn't happen but be safe)
    bpv_vol_daily = np.sqrt(np.maximum(bpv_var.values, 0.0))

    # annualised version for inspection (not used in jump detection)
    bpv_vol_ann = bpv_vol_daily * np.sqrt(252.0)

    # ------------------------------------------------------------------ #
    # 3. Jump flag: |ret_t| > 4 * bpv_vol_daily                          #
    # ------------------------------------------------------------------ #
    threshold = 4.0 * bpv_vol_daily
    is_jump = np.where(
        np.isfinite(abs_ret) & np.isfinite(threshold) & (threshold > 0.0),
        (abs_ret > threshold).astype(np.float64),
        np.nan,
    )
    jump_size = np.where(is_jump == 1.0, abs_ret, np.nan)

    # ------------------------------------------------------------------ #
    # 4. Rolling 60-day jump count and mean jump size                     #
    # ------------------------------------------------------------------ #
    s_jump = pd.Series(is_jump, index=df.index)
    s_jsize = pd.Series(jump_size, index=df.index)

    # sum of jump flags over 60 days
    jump_count60 = s_jump.rolling(60, min_periods=20).sum()

    # mean |ret| on jump days only (NaN if no jump in window)
    # = sum(jump_size) / count(jumps)  — guard zero denominator
    jump_size_sum60 = s_jsize.rolling(60, min_periods=1).sum()
    jump_count60_raw = s_jump.rolling(60, min_periods=20).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        jump_size_mean60 = np.where(
            jump_count60_raw.values > 0,
            jump_size_sum60.values / jump_count60_raw.values,
            np.nan,
        )

    # ------------------------------------------------------------------ #
    # 5. Attach columns                                                   #
    # ------------------------------------------------------------------ #
    df["ext4_jump_intensity_count60"] = jump_count60.values
    df["ext4_jump_intensity_size60"] = jump_size_mean60
    df["ext4_jump_intensity_bpv_vol"] = bpv_vol_ann

    return df
