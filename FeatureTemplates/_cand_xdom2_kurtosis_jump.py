"""
Realized jump intensity via bipower variation (Barndorff-Nielsen-Shephard).

Rolling 40-day decomposition of realized variance into jump vs continuous components
using bipower variation. jump_ratio_40 = max(0, 1 - BV/RV) captures the fraction of
variance attributable to price jumps. jump_count_40 counts days where |return| exceeds
4x the local bipower-implied volatility (a jump threshold). jump_ratio_slope is a
10-day change in jump_ratio_40 to capture acceleration of jump risk.

Per-ticker implementation from OHLCV close returns only -- faithful to the
Barndorff-Nielsen-Shephard (2004) framework.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_kurtosis_jump",
    "description": (
        "Realized jump intensity via bipower variation (Barndorff-Nielsen & Shephard). "
        "Rolling 40-day ratio of realized variance (sum r^2) to bipower variation "
        "(mu1^-2 * sum|r_t||r_{t-1}|, mu1=sqrt(2/pi)) identifies jump share of total "
        "variance. jump_ratio_40 = max(0, 1 - BV/RV); jump_count_40 = days per window "
        "where |r| > 4 * sqrt(BV/n) (jump threshold); jump_ratio_slope = 10d change "
        "in ratio. Pure per-ticker OHLCV proxy -- fully faithful to BNS framework."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_kurtosis_jump_ratio_40",
        "xdom2_kurtosis_jump_count_40",
        "xdom2_kurtosis_jump_ratio_slope",
    ],
    "tags": ["jump", "realized-variance", "bipower-variation", "volatility", "risk"],
    "version": "1.0",
    "author": "Barndorff-Nielsen & Shephard (2004) -- BNS bipower variation framework; per-ticker OHLCV implementation",
}

# mu_1 = E[|Z|] for Z ~ N(0,1) = sqrt(2/pi)
_MU1 = np.sqrt(2.0 / np.pi)
_MU1_SQ = _MU1 ** 2  # used to scale bipower variation to be comparable to RV
_WINDOW = 40
_SLOPE_WINDOW = 10
_JUMP_SIGMA_MULT = 4.0


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # Log returns; r[0] is NaN (no prior close)
    r = np.empty(n, dtype=np.float64)
    r[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(close[1:] / close[:-1])

    abs_r = np.abs(r)

    # |r_t| * |r_{t-1}|  -- product of adjacent absolute returns (index 1..n-1 usable)
    bv_products = np.empty(n, dtype=np.float64)
    bv_products[:2] = np.nan  # need two valid returns
    bv_products[2:] = abs_r[2:] * abs_r[1:-1]

    jump_ratio = np.full(n, np.nan, dtype=np.float64)
    jump_count = np.full(n, np.nan, dtype=np.float64)

    # Rolling computation over the window
    for i in range(_WINDOW - 1, n):
        r_win = r[i - _WINDOW + 1 : i + 1]  # shape (40,)
        bv_win = bv_products[i - _WINDOW + 1 : i + 1]  # shape (40,)

        # RV = sum(r^2) -- skip NaN (first element of entire series may be NaN)
        rv = np.nansum(r_win ** 2)

        # BV = (1/mu1^2) * sum(|r_t||r_{t-1}|) -- skip NaN pairs
        bv_raw = np.nansum(bv_win)
        bv = bv_raw / _MU1_SQ

        if rv <= 0.0 or np.isnan(rv) or np.isnan(bv):
            # Cannot compute ratio
            continue

        ratio = max(0.0, 1.0 - bv / rv)
        jump_ratio[i] = ratio

        # Jump threshold: 4 * sqrt(BV / n_valid) where n_valid = #valid bv_products in window
        n_bv = np.sum(~np.isnan(bv_win))
        if n_bv > 0:
            bv_vol = np.sqrt(bv / n_bv)  # per-day bipower vol estimate
        else:
            bv_vol = np.nan

        if not np.isnan(bv_vol) and bv_vol > 0.0:
            threshold = _JUMP_SIGMA_MULT * bv_vol
            # Count days in window where |r| exceeds threshold (exclude NaN)
            valid_r = r_win[~np.isnan(r_win)]
            jump_count[i] = float(np.sum(np.abs(valid_r) > threshold))
        else:
            jump_count[i] = np.nan

    # Slope: 10-day change in jump_ratio
    jump_ratio_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(_SLOPE_WINDOW, n):
        prev = jump_ratio[i - _SLOPE_WINDOW]
        curr = jump_ratio[i]
        if not np.isnan(prev) and not np.isnan(curr):
            jump_ratio_slope[i] = curr - prev

    df["xdom2_kurtosis_jump_ratio_40"] = jump_ratio
    df["xdom2_kurtosis_jump_count_40"] = jump_count
    df["xdom2_kurtosis_jump_ratio_slope"] = jump_ratio_slope

    return df
