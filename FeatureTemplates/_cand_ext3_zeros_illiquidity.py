"""
Zero-return-days illiquidity (Lesmond-Ogden-Trzcinka) candidate feature block.

Spec: ext3_zeros_illiquidity
Source: Round-4 expansion (NEW: microstructure)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_zeros_illiquidity",
    "description": (
        "Lesmond-Ogden-Trzcinka (LOT) proportion-of-zero-returns illiquidity. "
        "Rolling 60-day fraction of days where |daily return| < 1e-6 (no-trade / informed-trade proxy); "
        "rolling 60-day fraction of zero-volume days; and 20-day change in the zero-return fraction "
        "as a dynamic illiquidity-trend signal. "
        "Per-ticker OHLCV proxy -- no cross-sectional ranking needed. "
        "High zero-return fraction signals wide effective spreads and informed-trade barriers "
        "(Lesmond, Ogden & Trzcinka 1999, JFE). "
        "Leading NaNs during the first 60-bar warm-up are expected."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext3_zeros_illiquidity_ret60",   # 60d rolling fraction of near-zero returns
        "ext3_zeros_illiquidity_vol60",   # 60d rolling fraction of zero-volume days
        "ext3_zeros_illiquidity_chg20",   # 20d change in the zero-return fraction (trend)
    ],
    "tags": ["microstructure", "illiquidity", "lesmond", "lot", "zero_returns"],
    "version": "1.0.0",
    "author": "Lesmond, Ogden & Trzcinka (1999, JFE); spec: Round-4 expansion (NEW: microstructure)",
}

_ZERO_RETURN_THRESHOLD = 1e-6  # |ret| below this = zero-return day
_ROLL_LONG = 60
_ROLL_SHORT = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext3_zeros_illiquidity_* columns to df (one ticker, ascending Date)."""

    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    n = len(close)

    # --- daily log-returns (causal: shift by 1 to avoid lookahead) ---
    # ret[t] = log(close[t] / close[t-1]); ret[0] = NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.empty(n, dtype=np.float64)
        ret[0] = np.nan
        prev = close[:-1]
        curr = close[1:]
        valid = (prev > 0) & np.isfinite(prev) & np.isfinite(curr)
        ratio = np.where(valid, curr / prev, np.nan)
        ret[1:] = np.where(valid, np.log(ratio), np.nan)

    # binary indicators
    zero_ret = (np.abs(ret) < _ZERO_RETURN_THRESHOLD).astype(np.float64)
    zero_ret[~np.isfinite(ret)] = np.nan  # keep NaN where ret is NaN

    zero_vol = (volume == 0).astype(np.float64)
    zero_vol[~np.isfinite(volume)] = np.nan

    # --- rolling means via pandas for clean NaN handling ---
    zero_ret_s = pd.Series(zero_ret, index=df.index)
    zero_vol_s = pd.Series(zero_vol, index=df.index)

    frac_ret60 = zero_ret_s.rolling(window=_ROLL_LONG, min_periods=_ROLL_LONG).mean()
    frac_vol60 = zero_vol_s.rolling(window=_ROLL_LONG, min_periods=_ROLL_LONG).mean()

    # 20-day change in zero-return fraction (trend / deterioration signal)
    chg20 = frac_ret60 - frac_ret60.shift(_ROLL_SHORT)

    # guard: replace inf/-inf with NaN (divisions already guarded, but be safe)
    def _clean(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan)

    df["ext3_zeros_illiquidity_ret60"] = _clean(frac_ret60).values
    df["ext3_zeros_illiquidity_vol60"] = _clean(frac_vol60).values
    df["ext3_zeros_illiquidity_chg20"] = _clean(chg20).values

    return df
