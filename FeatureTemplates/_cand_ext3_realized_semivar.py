"""
Realized semivariance & signed-jump variation (Barndorff-Nielsen decomposition).

RS+ = sum of squared positive returns over rolling 21d window
RS- = sum of squared negative returns over rolling 21d window
Downside share = RS- / (RS+ + RS-)
Signed jump = (RS+ - RS-) / (RS+ + RS-)   [+1 = all upside, -1 = all downside]
20d momentum of downside share = diff(downside_share, 20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_realized_semivar",
    "description": (
        "Barndorff-Nielsen realized semivariances over a rolling 21-day window. "
        "RS- / (RS+ + RS-) is the downside variance share (high = downside-dominated vol). "
        "ext3_realized_semivar_sj = signed jump = (RS+ - RS-) / total (positive = upside jump dominance). "
        "ext3_realized_semivar_mom20 = 20-bar change in downside share (acceleration of downside risk). "
        "Pure OHLCV per-ticker proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_realized_semivar_ds",      # downside variance share  [0,1]
        "ext3_realized_semivar_sj",      # signed jump ratio        [-1,+1]
        "ext3_realized_semivar_mom20",   # 20-day change of ds
    ],
    "tags": ["volatility", "semivariance", "downside", "jump", "realized"],
    "version": "1.0",
    "author": "Round-4 expansion (xdom2_downside_beta); Barndorff-Nielsen et al.",
}

_WIN = 21
_MOM = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns (causal: current Close / previous Close)
    ret = df["Close"].pct_change()

    # Squared positive and negative returns
    ret_pos_sq = ret.clip(lower=0.0) ** 2   # 0 where ret <= 0
    ret_neg_sq = ret.clip(upper=0.0) ** 2   # 0 where ret >= 0

    # Rolling realized semivariances over 21-day window
    rs_plus  = ret_pos_sq.rolling(_WIN, min_periods=_WIN).sum()
    rs_minus = ret_neg_sq.rolling(_WIN, min_periods=_WIN).sum()

    total = rs_plus + rs_minus

    # Guard against divide-by-zero (flat price series)
    total_safe = total.replace(0.0, np.nan)

    # Downside variance share  RS- / (RS+ + RS-)  in [0, 1]
    ds = rs_minus / total_safe

    # Signed jump ratio  (RS+ - RS-) / total  in [-1, +1]
    sj = (rs_plus - rs_minus) / total_safe

    # 20-bar momentum of downside share
    mom20 = ds.diff(_MOM)

    df["ext3_realized_semivar_ds"]    = ds
    df["ext3_realized_semivar_sj"]    = sj
    df["ext3_realized_semivar_mom20"] = mom20

    return df
