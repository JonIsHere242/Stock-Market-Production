"""
Close-Location-Value (CLV) Accumulation / Distribution
Williams Accumulation/Distribution pressure per ticker.

Daily CLV = ((Close - Low) - (High - Close)) / (High - Low)
  = (2*Close - High - Low) / (High - Low)
Captures where the close sits within the day's High-Low range:
  +1 = closed at high (max accumulation), -1 = closed at low (max distribution).

Produced features:
  xdom2_clv_accum_20  : 20-day CLV*Volume cumsum normalised by 20-day rolling Volume sum
                        (volume-weighted accumulation/distribution pressure level)
  xdom2_clv_mean_20   : 20-day equal-weight mean of CLV (range-position pressure, no vol weight)
  xdom2_clv_slope     : 5-day rate-of-change of xdom2_clv_accum_20 (momentum of pressure)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_clv_accum",
    "description": (
        "Per-ticker Close-Location-Value (CLV) accumulation/distribution pressure. "
        "CLV = (2*Close - High - Low) / (High - Low) measures intraday close position "
        "relative to the day's range. clv_accum_20 is the 20-day rolling sum of "
        "CLV*Volume divided by rolling 20-day Volume sum (volume-weighted A/D pressure). "
        "clv_mean_20 is the 20-day equal-weight mean CLV. clv_slope is the 5-day change "
        "in clv_accum_20. Faithful per-ticker implementation of Williams A/D. "
        "Positive values signal accumulation (buying pressure); negative signal distribution."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "xdom2_clv_accum_20",
        "xdom2_clv_mean_20",
        "xdom2_clv_slope",
    ],
    "tags": ["accumulation_distribution", "volume", "pressure", "williams", "clv", "cross_domain"],
    "version": "1.0",
    "author": "Cross-domain / practitioner method transfer (batch 2) — Williams Accumulation/Distribution CLV",
}

_WINDOW = 20
_SLOPE_WINDOW = 5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Daily range; guard against zero-range (doji / halt) with np.nan
    hl_range = high - low
    hl_safe = hl_range.where(hl_range > 0, np.nan)

    # CLV in [-1, +1]
    clv = (2.0 * close - high - low) / hl_safe  # NaN where High == Low

    # Volume-weighted AD flow for each bar
    clv_vol = clv * volume

    # 20-day rolling volume sum (guard against zero)
    roll_vol = volume.rolling(_WINDOW, min_periods=_WINDOW).sum()
    roll_vol_safe = roll_vol.where(roll_vol > 0, np.nan)

    # clv_accum_20: rolling 20-day CLV*Volume sum / rolling 20-day Volume sum
    roll_clv_vol = clv_vol.rolling(_WINDOW, min_periods=_WINDOW).sum()
    df["xdom2_clv_accum_20"] = roll_clv_vol / roll_vol_safe

    # clv_mean_20: equal-weight 20-day mean of CLV (range-position pressure)
    df["xdom2_clv_mean_20"] = clv.rolling(_WINDOW, min_periods=_WINDOW).mean()

    # clv_slope: 5-day difference of clv_accum_20 (momentum of A/D pressure)
    accum = df["xdom2_clv_accum_20"]
    df["xdom2_clv_slope"] = accum.diff(_SLOPE_WINDOW)

    return df
