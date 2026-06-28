"""
Amihud-illiquidity-weighted short-term reversal.

Each daily return in a 21-day rolling window is weighted by that day's Amihud
illiquidity measure (|ret| / (Close * Volume)) before summing, so that price
moves on illiquid days contribute more to the reversal signal. The feature
captures the well-established finding that illiquid-day moves tend to reverse
harder than liquid-day moves, giving a different signal axis from the plain
equal-weighted short-term reversal (osap_streversal).

Three columns produced:
  ext_reversal_illiq_weighted_main   -- negated illiq-weighted 21d return sum
                                        (positive = oversold, expect bounce)
  ext_reversal_illiq_weighted_ratio  -- illiq-weighted sum / equal-weighted sum
                                        (>1 means illiquid days dominate recent
                                         move, amplifying the reversal signal)
  ext_reversal_illiq_weighted_illiq  -- trailing 21d mean Amihud illiquidity
                                        (conditioning variable / level)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_reversal_illiq_weighted",
    "description": (
        "21-day short-term reversal where each day's return is weighted by "
        "its Amihud illiquidity (|ret|/(Close*Volume)) before summing. "
        "Negated so positive values signal expected bounce. Produces the "
        "illiq-weighted reversal, ratio to equal-weighted reversal, and the "
        "trailing mean illiquidity level. Per-ticker OHLCV proxy; no "
        "cross-sectional component."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_reversal_illiq_weighted_main",
        "ext_reversal_illiq_weighted_ratio",
        "ext_reversal_illiq_weighted_illiq",
    ],
    "tags": ["reversal", "illiquidity", "amihud", "short_term", "microstructure"],
    "version": "1.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner osap_streversal; "
        "based on Amihud (2002) illiquidity ratio and short-term reversal "
        "literature (Jegadeesh 1990, Avramov, Chordia & Goyal 2006)."
    ),
}

_WINDOW = 21


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    n = len(df)

    # Daily log return (causal: uses today's and yesterday's close)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.log(close[1:] / close[:-1])

    # Amihud illiquidity: |ret| / (Close * Volume); guard zero denominator
    dollar_vol = close * volume
    with np.errstate(divide="ignore", invalid="ignore"):
        illiq = np.where(dollar_vol > 0, np.abs(ret) / dollar_vol, np.nan)

    # Weight = illiquidity ratio (normalised within window below)
    # Rolling sums over _WINDOW days (look-back, no lookahead)
    # We compute via pandas rolling for readability/speed
    s_ret = pd.Series(ret, index=df.index)
    s_illiq = pd.Series(illiq, index=df.index)

    roll_ret = s_ret.rolling(_WINDOW, min_periods=_WINDOW // 2)
    roll_illiq = s_illiq.rolling(_WINDOW, min_periods=_WINDOW // 2)

    # Equal-weighted sum of returns (plain reversal)
    ew_sum = roll_ret.sum()

    # Illiquidity-weighted sum: sum(illiq_t * ret_t) for t in window
    illiq_x_ret = s_illiq * s_ret
    illiq_x_ret_roll = illiq_x_ret.rolling(_WINDOW, min_periods=_WINDOW // 2)
    iw_sum = illiq_x_ret_roll.sum()

    # Mean illiquidity over window (level / conditioning variable)
    mean_illiq = roll_illiq.mean()

    # Reversal = NEGATED weighted sum (big negative move → big positive signal)
    main = -iw_sum

    # Ratio: illiq-weighted / equal-weighted; guard zero ew_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            ew_sum.abs() > 1e-15,
            iw_sum.values / ew_sum.values,
            np.nan,
        )

    df["ext_reversal_illiq_weighted_main"] = main.values
    df["ext_reversal_illiq_weighted_ratio"] = ratio
    df["ext_reversal_illiq_weighted_illiq"] = mean_illiq.values

    return df
