"""
ext4_signed_jump_skew — Net signed jump asymmetry (up vs down jump skew)

Flags days where |return| > 4x the local bipower variation estimate of vol,
then computes rolling 90-day metrics:
  - net signed jump ratio: (up_jumps - down_jumps) / total_jumps
  - signed jump magnitude sum: sum(ret * jump_flag * sign)
  - 20d change of the net signed ratio (momentum of asymmetry)

Pure OHLCV, no lookahead.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext4_signed_jump_skew",
    "description": (
        "Per-ticker rolling 90-day signed jump asymmetry. Identifies jumps as days where "
        "|log-return| exceeds 4x the local bipower-variation vol estimate (BV = pi/2 * "
        "mean(|r_t|*|r_{t-1}|), annualised to daily). Produces: (1) net signed jump ratio "
        "(up_count - down_count) / total_count over 90d window, (2) rolling signed jump "
        "magnitude sum, (3) 20d change of the ratio. Proxy for the xdom2_downside_beta axis "
        "from a pure per-ticker OHLCV perspective — captures whether a stock's large moves "
        "are skewed up or down, which is orthogonal to symmetric vol measures."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_signed_jump_skew_ratio",
        "ext4_signed_jump_skew_mag_sum",
        "ext4_signed_jump_skew_ratio_chg20",
    ],
    "tags": ["jump", "skew", "asymmetry", "ohlcv", "rolling"],
    "version": "1.0",
    "author": "Round-5 expansion spec (xdom2_downside_beta lineage)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns (no lookahead: shift(1) uses past bar)
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Bipower variation: BV_t = (pi/2) * |r_{t-1}| * |r_t|, rolling 22-day mean
    # This estimates the continuous-variation component of vol
    abs_ret = log_ret.abs()
    bv_prod = abs_ret * abs_ret.shift(1)  # |r_t| * |r_{t-1}|
    # rolling mean of bv_prod gives estimate of daily BV; scale by pi/2
    bv_rolling = bv_prod.rolling(22, min_periods=10).mean() * (np.pi / 2)
    # Convert to daily vol estimate (BV is already a variance proxy, take sqrt)
    bv_vol = np.sqrt(bv_rolling)

    # Threshold: 4x local bipower vol
    threshold = 4.0 * bv_vol

    # Jump flags
    is_jump = log_ret.abs() > threshold
    is_up_jump = is_jump & (log_ret > 0)
    is_down_jump = is_jump & (log_ret < 0)

    # Cast to float for rolling sums
    up_flag = is_up_jump.astype(float)
    down_flag = is_down_jump.astype(float)
    jump_flag = is_jump.astype(float)

    # Rolling 90-day counts
    window = 90
    min_p = 20

    up_count = up_flag.rolling(window, min_periods=min_p).sum()
    down_count = down_flag.rolling(window, min_periods=min_p).sum()
    total_count = jump_flag.rolling(window, min_periods=min_p).sum()

    # Net signed jump ratio: (up - down) / total  [range -1 to +1]
    # Guard: divide by total; if total == 0 -> NaN
    total_safe = total_count.where(total_count > 0, other=np.nan)
    ratio = (up_count - down_count) / total_safe

    # Signed jump magnitude sum over 90d:
    # Each jump day contributes +|ret| if up, -|ret| if down
    signed_mag = log_ret * jump_flag * np.sign(log_ret)
    mag_sum = signed_mag.rolling(window, min_periods=min_p).sum()

    # 20d momentum of ratio
    ratio_chg20 = ratio - ratio.shift(20)

    df["ext4_signed_jump_skew_ratio"] = ratio
    df["ext4_signed_jump_skew_mag_sum"] = mag_sum
    df["ext4_signed_jump_skew_ratio_chg20"] = ratio_chg20

    return df
