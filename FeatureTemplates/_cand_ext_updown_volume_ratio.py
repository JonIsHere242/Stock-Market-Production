"""
Up/Down-Day Volume Pressure Ratio
----------------------------------
Rolling ratio of total volume on up-days vs down-days, capturing directional
volume-pressure imbalance.  Distinct from volume-return autocorrelation (the
parent feature xdom2_autocorr_volume_return): that measures *lagged* correlation
between volume and return; this measures the *asymmetry* in accumulated volume
on bullish vs bearish days.  Three produced columns:
  - 40d ratio (primary level)
  - 60d ratio (longer-window confirmation)
  - 20d change in the 40d ratio (momentum of pressure shift)

Source: Extension/exploration of xdom2_autocorr_volume_return gate-validated winner.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_updown_volume_ratio",
    "description": (
        "Per-ticker rolling up-day / down-day volume pressure ratio. "
        "40d and 60d windows capture buying vs selling pressure imbalance; "
        "the 20d change in the 40d ratio captures momentum of that shift. "
        "Orthogonal to xdom2_autocorr_volume_return (level asymmetry vs lagged correlation)."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_updown_volume_ratio_40d",
        "ext_updown_volume_ratio_60d",
        "ext_updown_volume_ratio_change20d",
    ],
    "tags": ["volume", "pressure", "directional", "rolling"],
    "version": "1.0",
    "author": "Extension/exploration of gate-validated winner xdom2_autocorr_volume_return",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute up/down volume pressure ratio features."""
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    n = len(df)

    # Day return sign: up (>0) vs down (<=0); first row has no prior close -> NaN
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    if n > 1:
        ret[1:] = close[1:] - close[:-1]

    is_up = (ret > 0).astype(np.float64)   # 1 on up-day, 0 on flat/down
    is_down = (ret <= 0).astype(np.float64)  # 1 on flat or down-day
    # flat days contribute to 'down' volume; exclude first row (NaN)
    is_up[0] = np.nan
    is_down[0] = np.nan

    up_vol = volume * is_up      # volume on up-days, else 0
    dn_vol = volume * is_down    # volume on down-days, else 0

    up_s = pd.Series(up_vol, index=df.index)
    dn_s = pd.Series(dn_vol, index=df.index)

    def _ratio(win: int) -> pd.Series:
        sum_up = up_s.rolling(win, min_periods=max(1, win // 2)).sum()
        sum_dn = dn_s.rolling(win, min_periods=max(1, win // 2)).sum()
        # guard divide-by-zero
        denom = sum_dn.where(sum_dn != 0, other=np.nan)
        ratio = sum_up / denom
        # replace inf/-inf with nan
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        return ratio

    ratio_40 = _ratio(40)
    ratio_60 = _ratio(60)
    change_20 = ratio_40 - ratio_40.shift(20)

    df["ext_updown_volume_ratio_40d"] = ratio_40.values
    df["ext_updown_volume_ratio_60d"] = ratio_60.values
    df["ext_updown_volume_ratio_change20d"] = change_20.values

    return df
