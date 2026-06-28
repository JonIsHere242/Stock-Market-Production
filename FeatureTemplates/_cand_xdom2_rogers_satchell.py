"""
Rogers-Satchell drift-independent volatility (1991) — per-ticker block.

RS estimator: mean(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)) over a rolling 20-day window.
Drift-independent (no close-to-close term), unbiased under any constant drift.
The ratio rs_cc_ratio_20 probes intraday vs overnight variance contribution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_rogers_satchell",
    "description": (
        "Rolling 20-day Rogers-Satchell (1991) range-based volatility estimator. "
        "RS_t = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) is drift-independent and uses only "
        "within-day OHLC. Produces: (1) xdom2_rogers_satchell_vol20 = annualised RS vol "
        "(sqrt of 20-day mean RS, scaled by sqrt(252)); "
        "(2) xdom2_rogers_satchell_cc20 = classical close-to-close annualised vol over 20 days; "
        "(3) xdom2_rogers_satchell_ratio20 = RS vol / CC vol, an intraday-vs-overnight "
        "efficiency probe (>1 = large intraday range relative to overnight gaps, <1 = gap-driven). "
        "Per-ticker time-series implementation; no cross-sectional dependency."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "xdom2_rogers_satchell_vol20",
        "xdom2_rogers_satchell_cc20",
        "xdom2_rogers_satchell_ratio20",
    ],
    "tags": ["volatility", "range-based", "rogers-satchell", "intraday", "cross-domain"],
    "version": "1.0.0",
    "author": "Rogers-Satchell drift-independent volatility (1991); cross-domain-method batch 2",
}

_WINDOW = 20
_ANNUALISE = np.sqrt(252)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"].to_numpy(dtype=np.float64)
    h = df["High"].to_numpy(dtype=np.float64)
    lo = df["Low"].to_numpy(dtype=np.float64)
    c = df["Close"].to_numpy(dtype=np.float64)

    # Guard against non-positive prices to avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lh_c = np.where((h > 0) & (c > 0), np.log(h / c), np.nan)
        lh_o = np.where((h > 0) & (o > 0), np.log(h / o), np.nan)
        ll_c = np.where((lo > 0) & (c > 0), np.log(lo / c), np.nan)
        ll_o = np.where((lo > 0) & (o > 0), np.log(lo / o), np.nan)

    # Rogers-Satchell per-bar variance proxy
    rs_bar = lh_c * lh_o + ll_c * ll_o  # non-negative in theory

    # Close-to-close log returns for the CC estimator
    with np.errstate(divide="ignore", invalid="ignore"):
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan
        cc_ret = np.where((c > 0) & (prev_c > 0), np.log(c / prev_c), np.nan)

    # Convert to pd.Series for rolling (respects NaN naturally)
    rs_s = pd.Series(rs_bar, index=df.index)
    cc_s = pd.Series(cc_ret, index=df.index)

    # Rolling 20-day mean of RS bar variance -> annualised vol
    rs_mean = rs_s.rolling(_WINDOW, min_periods=_WINDOW).mean()
    # RS mean can be slightly negative due to fp noise; clip to 0 before sqrt
    rs_vol = np.sqrt(rs_mean.clip(lower=0)) * _ANNUALISE

    # Close-to-close vol: std of log returns (ddof=1) * sqrt(252)
    cc_vol = cc_s.rolling(_WINDOW, min_periods=_WINDOW).std(ddof=1) * _ANNUALISE

    # Ratio: RS vol / CC vol; guard against CC vol == 0
    ratio = rs_vol / cc_vol.replace(0, np.nan)
    # Replace inf/-inf with NaN
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["xdom2_rogers_satchell_vol20"] = rs_vol.values
    df["xdom2_rogers_satchell_cc20"] = cc_vol.values
    df["xdom2_rogers_satchell_ratio20"] = ratio.values

    return df
