from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_parkinson_gap",
    "description": (
        "Overnight-gap vs intraday-range decomposition. "
        "Decomposes total close-to-close variance into overnight (ln(Open/prevClose)) "
        "and intraday (ln(Close/Open)) log-return components over a rolling 40-day window. "
        "Produces: (1) xdom2_parkinson_gap_ovar_share_40 -- share of total variance explained "
        "by overnight gaps [0,1]; (2) xdom2_parkinson_gap_gap_skew_40 -- skewness of overnight "
        "log-gaps over the same window (microstructure flavour -- persistent positive skew implies "
        "upward gap drift; negative = downward gap pressure); "
        "(3) xdom2_parkinson_gap_gap_z -- z-score of today's overnight gap relative to the 40d "
        "rolling distribution, capturing whether the current gap is extreme. "
        "Per-ticker proxy; no cross-sectional component needed as the decomposition is stock-level. "
        "Method: Cross-domain / practitioner method transfer (batch 2)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "xdom2_parkinson_gap_ovar_share_40",
        "xdom2_parkinson_gap_gap_skew_40",
        "xdom2_parkinson_gap_gap_z",
    ],
    "tags": ["volatility", "microstructure", "overnight", "gap", "decomposition", "cross-domain"],
    "version": "1.0",
    "author": "Cross-domain / practitioner method transfer (batch 2) — Overnight-gap vs intraday-range decomposition",
}

_WINDOW = 40
_MIN_PERIODS = 10


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Overnight log-return: ln(Open_t / Close_{t-1})
    prev_close = df["Close"].shift(1)
    gap = np.log(df["Open"] / prev_close.replace(0, np.nan))

    # Intraday log-return: ln(Close_t / Open_t)
    open_ = df["Open"].replace(0, np.nan)
    intraday = np.log(df["Close"] / open_)

    # Rolling variance of each component (population-style via pandas default ddof=1)
    gap_var = gap.rolling(window=_WINDOW, min_periods=_MIN_PERIODS).var()
    intra_var = intraday.rolling(window=_WINDOW, min_periods=_MIN_PERIODS).var()

    # Overnight variance share: overnight_var / (overnight_var + intraday_var)
    total_var = gap_var + intra_var
    # Guard divide-by-zero
    ovar_share = gap_var / total_var.where(total_var != 0, np.nan)
    df["xdom2_parkinson_gap_ovar_share_40"] = ovar_share.clip(0.0, 1.0)

    # Rolling skewness of overnight gaps
    gap_skew = gap.rolling(window=_WINDOW, min_periods=_MIN_PERIODS).skew()
    df["xdom2_parkinson_gap_gap_skew_40"] = gap_skew

    # Z-score of today's gap relative to the 40d rolling distribution
    gap_mean = gap.rolling(window=_WINDOW, min_periods=_MIN_PERIODS).mean()
    gap_std = gap.rolling(window=_WINDOW, min_periods=_MIN_PERIODS).std()
    gap_z = (gap - gap_mean) / gap_std.where(gap_std != 0, np.nan)
    df["xdom2_parkinson_gap_gap_z"] = gap_z

    return df
