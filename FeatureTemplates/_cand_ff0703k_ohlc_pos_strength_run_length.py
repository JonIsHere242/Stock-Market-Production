from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703k_ohlc_pos_strength_run_length",
    "description": (
        "Signed run-length of consecutive 'extreme close' days based on intrabar closing "
        "position p_t=(Close-Low)/(High-Low) (High-Low<=0 guarded to NaN/neutral). A day is "
        "STRONG if p_t>=0.75 (close near the high, buyers-in-control), WEAK if p_t<=0.25 (close "
        "near the low, sellers-in-control), else neutral. "
        "ff0703k_ohlc_pos_strength_run_length_signed = +k if the last k days were all STRONG "
        "(broken by any non-STRONG day), -k if the last k days were all WEAK, 0 on a neutral day. "
        "ff0703k_ohlc_pos_strength_run_length_chg5 = signed run length today minus its value 5 "
        "trading days ago, capturing whether persistence of support/rejection at the extremes is "
        "building or breaking down. Both computed causally (grouping consecutive equal-state days "
        "via a backward-looking state-change cumsum), no future information used. Pure per-ticker "
        "OHLC construction; no cross-sectional inputs."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff0703k_ohlc_pos_strength_run_length_signed",
        "ff0703k_ohlc_pos_strength_run_length_chg5",
    ],
    "tags": ["ohlc_pos", "run-length", "candle", "momentum", "structure"],
    "version": "1.0",
    "author": "feature-factory ff0703k spec (faithful vectorised implementation)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df["ff0703k_ohlc_pos_strength_run_length_signed"] = np.nan
    df["ff0703k_ohlc_pos_strength_run_length_chg5"] = np.nan

    if n == 0:
        return df

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    hl = high - low
    hl_safe = hl.where(hl > 0, np.nan)
    p = (close - low) / hl_safe

    state = pd.Series(0, index=df.index, dtype=np.int64)
    state = state.mask(p >= 0.75, 1)
    state = state.mask(p <= 0.25, -1)
    # NaN p (zero/invalid range) stays neutral (0), which correctly breaks any run.

    # Causal grouping: a new group starts whenever state differs from the prior bar
    # (first bar always starts a new group). Cumcount within group -> run magnitude.
    changed = state.ne(state.shift(1))
    changed.iloc[0] = True
    group_id = changed.cumsum()

    run_len = group_id.groupby(group_id).cumcount() + 1
    signed_run = (state * run_len).astype(float)

    df["ff0703k_ohlc_pos_strength_run_length_signed"] = signed_run
    df["ff0703k_ohlc_pos_strength_run_length_chg5"] = signed_run - signed_run.shift(5)

    return df
