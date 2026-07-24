from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703a_microstructure_intraday_oc_momentum_level_60",
    "description": (
        "Intraday open-to-close momentum LEVEL: 60-day rolling mean of daily "
        "log(Close/Open), a proxy for persistent intraday drift distinct from "
        "overnight (close-to-open) gap moves. Guards Open>0 and Close>0 "
        "(non-positive treated as missing). NaN if fewer than 30 valid daily "
        "observations in the trailing 60-bar window (min_periods=30). A second "
        "companion column captures the SLOPE/dynamics of that level -- the "
        "change in the 60d mean intraday drift over the last 20 days -- to "
        "distinguish a stable persistent drift from one that is accelerating "
        "or decaying. Pure per-ticker OHLC, no cross-sectional or lookahead "
        "data used."
    ),
    "requires": ["Open", "Close"],
    "produces": [
        "ff0703a_microstructure_intraday_oc_momentum_level_60_mean",
        "ff0703a_microstructure_intraday_oc_momentum_level_60_slope",
    ],
    "tags": ["microstructure", "intraday", "momentum", "open-close", "drift"],
    "version": "1.0",
    "author": "ff0703a batch spec, direct faithful implementation (per-ticker OHLC proxy)",
}

_WINDOW = 60
_MIN_PERIODS = 30
_SLOPE_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_mean = "ff0703a_microstructure_intraday_oc_momentum_level_60_mean"
    col_slope = "ff0703a_microstructure_intraday_oc_momentum_level_60_slope"

    df[col_mean] = np.nan
    df[col_slope] = np.nan

    if len(df) == 0:
        return df

    open_ = df["Open"].astype(float)
    close_ = df["Close"].astype(float)

    # Guard non-positive prices -> NaN (undefined log ratio)
    valid = (open_ > 0) & (close_ > 0)
    ratio = close_.where(valid) / open_.where(valid)
    daily_oc_logret = np.log(ratio.where(ratio > 0))

    level = daily_oc_logret.rolling(_WINDOW, min_periods=_MIN_PERIODS).mean()

    df[col_mean] = level
    df[col_slope] = level - level.shift(_SLOPE_LAG)

    return df
