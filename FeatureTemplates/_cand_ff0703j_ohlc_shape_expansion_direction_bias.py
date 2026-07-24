from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703j_ohlc_shape_expansion_direction_bias",
    "description": (
        "Range-expansion release-direction bias. R = High-Low (0->NaN); rmean20 = 20-bar "
        "trailing rolling mean of R. A bar is an 'expansion day' when R_t > rmean20_t. On "
        "expansion days, intraday sign = sign(Close/Open - 1) (NaN if Open==0). "
        "ff0703j_ohlc_shape_expansion_direction_bias_expdir = the mean of that intraday sign "
        "taken ONLY over expansion days within the trailing 40 bars (NaN if none) -- captures "
        "whether released range-energy tends to resolve up or down intraday. "
        "ff0703j_ohlc_shape_expansion_direction_bias_expdir_chg = expdir_t - expdir_{t-20}, the "
        "20-bar drift in that release-direction bias. Pure per-ticker OHLC, no lookahead: rmean20 "
        "and the expansion mask are computed from trailing-only data, and the 40-bar aggregation "
        "window ends at t."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703j_ohlc_shape_expansion_direction_bias_expdir",
        "ff0703j_ohlc_shape_expansion_direction_bias_expdir_chg",
    ],
    "tags": ["ohlc", "range", "expansion", "direction", "bias", "candle"],
    "version": "1.0",
    "author": "ff0703j spec batch (faithful implementation)",
}

_RMEAN_WIN = 20
_AGG_WIN = 40
_CHG_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_level = "ff0703j_ohlc_shape_expansion_direction_bias_expdir"
    col_chg = "ff0703j_ohlc_shape_expansion_direction_bias_expdir_chg"

    df[col_level] = np.nan
    df[col_chg] = np.nan

    if len(df) == 0:
        return df

    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    close = df["Close"]

    rng = (high - low).replace(0, np.nan)
    rmean20 = rng.rolling(_RMEAN_WIN, min_periods=_RMEAN_WIN).mean()

    expansion = rng > rmean20  # NaN comparisons -> False, which is correct (no signal)

    open_safe = open_.where(open_ != 0, np.nan)
    intraday_ret = close / open_safe - 1.0
    intraday_sign = np.sign(intraday_ret)

    masked_sign = intraday_sign.where(expansion)

    expdir = masked_sign.rolling(_AGG_WIN, min_periods=1).mean()
    # rolling().mean() with all-NaN window yields NaN naturally.

    df[col_level] = expdir
    df[col_chg] = expdir - expdir.shift(_CHG_LAG)

    return df
