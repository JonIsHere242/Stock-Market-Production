from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703h_intraday_moment_signed_close_semivar_ratio",
    "description": (
        "Leverage-style intraday semivariance ratio conditioned on daily close direction. "
        "Computes the intraday open-to-close log return co=ln(Close/Open) and the sign of the "
        "day's close-to-close move s=sign(Close-prevClose). Over a trailing 60-day window, "
        "var_down = mean(co^2) over days where s<0 and var_up = mean(co^2) over days where s>0 "
        "(each requiring >=5 qualifying days in the window, else NaN). Level column is "
        "var_down/var_up (denominator zero/NaN -> NaN): >1 means intraday variance is larger on "
        "down-closing days (leverage-effect-like asymmetry), <1 means larger on up-closing days. "
        "Dynamic column is the 60-day change in that ratio: current ratio minus the ratio computed "
        "on the window ending 20 trading days earlier. Pure per-ticker OHLC construction, no "
        "cross-sectional or external data; causal fixed-length trailing windows only."
    ),
    "requires": ["Open", "Close"],
    "produces": [
        "ff0703h_intraday_moment_signed_close_semivar_ratio_level",
        "ff0703h_intraday_moment_signed_close_semivar_ratio_dyn",
    ],
    "tags": ["intraday", "semivariance", "leverage_effect", "volatility", "asymmetry"],
    "version": "1.0",
    "author": "ff0703h codegen: faithful per-ticker implementation of spec (conditional semivariance ratio)",
}

_WINDOW = 60
_LAG = 20
_MIN_DAYS = 5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    level_col = "ff0703h_intraday_moment_signed_close_semivar_ratio_level"
    dyn_col = "ff0703h_intraday_moment_signed_close_semivar_ratio_dyn"

    df[level_col] = np.nan
    df[dyn_col] = np.nan

    n = len(df)
    if n == 0:
        return df

    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)

    open_safe = open_.where(open_ > 0, np.nan)
    co = np.log(close.where(close > 0, np.nan) / open_safe)
    co2 = co * co

    prev_close = close.shift(1)
    delta = close - prev_close
    down_flag = (delta < 0).astype(float)
    up_flag = (delta > 0).astype(float)
    # Where delta is NaN (first bar) or co2 is NaN, exclude from both flags/values
    valid = co2.notna() & delta.notna()
    down_flag = down_flag.where(valid, 0.0)
    up_flag = up_flag.where(valid, 0.0)

    down_val = (co2 * down_flag).fillna(0.0)
    up_val = (co2 * up_flag).fillna(0.0)

    down_sum = down_val.rolling(_WINDOW, min_periods=_WINDOW).sum()
    up_sum = up_val.rolling(_WINDOW, min_periods=_WINDOW).sum()
    down_cnt = down_flag.rolling(_WINDOW, min_periods=_WINDOW).sum()
    up_cnt = up_flag.rolling(_WINDOW, min_periods=_WINDOW).sum()

    down_cnt_safe = down_cnt.where(down_cnt >= _MIN_DAYS, np.nan)
    up_cnt_safe = up_cnt.where(up_cnt >= _MIN_DAYS, np.nan)

    var_down = down_sum / down_cnt_safe
    var_up = up_sum / up_cnt_safe

    var_up_denom = var_up.where(var_up != 0, np.nan)
    ratio = var_down / var_up_denom

    df[level_col] = ratio
    df[dyn_col] = ratio - ratio.shift(_LAG)

    return df
