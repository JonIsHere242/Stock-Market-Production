"""
ff0703k_intraday_moment_cubed_std_intraday_ret — Realized skewness of the
intraday (open-to-close) session return.
Candidate block (unproven); prefixed with _ so discovery skips it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703k_intraday_moment_cubed_std_intraday_ret",
    "description": (
        "Realized skewness of the intraday session return i_t = Close/Open - 1 "
        "(NaN if Open<=0). i_t is standardized with a 60-day rolling mean/std "
        "(z_t = (i_t - mu60)/sigma60, NaN if sigma60<=1e-9); LEVEL "
        "(ff0703k_intraday_moment_cubed_std_intraday_ret_level) is the 20-day rolling "
        "mean of z_t^3 -- a direct cubed-standardized-return realized skew of the "
        "open-to-close move, structurally distinct from OHLC close-position features. "
        "DYNAMIC (ff0703k_intraday_moment_cubed_std_intraday_ret_dyn) is the z-score of "
        "the level vs its own 120-day rolling mean/std (NaN if std<=1e-9). Pure OHLC, "
        "causal fixed-window rolling ops only, no lookahead."
    ),
    "requires": ["Open", "Close"],
    "produces": [
        "ff0703k_intraday_moment_cubed_std_intraday_ret_level",
        "ff0703k_intraday_moment_cubed_std_intraday_ret_dyn",
    ],
    "tags": ["intraday", "skewness", "moments", "rolling", "ohlc"],
    "version": "1.0.0",
    "author": "Spec: ff0703k (intraday_moment vein) — realized skew of standardized intraday return",
}

_STD_WIN = 60
_SKEW_WIN = 20
_DYN_WIN = 120
_MIN_STD = max(20, _STD_WIN // 3)
_MIN_SKEW = max(10, _SKEW_WIN // 2)
_MIN_DYN = max(30, _DYN_WIN // 3)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    level_col = "ff0703k_intraday_moment_cubed_std_intraday_ret_level"
    dyn_col = "ff0703k_intraday_moment_cubed_std_intraday_ret_dyn"

    # Initialise produced columns up front on every code path.
    df[level_col] = np.nan
    df[dyn_col] = np.nan

    if len(df) == 0:
        return df

    open_ = df["Open"]
    close = df["Close"]

    open_safe = open_.where(open_ > 0)
    i_t = (close / open_safe) - 1.0

    mu60 = i_t.rolling(_STD_WIN, min_periods=_MIN_STD).mean()
    sigma60 = i_t.rolling(_STD_WIN, min_periods=_MIN_STD).std()
    sigma60_safe = sigma60.where(sigma60 > 1e-9)

    z_t = (i_t - mu60) / sigma60_safe

    z3 = z_t.pow(3)
    level = z3.rolling(_SKEW_WIN, min_periods=_MIN_SKEW).mean()
    df[level_col] = level

    dyn_mu = level.rolling(_DYN_WIN, min_periods=_MIN_DYN).mean()
    dyn_std = level.rolling(_DYN_WIN, min_periods=_MIN_DYN).std()
    dyn_std_safe = dyn_std.where(dyn_std > 1e-9)

    df[dyn_col] = (level - dyn_mu) / dyn_std_safe

    return df
