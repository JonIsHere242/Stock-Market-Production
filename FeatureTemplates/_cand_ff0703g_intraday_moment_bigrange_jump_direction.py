"""
ff0703g_intraday_moment_bigrange_jump_direction — Directional asymmetry of
big-range ("jump") sessions.

VEIN: intraday_moment
Candidate block (unproven); prefixed with _ so discovery skips it.

Definition (per spec):
  rng = High - Low (0 -> NaN)
  med60 = 60d rolling median of rng
  jump day: rng > 1.5 * med60
  dir = sign(Close - Open) on jump days only
  Over trailing 60d window:
    LEVEL      frac_down = (# jump days with dir<0) / (# jump days)   [0 jumps -> NaN]
    ASYMMETRY  mean(dir) over jump days in the window                  [0 jumps -> NaN]
    DYNAMIC    frac_down(t) - frac_down(t-20)

This isolates the directional skew of the largest-range (jump-like) sessions,
which dominate realized skewness -- distinct from ordinary gap/range/volatility
features that don't condition on the jump subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703g_intraday_moment_bigrange_jump_direction",
    "description": (
        "Per-ticker, OHLC-only directional asymmetry of big-range 'jump' sessions. "
        "rng=High-Low; a day is a jump day when rng > 1.5x its trailing 60d median "
        "rolling range; dir=sign(Close-Open) recorded only on jump days. Over a "
        "trailing 60d window: (1) ff0703g_intraday_moment_bigrange_jump_direction_frac_down "
        "= fraction of jump days that closed down (LEVEL); "
        "(2) ff0703g_intraday_moment_bigrange_jump_direction_asym = mean(dir) over jump "
        "days in the window (ASYMMETRY, in [-1,1]); "
        "(3) ff0703g_intraday_moment_bigrange_jump_direction_dyn = frac_down(t) - "
        "frac_down(t-20) (DYNAMIC, trend in jump-direction skew). Windows with zero "
        "jump days in the trailing 60d -> NaN. Faithful direct implementation of the "
        "spec; pure OHLC, causal, no lookahead."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703g_intraday_moment_bigrange_jump_direction_frac_down",
        "ff0703g_intraday_moment_bigrange_jump_direction_asym",
        "ff0703g_intraday_moment_bigrange_jump_direction_dyn",
    ],
    "tags": ["intraday-moment", "jump", "skewness", "asymmetry", "candle-structure", "rolling"],
    "version": "1.0.0",
    "author": "Spec ff0703g (intraday_moment vein) — big-range jump-day directional asymmetry",
}

_MED_WINDOW = 60
_ASYM_WINDOW = 60
_DYN_LAG = 20
_MIN_PERIODS_MED = 20
_MIN_PERIODS_ASYM = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    col_level = "ff0703g_intraday_moment_bigrange_jump_direction_frac_down"
    col_asym = "ff0703g_intraday_moment_bigrange_jump_direction_asym"
    col_dyn = "ff0703g_intraday_moment_bigrange_jump_direction_dyn"

    # Initialise on every path
    df[col_level] = np.nan
    df[col_asym] = np.nan
    df[col_dyn] = np.nan

    if n == 0:
        return df

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    rng = (high - low)
    rng = rng.where(rng != 0, np.nan)

    med60 = rng.rolling(_MED_WINDOW, min_periods=_MIN_PERIODS_MED).median()

    is_jump = (rng > 1.5 * med60).fillna(False)

    dir_raw = np.sign(close - open_)
    # dir only defined on jump days; elsewhere NaN so rolling sums ignore them
    dir_on_jump = dir_raw.where(is_jump)

    jump_flag = is_jump.astype(float)
    down_flag = (dir_on_jump < 0).astype(float).where(is_jump)

    roll_jump_count = jump_flag.rolling(_ASYM_WINDOW, min_periods=1).sum()
    roll_down_count = down_flag.rolling(_ASYM_WINDOW, min_periods=1).sum()
    roll_dir_sum = dir_on_jump.rolling(_ASYM_WINDOW, min_periods=1).sum()

    denom = roll_jump_count.replace(0, np.nan)

    frac_down = roll_down_count / denom
    asym = roll_dir_sum / denom

    df[col_level] = frac_down
    df[col_asym] = asym
    df[col_dyn] = frac_down - frac_down.shift(_DYN_LAG)

    return df
