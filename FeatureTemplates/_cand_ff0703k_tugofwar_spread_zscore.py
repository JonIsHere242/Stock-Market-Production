"""
ff0703k_tugofwar_spread_zscore — normalized tug-of-war spread (overnight vs
intraday return dominance), z-scored against its own longer-run distribution.

Definition (per spec ff0703k_tugofwar_spread_zscore):
  o_t = Open_t / Close_{t-1} - 1          (overnight gap return)
  i_t = Close_t / Open_t - 1              (intraday return)
  s_t = o_t - i_t                          (tug spread: overnight vs intraday "winner")

  spread_z_126_t = (mean(s over last 21d) - mean(s over last 126d)) / std(s over last 126d)
    guarded: std < 1e-9 or fewer than 126 finite values in the trailing window -> NaN

  spread_z_slope_t = spread_z_126_t - spread_z_126_{t-21}

This is a normalization-based cousin of raw spread-momentum: instead of the raw
cumulative drift of overnight-vs-intraday dominance, it standardizes the recent
(21d) spread mean against the ticker's own trailing 126d distribution, making the
signal regime-robust and more cross-sectionally comparable (removes each ticker's
idiosyncratic baseline spread scale/variance).

Per-ticker OHLCV-only computation; fully causal (all rolling windows use only
current-and-past bars; the 21d-ago lookup for the slope is a backward shift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703k_tugofwar_spread_zscore",
    "description": (
        "Tug-of-war spread s_t = (Open_t/Close_{t-1}-1) - (Close_t/Open_t-1) "
        "(overnight-gap return minus intraday return), standardized: "
        "ff0703k_spread_z_126 = (mean(s,21d) - mean(s,126d)) / std(s,126d), guarded "
        "for std<1e-9 or <126 finite trailing obs. ff0703k_spread_z_slope = 21-day "
        "change in that z-score. Normalization-based cousin of spread-momentum: "
        "removes the ticker's baseline spread scale/variance for a regime-robust, "
        "more cross-sectionally comparable overnight-vs-intraday tug-of-war signal."
    ),
    "requires": ["Open", "Close"],
    "produces": ["ff0703k_spread_z_126", "ff0703k_spread_z_slope"],
    "tags": ["tugofwar", "overnight", "intraday", "zscore", "spread", "mean-reversion"],
    "version": "1.0.0",
    "author": "ff0703k batch spec — faithful direct implementation (OHLCV-only, no proxy needed)",
}

_SHORT = 21
_LONG = 126


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    df["ff0703k_spread_z_126"] = np.nan
    df["ff0703k_spread_z_slope"] = np.nan

    if n == 0:
        return df

    open_arr = df["Open"].to_numpy(dtype=float)
    close_arr = df["Close"].to_numpy(dtype=float)

    if n < _LONG + 1:
        return df

    prev_close = np.empty(n)
    prev_close[:] = np.nan
    prev_close[1:] = close_arr[:-1]

    # overnight gap return: Open_t / Close_{t-1} - 1
    o_t = np.where(
        (prev_close != 0) & np.isfinite(prev_close),
        open_arr / np.where(prev_close != 0, prev_close, np.nan) - 1.0,
        np.nan,
    )

    # intraday return: Close_t / Open_t - 1
    i_t = np.where(
        (open_arr != 0) & np.isfinite(open_arr),
        close_arr / np.where(open_arr != 0, open_arr, np.nan) - 1.0,
        np.nan,
    )

    s_t = o_t - i_t
    s = pd.Series(s_t, index=df.index)

    # trailing means/std, causal (min_periods = full window required)
    mean_short = s.rolling(window=_SHORT, min_periods=_SHORT).mean()
    mean_long = s.rolling(window=_LONG, min_periods=_LONG).mean()
    std_long = s.rolling(window=_LONG, min_periods=_LONG).std()

    # count of finite values within trailing 126d window (guard for NaNs inside window)
    finite_count = (
        s.notna().rolling(window=_LONG, min_periods=1).sum()
    )

    valid = (
        std_long.notna()
        & (std_long.abs() >= 1e-9)
        & (finite_count >= _LONG)
        & mean_short.notna()
        & mean_long.notna()
    )

    spread_z = (mean_short - mean_long) / std_long.replace(0, np.nan)
    spread_z = spread_z.where(valid, np.nan)
    spread_z = spread_z.replace([np.inf, -np.inf], np.nan)

    spread_z_slope = spread_z - spread_z.shift(_SHORT)

    df["ff0703k_spread_z_126"] = spread_z.to_numpy()
    df["ff0703k_spread_z_slope"] = spread_z_slope.to_numpy()

    return df
