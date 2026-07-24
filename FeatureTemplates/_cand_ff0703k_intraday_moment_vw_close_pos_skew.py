from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name": "ff0703k_intraday_moment_vw_close_pos_skew",
    "description": (
        "Volume-weighted 20-day close-location value (CLV): cp_t = (Close-(High+Low)/2)/(High-Low) "
        "(NaN when High==Low), aggregated as sum(Volume*cp)/sum(Volume) over a trailing 20-day window. "
        "This re-weights the equal-weight close-position-in-range proxy for intraday realized skew by "
        "participation, so heavy-volume asymmetric closes dominate the level rather than every bar "
        "counting equally -- makes it orthogonal to plain-average CLV features. Also emits the 10-day "
        "change in this level (dynamic/momentum-of-the-skew variant). Per-ticker OHLCV proxy; "
        "true realized intraday skew would need tick data, so daily-bar close-location is used as the "
        "closest faithful daily-bar proxy for intraday asymmetry."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ff0703k_intraday_moment_vw_close_pos_skew_20",
        "ff0703k_intraday_moment_vw_close_pos_skew_chg10",
    ],
    "tags": ["volume", "intraday", "close-location", "skew", "participation-weighted"],
    "version": "1.0",
    "author": "ff0703k spec (VEIN intraday_moment); faithful daily-bar CLV proxy for intraday skew",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    level_col = "ff0703k_intraday_moment_vw_close_pos_skew_20"
    chg_col = "ff0703k_intraday_moment_vw_close_pos_skew_chg10"

    level = np.full(n, np.nan, dtype=np.float64)
    chg = np.full(n, np.nan, dtype=np.float64)

    WINDOW = 20
    LAG = 10

    if n == 0:
        df[level_col] = level
        df[chg_col] = chg
        return df

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    rng = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.where(rng > 0.0, (close - (high + low) / 2.0) / rng, np.nan)

    if n >= WINDOW:
        cp_wins = sliding_window_view(cp, WINDOW)       # (M, WINDOW)
        v_wins = sliding_window_view(volume, WINDOW)    # (M, WINDOW)

        valid_mask = np.isfinite(cp_wins) & np.isfinite(v_wins) & (v_wins > 0.0)
        cp_masked = np.where(valid_mask, cp_wins, 0.0)
        v_masked = np.where(valid_mask, v_wins, 0.0)

        sum_v = v_masked.sum(axis=1)
        sum_cv = (cp_masked * v_masked).sum(axis=1)
        count_valid = valid_mask.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.where((sum_v > 0.0) & (count_valid >= 2), sum_cv / sum_v, np.nan)

        level[WINDOW - 1:] = vw

    if n > LAG:
        curr = level[LAG:]
        prev = level[: n - LAG]
        diff = curr - prev
        diff[~(np.isfinite(curr) & np.isfinite(prev))] = np.nan
        chg[LAG:] = diff

    level = np.where(np.isinf(level), np.nan, level)
    chg = np.where(np.isinf(chg), np.nan, chg)

    df[level_col] = level
    df[chg_col] = chg

    return df
