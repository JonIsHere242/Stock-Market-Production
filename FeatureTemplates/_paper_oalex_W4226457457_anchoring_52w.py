"""
_paper_oalex_W4226457457_anchoring_52w.py  --  52-week high/low anchoring features.

Per-ticker proxy for the George & Hwang (2004) 52-week-high anchoring effect,
motivated by openalex W4226457457 ("Forecasting the cross-sectional stock returns:
Evidence from the United Kingdom"), which uses firm-specific anchoring variables
in a Fama-MacBeth framework.

All features use a trailing 252-bar window ending at (and including) the current
row, so there is no look-ahead. Rolling argmax/argmin are computed via a rolling
apply that returns the within-window distance from the current row to the
extreme — purely a function of past and present observations.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_oalex_W4226457457_anchoring_52w",
    "description": (
        "52-week-high/low anchoring features: proximity to 1-year price extremes, "
        "range position, bars-since-high/low, and near-high flag (George & Hwang proxy)."
    ),
    "requires":    ["High", "Low", "Close"],
    "produces":    [
        "anc_pct_off_252d_high",
        "anc_pct_off_252d_low",
        "anc_range_position_252d",
        "anc_days_since_252d_high",
        "anc_days_since_252d_low",
        "anc_near_high_flag",
    ],
    "tags":        ["momentum", "mean_reversion", "technical", "anchoring"],
    "version":     "1.0",
    "author":      "paper openalex W4226457457 — George & Hwang 52-week-high proxy",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 52-week (252-bar) anchoring features from OHLCV columns.

    Window = 252 bars, trailing, includes the current row.
    Leading NaN rows (fewer than 252 bars of history) are expected and fine.

    Columns added
    -------------
    anc_pct_off_252d_high   : Close / rolling_max(High, 252) - 1   (<=0)
    anc_pct_off_252d_low    : Close / rolling_min(Low,  252) - 1   (>=0)
    anc_range_position_252d : (Close - min252) / (max252 - min252)  in [0,1];
                              NaN when max252 == min252
    anc_days_since_252d_high: bars elapsed since the 252-bar High was last reached
    anc_days_since_252d_low : bars elapsed since the 252-bar Low  was last reached
    anc_near_high_flag      : 1.0 when Close is within 2% of the 252-bar High,
                              else 0.0 (NaN when the rolling max is unavailable)
    """
    WINDOW = 252

    high  = df["High"].to_numpy(dtype=np.float64)
    low   = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)
    n     = len(df)

    # ------------------------------------------------------------------
    # Rolling max(High) and min(Low) over trailing WINDOW bars
    # ------------------------------------------------------------------
    roll_max = df["High"].rolling(WINDOW, min_periods=WINDOW).max().to_numpy()
    roll_min = df["Low"].rolling(WINDOW, min_periods=WINDOW).min().to_numpy()

    # ------------------------------------------------------------------
    # 1. anc_pct_off_252d_high  =  Close / roll_max - 1  (<=0)
    #    Guard: roll_max <= 0 or NaN -> NaN
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_off_high = np.where(
            np.isfinite(roll_max) & (roll_max > 0),
            close / roll_max - 1.0,
            np.nan,
        )

    # ------------------------------------------------------------------
    # 2. anc_pct_off_252d_low   =  Close / roll_min - 1  (>=0)
    #    Guard: roll_min <= 0 or NaN -> NaN
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_off_low = np.where(
            np.isfinite(roll_min) & (roll_min > 0),
            close / roll_min - 1.0,
            np.nan,
        )

    # ------------------------------------------------------------------
    # 3. anc_range_position_252d  =  (Close - min) / (max - min)
    #    Guard: max == min -> NaN
    # ------------------------------------------------------------------
    rng = roll_max - roll_min
    with np.errstate(divide="ignore", invalid="ignore"):
        range_pos = np.where(
            np.isfinite(roll_max) & np.isfinite(roll_min) & (rng > 0),
            (close - roll_min) / rng,
            np.nan,
        )

    # ------------------------------------------------------------------
    # 4 & 5. anc_days_since_252d_high / low
    #
    # For each row t, we want the number of bars since the High in
    # rows [t-251 .. t] was last at its maximum.
    # argmax of the window returns the 0-based index of the maximum
    # element within the window; the current bar is at position
    # (window_len - 1), so "bars since" = (window_len - 1) - argmax.
    #
    # Rolling apply receives the raw numpy slice; raw=True is used for
    # speed. min_periods=WINDOW so values before the warmup are NaN.
    # ------------------------------------------------------------------
    def _days_since_max(x: np.ndarray) -> float:
        # x is a 1-D float array of length WINDOW (ascending in time)
        idx = int(np.nanargmax(x))
        return float((len(x) - 1) - idx)

    def _days_since_min(x: np.ndarray) -> float:
        idx = int(np.nanargmin(x))
        return float((len(x) - 1) - idx)

    days_since_high = (
        df["High"]
        .rolling(WINDOW, min_periods=WINDOW)
        .apply(_days_since_max, raw=True)
        .to_numpy()
    )

    days_since_low = (
        df["Low"]
        .rolling(WINDOW, min_periods=WINDOW)
        .apply(_days_since_min, raw=True)
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # 6. anc_near_high_flag  =  1.0 if Close >= roll_max * 0.98 else 0.0
    #    NaN where roll_max is NaN (insufficient history).
    # ------------------------------------------------------------------
    near_high_flag = np.where(
        np.isfinite(roll_max),
        np.where(close >= roll_max * 0.98, 1.0, 0.0),
        np.nan,
    )

    # ------------------------------------------------------------------
    # Assign new columns onto df (never modify existing columns)
    # ------------------------------------------------------------------
    idx = df.index
    df["anc_pct_off_252d_high"]   = pd.array(pct_off_high,   dtype="Float64")
    df["anc_pct_off_252d_low"]    = pd.array(pct_off_low,    dtype="Float64")
    df["anc_range_position_252d"] = pd.array(range_pos,      dtype="Float64")
    df["anc_days_since_252d_high"]= pd.array(days_since_high,dtype="Float64")
    df["anc_days_since_252d_low"] = pd.array(days_since_low, dtype="Float64")
    df["anc_near_high_flag"]      = pd.array(near_high_flag, dtype="Float64")

    return df
