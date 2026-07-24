from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703h_intraday_moment_range_pos_third_moment",
    "description": (
        "Uncentered third moment of the intraday close-location value (cp), where "
        "cp = (Close - (High+Low)/2) / (High-Low), guarded to NaN when High==Low. "
        "LEVEL column = 60d rolling mean of cp**3: cubing preserves sign while cubically "
        "up-weighting extreme close-positions near the bar's high or low, so persistent "
        "right- (close-near-high) or left- (close-near-low) positioning accumulates into a "
        "large positive or negative value, distinct from centered skewness or the plain CLV mean. "
        "DYNAMIC column = level minus its value 20 rows earlier, capturing recent acceleration/ "
        "deceleration of that close-position bias. Pure per-ticker OHLC, no cross-sectional data."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff0703h_intraday_moment_range_pos_third_moment_level",
        "ff0703h_intraday_moment_range_pos_third_moment_dyn",
    ],
    "tags": ["intraday", "close_location", "moment", "skew", "range_position"],
    "version": "1.0",
    "author": "ff0703h spec batch (faithful direct implementation)",
}

_WINDOW = 60
_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    level_col = "ff0703h_intraday_moment_range_pos_third_moment_level"
    dyn_col = "ff0703h_intraday_moment_range_pos_third_moment_dyn"

    df[level_col] = np.nan
    df[dyn_col] = np.nan

    if len(df) == 0:
        return df

    hl = df["High"] - df["Low"]
    hl_safe = hl.where(hl != 0, np.nan)

    cp = (df["Close"] - (df["High"] + df["Low"]) / 2.0) / hl_safe
    cp3 = cp ** 3

    level = cp3.rolling(_WINDOW, min_periods=_WINDOW).mean()
    dyn = level - level.shift(_LAG)

    df[level_col] = level
    df[dyn_col] = dyn

    return df
