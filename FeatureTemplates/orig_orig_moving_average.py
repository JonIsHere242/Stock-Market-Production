import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_moving_average",
    "description": "AlphaSensitivity moving-average % indicators ported verbatim "
                   "(14ma%, 14ma%_change, 14ma%_count, ATR%). Intermediates "
                   "(14ma, std_14, ATR) computed internally; only the % derivatives "
                   "are emitted to avoid colliding with vix_features' canonical ATR "
                   "(which uses min_periods=14 -- this ATR uses min_periods=1).",
    "requires":    ["Close", "High", "Low"],
    "produces":    ["14ma%", "14ma%_change", "14ma%_count", "ATR%"],
    "tags":        ["moving_average", "atr", "alphasens"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    epsilon = 1e-10

    # 14-day moving average (min_periods=1)
    ma14 = close.rolling(window=14, min_periods=1).mean()

    # Percentage difference from 14-day MA
    df["14ma%"] = ((close - ma14) / (ma14 + epsilon)) * 100

    # Percentage change of 14ma%
    df["14ma%_change"] = df["14ma%"].pct_change(fill_method=None)

    # Count of positive 14ma% days in a 14-day window
    df["14ma%_count"] = df["14ma%"].gt(0).rolling(window=14, min_periods=1).sum()

    # True Range using shift(1) so only past data is used
    close_shift_1 = close.shift(1)
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close_shift_1),
            np.abs(low - close_shift_1),
        ),
    )

    # ATR (min_periods=1) -- internal, not emitted as 'ATR' (owned by vix_features)
    atr = true_range.rolling(window=14, min_periods=1).mean()

    # ATR as percentage of price
    df["ATR%"] = (atr / (close + epsilon)) * 100

    return df
