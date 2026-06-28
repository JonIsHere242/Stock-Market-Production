import pandas as pd
import numpy as np

METADATA = {
    "name":        "moving_average_metrics",
    "description": "14-period moving average, ATR, and related deviation metrics",
    "requires":    ["Close", "High", "Low"],
    "produces":    ["ma_14", "ma_14_pct", "std_14", "ma_14_pct_change", "ma_14_pct_count", "atr", "atr_pct"],
    "tags":        ["trend", "volatility", "technical"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 14-period moving average indicators, True Range, and ATR.

    - ma_14: 14-period moving average of Close
    - ma_14_pct: percentage deviation of Close from ma_14
    - std_14: 14-period standard deviation of Close
    - ma_14_pct_change: percentage change of ma_14_pct
    - ma_14_pct_count: count of positive ma_14_pct values in 14-day window
    - atr: 14-period Average True Range
    - atr_pct: ATR as percentage of Close price
    """

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Create all columns in a dictionary first
    new_columns = {}

    # Add 14-day moving average with min_periods=1
    new_columns["ma_14"] = close.rolling(window=14, min_periods=1).mean()

    # Calculate percentage difference from 14-day MA
    # Using epsilon to avoid division by zero
    epsilon = 1e-10
    new_columns["ma_14_pct"] = ((close - new_columns["ma_14"]) / (new_columns["ma_14"] + epsilon)) * 100

    # Standard deviation with proper min_periods
    new_columns["std_14"] = close.rolling(window=14, min_periods=1).std()

    # Percentage change of ma_14_pct with proper handling of NAs
    new_columns["ma_14_pct_change"] = new_columns["ma_14_pct"].pct_change(fill_method=None)

    # Count of positive days in 14-day window
    new_columns["ma_14_pct_count"] = new_columns["ma_14_pct"].gt(0).rolling(window=14, min_periods=1).sum()

    # True Range calculation using shift(1) to ensure we only use past data
    close_shift_1 = close.shift(1)
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close_shift_1),
            np.abs(low - close_shift_1)
        )
    )

    # Calculate ATR using rolling mean with min_periods=1
    new_columns["atr"] = true_range.rolling(window=14, min_periods=1).mean()

    # ATR as percentage of price
    new_columns["atr_pct"] = (new_columns["atr"] / (close + epsilon)) * 100

    # Add all columns at once to minimize DataFrame operations
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['atr_pct']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
