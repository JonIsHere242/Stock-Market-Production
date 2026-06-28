import pandas as pd
import numpy as np

METADATA = {
    "name":        "market_regime_features",
    "description": "Price position ranges, trend direction across timeframes, and market regime composite",
    "requires":    ["Close", "High", "Low"],
    "produces": [
        "price_position_20d",
        "price_position_50d",
        "price_position_100d",
        "trend_direction_10d",
        "distance_from_sma_10d",
        "trend_direction_20d",
        "distance_from_sma_20d",
        "trend_direction_50d",
        "distance_from_sma_50d",
        "market_regime_composite",
    ],
    "tags":        ["market_regime", "trend"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract market regime features including price position in recent ranges,
    trend direction across multiple timeframes, distance from simple moving averages,
    and a composite market regime score.
    """

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    new_cols = {}

    # Price position in recent ranges
    for window in [20, 50, 100]:
        high_window = high.rolling(window, min_periods=window // 2).max()
        low_window = low.rolling(window, min_periods=window // 2).min()
        price_range = high_window - low_window
        new_cols[f"price_position_{window}d"] = (close - low_window) / (
            price_range + 1e-8
        )

    # Trend direction using multiple timeframes
    for window in [10, 20, 50]:
        sma = close.rolling(window, min_periods=window // 2).mean()
        new_cols[f"trend_direction_{window}d"] = np.where(close > sma, 1.0, -1.0)
        new_cols[f"distance_from_sma_{window}d"] = (close - sma) / (sma + 1e-8)

    # Market regime composite (trend + volatility)
    vol_regime = df.get("vix_regime_scale", pd.Series(1.0, index=df.index))
    trend_regime = new_cols.get(
        "trend_direction_20d", pd.Series(0.0, index=df.index)
    )
    new_cols["market_regime_composite"] = vol_regime * trend_regime

    result_df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )

    return result_df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['trend_direction_20d', 'trend_direction_50d']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
