"""Faithful AlphaSensitivity port: can_buy market-regime trend_direction features.

Source: calculate_market_regime_features (L3199-3202) in
_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py

Only the two trend_direction columns present in the 204-feature set are emitted
(20d, 50d); the backup also makes 10d, distance_from_sma_*, and
market_regime_composite, which are NOT in the 204 and are intentionally skipped.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "orig_orig_canbuy_regime",
    "description": "can_buy market-regime trend direction (Close vs SMA) for 20d/50d windows.",
    "requires": ["Close"],
    "produces": [
        "trend_direction_20d",
        "trend_direction_50d",
    ],
    "tags": ["regime", "trend", "canbuy", "alphasens"],
    "version": "1.0",
    "author": "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    for window in [20, 50]:
        sma = close.rolling(window, min_periods=window // 2).mean()
        df[f"trend_direction_{window}d"] = np.where(close > sma, 1.0, -1.0)
    return df
