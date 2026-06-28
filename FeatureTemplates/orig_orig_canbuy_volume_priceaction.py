import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_canbuy_volume_priceaction",
    "description": "AlphaSensitivity can_buy-block originals: dollar-volume trend strength and high/close ratio (OHLCV only).",
    "requires":    ["High", "Close", "Volume"],
    "produces":    ["volume_trend_strength", "high_close_ratio"],
    "tags":        ["volume", "price_action", "alphasens_port"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- volume_trend_strength (calculate_volume_momentum_features L3151-3169) ---
    volume = df["Volume"]
    close = df["Close"]
    dollar_volume = volume * close

    dv_5d = dollar_volume.shift(1).rolling(5, min_periods=3).mean()
    dv_20d = dollar_volume.shift(1).rolling(20, min_periods=10).mean()
    volume_momentum_ratio = dv_5d / (dv_20d + 1e-8)

    df["volume_trend_strength"] = np.where(
        volume_momentum_ratio > 1.0,
        np.log(volume_momentum_ratio),
        -np.log(1.0 / (volume_momentum_ratio + 1e-8)),
    )

    # --- high_close_ratio (calculate_price_action_features L3212-3227) ---
    high = df["High"]
    df["high_close_ratio"] = (high - close) / (close + 1e-8)

    return df
