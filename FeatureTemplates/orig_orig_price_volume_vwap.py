import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_price_volume_vwap",
    "description": "AlphaSensitivity price/volume/VWAP family ported verbatim "
                   "(original column names). VWAP_std is a stale/typo entry with "
                   "no faithful source -- aliased to VWAP_std14.",
    "requires":    ["High", "Low", "Close", "Volume"],
    "produces": [
        "VWAP%",
        "VWAP%_from_high",
        "VWAP_std14",
        "VWAP_std200",
        "VWAP_std",
        "Volume%",
        "Volume%_rolling_90",
        "Volume_rolling_28",
        "Volume_rolling_90",
        "Volume_std",
        "Volume_lag_1",
        "Weighted_Close_Change_Velocity",
    ],
    "tags": ["vwap", "volume", "price", "alphasens"],
    "version": "1.0",
    "author": "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Expanding max of close (no future leakage), used by VWAP%_from_high.
    expanding_max = close.expanding().max()

    # VWAP calculations using shifted data.
    typical_price_shifted = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    volume_shifted = volume.shift(1)

    vwap = (
        (typical_price_shifted * volume_shifted).rolling(window=14).sum()
        / volume_shifted.rolling(window=14).sum()
    )

    df["VWAP_std14"] = vwap.rolling(window=14).std()
    # WINDOW IS 20 despite the '200' name (verbatim from backup).
    df["VWAP_std200"] = vwap.rolling(window=20).std()
    df["VWAP%"] = ((close - vwap) / vwap) * 100
    df["VWAP%_from_high"] = ((vwap - expanding_max) / expanding_max) * 100

    # VWAP_std (bare) has NO faithful source in any backup; alias to VWAP_std14.
    df["VWAP_std"] = df["VWAP_std14"]

    # Volume metrics -- use shifted volume for rolling calculations.
    df["Volume_rolling_28"] = volume.shift(1).rolling(window=28).mean()
    df["Volume_rolling_90"] = volume.shift(1).rolling(window=90).mean()
    df["Volume%"] = ((volume - df["Volume_rolling_28"]) / df["Volume_rolling_28"]) * 100
    df["Volume%_rolling_90"] = ((volume - df["Volume_rolling_90"]) / df["Volume_rolling_90"]) * 100
    df["Volume_std"] = volume.shift(1).rolling(window=28).std()
    df["Volume_lag_1"] = volume.shift(1)

    # Weighted velocity -- use shifted price changes.
    window = 10
    price_change = close.diff().shift(1).fillna(0)
    weights = np.linspace(1, 0, window)
    weights /= np.sum(weights)
    weighted_velocity = price_change.rolling(window=window).apply(
        lambda x: np.dot(x, weights), raw=True
    )
    df["Weighted_Close_Change_Velocity"] = weighted_velocity

    return df
