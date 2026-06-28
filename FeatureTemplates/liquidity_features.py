import pandas as pd
import numpy as np

METADATA = {
    "name": "liquidity_features",
    "description": "Dollar volume statistics, liquidity stress, and volume trend metrics",
    "requires": ["Close", "Volume"],
    "produces": [
        "dollar_volume_ma_10",
        "dollar_volume_ma_252",
        "dollar_volume_std_252",
        "dollar_volume_ratio_10d",
        "dollar_volume_ratio_252d",
        "dollar_volume_zscore",
        "dollar_volume_percentile",
        "volume_trend_5d_20d",
        "liquidity_stress_3d",
        "volume_spike_ratio",
        "sustained_volume_burst_count",
    ],
    "tags": ["volume", "liquidity"],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity and dollar-volume-related features.

    Includes rolling averages, percentile ranks, trend analysis,
    stress indicators, and spike detection.
    """

    current_close = df["Close"]
    current_volume = df["Volume"]

    # Dollar volume calculations
    current_dollar_volume = current_volume * current_close

    # Historical dollar volume statistics (using shifted data)
    dollar_vol_shifted = current_dollar_volume.shift(1)
    new_cols = {}
    new_cols["dollar_volume_ma_10"] = dollar_vol_shifted.rolling(
        10, min_periods=5
    ).mean()
    new_cols["dollar_volume_ma_252"] = dollar_vol_shifted.rolling(
        252, min_periods=50
    ).mean()
    new_cols["dollar_volume_std_252"] = dollar_vol_shifted.rolling(
        252, min_periods=50
    ).std()

    # Current dollar volume relative measures
    new_cols["dollar_volume_ratio_10d"] = current_dollar_volume / (
        new_cols["dollar_volume_ma_10"] + 1e-8
    )
    new_cols["dollar_volume_ratio_252d"] = current_dollar_volume / (
        new_cols["dollar_volume_ma_252"] + 1e-8
    )
    new_cols["dollar_volume_zscore"] = (
        current_dollar_volume - new_cols["dollar_volume_ma_252"]
    ) / (new_cols["dollar_volume_std_252"] + 1e-8)

    # Volume percentile rank
    vol_window_252 = dollar_vol_shifted.rolling(252, min_periods=50)
    new_cols["dollar_volume_percentile"] = vol_window_252.apply(
        lambda x: (x <= current_dollar_volume.iloc[x.index[-1]]).mean() * 100
        if len(x) > 0
        else 50,
        raw=False,
    )

    # Volume trend analysis (5d vs 20d)
    new_cols["volume_trend_5d_20d"] = (
        dollar_vol_shifted.rolling(5, min_periods=3).mean()
        / (dollar_vol_shifted.rolling(20, min_periods=10).mean() + 1e-8)
    )

    # Short-term liquidity stress (3d vs median)
    new_cols["liquidity_stress_3d"] = (
        dollar_vol_shifted.rolling(3, min_periods=2).mean()
        / (dollar_vol_shifted.rolling(252, min_periods=50).median() + 1e-8)
    )

    # Volume spike detection (relative to recent average)
    vol_10d_avg = current_volume.shift(1).rolling(10, min_periods=5).mean()
    new_cols["volume_spike_ratio"] = current_volume / (vol_10d_avg + 1e-8)

    # Sustained volume burst (count of high volume days in last 5)
    volume_burst_threshold = vol_10d_avg * 3
    recent_volumes = current_volume.rolling(5, min_periods=3)
    new_cols["sustained_volume_burst_count"] = recent_volumes.apply(
        lambda x: (x > volume_burst_threshold.iloc[x.index[-1]]).sum()
        if len(x) > 0
        else 0,
        raw=False,
    )

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
