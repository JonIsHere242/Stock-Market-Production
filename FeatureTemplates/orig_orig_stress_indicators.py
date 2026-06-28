import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_stress_indicators",
    "description": "Open-to-low drawdown stress indicators (v6 EMA, v7 multi-level frequency, volume-enhanced v7) ported verbatim from AlphaSensitivity.add_top_stress_indicators",
    "requires":    ["Open", "Low", "Volume"],
    "produces":    ["stress_v6", "stress_v7", "stress_vol_v7"],
    "tags":        ["stress", "drawdown", "ohlcv"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('Date')

    v6_window = 10
    v7_window = 15

    open_to_low_dd = (df['Low'] - df['Open']) / df['Open'] * 100

    # stress_v6: Simple EMA on -3% events
    alpha_v6 = 2 / (v6_window + 1)
    df['stress_v6'] = (open_to_low_dd <= -3).ewm(alpha=alpha_v6, min_periods=1).mean()

    # Common masks for v7 indicators
    moderate_mask = open_to_low_dd <= -2
    severe_mask = open_to_low_dd <= -4
    extreme_mask = open_to_low_dd <= -6

    # stress_v7: Multi-level frequency
    moderate_freq = moderate_mask.rolling(v7_window, min_periods=1).mean()
    severe_freq = severe_mask.rolling(v7_window, min_periods=1).mean()
    extreme_freq = extreme_mask.rolling(v7_window, min_periods=1).mean()
    df['stress_v7'] = (moderate_freq * 0.3 + severe_freq * 0.5 + extreme_freq * 0.2).clip(0, 1)

    # stress_vol_v7: Volume-enhanced
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20, min_periods=1).mean()
        vol_ratio = df['Volume'] / vol_ma
        moderate_stress = ((moderate_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        severe_stress = ((severe_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        extreme_stress = ((extreme_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        alpha_v7 = 2 / (v7_window + 1)
        ema_moderate = ((moderate_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        ema_severe = ((severe_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        ema_extreme = ((extreme_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        df['stress_vol_v7'] = (
            (moderate_stress + ema_moderate) * 0.2 +
            (severe_stress + ema_severe) * 0.4 +
            (extreme_stress + ema_extreme) * 0.4
        ).clip(0, 2)

    return df
