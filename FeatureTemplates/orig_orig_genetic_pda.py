import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_genetic_pda",
    "description": "Genetic Price Differential Analyzer features (info decay, autocorr, log z-score, weighted signal strength) ported verbatim from AlphaSensitivity.",
    "requires":    ["High", "Low"],
    "produces":    [
        "G_PDA_Info_Decay_3d",
        "G_PDA_Autocorr_3d",
        "G_PDA_Log_Zscore_20d",
        "G_PDA_Weighted_Signal_Strength",
    ],
    "tags":        ["price_structure", "genetic"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-8

    # Shared base genetic price differential (computed once)
    base = (0.1673 / (df["High"] + epsilon) - df["Low"]) / (df["High"] + epsilon)

    # --- G_PDA_Autocorr_3d / G_PDA_Info_Decay_3d ---
    lagged_genetic = base.shift(3)
    rolling_autocorr = lagged_genetic.rolling(window=20, min_periods=10).corr(base)
    df["G_PDA_Autocorr_3d"] = rolling_autocorr
    df["G_PDA_Info_Decay_3d"] = 1 - rolling_autocorr

    # --- G_PDA_Log_Zscore_20d ---
    log_genetic = np.log(np.abs(base) + epsilon) * np.sign(base)
    rolling_mean = log_genetic.rolling(window=20, min_periods=10).mean()
    rolling_std = log_genetic.rolling(window=20, min_periods=10).std()
    df["G_PDA_Log_Zscore_20d"] = (log_genetic - rolling_mean) / (rolling_std + epsilon)

    # --- G_PDA_Weighted_Signal_Strength ---
    ss_mean = base.rolling(20, min_periods=10).mean()
    ss_std = base.rolling(20, min_periods=10).std()

    z_score = (base - ss_mean) / (ss_std + epsilon)
    extreme_high = (z_score > 2).astype(int) * 3
    extreme_low = (z_score < -2).astype(int) * 3

    genetic_roc_3d = base.pct_change(3)
    roc_rolling_80th = genetic_roc_3d.rolling(10, min_periods=5).quantile(0.8)
    roc_rolling_20th = genetic_roc_3d.rolling(10, min_periods=5).quantile(0.2)

    momentum_accelerating = (genetic_roc_3d > roc_rolling_80th).astype(int) * 2
    momentum_decelerating = (genetic_roc_3d < roc_rolling_20th).astype(int) * 2

    genetic_sma_short = base.rolling(5, min_periods=3).mean()
    genetic_sma_long = base.rolling(20, min_periods=10).mean()

    bullish_cross = ((genetic_sma_short > genetic_sma_long) &
                     (genetic_sma_short.shift(1) <= genetic_sma_long.shift(1))).astype(int) * 2
    bearish_cross = ((genetic_sma_short < genetic_sma_long) &
                     (genetic_sma_short.shift(1) >= genetic_sma_long.shift(1))).astype(int) * 2

    rolling_vol = base.rolling(20, min_periods=10).std()
    vol_75th = rolling_vol.rolling(60, min_periods=30).quantile(0.75)
    vol_25th = rolling_vol.rolling(60, min_periods=30).quantile(0.25)

    high_vol_regime = (rolling_vol > vol_75th).astype(int) * 1
    low_vol_regime = (rolling_vol < vol_25th).astype(int) * 1

    df["G_PDA_Weighted_Signal_Strength"] = (
        extreme_high + extreme_low +
        momentum_accelerating + momentum_decelerating +
        bullish_cross + bearish_cross +
        high_vol_regime + low_vol_regime
    )

    return df
