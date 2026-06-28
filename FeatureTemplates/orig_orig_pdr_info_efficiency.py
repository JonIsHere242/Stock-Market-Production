import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_pdr_info_efficiency",
    "description": "Original AlphaSensitivity information-efficiency features on "
                   "Price_Differential_Ratio (InformationRatio + Autocorr_{1,3,5}d + "
                   "InfoDecay_{1,3,5}d).",
    "requires":    ["Price_Differential_Ratio"],
    "produces": [
        "Price_Differential_Ratio_InformationRatio",
        "Price_Differential_Ratio_Autocorr_1d",
        "Price_Differential_Ratio_Autocorr_3d",
        "Price_Differential_Ratio_Autocorr_5d",
        "Price_Differential_Ratio_InfoDecay_1d",
        "Price_Differential_Ratio_InfoDecay_3d",
        "Price_Differential_Ratio_InfoDecay_5d",
    ],
    "tags":    ["price_structure", "information_efficiency"],
    "version": "1.0",
    "author":  "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Faithful port of add_information_efficiency_features(df,
    # base_column='Price_Differential_Ratio', window=20, epsilon=1e-6) from the
    # backup (L2039-2058). Exact math, windows, min_periods, clip, fillna, order.
    base_column = "Price_Differential_Ratio"
    window = 20
    epsilon = 1e-6

    # Compute the base column if absent (self-contained). The live pipeline built
    # Price_Differential_Ratio via add_price_differential_ratio with epsilon=1e-10
    # (backup L1867-1870, called L3594 w/ epsilon=1e-10) -- match that for parity.
    if base_column not in df.columns:
        base_eps = 1e-10
        df[base_column] = (
            (0.1673 / (df["High"] + base_eps) - df["Low"])
            / (df["High"] + base_eps)
        )

    feature_returns = df[base_column].pct_change()
    rolling_mean = feature_returns.rolling(window).mean()
    rolling_std = feature_returns.rolling(window).std()
    rolling_std = rolling_std.fillna(epsilon).replace(0, epsilon)
    df[f"{base_column}_InformationRatio"] = rolling_mean / rolling_std

    for lag in [1, 3, 5]:
        lagged_feature = df[base_column].shift(lag)
        current_feature = df[base_column]
        # NOTE: the backup source (L2053) passes min_periods=max(1, window//2)=10,
        # but the ground-truth Data/ProcessedData parquets were generated with the
        # rolling DEFAULT min_periods (== window). Using min_periods=window reproduces
        # the ground truth bit-for-bit (the only divergence was on rows 10-19).
        rolling_autocorr = lagged_feature.rolling(window).corr(current_feature)
        df[f"{base_column}_Autocorr_{lag}d"] = rolling_autocorr
        autocorr_clamped = rolling_autocorr.clip(0, 1).fillna(0)
        df[f"{base_column}_InfoDecay_{lag}d"] = 1 - autocorr_clamped

    return df
