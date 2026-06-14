import pandas as pd
import numpy as np

METADATA = {
    "name":        "information_efficiency",
    "description": "Information ratio, autocorrelation, and information decay metrics for price_differential_ratio",
    "requires":    ["price_differential_ratio"],
    "produces": [
        "price_differential_ratio_informationratio",

        "price_differential_ratio_autocorr_1d",
        "price_differential_ratio_autocorr_3d",
        "price_differential_ratio_autocorr_5d",

        "price_differential_ratio_infodecay_1d",
        "price_differential_ratio_infodecay_3d",
        "price_differential_ratio_infodecay_5d",
    ],
    "tags": [
        "information_theory",
        "market_regime",
        "autocorrelation",
        "experimental",
    ],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    base = "price_differential_ratio"

    epsilon = 1e-6
    window = 20

    feature = df[base]

    # ---------------------------------------------------------
    # Information Ratio
    # ---------------------------------------------------------

    feature_returns = feature.pct_change()

    rolling_mean = (
        feature_returns
        .rolling(window)
        .mean()
    )

    rolling_std = (
        feature_returns
        .rolling(window)
        .std()
        .fillna(epsilon)
        .replace(0, epsilon)
    )

    df[f"{base}_informationratio"] = (
        rolling_mean / rolling_std
    )

    # ---------------------------------------------------------
    # Rolling Autocorrelation
    # ---------------------------------------------------------

    min_periods = max(1, window // 2)

    for lag in [1, 3, 5]:

        lagged_feature = feature.shift(lag)

        rolling_autocorr = (
            lagged_feature
            .rolling(
                window=window,
                min_periods=min_periods
            )
            .corr(feature)
        )

        df[f"{base}_autocorr_{lag}d"] = rolling_autocorr

        autocorr_clamped = (
            rolling_autocorr
            .clip(0, 1)
            .fillna(0)
        )

        df[f"{base}_infodecay_{lag}d"] = (
            1 - autocorr_clamped
        )

    return df