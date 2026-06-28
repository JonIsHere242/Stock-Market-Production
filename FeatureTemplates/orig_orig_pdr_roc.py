import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_pdr_roc",
    "description": "Original AlphaSensitivity rate-of-change features on Price_Differential_Ratio "
                   "(PctChange/Diff/LogDiff over [1,3,5,10]d + Acceleration_3d/5d).",
    "requires":    ["Price_Differential_Ratio"],
    "produces": [
        "Price_Differential_Ratio_PctChange_1d",
        "Price_Differential_Ratio_PctChange_3d",
        "Price_Differential_Ratio_PctChange_5d",
        "Price_Differential_Ratio_PctChange_10d",
        "Price_Differential_Ratio_Diff_1d",
        "Price_Differential_Ratio_Diff_3d",
        "Price_Differential_Ratio_Diff_5d",
        "Price_Differential_Ratio_Diff_10d",
        "Price_Differential_Ratio_LogDiff_1d",
        "Price_Differential_Ratio_LogDiff_3d",
        "Price_Differential_Ratio_LogDiff_5d",
        "Price_Differential_Ratio_LogDiff_10d",
        "Price_Differential_Ratio_Acceleration_3d",
        "Price_Differential_Ratio_Acceleration_5d",
    ],
    "tags":    ["price_structure", "momentum"],
    "version": "1.0",
    "author":  "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Faithful port of add_rate_of_change_features(df, base_column='Price_Differential_Ratio',
    # periods=[1, 3, 5, 10]) from the backup (L1910-1933). Exact math, windows, epsilon, order.
    base_column = "Price_Differential_Ratio"
    periods = [1, 3, 5, 10]

    series = df[base_column]
    abs_series = np.abs(series)
    median_abs = abs_series.median()
    adaptive_epsilon = max(1e-8, median_abs * 1e-8) if median_abs > 0 else 1e-8

    with np.errstate(invalid="ignore", divide="ignore"):
        log_series = np.log(abs_series + adaptive_epsilon) * np.sign(series)
        log_series = pd.Series(log_series, index=series.index).replace([np.inf, -np.inf], np.nan)

    for period in periods:
        # pct_change uses the pandas DEFAULT fill_method (matches the backup exactly).
        df[f"{base_column}_PctChange_{period}d"] = series.pct_change(period)
        df[f"{base_column}_Diff_{period}d"] = series.diff(period)
        df[f"{base_column}_LogDiff_{period}d"] = log_series.diff(period)

    df[f"{base_column}_Acceleration_3d"] = df[f"{base_column}_PctChange_3d"].pct_change(1)
    df[f"{base_column}_Acceleration_5d"] = df[f"{base_column}_PctChange_5d"].pct_change(1)

    return df
