import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_pdr_logscale",
    "description": "Original AlphaSensitivity rolling log-scaled features of Price_Differential_Ratio (zscore/minmax/rank over 10/20/50d windows).",
    "requires":    [],
    "produces": [
        "Price_Differential_Ratio_LogScale_Zscore_10d",
        "Price_Differential_Ratio_LogScale_Minmax_10d",
        "Price_Differential_Ratio_LogScale_Rank_10d",
        "Price_Differential_Ratio_LogScale_Zscore_20d",
        "Price_Differential_Ratio_LogScale_Minmax_20d",
        "Price_Differential_Ratio_LogScale_Rank_20d",
        "Price_Differential_Ratio_LogScale_Zscore_50d",
        "Price_Differential_Ratio_LogScale_Minmax_50d",
        "Price_Differential_Ratio_LogScale_Rank_50d",
    ],
    "tags":    ["momentum", "alphasens_port"],
    "version": "1.0",
    "author":  "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    base_column = "Price_Differential_Ratio"

    # --- base feature (add_price_differential_ratio, L1867-1870) ---
    if base_column in df.columns:
        base = df[base_column]
    else:
        base = (0.1673 / (df["High"] + epsilon) - df["Low"]) / (df["High"] + epsilon)

    def apply_log_scaling(series, window, method):
        clean_series = series.replace([np.inf, -np.inf], np.nan)
        abs_series = np.abs(clean_series)
        adaptive_epsilon = max(epsilon, abs_series.median() * 1e-6) if abs_series.median() > 0 else epsilon

        with np.errstate(invalid="ignore", divide="ignore"):
            log_abs = np.log(abs_series + adaptive_epsilon)
            log_series = log_abs * np.sign(clean_series)
            log_series = pd.Series(log_series, index=series.index).replace([np.inf, -np.inf], np.nan)

        if method == "zscore":
            rolling_mean = log_series.rolling(window=window, min_periods=max(1, window // 2)).mean()
            rolling_std = log_series.rolling(window=window, min_periods=max(1, window // 2)).std()
            rolling_std = rolling_std.fillna(1.0).replace(0, 1.0)
            result = (log_series - rolling_mean) / rolling_std
        elif method == "minmax":
            rolling_min = log_series.rolling(window=window, min_periods=max(1, window // 2)).min()
            rolling_max = log_series.rolling(window=window, min_periods=max(1, window // 2)).max()
            rolling_range = rolling_max - rolling_min
            rolling_range = rolling_range.replace(0, 1.0)
            result = (log_series - rolling_min) / rolling_range
        elif method == "rank":
            result = log_series.rolling(window=window, min_periods=max(1, window // 2)).rank(pct=True)

        return result

    for window in [10, 20, 50]:
        for method in ["zscore", "minmax", "rank"]:
            feature_name = f"{base_column}_LogScale_{method.title()}_{window}d"
            df[feature_name] = apply_log_scaling(base, window, method)

    return df
