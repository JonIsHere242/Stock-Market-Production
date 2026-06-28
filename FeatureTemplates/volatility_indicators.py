import pandas as pd
import numpy as np

METADATA = {
    "name":        "volatility_indicators",
    "description": "Volatility, Keltner Channel, mean-reversion z-scores, and complexity metrics from Close, ATR, and smoothed_close",
    "requires":    ["Close", "atr", "smoothed_close"],
    "produces": [
        "percent_change_close",
        "pct_change_std",
        "percent_change_close_lag_1",
        "percent_change_close_lag_5",
        "percent_change_close_lag_10",
        "pct_change_std_rolling",
        "direction_flipper_count5",
        "direction_flipper_count_10",
        "kc_upper_pct",
        "kc_lower_pct",
        "rolling_mean_28",
        "rolling_std_28",
        "mean_reversion_z_score_28_std_1",
        "rolling_mean_90",
        "rolling_std_90",
        "mean_reversion_z_score_90_std_1",
        "rolling_mean_151",
        "rolling_std_151",
        "mean_reversion_z_score_151_std_3",
        "complexity_invariant_distance",
        "cid_mean",
        "cid_sd",
    ],
    "tags":    ["volatility", "mean_reversion"],
    "version": "1.0",
    "author":  "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility indicators migrated from calculate_volatility_indicators() in
    3__AlphaSensitivity.py (lines 2675-2735).

    Sections
    --------
    1. Percent-change and rolling std/mean (windows 20, 14).
    2. Direction flipper — rolling count of up-days over 5 and 10 bars.
    3. Keltner Channel upper/lower distance as % of Close (EWM span=20, range=ATR*1.5).
    4. Mean-reversion z-scores for windows [28, 90, 151] on smoothed_close.
    5. Complexity Invariant Distance from rolling variance of Close (window 90).
    """

    close = df["Close"]

    # ------------------------------------------------------------------
    # 1. Percent-change series and rolling statistics
    # ------------------------------------------------------------------
    pct_change_close = close.pct_change()
    rolling_20 = close.rolling(window=20)

    df["percent_change_close"]      = pct_change_close
    df["pct_change_std"]            = rolling_20.std()
    df["percent_change_close_lag_1"]  = pct_change_close.shift(1)
    df["percent_change_close_lag_5"]  = pct_change_close.shift(5)
    df["percent_change_close_lag_10"] = pct_change_close.shift(10)
    df["pct_change_std_rolling"]    = rolling_20.mean()

    # ------------------------------------------------------------------
    # 2. Direction flipper — rolling count of positive-return bars
    # ------------------------------------------------------------------
    direction_flipper = (pct_change_close > 0).astype(int)
    df["direction_flipper_count5"]  = direction_flipper.rolling(window=5).sum()
    df["direction_flipper_count_10"] = direction_flipper.rolling(window=10).sum()

    # ------------------------------------------------------------------
    # 3. Keltner Channel distances as % of Close
    #    Central line : EWM(span=20) of Close
    #    Band range   : ATR * 1.5
    # ------------------------------------------------------------------
    keltner_central = close.ewm(span=20).mean()
    keltner_range   = df["atr"] * 1.5

    df["kc_upper_pct"] = ((keltner_central + keltner_range) - close) / close * 100.0
    df["kc_lower_pct"] = (close - (keltner_central - keltner_range)) / close * 100.0

    # ------------------------------------------------------------------
    # 4. Mean-reversion z-scores on smoothed_close
    #    windows       : [28, 90, 151]
    #    std_multipliers: [1,   1,   3]
    # ------------------------------------------------------------------
    price = df["smoothed_close"]

    windows        = [28,  90,  151]
    std_multipliers = [1,   1,   3]

    for window, std_mult in zip(windows, std_multipliers):
        roll = price.rolling(window=window)
        mean_val = roll.mean()
        std_val  = roll.std()

        df[f"rolling_mean_{window}"] = mean_val
        df[f"rolling_std_{window}"]  = std_val
        df[f"mean_reversion_z_score_{window}_std_{std_mult}"] = (
            (price - mean_val) / (std_val * std_mult)
        )

    # ------------------------------------------------------------------
    # 5. Complexity Invariant Distance (CID) from rolling variance of Close
    #    window: 90
    # ------------------------------------------------------------------
    cid_window = 90
    rolling_variance = close.rolling(window=cid_window).var()

    complexity_invariant_distance = rolling_variance.diff().abs()
    df["complexity_invariant_distance"] = complexity_invariant_distance
    df["cid_mean"] = complexity_invariant_distance.rolling(window=cid_window).mean()
    df["cid_sd"]   = complexity_invariant_distance.rolling(window=cid_window).std()

    return df
