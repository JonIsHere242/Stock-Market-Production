import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_pdr_signals",
    "description": "Original AlphaSensitivity signal-engineering suite on Price_Differential_Ratio "
                   "(z-score/extremes, momentum accel/decel, trend-reversal crosses, volatility "
                   "regimes, signal persistence, composite + weighted signal strength).",
    "requires":    ["Price_Differential_Ratio"],
    "produces": [
        "Price_Differential_Ratio_ZScore",
        "Price_Differential_Ratio_ExtremeHigh",
        "Price_Differential_Ratio_ExtremeLow",
        "Price_Differential_Ratio_MomentumAccelerating",
        "Price_Differential_Ratio_MomentumDecelerating",
        "Price_Differential_Ratio_BullishCross",
        "Price_Differential_Ratio_BearishCross",
        "Price_Differential_Ratio_HighVolatilityRegime",
        "Price_Differential_Ratio_LowVolatilityRegime",
        "Price_Differential_Ratio_SignalPersistence",
        "Price_Differential_Ratio_SignalStrength",
        "Price_Differential_Ratio_WeightedSignalStrength",
    ],
    "tags":    ["price_structure", "momentum", "mean_reversion", "volatility", "market_regime"],
    "version": "1.0",
    "author":  "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Faithful port of the backup signal functions
    # (3__AlphaSensitivity_pre_tsz_20260530.py), all called with
    # base_column='Price_Differential_Ratio':
    #   add_z_score_signals            L1935-1950
    #   add_momentum_signals           L1952-1963
    #   add_trend_reversal_signals     L1965-1979
    #   add_volatility_regime_signals  L1981-1994
    #   add_signal_persistence_features L1996-2008
    #   add_composite_signal_strength  L2010-2037
    # Exact math, windows, min_periods, shifts, epsilon, order of ops.
    base_column = "Price_Differential_Ratio"
    feature = df[base_column]

    # ------------------------------------------------------------------
    # add_z_score_signals (window=20, epsilon=1e-6)
    # ------------------------------------------------------------------
    epsilon = 1e-6
    window = 20
    rolling_mean = feature.rolling(window).mean()
    rolling_std = feature.rolling(window).std()
    rolling_std = rolling_std.fillna(epsilon).replace(0, epsilon)

    z_score = (feature - rolling_mean) / rolling_std
    df[f"{base_column}_ZScore"] = z_score
    extremehigh = (z_score > 2).astype(int)
    extremelow = (z_score < -2).astype(int)
    df[f"{base_column}_ExtremeHigh"] = extremehigh
    df[f"{base_column}_ExtremeLow"] = extremelow

    # ------------------------------------------------------------------
    # add_momentum_signals (window=10)
    # roc_col = Price_Differential_Ratio_PctChange_3d; if absent, the backup
    # calls add_rate_of_change_features(df, base_column, [3]) -- replicate that
    # path EXACTLY so the result is identical whether or not orig_pdr_roc ran first.
    # ------------------------------------------------------------------
    mom_window = 10
    roc_col = f"{base_column}_PctChange_3d"
    if roc_col not in df.columns:
        # add_rate_of_change_features(df, base_column, periods=[3]) -- L1910-1933
        period = 3
        series = df[base_column]
        df[f"{base_column}_PctChange_{period}d"] = series.pct_change(period)
        df[f"{base_column}_Diff_{period}d"] = series.diff(period)
        abs_series = np.abs(series)
        median_abs = abs_series.median()
        adaptive_epsilon = max(1e-8, median_abs * 1e-8) if median_abs > 0 else 1e-8
        with np.errstate(invalid="ignore", divide="ignore"):
            log_series = np.log(abs_series + adaptive_epsilon) * np.sign(series)
            log_series = pd.Series(log_series, index=series.index).replace([np.inf, -np.inf], np.nan)
            df[f"{base_column}_LogDiff_{period}d"] = log_series.diff(period)
        df[f"{base_column}_Acceleration_3d"] = df[f"{base_column}_PctChange_3d"].pct_change(1)

    roc_3d = df[roc_col]
    momentumaccelerating = (roc_3d > roc_3d.rolling(mom_window).quantile(0.8)).astype(int)
    momentumdecelerating = (roc_3d < roc_3d.rolling(mom_window).quantile(0.2)).astype(int)
    df[f"{base_column}_MomentumAccelerating"] = momentumaccelerating
    df[f"{base_column}_MomentumDecelerating"] = momentumdecelerating

    # ------------------------------------------------------------------
    # add_trend_reversal_signals (short_window=5, long_window=20)
    # ------------------------------------------------------------------
    short_window = 5
    long_window = 20
    feature_sma_short = feature.rolling(short_window).mean()
    feature_sma_long = feature.rolling(long_window).mean()

    bullishcross = (
        (feature_sma_short > feature_sma_long)
        & (feature_sma_short.shift(1) <= feature_sma_long.shift(1))
    ).astype(int)
    bearishcross = (
        (feature_sma_short < feature_sma_long)
        & (feature_sma_short.shift(1) >= feature_sma_long.shift(1))
    ).astype(int)
    df[f"{base_column}_BullishCross"] = bullishcross
    df[f"{base_column}_BearishCross"] = bearishcross

    # ------------------------------------------------------------------
    # add_volatility_regime_signals (vol_window=20, regime_window=60)
    # ------------------------------------------------------------------
    vol_window = 20
    regime_window = 60
    rolling_vol = feature.rolling(vol_window).std()
    vol_threshold_high = rolling_vol.rolling(regime_window).quantile(0.75)
    vol_threshold_low = rolling_vol.rolling(regime_window).quantile(0.25)

    df[f"{base_column}_HighVolatilityRegime"] = (rolling_vol > vol_threshold_high).astype(int)
    df[f"{base_column}_LowVolatilityRegime"] = (rolling_vol < vol_threshold_low).astype(int)

    # ------------------------------------------------------------------
    # add_signal_persistence_features (window=20)
    # ------------------------------------------------------------------
    persist_window = 20
    rolling_mean_p = feature.rolling(persist_window).mean()
    feature_direction = np.sign(feature - rolling_mean_p)
    direction_changes = (feature_direction != feature_direction.shift(1)).astype(int)
    persistence = direction_changes.cumsum()
    df[f"{base_column}_SignalPersistence"] = persistence.groupby(persistence).cumcount() + 1

    # ------------------------------------------------------------------
    # add_composite_signal_strength
    # ------------------------------------------------------------------
    signal_endings = ["ExtremeHigh", "ExtremeLow", "MomentumAccelerating",
                      "MomentumDecelerating", "BullishCross", "BearishCross"]
    signal_cols = []
    for ending in signal_endings:
        col_name = f"{base_column}_{ending}"
        if col_name in df.columns:
            signal_cols.append(col_name)

    if signal_cols:
        df[f"{base_column}_SignalStrength"] = df[signal_cols].sum(axis=1)

        weights = {
            "ExtremeHigh": 3, "ExtremeLow": 3, "MomentumAccelerating": 2,
            "MomentumDecelerating": 2, "BullishCross": 2, "BearishCross": 2,
        }
        weighted_strength = pd.Series(0, index=df.index)
        for col in signal_cols:
            signal_type = col.split("_")[-1]
            weight = weights.get(signal_type, 1)
            weighted_strength += df[col] * weight
        df[f"{base_column}_WeightedSignalStrength"] = weighted_strength

    return df
