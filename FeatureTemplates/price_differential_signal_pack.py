import numpy as np
import pandas as pd

METADATA = {
    "name":        "price_differential_signal_pack",
    "description": "Signal engineering suite derived from price_differential_ratio",
    "requires":    ["price_differential_ratio"],
    "produces": [
        # rolling log scaled
        "price_differential_ratio_logscale_zscore_10d",
        "price_differential_ratio_logscale_minmax_10d",
        "price_differential_ratio_logscale_rank_10d",

        "price_differential_ratio_logscale_zscore_20d",
        "price_differential_ratio_logscale_minmax_20d",
        "price_differential_ratio_logscale_rank_20d",

        "price_differential_ratio_logscale_zscore_50d",
        "price_differential_ratio_logscale_minmax_50d",
        "price_differential_ratio_logscale_rank_50d",

        # roc
        "price_differential_ratio_pctchange_1d",
        "price_differential_ratio_pctchange_3d",
        "price_differential_ratio_pctchange_5d",
        "price_differential_ratio_pctchange_10d",

        "price_differential_ratio_diff_1d",
        "price_differential_ratio_diff_3d",
        "price_differential_ratio_diff_5d",
        "price_differential_ratio_diff_10d",

        "price_differential_ratio_logdiff_1d",
        "price_differential_ratio_logdiff_3d",
        "price_differential_ratio_logdiff_5d",
        "price_differential_ratio_logdiff_10d",

        "price_differential_ratio_acceleration_3d",
        "price_differential_ratio_acceleration_5d",

        # z-score
        "price_differential_ratio_zscore",
        "price_differential_ratio_extremehigh",
        "price_differential_ratio_extremelow",

        # momentum
        "price_differential_ratio_momentumaccelerating",
        "price_differential_ratio_momentumdecelerating",

        # reversals
        "price_differential_ratio_bullishcross",
        "price_differential_ratio_bearishcross",

        # volatility regimes
        "price_differential_ratio_highvolatilityregime",
        "price_differential_ratio_lowvolatilityregime",

        # persistence
        "price_differential_ratio_signalpersistence",

        # composite
        "price_differential_ratio_signalstrength",
        "price_differential_ratio_weightedsignalstrength",

        # information efficiency
        "price_differential_ratio_informationratio",

        "price_differential_ratio_autocorr_1d",
        "price_differential_ratio_autocorr_3d",
        "price_differential_ratio_autocorr_5d",

        "price_differential_ratio_infodecay_1d",
        "price_differential_ratio_infodecay_3d",
        "price_differential_ratio_infodecay_5d",
    ],
    "tags": [
        "experimental",
        "momentum",
        "mean_reversion",
        "volatility",
        "market_regime",
    ],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    epsilon = 1e-6
    base = "price_differential_ratio"
    s = df[base]
    new_cols: dict = {}

    # ------------------------------------------------------------------
    # Rolling Log Scaled Features
    # ------------------------------------------------------------------

    for window in [10, 20, 50]:
        clean = s.replace([np.inf, -np.inf], np.nan)
        abs_series = np.abs(clean)
        median_abs = abs_series.median()
        adaptive_epsilon = max(epsilon, median_abs * 1e-6) if median_abs > 0 else epsilon

        with np.errstate(invalid="ignore", divide="ignore"):
            log_abs = np.log(abs_series + adaptive_epsilon)
            log_series = (
                pd.Series(log_abs * np.sign(clean), index=s.index)
                .replace([np.inf, -np.inf], np.nan)
            )

        mp = max(1, window // 2)
        rolling_mean = log_series.rolling(window, min_periods=mp).mean()
        rolling_std = (
            log_series.rolling(window, min_periods=mp).std()
            .fillna(1.0).replace(0, 1.0)
        )
        new_cols[f"{base}_logscale_zscore_{window}d"] = (log_series - rolling_mean) / rolling_std

        rolling_min = log_series.rolling(window, min_periods=mp).min()
        rolling_max = log_series.rolling(window, min_periods=mp).max()
        rolling_range = (rolling_max - rolling_min).replace(0, 1.0)
        new_cols[f"{base}_logscale_minmax_{window}d"] = (log_series - rolling_min) / rolling_range

        new_cols[f"{base}_logscale_rank_{window}d"] = (
            log_series.rolling(window, min_periods=mp).rank(pct=True)
        )

    # ------------------------------------------------------------------
    # ROC / Diff / LogDiff
    # ------------------------------------------------------------------

    abs_series = np.abs(s)
    median_abs = abs_series.median()
    adaptive_epsilon = max(1e-8, median_abs * 1e-8) if median_abs > 0 else 1e-8

    with np.errstate(invalid="ignore", divide="ignore"):
        log_series = (
            pd.Series(np.log(abs_series + adaptive_epsilon) * np.sign(s), index=s.index)
            .replace([np.inf, -np.inf], np.nan)
        )

    pctchange: dict = {}
    for period in [1, 3, 5, 10]:
        pctchange[period] = s.pct_change(period)
        new_cols[f"{base}_pctchange_{period}d"] = pctchange[period]
        new_cols[f"{base}_diff_{period}d"]      = s.diff(period)
        new_cols[f"{base}_logdiff_{period}d"]   = log_series.diff(period)

    new_cols[f"{base}_acceleration_3d"] = pctchange[3].pct_change()
    new_cols[f"{base}_acceleration_5d"] = pctchange[5].pct_change()

    # ------------------------------------------------------------------
    # Z-score Signals
    # ------------------------------------------------------------------

    rolling_mean = s.rolling(20).mean()
    rolling_std  = s.rolling(20).std().fillna(epsilon).replace(0, epsilon)
    z = (s - rolling_mean) / rolling_std

    new_cols[f"{base}_zscore"]      = z
    extremehigh = (z > 2).astype(int)
    extremelow  = (z < -2).astype(int)
    new_cols[f"{base}_extremehigh"] = extremehigh
    new_cols[f"{base}_extremelow"]  = extremelow

    # ------------------------------------------------------------------
    # Momentum Signals
    # ------------------------------------------------------------------

    roc = pctchange[3]
    momentumaccelerating = (roc > roc.rolling(10).quantile(0.80)).astype(int)
    momentumdecelerating = (roc < roc.rolling(10).quantile(0.20)).astype(int)
    new_cols[f"{base}_momentumaccelerating"] = momentumaccelerating
    new_cols[f"{base}_momentumdecelerating"] = momentumdecelerating

    # ------------------------------------------------------------------
    # Trend Reversal Signals
    # ------------------------------------------------------------------

    sma_short = s.rolling(5).mean()
    sma_long  = s.rolling(20).mean()
    bullishcross = ((sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))).astype(int)
    bearishcross = ((sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))).astype(int)
    new_cols[f"{base}_bullishcross"] = bullishcross
    new_cols[f"{base}_bearishcross"] = bearishcross

    # ------------------------------------------------------------------
    # Volatility Regime
    # ------------------------------------------------------------------

    rolling_vol  = s.rolling(20).std()
    vol_hi       = rolling_vol.rolling(60).quantile(0.75)
    vol_lo       = rolling_vol.rolling(60).quantile(0.25)
    highvol = (rolling_vol > vol_hi).astype(int)
    lowvol  = (rolling_vol < vol_lo).astype(int)
    new_cols[f"{base}_highvolatilityregime"] = highvol
    new_cols[f"{base}_lowvolatilityregime"]  = lowvol

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    rolling_mean = s.rolling(20).mean()
    direction = np.sign(s - rolling_mean)
    changes   = (direction != direction.shift(1)).astype(int)
    groups    = changes.cumsum()
    new_cols[f"{base}_signalpersistence"] = groups.groupby(groups).cumcount() + 1

    # ------------------------------------------------------------------
    # Composite Signal Strength
    # ------------------------------------------------------------------

    signalstrength = (
        extremehigh + extremelow
        + momentumaccelerating + momentumdecelerating
        + bullishcross + bearishcross
    )
    new_cols[f"{base}_signalstrength"] = signalstrength
    new_cols[f"{base}_weightedsignalstrength"] = (
        extremehigh * 3 + extremelow * 3
        + momentumaccelerating * 2 + momentumdecelerating * 2
        + bullishcross * 2 + bearishcross * 2
    )

    # ------------------------------------------------------------------
    # Information Efficiency
    # ------------------------------------------------------------------

    returns      = s.pct_change()
    rolling_mean = returns.rolling(20).mean()
    rolling_std  = returns.rolling(20).std().fillna(epsilon).replace(0, epsilon)
    new_cols[f"{base}_informationratio"] = rolling_mean / rolling_std

    for lag in [1, 3, 5]:
        autocorr = s.shift(lag).rolling(20, min_periods=10).corr(s)
        new_cols[f"{base}_autocorr_{lag}d"] = autocorr
        autocorr_clamped = autocorr.clip(0, 1).fillna(0)
        new_cols[f"{base}_infodecay_{lag}d"] = 1 - autocorr_clamped

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)