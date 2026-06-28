import pandas as pd
import numpy as np

METADATA = {
    "name":        "genetic_pda_pack",
    "description": "Genetic price differential analysis (PDA) family: info decay, autocorr, log z-score, and weighted signal strength",
    "requires":    ["High", "Low"],
    "produces":    ["g_pda_info_decay_3d", "g_pda_autocorr_3d", "g_pda_log_zscore_20d", "g_pda_weighted_signal_strength"],
    "tags":        ["experimental", "mean_reversion"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute four genetic price differential (PDA) features from the same base primitive.

    Base primitive: g = (0.1673 / (High + eps) - Low) / (High + eps)

    Four derived outputs:
    1. g_pda_info_decay_3d: 1 - rolling correlation of g vs g.shift(3)
    2. g_pda_autocorr_3d: rolling correlation of g vs g.shift(3)
    3. g_pda_log_zscore_20d: 20-day rolling z-score of log-scaled g
    4. g_pda_weighted_signal_strength: weighted combination of extreme/momentum/trend/vol signals
    """

    epsilon = 1e-8

    # Step 1: Compute base genetic price differential (local variable, not a column)
    g = (0.1673 / (df["High"] + epsilon) - df["Low"]) / (df["High"] + epsilon)

    # ========================================================================
    # Feature 1: Information Decay (1 - autocorrelation with 3-day lag)
    # ========================================================================
    g_lagged_3d = g.shift(3)
    rolling_autocorr_3d = g_lagged_3d.rolling(window=20, min_periods=10).corr(g)
    df["g_pda_info_decay_3d"] = 1 - rolling_autocorr_3d

    # ========================================================================
    # Feature 2: Autocorrelation (correlation with 3-day lag)
    # ========================================================================
    df["g_pda_autocorr_3d"] = rolling_autocorr_3d

    # ========================================================================
    # Feature 3: Log Z-Score (20-day rolling normalization of log-scaled g)
    # ========================================================================
    log_g = np.log(np.abs(g) + epsilon) * np.sign(g)
    rolling_mean_20d = log_g.rolling(window=20, min_periods=10).mean()
    rolling_std_20d = log_g.rolling(window=20, min_periods=10).std()
    df["g_pda_log_zscore_20d"] = (log_g - rolling_mean_20d) / (rolling_std_20d + epsilon)

    # ========================================================================
    # Feature 4: Weighted Signal Strength
    # ========================================================================

    # Rolling statistics for signal detection
    rolling_mean_20d = g.rolling(20, min_periods=10).mean()
    rolling_std_20d = g.rolling(20, min_periods=10).std()

    # Z-score based extreme signals (weight: 3 each)
    z_score = (g - rolling_mean_20d) / (rolling_std_20d + epsilon)
    extreme_high = (z_score > 2).astype(int) * 3
    extreme_low = (z_score < -2).astype(int) * 3

    # Momentum signals (weight: 2 each)
    g_roc_3d = g.pct_change(3)
    roc_rolling_80th = g_roc_3d.rolling(10, min_periods=5).quantile(0.8)
    roc_rolling_20th = g_roc_3d.rolling(10, min_periods=5).quantile(0.2)

    momentum_accelerating = (g_roc_3d > roc_rolling_80th).astype(int) * 2
    momentum_decelerating = (g_roc_3d < roc_rolling_20th).astype(int) * 2

    # Trend cross signals (weight: 2 each)
    g_sma_short = g.rolling(5, min_periods=3).mean()
    g_sma_long = g.rolling(20, min_periods=10).mean()

    bullish_cross = ((g_sma_short > g_sma_long) &
                     (g_sma_short.shift(1) <= g_sma_long.shift(1))).astype(int) * 2
    bearish_cross = ((g_sma_short < g_sma_long) &
                     (g_sma_short.shift(1) >= g_sma_long.shift(1))).astype(int) * 2

    # Volatility regime signals (weight: 1 each)
    rolling_vol = g.rolling(20, min_periods=10).std()
    vol_75th = rolling_vol.rolling(60, min_periods=30).quantile(0.75)
    vol_25th = rolling_vol.rolling(60, min_periods=30).quantile(0.25)

    high_vol_regime = (rolling_vol > vol_75th).astype(int) * 1
    low_vol_regime = (rolling_vol < vol_25th).astype(int) * 1

    # Combine into weighted signal strength
    df["g_pda_weighted_signal_strength"] = (
        extreme_high + extreme_low +
        momentum_accelerating + momentum_decelerating +
        bullish_cross + bearish_cross +
        high_vol_regime + low_vol_regime
    )

    return df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['g_pda_info_decay_3d', 'g_pda_log_zscore_20d']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
