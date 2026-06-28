import pandas as pd
import numpy as np

METADATA = {
    "name":        "trading_signal_composite",
    "description": "Composite trading signal scores aggregating liquidity, volatility, and momentum conditions",
    "requires": [
        "dollar_volume_percentile",
        "volume_trend_5d_20d",
        "liquidity_stress_3d",
        "vix_percentile_rank",
        "atr_percentile_rank",
        "cv_20d_percentile",
        "momentum_strength_continuous",
        "volume_momentum_ratio",
        "trend_consistency_10d",
    ],
    "produces": [
        "liquidity_score",
        "volatility_score",
        "momentum_score",
        "trading_signal_composite",
    ],
    "tags":        ["market_regime", "experimental"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite scores that summarize trading signal strength.
    These aggregate liquidity, volatility, and momentum conditions into
    high-level features for the ML model.
    """

    new_cols = {}

    # Liquidity score (0-1, higher is better liquidity)
    liquidity_components = [
        df.get("dollar_volume_percentile", 50) / 100,
        np.clip(df.get("volume_trend_5d_20d", 1.0), 0, 2) / 2,
        1 - np.clip(df.get("liquidity_stress_3d", 1.0), 0, 2) / 2,
    ]
    new_cols["liquidity_score"] = np.mean(liquidity_components, axis=0)

    # Volatility score (0-1, higher indicates better trading conditions)
    vol_components = [
        df.get("vix_percentile_rank", 50) / 100,
        np.clip(df.get("atr_percentile_rank", 50), 20, 80) / 100,
        1 - np.clip(df.get("cv_20d_percentile", 50), 0, 90) / 100,
    ]
    new_cols["volatility_score"] = np.mean(vol_components, axis=0)

    # Momentum score (0-1, higher indicates strong momentum)
    momentum_components = [
        np.clip(df.get("momentum_strength_continuous", 0), -1, 1) / 2 + 0.5,
        np.clip(df.get("volume_momentum_ratio", 1.0), 0.5, 1.5) / 2,
        np.clip(df.get("trend_consistency_10d", 0.5), 0, 1),
    ]
    new_cols["momentum_score"] = np.mean(momentum_components, axis=0)

    # Overall trading signal score
    new_cols["trading_signal_composite"] = (
        new_cols["liquidity_score"] * 0.3
        + new_cols["volatility_score"] * 0.4
        + new_cols["momentum_score"] * 0.3
    )

    result_df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )

    return result_df
