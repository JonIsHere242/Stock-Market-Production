import pandas as pd
import numpy as np

METADATA = {
    "name": "volume_momentum_features",
    "description": "Dollar volume momentum, trend strength, persistence, and volatility metrics",
    "requires": ["Close", "Volume"],
    "produces": [
        "volume_momentum_ratio",
        "volume_persistence",
        "volume_volatility",
    ],
    "tags": ["volume", "momentum"],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume momentum and related features.

    Includes dollar volume momentum ratio, trend strength,
    persistence (count of high-volume days), and volume volatility.
    """

    volume = df["Volume"]
    close = df["Close"]
    dollar_volume = volume * close

    # Dollar volume momentum (5d vs 20d comparison)
    dv_5d = dollar_volume.shift(1).rolling(5, min_periods=3).mean()
    dv_20d = dollar_volume.shift(1).rolling(20, min_periods=10).mean()
    new_cols = {}
    new_cols["volume_momentum_ratio"] = dv_5d / (dv_20d + 1e-8)

    # NOTE: volume_trend_strength is no longer emitted by this block (the faithful
    # version is produced by orig_orig_canbuy_volume_priceaction). It is not used
    # internally by any other feature here, so its computation has been removed.

    # Volume persistence (how many days has volume been above/below average)
    vol_ma_20 = volume.shift(1).rolling(20, min_periods=10).mean()
    vol_above_avg = volume > vol_ma_20
    new_cols["volume_persistence"] = vol_above_avg.rolling(10, min_periods=5).sum()

    # Volume volatility
    vol_returns = volume.pct_change()
    new_cols["volume_volatility"] = vol_returns.rolling(20, min_periods=10).std()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
# volume_trend_strength is no longer produced by this block (ownership ceded to
# orig_orig_canbuy_volume_priceaction), so the cull list is now empty.
_CULL_2026_06_13 = []
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
