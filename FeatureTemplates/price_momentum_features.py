import pandas as pd
import numpy as np

METADATA = {
    "name":        "price_momentum_features",
    "description": "Multi-timeframe returns, momentum strength, price acceleration, and trend consistency",
    "requires":    ["Close"],
    "produces": [
        "return_3d",
        "return_3d_abs",
        "return_5d",
        "return_5d_abs",
        "return_10d",
        "return_10d_abs",
        "return_20d",
        "return_20d_abs",
        "momentum_strength_5d",
        "momentum_strength_continuous",
        "rapid_price_change_zscore",
        "price_acceleration",
        "trend_consistency_10d",
        "trend_consistency_20d",
    ],
    "tags":        ["momentum"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract price momentum features including multi-timeframe returns,
    momentum strength indicators, rapid price changes, price acceleration,
    and trend consistency measures.
    """

    close = df["Close"]

    new_cols = {}

    # Multi-timeframe returns
    for period in [3, 5, 10, 20]:
        new_cols[f"return_{period}d"] = close.pct_change(period)
        new_cols[f"return_{period}d_abs"] = np.abs(new_cols[f"return_{period}d"])

    # Momentum strength indicators
    new_cols["momentum_strength_5d"] = np.where(
        new_cols["return_5d"] > 0.005, 1.0, 0.0
    )
    new_cols["momentum_strength_continuous"] = np.tanh(new_cols["return_5d"] * 100)

    # Rapid price movement detection
    # NOTE: rapid_price_change_5d is no longer emitted as an output column
    # (the faithful version is produced by orig_orig_canbuy_momentum). It is
    # retained here only as a local intermediate for rapid_price_change_zscore.
    _rapid_price_change_5d = close / close.shift(5) - 1
    new_cols["rapid_price_change_zscore"] = (
        _rapid_price_change_5d
        - _rapid_price_change_5d.shift(1).rolling(50, min_periods=20).mean()
    ) / (
        _rapid_price_change_5d.shift(1).rolling(50, min_periods=20).std()
        + 1e-8
    )

    # Price acceleration
    new_cols["price_acceleration"] = new_cols["return_5d"] - new_cols["return_5d"].shift(
        5
    )

    # Trend consistency (what % of last N days were positive)
    # NOTE: trend_consistency_5d is no longer emitted here (the faithful version
    # is produced by orig_orig_canbuy_momentum); only the 10d/20d windows remain.
    for window in [10, 20]:
        daily_returns = close.pct_change()
        new_cols[f"trend_consistency_{window}d"] = daily_returns.rolling(
            window, min_periods=window // 2
        ).apply(lambda x: (x > 0).mean(), raw=True)

    result_df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )

    return result_df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
# rapid_price_change_5d / trend_consistency_5d are no longer produced by this block
# (ownership ceded to orig_orig_canbuy_momentum), so the cull list is now empty.
_CULL_2026_06_13 = []
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
