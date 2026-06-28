import pandas as pd
import numpy as np

METADATA = {
    "name": "vix_regime_features",
    "description": "VIX-based regime classification, percentile rank, momentum, and acceleration",
    "requires": ["vix_close"],
    "produces": [
        "vix_percentile_rank",
        "vix_regime_scale",
        "vix_regime_acceleration",
        "vix_zscore",
        "vix_regime_low",
        "vix_regime_high",
    ],
    "tags": ["market_regime", "volatility"],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIX regime features from VIX close price.

    Includes percentile rank, regime scaling, momentum, acceleration,
    z-score, and regime classification (low/high).
    """

    current_vix = df["vix_close"]

    # Rolling VIX statistics (using only past data)
    vix_ma_20 = current_vix.shift(1).rolling(20, min_periods=10).mean()
    vix_std_20 = current_vix.shift(1).rolling(20, min_periods=10).std()

    # VIX percentile rank (relative measure)
    vix_window_252 = current_vix.shift(1).rolling(252, min_periods=50)
    new_cols = {}
    new_cols["vix_percentile_rank"] = vix_window_252.apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100 if len(x) > 0 else 50, raw=False
    )

    # VIX regime scaling factor (continuous)
    new_cols["vix_regime_scale"] = np.where(
        new_cols["vix_percentile_rank"] >= 80, 1.20,
        np.where(
            new_cols["vix_percentile_rank"] >= 60, 1.10,
            np.where(new_cols["vix_percentile_rank"] <= 20, 0.95, 1.0)
        )
    )

    # VIX momentum and acceleration
    # NOTE: vix_momentum_5d is no longer emitted as an output column (the faithful
    # version is produced by orig_orig_canbuy_momentum). It is retained here only
    # as a local intermediate for vix_regime_acceleration.
    _vix_momentum_5d = current_vix.pct_change(5)
    new_cols["vix_regime_acceleration"] = _vix_momentum_5d.diff()

    # VIX z-score
    new_cols["vix_zscore"] = (current_vix - vix_ma_20) / (vix_std_20 + 1e-8)

    # VIX regime transitions
    new_cols["vix_regime_low"] = (new_cols["vix_percentile_rank"] <= 25).astype(float)
    new_cols["vix_regime_high"] = (new_cols["vix_percentile_rank"] >= 75).astype(float)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
# vix_momentum_5d is no longer produced by this block (ownership ceded to
# orig_orig_canbuy_momentum), so the cull list is now empty.
_CULL_2026_06_13 = []
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
