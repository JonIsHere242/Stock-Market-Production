import pandas as pd
import numpy as np

METADATA = {
    "name":        "price_structure_metrics",
    "description": "Price structure and distance-from-high metrics",
    "requires":    ["Close", "High", "Low"],
    "produces": [
        "percent_from_high",
        "new_high",
        "days_since_high",
        "percent_range",
        "high_close_ratio_norm",
    ],
    "tags": [
        "price_structure",
        "momentum",
        "trend",
    ],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    expanding_max = close.expanding().max()

    df["percent_from_high"] = (
        (close - expanding_max)
        / expanding_max
        * 100.0
    )

    df["new_high"] = (
        close == expanding_max
    ).astype(int)

    high_mask = ~df["new_high"].astype(bool)

    cumsum_mask = high_mask.cumsum()

    df["days_since_high"] = (
        cumsum_mask
        - cumsum_mask.where(df["new_high"].astype(bool))
        .ffill()
        .fillna(0)
    )

    df["percent_range"] = (
        (high - low)
        / close
        * 100.0
    )

    # NOTE: high_close_ratio is no longer emitted as an output column (the
    # faithful version is produced by orig_orig_canbuy_volume_priceaction). It
    # is retained here only as a local intermediate for high_close_ratio_norm.
    _high_close_ratio = (
        (high - close)
        / (close + 1e-10)
    )

    shifted_ratio = _high_close_ratio.shift(1)

    norm = (
        (
            _high_close_ratio
            - shifted_ratio.rolling(50).mean()
        )
        /
        shifted_ratio.rolling(50).std()
    )

    df["high_close_ratio_norm"] = norm.clip(-3, 3)

    return df

# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
# high_close_ratio is no longer produced by this block (ownership ceded to
# orig_orig_canbuy_volume_priceaction); only high_close_ratio_norm remains culled.
_CULL_2026_06_13 = ['high_close_ratio_norm']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
