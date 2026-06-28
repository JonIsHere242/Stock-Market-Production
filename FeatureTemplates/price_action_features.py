import pandas as pd
import numpy as np

METADATA = {
    "name":        "price_action_features",
    "description": "Intraday price structure patterns including OHLC ratios, doji, and hammer patterns",
    "requires":    ["Open", "High", "Low", "Close"],
    "produces": [
        "price_quality_score",
        "intraday_range_pct",
        "open_close_ratio",
        "pa_high_close_ratio",
        "low_close_ratio",
        "doji_pattern",
        "hammer_pattern",
    ],
    "tags":        ["price_structure"],
    "version":     "1.0",
    "author":      "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract price action features including intraday range characteristics
    and candlestick patterns (doji, hammer).
    """

    close = df["Close"]
    open_price = df["Open"]
    high = df["High"]
    low = df["Low"]

    new_cols = {}

    # Price floor relative measure (distance from minimum viable price)
    new_cols["price_quality_score"] = (
        np.log(close / 1.10)
        if (close > 1.10).all()
        else np.log(close / close.min())
    )

    # Intraday range characteristics
    new_cols["intraday_range_pct"] = (high - low) / close
    new_cols["open_close_ratio"] = (close - open_price) / (open_price + 1e-8)
    high_close_ratio_local = (high - close) / (close + 1e-8)
    new_cols["pa_high_close_ratio"] = high_close_ratio_local
    new_cols["low_close_ratio"] = (close - low) / (close + 1e-8)

    # Price action patterns
    new_cols["doji_pattern"] = np.where(
        np.abs(new_cols["open_close_ratio"]) < 0.001, 1.0, 0.0
    )
    new_cols["hammer_pattern"] = np.where(
        (new_cols["low_close_ratio"] > high_close_ratio_local * 2)
        & (new_cols["open_close_ratio"] > 0),
        1.0,
        0.0,
    )

    result_df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )

    return result_df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['pa_high_close_ratio']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
