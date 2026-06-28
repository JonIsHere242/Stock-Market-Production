import pandas as pd
import numpy as np

METADATA = {
    "name": "volatility_atr_features",
    "description": "ATR (Average True Range), ATR percentile rank, gap analysis, and regime classification",
    "requires": ["High", "Low", "Close", "Open"],
    "produces": [
        "atr_14",
        "atr_percentage",
        "atr_percentile_rank",
        "gap_percentage",
        "gap_absolute",
        "gap_in_atr_terms",
        "atr_regime_low",
        "atr_regime_high",
    ],
    "tags": ["volatility", "trend"],
    "version": "1.0",
    "author": "migration from monolith",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ATR (Average True Range) and volatility-related features.

    Includes ATR, percentage ATR, percentile rank, gap analysis,
    and ATR regime classification (low/high).
    """

    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    # True Range calculation (using shifted close)
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # ATR calculation
    new_cols = {}
    new_cols["atr_14"] = true_range.rolling(14, min_periods=10).mean()
    new_cols["atr_percentage"] = new_cols["atr_14"] / df["Close"]

    # ATR percentile rank
    atr_pct_shifted = new_cols["atr_percentage"].shift(1)
    atr_window_252 = atr_pct_shifted.rolling(252, min_periods=50)
    new_cols["atr_percentile_rank"] = atr_window_252.apply(
        lambda x: (x <= new_cols["atr_percentage"].iloc[x.index[-1]]).mean() * 100
        if len(x) > 0
        else 50,
        raw=False,
    )

    # Gap analysis
    current_open = df["Open"]
    prev_close = df["Close"].shift(1)
    gap_pct = (current_open - prev_close) / (prev_close + 1e-8)
    new_cols["gap_percentage"] = gap_pct
    new_cols["gap_absolute"] = np.abs(gap_pct)

    # Gap in ATR terms
    prev_atr_pct = new_cols["atr_percentage"].shift(1)
    new_cols["gap_in_atr_terms"] = gap_pct / (prev_atr_pct + 1e-8)

    # ATR regime classification
    new_cols["atr_regime_low"] = (new_cols["atr_percentile_rank"] <= 25).astype(float)
    new_cols["atr_regime_high"] = (new_cols["atr_percentile_rank"] >= 75).astype(float)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
