"""
osap_price — Log of absolute price level (Blume & Husic 1973, via OpenSourceAP / Chen-Zimmermann).

Cross-sectional interpretation: low-priced stocks tend to outperform (predicted sign -1).
Per-ticker proxy: log(|Close|) tracks the stock's own price level through time, capturing
the same economic signal (nominal price) on a rolling basis. We also emit a 12-month
z-score of this level to capture where the stock sits relative to its own price history,
and a 1-month change in log-price to detect recent price-level drift.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_price",
    "description": (
        "Log of absolute price level (Close). Implements Blume & Husic (1973) nominal-price "
        "anomaly: low-priced stocks outperform cross-sectionally (predicted sign -1). "
        "Per-ticker proxy: osap_price_log = log(|Close|); "
        "osap_price_zscore = rolling 252-day z-score of log-price (where is the stock vs "
        "its own price history); osap_price_chg1m = 1-month change in log-price level "
        "(captures recent price drift). Inherently cross-sectional rank is not computable "
        "per-ticker, so we expose the raw level and its within-ticker dynamics."
    ),
    "requires": ["Close"],
    "produces": ["osap_price_log", "osap_price_zscore", "osap_price_chg1m"],
    "tags": ["price_level", "anomaly", "osap", "blume_husic", "low_price_effect"],
    "version": "1.0",
    "author": "Blume and Husic 1973; OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].copy()

    # Guard: replace non-positive prices with NaN so log is safe
    safe_close = close.where(close > 0, np.nan)

    # Primary feature: log of absolute price level
    log_price = np.log(safe_close.abs())
    df["osap_price_log"] = log_price

    # Within-ticker z-score over a 252-trading-day (1-year) rolling window.
    # Tells us whether the stock is cheap/expensive relative to its own history.
    roll_mean = log_price.rolling(window=252, min_periods=60).mean()
    roll_std  = log_price.rolling(window=252, min_periods=60).std()
    zscore = (log_price - roll_mean) / roll_std.replace(0, np.nan)
    # Replace inf/-inf with NaN (guard for zero-std edge case)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    df["osap_price_zscore"] = zscore

    # 1-month (21-trading-day) change in log-price level.
    # Positive = stock has risen in nominal price; negative = fallen.
    chg1m = log_price.diff(21)
    df["osap_price_chg1m"] = chg1m

    return df
