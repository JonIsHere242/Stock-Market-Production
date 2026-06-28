"""
VPIN-lite order-flow toxicity proxy from daily OHLCV.

Implements a per-ticker VPIN (Volume-synchronized Probability of Informed
Trading) approximation using the close-location value to classify daily
volume into buyer- vs seller-initiated, then computes a rolling order-
imbalance (toxicity) measure and its rate of change.

No cross-sectional data needed -- pure OHLCV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_vpin_proxy",
    "description": (
        "VPIN-lite order-flow toxicity from daily OHLCV. "
        "Uses the close-location value CLV = ((Close-Low)-(High-Close))/(High-Low) "
        "to split daily volume into estimated buy vs sell volume. "
        "Order imbalance OI = |buy_vol - sell_vol| / total_vol is a per-day "
        "microstructure toxicity proxy (high OI = one-sided flow = higher "
        "probability of informed trading). Rolling 50-day mean OI gives the "
        "toxicity level (ext3_vpin_proxy_tox); 20-day change in that level "
        "gives momentum of toxicity (ext3_vpin_proxy_tox_chg); and the raw "
        "daily OI smoothed over 10 days gives a fast signal (ext3_vpin_proxy_oi10). "
        "Per-ticker proxy -- no cross-sectional rank available here. "
        "Leakage-free: all windows use only past data."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext3_vpin_proxy_tox",      # 50-day rolling mean order-imbalance (toxicity level)
        "ext3_vpin_proxy_tox_chg",  # 20-day change in toxicity (toxicity momentum)
        "ext3_vpin_proxy_oi10",     # 10-day smoothed daily order-imbalance (fast signal)
    ],
    "tags": ["microstructure", "order_flow", "vpin", "toxicity", "volume"],
    "version": "1.0",
    "author": "Round-4 expansion spec (NEW: microstructure); VPIN concept from Easley, Lopez de Prado & O'Hara (2012)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VPIN-lite toxicity features.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # --- Close-location value (CLV) in [-1, +1] ---
    # CLV = ((Close - Low) - (High - Close)) / (High - Low)
    #      = (2*Close - High - Low) / (High - Low)
    # Positive => close near high => buyer-initiated flow
    # Negative => close near low  => seller-initiated flow
    hl_range = high - low
    # Guard: when High == Low the range is 0 (doji / halt); CLV = 0 (neutral)
    clv = np.where(
        hl_range > 0,
        (2.0 * close - high - low) / hl_range,
        0.0,
    )
    clv = pd.Series(clv, index=df.index)

    # --- Estimated buy/sell volume ---
    # buy_frac  = (1 + CLV) / 2  in [0, 1]
    # sell_frac = (1 - CLV) / 2  in [0, 1]
    buy_frac = (1.0 + clv) / 2.0
    sell_frac = 1.0 - buy_frac

    buy_vol = buy_frac * volume
    sell_vol = sell_frac * volume
    total_vol = volume  # same as buy_vol + sell_vol by construction

    # --- Daily order imbalance OI = |buy_vol - sell_vol| / total_vol ---
    # Guard against zero-volume days
    oi = np.where(
        total_vol > 0,
        np.abs(buy_vol - sell_vol) / total_vol,
        np.nan,
    )
    oi = pd.Series(oi, index=df.index)

    # --- Feature 1: 50-day rolling mean OI (toxicity level) ---
    tox50 = oi.rolling(window=50, min_periods=25).mean()

    # --- Feature 2: 20-day change in toxicity level ---
    tox_chg = tox50 - tox50.shift(20)

    # --- Feature 3: 10-day smoothed daily OI (fast signal) ---
    oi10 = oi.rolling(window=10, min_periods=5).mean()

    # --- Attach to dataframe ---
    df = df.copy()
    df["ext3_vpin_proxy_tox"] = tox50
    df["ext3_vpin_proxy_tox_chg"] = tox_chg
    df["ext3_vpin_proxy_oi10"] = oi10

    return df
