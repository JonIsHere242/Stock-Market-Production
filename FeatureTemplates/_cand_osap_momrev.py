"""
osap_momrev — Momentum + Long-Term Reversal interaction (Chan & Ko 2006)
via OpenSourceAP (Chen-Zimmermann).

Original signal: Binary cross-sectional flag = 1 if stock is in the TOP quintile
of 6-month momentum AND BOTTOM quintile of 36-month momentum (i.e. recent winner
/ long-term loser), = 0 for the opposite.  Predicted to be long-favoured (+1).

Per-ticker proxy: We cannot form cross-sectional quintile ranks inside a
single-stock compute(), so we capture the same economic interaction directly:
  - osap_momrev_mom6   : raw 6-month total return (21*6 trading days)
  - osap_momrev_ltrev36: raw 36-month total return (21*36 trading days)
  - osap_momrev_signal : continuous proxy = mom6 – ltrev36 / 6  (scaled).
    High values → recent winner + long-term underperformer → long-favoured.
    Zero or low values → the opposite pattern.
    Rows where Close < 5 are set to NaN (matching the original price filter).

All values are leakage-free (no negative shift, no future data), vectorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momrev",
    "description": (
        "Per-ticker proxy for Chan & Ko (2006) momentum × long-term reversal "
        "cross-sectional signal (OpenSourceAP / Chen-Zimmermann). "
        "Computes 6-month and 36-month cumulative returns per ticker; their "
        "difference captures 'recent winner but long-term loser' (+1 predicted). "
        "Price < 5 rows set to NaN per original filter. "
        "Cross-sectional quintile ranking is inherently multi-stock, so this "
        "block provides the raw continuous exposures instead."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momrev_mom6",
        "osap_momrev_ltrev36",
        "osap_momrev_signal",
    ],
    "tags": ["momentum", "reversal", "price", "medium_term", "long_term"],
    "version": "1.0",
    "author": "Chan and Ko (2006) via OpenSourceAP (Chen-Zimmermann); block by Claude",
}

# Trading-day window constants
_DAYS_PER_MONTH = 21
_MOM6_WINDOW = 6 * _DAYS_PER_MONTH    # 126 trading days
_LTREV_WINDOW = 36 * _DAYS_PER_MONTH  # 756 trading days

_PRICE_FLOOR = 5.0


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # --- 6-month momentum: return over past 126 trading days ---
    # shift(1) skips the current bar (standard momentum convention: skip most
    # recent month is not required here per spec, but we use close-to-close)
    past_close_6m = close.shift(_MOM6_WINDOW)
    mom6 = (close - past_close_6m) / past_close_6m.replace(0, np.nan)

    # --- 36-month long-term reversal: return over past 756 trading days ---
    past_close_36m = close.shift(_LTREV_WINDOW)
    ltrev36 = (close - past_close_36m) / past_close_36m.replace(0, np.nan)

    # --- Composite signal: high mom6 + low ltrev36 is the favoured quadrant ---
    # Scaled so units are comparable (ltrev36 covers 6× the period).
    # signal = mom6 - (ltrev36 / 6)
    # High signal → recent outperformer who has underperformed long-run (mean-rev setup).
    signal = mom6 - (ltrev36 / 6.0)

    # --- Apply price filter: mask rows where Close < $5 ---
    price_mask = close < _PRICE_FLOOR
    mom6 = mom6.where(~price_mask, other=np.nan)
    ltrev36 = ltrev36.where(~price_mask, other=np.nan)
    signal = signal.where(~price_mask, other=np.nan)

    # Guard against any stray inf values introduced by zero denominators
    mom6 = mom6.replace([np.inf, -np.inf], np.nan)
    ltrev36 = ltrev36.replace([np.inf, -np.inf], np.nan)
    signal = signal.replace([np.inf, -np.inf], np.nan)

    df["osap_momrev_mom6"] = mom6
    df["osap_momrev_ltrev36"] = ltrev36
    df["osap_momrev_signal"] = signal

    return df
