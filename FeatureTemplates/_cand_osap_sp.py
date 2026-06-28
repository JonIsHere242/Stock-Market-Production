"""
osap_sp  --  Sales-to-Price ratio (per-ticker proxy)

SOURCE: OpenSourceAP / Chen-Zimmermann open-source factor library.
ORIGINAL: Barbee, Mukherji & Raines (1996) "Do Sales–Price and Debt–Equity Ratios
          Explain Stock Returns Better than Book-Market and Firm Size?"
          The Journal of Financial Analysts, 52(2), 56-60.
          Also widely used as a value signal in the academic factor zoo.

ECONOMIC SIGNAL (cross-sectional, sign = -1: SHORT high SP, i.e. cheap stocks win):
  SP = annual revenue / market-cap.  High SP = cheap on a revenue basis (value).
  Cross-sectionally, firms with higher SP tend to outperform (value premium).
  Per-ticker proxy: we track the time-series of the stock's own SP level and its
  12-month change, which capture the same cheap-vs-expensive dynamic within the
  stock's own history.

PER-TICKER NOTE: True SP is a cross-sectional rank (cheap vs peers). Here we
  implement the per-stock level (revenue_ttm / market_cap) and its 1-year change
  (delta_sp). The level captures absolute cheapness; the change captures whether
  the stock is getting cheaper or more expensive relative to its own history.
  Fundamentals are PIT via _fundamentals.as_of().
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by file path -- no package import)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_sp",
    "description": (
        "Sales-to-Price ratio: revenue_ttm divided by market_cap (Close x shares_outstanding). "
        "Captures value cheapness on a revenue basis. Per-ticker proxy for the cross-sectional "
        "SP factor from OpenSourceAP/Chen-Zimmermann. Produces the level (osap_sp_level), "
        "its 252-trading-day change (osap_sp_delta_1y), and a z-score over a 3-year trailing "
        "window (osap_sp_zscore_3y). Sign per original paper: -1 (high SP = cheap = long). "
        "Coverage ~84% (ETFs/foreign = NaN). True cross-sectional ranking is NOT performed; "
        "within-stock dynamics are used as proxy."
    ),
    "requires": ["Close"],
    "produces": ["osap_sp_level", "osap_sp_delta_1y", "osap_sp_zscore_3y"],
    "tags": ["value", "fundamentals", "sales_to_price", "osap"],
    "version": "1.0",
    "author": (
        "Barbee, Mukherji & Raines (1996); OpenSourceAP / Chen-Zimmermann factor library. "
        "Per-ticker implementation."
    ),
}

# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker Sales-to-Price features."""

    # --- Pull PIT fundamentals: revenue_ttm + shares_outstanding ----------
    df = _fundamentals.as_of(df, fields=["revenue_ttm", "shares_outstanding"])

    rev_ttm = df["fund_revenue_ttm"]          # trailing-12-month revenue ($)
    shares  = df["fund_shares_outstanding"]   # shares outstanding

    # Market cap = Close * shares_outstanding (guard: shares must be > 0)
    mktcap = df["Close"] * shares.where(shares > 0, other=np.nan)

    # SP level = revenue / market cap (guard against zero / NaN mktcap)
    sp_level = rev_ttm / mktcap.where(mktcap > 0, other=np.nan)

    # Replace inf with NaN (negative market cap edge-case is already NaN)
    sp_level = sp_level.replace([np.inf, -np.inf], np.nan)

    # --- 1-year (252 trading-day) change in SP ----------------------------
    sp_delta_1y = sp_level - sp_level.shift(252)
    sp_delta_1y = sp_delta_1y.replace([np.inf, -np.inf], np.nan)

    # --- 3-year rolling z-score (756 trading days, min 63 obs = ~3 months) -
    roll_3y_mean = sp_level.rolling(window=756, min_periods=63).mean()
    roll_3y_std  = sp_level.rolling(window=756, min_periods=63).std(ddof=1)
    # Guard against zero / NaN std
    roll_3y_std_safe = roll_3y_std.where(roll_3y_std > 0, other=np.nan)
    sp_zscore_3y = (sp_level - roll_3y_mean) / roll_3y_std_safe
    sp_zscore_3y = sp_zscore_3y.replace([np.inf, -np.inf], np.nan)

    # --- Assign produced columns ------------------------------------------
    df["osap_sp_level"]     = sp_level
    df["osap_sp_delta_1y"]  = sp_delta_1y
    df["osap_sp_zscore_3y"] = sp_zscore_3y

    # --- Drop scratch fund_ columns not in produces ----------------------
    for col in ["fund_revenue_ttm", "fund_shares_outstanding"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
