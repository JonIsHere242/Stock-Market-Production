"""
Net Debt to Price (osap_netdebtprice)

Source: OpenSourceAP (Chen-Zimmermann), based on Penman, Richardson and Tuna (2007).
Predicted sign: -1 (high net-debt/price → negative future returns).

Per-ticker proxy notes:
- True cross-sectional signal requires ranking within B/M quintile 3 (Table 4).
  Here we compute the per-ticker LEVEL and a trailing z-score (proxy for how
  extreme the current ratio is relative to its own history), which capture the
  same economic information without needing a cross-sectional panel.
- Net debt = long_term_debt + liabilities_current + long_term_debt (proxy for
  preferred; not separately available) - cash. We approximate:
    net_debt = total_debt - cash
  using PIT fundamentals fields [total_debt, cash, shares_outstanding].
- Market cap = Close * shares_outstanding.
- Exclusions: ETFs/foreign (fund_ columns will be NaN → feature NaN, expected).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_netdebtprice",
    "description": (
        "Net Debt to Price ratio: (total_debt - cash) / market_cap, using "
        "point-in-time fundamentals (filed_date safe). Predicted sign -1: "
        "high leverage relative to price forecasts lower returns. "
        "Per-ticker proxy — the original (Penman, Richardson, Tuna 2007 / "
        "Chen-Zimmermann OpenSourceAP) is cross-sectional within B/M quintile 3; "
        "here we produce the level ratio and a 252-day trailing z-score to capture "
        "time-series variation of the same signal within each stock."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_netdebtprice_ratio",   # (total_debt - cash) / market_cap
        "osap_netdebtprice_zscore",  # 252-day trailing z-score of ratio
        "osap_netdebtprice_chg1y",   # 1-year change in ratio (momentum of leverage)
    ],
    "tags": ["leverage", "fundamentals", "accounting", "value"],
    "version": "1.0",
    "author": "Penman, Richardson and Tuna (2007); OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: total_debt, cash, shares_outstanding
    df = _fundamentals.as_of(df, fields=["total_debt", "cash", "shares_outstanding"])

    total_debt = df["fund_total_debt"]
    cash = df["fund_cash"]
    shares = df["fund_shares_outstanding"]
    close = df["Close"]

    # Market cap (shares in units matching fundamentals — usually millions)
    mktcap = close * shares
    mktcap_safe = mktcap.replace(0, np.nan)

    # Net debt (proxy: total_debt - cash; negative = net cash)
    net_debt = total_debt - cash

    # Level ratio
    ratio = net_debt / mktcap_safe
    # Clamp extreme outliers (e.g. near-zero mktcap on delisting noise)
    ratio = ratio.clip(-50.0, 50.0)
    df["osap_netdebtprice_ratio"] = ratio

    # 252-day trailing z-score (own-history normalisation, lookahead-free)
    roll_mean = ratio.rolling(252, min_periods=63).mean()
    roll_std = ratio.rolling(252, min_periods=63).std()
    roll_std_safe = roll_std.where(roll_std > 1e-9, np.nan)
    df["osap_netdebtprice_zscore"] = (ratio - roll_mean) / roll_std_safe

    # 1-year change in ratio (captures rising/falling leverage trend)
    df["osap_netdebtprice_chg1y"] = ratio.diff(252)

    # Drop scratch fund_ columns not in produces
    df = df.drop(
        columns=["fund_total_debt", "fund_cash", "fund_shares_outstanding"],
        errors="ignore",
    )

    return df
