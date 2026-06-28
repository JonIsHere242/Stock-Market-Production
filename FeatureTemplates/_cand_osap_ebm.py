"""
osap_ebm — Enterprise component of Book-to-Market (Penman, Richardson & Tuna 2007)
via OpenSourceAP (Chen-Zimmermann).

Compustat formula:
  numerator   = ceq + che - dltt - dlc - dc - dvpa + tstkp
  denominator = mve_c + che - dltt - dlc - dc - dvpa + tstkp

Where:
  ceq    = common equity            → fund_equity
  che    = cash & short-term inv    → fund_cash
  dltt   = long-term debt           → fund_long_term_debt
  dlc    = debt in current liab     → (fund_total_debt - fund_long_term_debt), proxy
  dc     = deferred charges         → not available → 0
  dvpa   = pref stock div arrears   → not available → 0
  tstkp  = treasury stock (pref)    → not available → 0
  mve_c  = market cap               → fund_shares_outstanding * Close

Exclude rows where price < $5 (set to NaN per spec).

Cross-sectional note: the sign of EBM is predicted +1 (long high EBM). Per-ticker
time series of this ratio still captures the same economic signal: when the
enterprise-adjusted book fraction of enterprise value rises, the stock tends to
appreciate. Missing fundamentals (ETFs, foreign) produce NaN — expected.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load PIT-fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_ebm",
    "description": (
        "Enterprise Book-to-Market (EBM) as defined by Penman, Richardson & Tuna (2007), "
        "replicated from OpenSourceAP (Chen-Zimmermann). "
        "Numerator = common_equity + cash - total_net_debt; "
        "Denominator = market_cap + cash - total_net_debt. "
        "dlc (debt in current liabilities) proxied as total_debt - long_term_debt. "
        "dc/dvpa/tstkp unavailable, treated as zero. "
        "Rows where Close < 5 are set to NaN per spec exclusion rule. "
        "Per-ticker time-series proxy: captures same valuation signal as cross-sectional EBM."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_ebm_ratio",       # raw EBM value
        "osap_ebm_12m_chg",     # 12-month change in EBM (dynamic/slope signal)
    ],
    "tags": ["valuation", "fundamentals", "accounting", "book_to_market", "osap"],
    "version": "1.0",
    "author": "Penman, Richardson & Tuna (2007); OpenSourceAP / Chen-Zimmermann",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals
    # ------------------------------------------------------------------
    fields = [
        "equity",           # ceq: common equity
        "cash",             # che: cash & short-term investments
        "long_term_debt",   # dltt
        "total_debt",       # dltt + dlc combined; dlc = total_debt - long_term_debt
        "shares_outstanding",
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # ------------------------------------------------------------------
    # 2. Derive components
    # ------------------------------------------------------------------
    ceq   = df["fund_equity"]
    che   = df["fund_cash"]
    dltt  = df["fund_long_term_debt"]
    # dlc proxy: current portion of debt = total_debt - long_term_debt (floor at 0)
    total_debt = df["fund_total_debt"]
    dlc   = (total_debt - dltt).clip(lower=0)

    # dc, dvpa, tstkp all unavailable → treated as 0
    # net adjustment term shared by both numerator and denominator
    adj = che - dltt - dlc  # + 0 - 0 + 0

    # market cap
    shares = df["fund_shares_outstanding"]
    mve_c  = shares * df["Close"]

    # ------------------------------------------------------------------
    # 3. Compute EBM = (ceq + adj) / (mve_c + adj)
    # ------------------------------------------------------------------
    numerator   = ceq + adj
    denominator = mve_c + adj

    # Guard divide-by-zero
    denom_safe = denominator.where(denominator != 0, other=np.nan)
    ebm = numerator / denom_safe

    # Replace inf/-inf with NaN (can happen if denominator is near zero)
    ebm = ebm.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 4. Apply price < $5 exclusion (spec rule)
    # ------------------------------------------------------------------
    price_mask = df["Close"] < 5.0
    ebm = ebm.where(~price_mask, other=np.nan)

    df["osap_ebm_ratio"] = ebm

    # ------------------------------------------------------------------
    # 5. 12-month change in EBM (dynamic signal)
    #    Use ~252 trading days as proxy for 1 year
    # ------------------------------------------------------------------
    ebm_lag = ebm.shift(252)
    df["osap_ebm_12m_chg"] = ebm - ebm_lag

    # ------------------------------------------------------------------
    # 6. Drop scratch fund_ columns not listed in produces
    # ------------------------------------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
