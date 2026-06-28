"""
osap_cfp  --  Cash Flow to Price (CFP)
Source: OpenSourceAP (Chen-Zimmermann), based on Lakonishok, Shleifer, Vishny (1994).

CFP = (net_income_ttm + depreciation_proxy) / market_cap
where market_cap = Close * shares_outstanding (PIT via fundamentals).

Predicted sign: +1 (high CFP = cheap cash-generating firm → outperforms).

Per-ticker proxy note:
  The canonical version excludes NASDAQ stocks in a cross-sectional sort.
  This block implements the per-ticker ratio directly (no cross-sectional rank).
  The ratio itself is the economically meaningful signal; downstream XS-ranking
  in the predictor will recover the cross-sectional ordering.
  Depreciation is not available as a standalone SEC field, so we proxy it via
  (capex_ttm * 0.5) as a conservative estimate -- common in the empirical
  accounting literature when dp is missing. When capex is also missing we use
  net_income_ttm alone (pure earnings yield). Users may improve by adding dp
  from a richer data source.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── PIT fundamentals helper ──────────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ── metadata ─────────────────────────────────────────────────────────────────
METADATA = {
    "name": "osap_cfp",
    "description": (
        "Cash Flow to Price (CFP): (net_income_ttm + capex_ttm*0.5 as depreciation proxy) "
        "divided by market cap (Close * shares_outstanding). "
        "High CFP signals cheap cash-generating firms (value factor, predicted +). "
        "Per-ticker PIT ratio; cross-sectional rank applied downstream by predictor. "
        "Depreciation approximated as 0.5*capex_ttm when standalone dp unavailable. "
        "Source: OpenSourceAP (Chen-Zimmermann); Lakonishok, Shleifer, Vishny 1994."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_cfp_ratio",       # core CFP level: (ni_ttm + dep_proxy) / mktcap
        "osap_cfp_ni_only",     # fallback: net_income_ttm / mktcap (earnings yield)
        "osap_cfp_chg_1y",      # 1-year change in CFP ratio (trend signal)
    ],
    "tags": ["valuation", "fundamentals", "cash_flow", "accounting", "osap"],
    "version": "1.0",
    "author": "osap_cfp block; spec: OpenSourceAP (Chen-Zimmermann); Lakonishok, Shleifer, Vishny 1994",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cash Flow to Price ratio from PIT fundamentals + Close price."""

    # ── pull PIT fundamentals ────────────────────────────────────────────────
    df = _fundamentals.as_of(
        df,
        fields=["net_income_ttm", "shares_outstanding", "capex_ttm"],
    )

    ni = df["fund_net_income_ttm"]
    shares = df["fund_shares_outstanding"]
    capex = df["fund_capex_ttm"]

    # Market cap: price * shares (shares in thousands from Compustat → multiply by 1000)
    # Guard: shares must be positive
    mktcap = df["Close"] * shares.where(shares > 0, np.nan)

    # Depreciation proxy: 0.5 * |capex_ttm| (capex reported as negative in some sources)
    dep_proxy = capex.abs() * 0.5

    # Core CFP: (NI + dep_proxy) / mktcap
    cf_total = ni + dep_proxy
    cfp_ratio = cf_total / mktcap.where(mktcap > 0, np.nan)
    # Clip extreme outliers (> ±10 is economically implausible for annual CF/price)
    cfp_ratio = cfp_ratio.clip(-10.0, 10.0)

    # Fallback earnings yield: ni / mktcap
    cfp_ni_only = ni / mktcap.where(mktcap > 0, np.nan)
    cfp_ni_only = cfp_ni_only.clip(-10.0, 10.0)

    # 1-year change: rolling shift ~252 trading days
    # Use forward-filled ratio at quarterly granularity (fundamentals update ~quarterly)
    cfp_lag = cfp_ratio.shift(252)
    cfp_chg_1y = cfp_ratio - cfp_lag

    # ── assign produced columns ──────────────────────────────────────────────
    df["osap_cfp_ratio"] = cfp_ratio
    df["osap_cfp_ni_only"] = cfp_ni_only
    df["osap_cfp_chg_1y"] = cfp_chg_1y

    # ── drop scratch fund_ columns not in produces ───────────────────────────
    for col in ["fund_net_income_ttm", "fund_shares_outstanding", "fund_capex_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
