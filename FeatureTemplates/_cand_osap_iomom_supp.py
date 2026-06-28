"""
Spec: osap_iomom_supp — Suppliers Momentum (Menzly & Ozbas 2010)
Per-ticker OHLCV+index proxy.

The original method requires BEA Input-Output tables matched to Compustat
NAICS codes to build industry-level supplier return portfolios. That is
inherently cross-sectional and cannot be reproduced per-ticker. This proxy
captures the same economic signal: upstream (supplier) industries tend to
LEAD downstream (customer) industries by roughly 1-3 months, so a stock's
future return is predicted by returns from its "supply chain". We approximate:

  1. osap_iomom_supp_mkt_lag : Market (SPY) momentum lagged 1-3 months vs
     current stock return. The broad market acts as a common upstream driver;
     the lagged market return relative to the stock's own recent return captures
     how much of the supply-chain lead the ticker has already incorporated.

  2. osap_iomom_supp_lead_ratio : Ratio of stock's 1-month return to its own
     3-month return. A positive IO-momentum stock should have improving near-
     term momentum relative to the medium-term (the supplier pull coming
     through). Positive = supplier momentum arriving now.

  3. osap_iomom_supp_mkt_diff : Difference between lagged 2-month SPY return
     and current 1-month stock return — captures how much the market (upstream
     aggregate) moved before the ticker did, i.e. the lead-lag spread that the
     IO-mom paper exploits in cross-section.

All windows are expressed in trading days. No lookahead: all shifts forward.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load _indexes helper ──────────────────────────────────────────────────────
try:
    _s = _ilu.spec_from_file_location(
        "_indexes", _P(__file__).resolve().parent / "_indexes.py"
    )
    _indexes = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_indexes)
    _HAS_INDEXES = True
except Exception:
    _HAS_INDEXES = False

# ── constants ─────────────────────────────────────────────────────────────────
_D1 = 21    # ~1 month trading days
_D2 = 42    # ~2 months
_D3 = 63    # ~3 months

METADATA = {
    "name": "osap_iomom_supp",
    "description": (
        "Per-ticker proxy for Suppliers Momentum (Menzly & Ozbas 2010 / "
        "Chen-Zimmermann OpenSourceAP). True method: BEA Input-Output tables "
        "matched by NAICS -> industry-level supplier returns -> cross-sectional "
        "decile sort. Per-ticker proxy: (1) lagged market (SPY) return minus "
        "current stock return -- upstream aggregate lead-lag spread; "
        "(2) 1m/3m return ratio -- supplier momentum arriving now; "
        "(3) SPY 2m-lagged return minus stock 1m return. Predicted sign: +1 "
        "(high supplier momentum -> long)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_iomom_supp_mkt_lag",
        "osap_iomom_supp_lead_ratio",
        "osap_iomom_supp_mkt_diff",
    ],
    "tags": ["momentum", "lead_lag", "supply_chain", "io_tables", "proxy"],
    "version": "1.0",
    "author": "Menzly and Ozbas 2010; Chen-Zimmermann OpenSourceAP; per-ticker proxy",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # ── stock returns at multiple horizons ───────────────────────────────────
    # 1-month return (current close vs close 21 days ago)
    ret_1m = close / close.shift(_D1) - 1.0
    # 3-month return
    ret_3m = close / close.shift(_D3) - 1.0

    # ── market (SPY) index returns ───────────────────────────────────────────
    spy_1m_lagged = pd.Series(np.nan, index=df.index)   # SPY ret ~2m ago
    spy_2m_lagged = pd.Series(np.nan, index=df.index)   # SPY ret ~2m ago

    if _HAS_INDEXES:
        try:
            spy_close = _indexes.index_close("SPY")  # DatetimeIndex -> float
            # Align to our df by Date (backward safe merge_asof)
            df_dates = pd.DataFrame({"Date": df["Date"]})
            spy_df = spy_close.rename("spy_close").reset_index()
            spy_df.columns = ["Date", "spy_close"]
            spy_df["Date"] = pd.to_datetime(spy_df["Date"])
            df_dates["Date"] = pd.to_datetime(df_dates["Date"])

            merged = pd.merge_asof(
                df_dates.sort_values("Date"),
                spy_df.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            # restore original order
            merged = merged.set_index(df_dates.sort_values("Date").index)
            spy_c = merged["spy_close"].reindex(df.index)

            # SPY 1-month return lagged by 1 month
            #   = SPY return from 2m ago to 1m ago  (fully in the past)
            spy_1m_lagged = (spy_c.shift(_D1) / spy_c.shift(_D2) - 1.0)
            # SPY 2-month return lagged by 1 month
            #   = SPY return from 3m ago to 1m ago
            spy_2m_lagged = (spy_c.shift(_D1) / spy_c.shift(_D3) - 1.0)
        except Exception:
            pass

    # ── feature 1: market lead-lag spread (supplier proxy) ──────────────────
    # High value => market was strong upstream before stock caught up -> bullish
    mkt_lag = spy_1m_lagged - ret_1m
    mkt_lag = mkt_lag.replace([np.inf, -np.inf], np.nan)

    # ── feature 2: near/medium momentum ratio ────────────────────────────────
    # ret_3m denominator guarded
    denom_3m = ret_3m.copy()
    denom_3m = denom_3m.where(denom_3m.abs() > 1e-8, np.nan)
    lead_ratio = ret_1m / denom_3m
    lead_ratio = lead_ratio.replace([np.inf, -np.inf], np.nan)

    # ── feature 3: 2m lagged SPY spread ─────────────────────────────────────
    mkt_diff = spy_2m_lagged - ret_1m
    mkt_diff = mkt_diff.replace([np.inf, -np.inf], np.nan)

    df["osap_iomom_supp_mkt_lag"]   = mkt_lag.values
    df["osap_iomom_supp_lead_ratio"] = lead_ratio.values
    df["osap_iomom_supp_mkt_diff"]  = mkt_diff.values

    return df
