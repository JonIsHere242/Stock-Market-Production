"""
osap_growthinv — Inventory Growth (Belo and Lin 2012)

Signal: deflated inventory growth rate from fiscal year t to fiscal year t-1.
Cross-sectional anomaly (long low-inventory-growth / short high-inventory-growth; sign = -1).

Per-ticker proxy: uses PIT fundamentals (inventory from SEC EDGAR) to compute
YoY inventory growth (nominal, since GNP deflator is unavailable). The level and
a lagged-change variant are produced. This is per-ticker only; the cross-sectional
rank is not computed here (that happens downstream). Coverage ~84% (ETFs/foreign = NaN).

Exclusions noted in original spec (SIC 4/6, at/ppent <= 0) cannot be enforced per-ticker
without sector data; consumers of this feature should apply those filters externally.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

METADATA = {
    "name": "osap_growthinv",
    "description": (
        "Inventory growth (YoY) from PIT SEC fundamentals. "
        "Original method (Belo & Lin 2012): deflated inventory growth fiscal-yr t vs t-1, "
        "negative predicted sign (high inventory growth → lower future returns). "
        "Per-ticker proxy: nominal YoY inventory growth (GNP deflator unavailable). "
        "Three columns: level, momentum-of-growth (change in growth rate), and "
        "a growth-vs-revenue ratio to normalise for sector scale."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_growthinv_yoy",      # YoY inventory growth rate (current period vs prior)
        "osap_growthinv_accel",    # Change in growth rate (acceleration; 2nd derivative proxy)
        "osap_growthinv_rel_rev",  # Inventory growth relative to revenue growth (divergence signal)
    ],
    "tags": ["fundamentals", "accounting", "profitability", "inventory", "osap"],
    "version": "1.0",
    "author": "Belo and Lin (2012), via OpenSourceAP (Chen-Zimmermann); per-ticker impl.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # Load fundamentals helper
    # ------------------------------------------------------------------
    _spec = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fundamentals)

    # Pull PIT inventory and revenue (TTM for revenue stability)
    df = _fundamentals.as_of(df, fields=["inventory", "revenue_ttm"])

    inv = df["fund_inventory"]          # raw inventory level (PIT)
    rev = df["fund_revenue_ttm"]        # TTM revenue (PIT)

    # ------------------------------------------------------------------
    # YoY inventory growth
    # Fundamentals update ~quarterly; 252 trading days ≈ 1 fiscal year.
    # We use a 252-day backward shift to get "prior year" value.
    # Where fundamentals haven't changed (same report), the ratio = 1 (growth=0),
    # which is correct — no inventory change until new report.
    # ------------------------------------------------------------------
    LAG = 252  # ~1 trading year

    inv_lag = inv.shift(LAG)
    # Guard zero denominator; negative inventory is also anomalous → NaN
    inv_lag_safe = inv_lag.where((inv_lag > 0) & inv_lag.notna(), other=np.nan)
    inv_safe = inv.where(inv.notna(), other=np.nan)

    yoy_growth = (inv_safe - inv_lag_safe) / inv_lag_safe  # unbounded ratio

    # Clip extreme outliers (>±5 = 500%) to reduce noise; keeps sign
    yoy_growth = yoy_growth.clip(-5.0, 5.0)

    df["osap_growthinv_yoy"] = yoy_growth

    # ------------------------------------------------------------------
    # Acceleration: change in YoY growth rate (rolling difference of yoy)
    # Shift by another LAG quarter (~63 days) to get prior growth rate
    # ------------------------------------------------------------------
    QLAG = 63  # ~1 quarter
    yoy_lag1 = yoy_growth.shift(QLAG)
    accel = yoy_growth - yoy_lag1
    accel = accel.clip(-5.0, 5.0)
    df["osap_growthinv_accel"] = accel

    # ------------------------------------------------------------------
    # Inventory growth relative to revenue growth
    # High inv growth + low/negative rev growth = inventory overhang (bad signal)
    # Low/negative inv growth + high rev growth = lean operations (good signal)
    # Metric: inv_growth - rev_growth  (negative value = lean; positive = overhang)
    # ------------------------------------------------------------------
    rev_lag = rev.shift(LAG)
    rev_lag_safe = rev_lag.where((rev_lag > 0) & rev_lag.notna(), other=np.nan)
    rev_safe = rev.where(rev.notna(), other=np.nan)

    rev_yoy = (rev_safe - rev_lag_safe) / rev_lag_safe
    rev_yoy = rev_yoy.clip(-5.0, 5.0)

    rel_rev = yoy_growth - rev_yoy  # positive = inventory growing faster than revenue
    rel_rev = rel_rev.clip(-5.0, 5.0)
    df["osap_growthinv_rel_rev"] = rel_rev

    # ------------------------------------------------------------------
    # Drop scratch fundamentals columns not in produces
    # ------------------------------------------------------------------
    df = df.drop(columns=["fund_inventory", "fund_revenue_ttm"], errors="ignore")

    return df
