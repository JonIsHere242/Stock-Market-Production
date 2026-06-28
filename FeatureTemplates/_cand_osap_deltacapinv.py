"""
osap_deltacapinv — Change in Capital Investment (industry-adjusted proxy)

Source: Abarbanell and Bushee (1998), "Abnormal Returns to a Fundamental Analysis Strategy",
        The Accounting Review, Vol. 73, No. 1, pp. 19-45.

Signal: Growth in capital expenditure (capx) relative to the average of the prior two years,
        minus the same growth averaged across the firm's industry (2-digit SIC).
        High capex growth is bearish (over-investment signal, sign = -1).

Per-ticker proxy: The industry-adjustment requires a cross-sectional panel and cannot be
        computed faithfully per-stock. We implement the firm-level capex growth term
        (which captures the core over-investment signal) using PIT SEC fundamentals:
        capex_ttm as the primary capex proxy; if missing, change in ppe_net as fallback.
        The industry-mean subtraction is OMITTED — this is a proxy, not the exact signal.
        Three columns are produced:
          osap_deltacapinv_raw   : capex growth vs 1-year-ago (single-period, always available)
          osap_deltacapinv_2y    : capex growth vs 2-year average (closer to paper definition)
          osap_deltacapinv_accel : change in growth rate (acceleration/deceleration of capex)
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------- PIT fundamentals helper ----------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_deltacapinv",
    "description": (
        "Per-ticker proxy for Abarbanell & Bushee (1998) industry-adjusted capex growth signal. "
        "Measures growth in capital expenditure (capex_ttm or ppe_net change) relative to prior "
        "periods. High capex growth is bearish (over-investment). Industry-mean adjustment is "
        "omitted (requires cross-sectional panel); this is a faithful firm-level proxy. "
        "Produces: raw 1yr growth, 2yr-avg growth (closer to paper), and growth acceleration."
    ),
    "requires": [],
    "produces": [
        "osap_deltacapinv_raw",
        "osap_deltacapinv_2y",
        "osap_deltacapinv_accel",
    ],
    "tags": ["fundamental", "investment", "capex", "accounting", "osap"],
    "version": "1.0.0",
    "author": "Abarbanell and Bushee (1998), AR; proxy impl by codegen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: capex_ttm (primary), ppe_net (fallback)
    df = _fundamentals.as_of(df, fields=["capex_ttm", "ppe_net"])

    # ---- Build capex proxy series (annual frequency, ffilled to daily) ----
    # capex_ttm is already trailing-12-month capital expenditure (positive convention assumed)
    capex = df["fund_capex_ttm"].copy()

    # Fallback: where capex_ttm is NaN, use year-over-year change in ppe_net as capex proxy
    ppe = df["fund_ppe_net"].copy()
    ppe_delta = ppe.diff()          # positive = net new investment (after depr)
    capex_fill = capex.where(capex.notna(), other=ppe_delta.where(ppe_delta > 0))

    # ---- Lagged capex values (approximate annual lags via 252-trading-day windows) ----
    # Use rolling mean of the last 252 rows as "prior-year" value — this smooths noise
    # from the asof-merge point-in-time data landing on different days each year.
    # We use expanding min_periods to avoid excessive NaN at the start.

    # Shift by ~252 trading days to approximate t-1 year value
    CAP_T1 = capex_fill.shift(252)   # approximate 1-year-ago capex
    CAP_T2 = capex_fill.shift(504)   # approximate 2-year-ago capex

    # ---- 1. Raw 1-year growth: (capex_t - capex_{t-1}) / |capex_{t-1}| ----
    denom_raw = CAP_T1.abs().replace(0, np.nan)
    raw_growth = (capex_fill - CAP_T1) / denom_raw

    # ---- 2. Two-year-average growth (closer to Abarbanell & Bushee definition):
    #    growth = (capex_t - avg(capex_{t-1}, capex_{t-2})) / avg(capex_{t-1}, capex_{t-2})
    avg_2y = (CAP_T1 + CAP_T2) / 2.0
    # Fall back to single-year if t-2 is missing
    avg_2y_filled = avg_2y.where(CAP_T2.notna(), other=CAP_T1)
    denom_2y = avg_2y_filled.abs().replace(0, np.nan)
    growth_2y = (capex_fill - avg_2y_filled) / denom_2y

    # ---- 3. Acceleration: change in 1-year growth vs prior year's 1-year growth ----
    raw_growth_t1 = raw_growth.shift(252)
    accel = raw_growth - raw_growth_t1

    # ---- Clip extreme outliers (>5 std dev) without filling NaN ----
    def _winsor(s: pd.Series, n_std: float = 5.0) -> pd.Series:
        mu = s.mean()
        sd = s.std()
        if pd.isna(sd) or sd == 0:
            return s
        return s.clip(lower=mu - n_std * sd, upper=mu + n_std * sd)

    raw_growth = _winsor(raw_growth)
    growth_2y = _winsor(growth_2y)
    accel = _winsor(accel)

    # Assign produced columns
    df["osap_deltacapinv_raw"] = raw_growth
    df["osap_deltacapinv_2y"] = growth_2y
    df["osap_deltacapinv_accel"] = accel

    # Drop scratch fundamentals columns
    df = df.drop(columns=["fund_capex_ttm", "fund_ppe_net"], errors="ignore")

    return df
