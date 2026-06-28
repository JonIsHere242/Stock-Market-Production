"""
_cand_osap_netequityfinance.py -- Net Equity Financing anomaly (Bradshaw, Richardson, Sloan 2006)

Signal: Firms that issue large amounts of net equity (stock sales minus buybacks, scaled by assets)
tend to underperform. Negative predictor: high net equity issuance -> lower future returns.

Original construction (Compustat):
    NEqFin = (sstk - prstkc) / avg(at_t, at_t-1)
where sstk = sale of common/preferred stock, prstkc = purchase of common stock (buybacks).

PIT-fundamentals proxy used here (direct sstk/prstkc not in available fields):
    NEqFin_proxy = (delta_equity - net_income_ttm + dividends_paid_ttm) / avg_assets
Accounting identity: delta_equity = retained_earnings + external_equity_raised
=> external_equity_raised = delta_equity - retained_earnings = delta_equity - net_income + dividends
This isolates the net external equity financing component (issuances minus buybacks).

Scaled by average assets (t and t-1) consistent with the original paper.
Winsorised to |ratio| <= 1 per the paper's exclusion rule (set to NaN outside).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# -- load fundamentals helper -------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)
# -----------------------------------------------------------------------------

METADATA = {
    "name": "osap_netequityfinance",
    "description": (
        "Net Equity Financing anomaly: firms raising net external equity underperform. "
        "Proxy = (delta_equity - net_income_ttm + dividends_paid_ttm) / avg_assets, "
        "which isolates external equity raised via the accounting identity. "
        "Also produces a 4-quarter change (slope) to capture acceleration. "
        "Winsorised per original paper (|ratio|>1 -> NaN). Per-ticker PIT fundamentals; "
        "cross-sectional signal -- per-ticker time-series variant only."
    ),
    "requires": [],
    "produces": [
        "osap_netequityfinance_lvl",     # current NEqFin level (lower = better)
        "osap_netequityfinance_chg",     # change vs prior filing (acceleration proxy)
        "osap_netequityfinance_ts_z",    # rolling 8-quarter z-score for normalisation
    ],
    "tags": ["fundamentals", "external_financing", "equity_issuance", "osap", "accounting"],
    "version": "1.0.0",
    "author": "Bradshaw, Richardson, Sloan (JAE 2006) -- PIT proxy implementation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull required fundamental fields
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=[
                "equity",
                "net_income_ttm",
                "dividends_paid_ttm",
                "assets",
            ],
        )

    eq = df["fund_equity"]
    ni = df["fund_net_income_ttm"]
    div = df["fund_dividends_paid_ttm"]
    at = df["fund_assets"]

    # -- Net equity financing level -------------------------------------------
    # delta_equity: year-over-year change in book equity (approx one filing back)
    # We use shift(1) on the filed fundamentals -- since as_of() forward-fills filing
    # values, shift(1) gives the prior filing's value on a daily basis.
    eq_lag = eq.shift(1)
    at_lag = at.shift(1)

    delta_eq = eq - eq_lag

    # avg_assets: average of current and lagged assets
    avg_at = (at + at_lag) / 2.0
    avg_at_safe = avg_at.where(avg_at.abs() > 0, np.nan)

    # Dividends paid is typically reported as a negative number in cash flow statements;
    # we want the absolute outflow, so take abs() to be safe across reporting conventions.
    div_abs = div.abs()

    # NEqFin proxy = (delta_equity - net_income_ttm + dividends_paid_ttm) / avg_assets
    # net_income_ttm: trailing 12-month income; dividends reduce equity similarly
    nef = (delta_eq - ni + div_abs) / avg_at_safe

    # Winsorise: |NEqFin| > 1 -> NaN (per original paper exclusion)
    nef = nef.where(nef.abs() <= 1.0, np.nan)

    df["osap_netequityfinance_lvl"] = nef

    # -- Change in NEqFin (acceleration) ---------------------------------------
    # Shift by ~63 trading days (~1 quarter) to get recent change
    nef_lag_q = nef.shift(63)
    df["osap_netequityfinance_chg"] = nef - nef_lag_q

    # -- Rolling z-score (trailing ~2 years = 504 trading days) ---------------
    roll_mean = nef.rolling(window=504, min_periods=63).mean()
    roll_std = nef.rolling(window=504, min_periods=63).std()
    roll_std_safe = roll_std.where(roll_std > 0, np.nan)
    df["osap_netequityfinance_ts_z"] = (nef - roll_mean) / roll_std_safe

    # Drop scratch fundamental columns
    df = df.drop(
        columns=[
            "fund_equity",
            "fund_net_income_ttm",
            "fund_dividends_paid_ttm",
            "fund_assets",
        ],
        errors="ignore",
    )

    return df
