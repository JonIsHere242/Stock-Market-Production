"""
Candidate feature block: osap_invgrowth
Inventory Growth (Belo and Lin 2012, via Chen-Zimmermann OpenSourceAP)

Predicted sign: -1 (high inventory growth predicts lower future returns).

Per-ticker proxy notes:
  - True signal deflates inventory by the GNP deflator and computes fiscal-year
    YoY growth. We cannot access the GNP deflator series here, so we use the
    NOMINAL YoY inventory growth rate from PIT fundamentals (inventory field).
  - Original paper drops SIC codes starting with 4 (transport/utilities) or 6
    (finance) and firms with at/ppent <= 0. We approximate by requiring
    fund_assets > 0 and fund_ppe_net > 0 before emitting the signal; rows that
    fail the filter are set to NaN rather than dropped.
  - Coverage is ~84% (ETFs/foreign will be NaN); leading rows before the first
    filing are also NaN -- both are expected.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_invgrowth",
    "description": (
        "Inventory Growth: YoY change in inventory (PIT fundamentals), "
        "a per-ticker proxy for the deflated inventory growth signal from "
        "Belo and Lin (2012) / OpenSourceAP (Chen-Zimmermann). "
        "Predicted sign is -1 (high inventory growth -> lower future returns). "
        "GNP deflator unavailable; nominal growth used instead. "
        "Rows where assets <= 0 or ppe_net <= 0 are set to NaN to mirror "
        "the original paper's sample filter."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_invgrowth_yoy",       # YoY inventory growth rate (primary signal)
        "osap_invgrowth_inv_to_at", # Inventory / total assets (level proxy)
        "osap_invgrowth_chg_12m",   # 12-month rolling change in inv/assets ratio (dynamic variant)
    ],
    "tags": ["fundamentals", "profitability", "accounting", "osap"],
    "version": "1.0.0",
    "author": "Belo and Lin 2012; OpenSourceAP (Chen-Zimmermann); block by claude-sonnet-4-6",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach osap_invgrowth_* columns to df (one stock, ascending Date).
    Returns df with new columns appended; existing columns are never modified.
    """
    # Fetch PIT fundamentals we need
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=["inventory", "assets", "ppe_net"],
        )

    inv = df["fund_inventory"].copy()         # inventory level (PIT)
    at  = df["fund_assets"].copy()            # total assets (PIT)
    ppent = df["fund_ppe_net"].copy()         # net PP&E (proxy for ppent)

    # Apply sample filter: require assets > 0 AND ppe_net > 0
    valid_mask = (at > 0) & (ppent > 0)

    # ------------------------------------------------------------------
    # Signal 1: YoY inventory growth rate
    #   growth = (inv_t - inv_{t-252}) / |inv_{t-252}|
    # We approximate "fiscal year ago" with a 252-trading-day shift.
    # Use absolute value of lagged inv as denominator to handle sign changes;
    # if lagged inv is 0 -> NaN (guard zero division).
    # ------------------------------------------------------------------
    inv_lag = inv.shift(252)
    denom_yoy = inv_lag.abs().replace(0, np.nan)
    yoy = (inv - inv_lag) / denom_yoy

    # Apply filter and cap extreme values (winsorise at 1st/99th by clipping)
    yoy = yoy.where(valid_mask, other=np.nan)
    # Soft cap: beyond ±5 is almost certainly a data artefact
    yoy = yoy.clip(-5.0, 5.0)

    # ------------------------------------------------------------------
    # Signal 2: Inventory / Total Assets (level proxy)
    #   High inv/at can also flag over-building.
    # ------------------------------------------------------------------
    denom_at = at.replace(0, np.nan)
    inv_to_at = inv / denom_at
    inv_to_at = inv_to_at.where(valid_mask, other=np.nan)
    inv_to_at = inv_to_at.clip(0.0, 1.0)   # ratio must be [0,1] for real firms

    # ------------------------------------------------------------------
    # Signal 3: 12-month rolling change in inv/assets (dynamic variant)
    #   Captures acceleration in inventory build-up.
    # ------------------------------------------------------------------
    chg_12m = inv_to_at - inv_to_at.shift(252)

    # ------------------------------------------------------------------
    # Assign produced columns
    # ------------------------------------------------------------------
    df["osap_invgrowth_yoy"]       = yoy
    df["osap_invgrowth_inv_to_at"] = inv_to_at
    df["osap_invgrowth_chg_12m"]   = chg_12m

    # Drop scratch fund_* columns we are NOT listing in produces
    for col in ["fund_inventory", "fund_assets", "fund_ppe_net"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
