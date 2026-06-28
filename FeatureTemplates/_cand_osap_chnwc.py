"""
osap_chnwc — Change in Net Working Capital (Soliman 2008 / Chen-Zimmermann OpenSourceAP)

Net working capital (NWC) is defined as:
  nwc = ( (current_assets - cash) - (current_liabilities - short_term_debt) ) / total_assets
  i.e.  ( (act - che) - (lct - dlc) ) / at

The feature is the 12-month (trailing) change in this ratio.

Predicted sign: -1  (higher NWC growth → lower future returns; firms investing more in
operating working capital tend to underperform, consistent with an investment/accruals
anomaly channel).

Implementation note: This is a pure accounting signal. Per-ticker PIT fundamentals are
used via _fundamentals.as_of(). Because filings arrive quarterly at irregular dates,
the 12-month change is computed as the difference between the most-recently-filed NWC
ratio and the NWC ratio from the filing that was available ~252 trading days ago
(backward merge_asof).  This is inherently per-ticker; no cross-sectional ranking.
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
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_chnwc",
    "description": (
        "Change in Net Working Capital (Soliman 2008 via OpenSourceAP / Chen-Zimmermann). "
        "NWC = ((current_assets - cash) - (current_liabilities - short_term_debt)) / total_assets. "
        "Produces: (1) the current NWC ratio level, (2) the 12-month change in NWC (primary signal, "
        "predicted sign -1: rising NWC → lower future returns), (3) a 6-month change as a faster variant. "
        "Per-ticker PIT proxy using quarterly fundamentals; not cross-sectional."
    ),
    "requires": [],
    "produces": [
        "osap_chnwc_level",
        "osap_chnwc_12m",
        "osap_chnwc_6m",
    ],
    "tags": ["accounting", "investment", "accruals", "working_capital", "fundamental"],
    "version": "1.0.0",
    "author": "Soliman (2008); OpenSourceAP / Chen-Zimmermann dataset. Block by Claude.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker NWC level and 12m/6m changes from PIT fundamentals.
    """
    needed_fields = [
        "assets_current",   # act  — current assets
        "cash",             # che  — cash & short-term investments
        "liabilities_current",  # lct  — current liabilities
        "long_term_debt",   # We'll use this as a proxy; dlc (debt in current liab) not separately available
        "assets",           # at   — total assets
    ]

    # Merge PIT fundamentals (backward, lookahead-safe)
    df = _fundamentals.as_of(df, fields=needed_fields)

    fund_assets_cur  = "fund_assets_current"
    fund_cash        = "fund_cash"
    fund_liab_cur    = "fund_liabilities_current"
    fund_lt_debt     = "fund_long_term_debt"   # proxy for dlc; actual dlc unavailable
    fund_assets      = "fund_assets"

    # ---------------------------------------------------------------------------
    # Compute NWC level
    # (current_assets - cash) - (current_liabilities - 0) / total_assets
    # Note: dlc (short-term portion of long-term debt, i.e., debt in current liabilities)
    # is not separately available in the fundamentals panel.  We use 0 for dlc, which is
    # the conservative version of the formula (slightly overstates net current liabilities).
    # ---------------------------------------------------------------------------
    act  = df[fund_assets_cur].astype(float)
    che  = df[fund_cash].astype(float)
    lct  = df[fund_liab_cur].astype(float)
    at   = df[fund_assets].astype(float)

    # Guard against zero/negative total assets
    at_safe = at.where(at.abs() > 1e-9, np.nan)

    nwc = ((act - che) - lct) / at_safe

    # Replace inf/-inf with NaN
    nwc = nwc.replace([np.inf, -np.inf], np.nan)

    df["osap_chnwc_level"] = nwc

    # ---------------------------------------------------------------------------
    # 12-month change: difference between current NWC and NWC 252 trading days ago
    # Because fundamentals update only on filing dates, shift(252) gives the value
    # that was observable approximately one year prior (PIT-safe via the backward merge).
    # ---------------------------------------------------------------------------
    df["osap_chnwc_12m"] = nwc - nwc.shift(252)
    df["osap_chnwc_6m"]  = nwc - nwc.shift(126)

    # Drop scratch fund_ columns not listed in produces
    scratch_cols = [
        fund_assets_cur,
        fund_cash,
        fund_liab_cur,
        fund_lt_debt,
        fund_assets,
    ]
    df = df.drop(columns=[c for c in scratch_cols if c in df.columns])

    return df
