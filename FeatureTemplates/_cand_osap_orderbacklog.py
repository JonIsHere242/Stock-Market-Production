"""
Candidate feature block: osap_orderbacklog
Spec: Order backlog / average total assets (Rajgopal, Shevlin, Venkatachalam 2003)
via OpenSourceAP (Chen-Zimmermann).

Predicted sign: -1 (high backlog/assets predicts lower returns cross-sectionally).

PROXY NOTE: "Order backlog" is not reported in standard bulk SEC fundamentals.
We approximate it as accounts receivable (a near-term claim on future revenue,
i.e. orders already placed but not yet collected) scaled by average total assets
over consecutive filing periods -- capturing the same economic signal of sales
pipeline depth relative to asset base.  The directional prediction should be
preserved: large receivables/assets implies management is stretching the balance
sheet to fulfil forward orders, and the cross-sectional premium is negative
(growth already priced in).

A second variant (slope) captures the quarterly change in the ratio, which
isolates acceleration in the backlog proxy.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

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
    "name": "osap_orderbacklog",
    "description": (
        "Per-ticker proxy for the Order Backlog / Average Total Assets anomaly "
        "(Rajgopal, Shevlin, Venkatachalam 2003; OpenSourceAP Chen-Zimmermann). "
        "True order backlog is not in bulk SEC filings; we proxy with accounts "
        "receivable (a forward-looking claim on booked but uncollected revenue) "
        "scaled by the average of current and prior-period total assets. "
        "Cross-sectional predicted sign is -1 (high ratio -> lower future returns). "
        "Produces: level ratio, 1-period change (slope), and a 2-period rolling "
        "mean for smoothing."
    ),
    "requires": [],   # OHLCV not directly needed; only PIT fundamentals
    "produces": [
        "osap_orderbacklog_ratio",   # receivables / avg(assets_t, assets_t-1)
        "osap_orderbacklog_chg",     # QoQ change in ratio (slope signal)
        "osap_orderbacklog_smooth",  # 2-period rolling mean of ratio
    ],
    "tags": ["fundamentals", "accounting", "sales_growth", "openSourceAP"],
    "version": "1.0",
    "author": (
        "Rajgopal S., Shevlin T., Venkatachalam M. (2003) via "
        "OpenSourceAP / Chen-Zimmermann replication"
    ),
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: receivables and total assets
    df = _fundamentals.as_of(df, fields=["receivables", "assets"])

    recv = df["fund_receivables"]
    assets = df["fund_assets"]

    # Average assets: current + prior observation (shift(1) = previous filing value
    # as seen by merge_asof; since as_of is backward-merged on filed_date, two
    # consecutive rows already represent two different filing dates in time order).
    assets_prev = assets.shift(1)
    avg_assets = (assets + assets_prev) / 2.0

    # Guard: exclude when assets average is zero or NaN, or when receivables == 0
    # (consistent with spec's "exclude if order backlog is 0")
    valid_recv = recv.replace(0, np.nan)
    avg_assets_safe = avg_assets.replace(0, np.nan)

    ratio = valid_recv / avg_assets_safe
    # Replace any inf that might slip through
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_orderbacklog_ratio"] = ratio

    # Slope: change from prior period ratio
    chg = ratio - ratio.shift(1)
    chg = chg.replace([np.inf, -np.inf], np.nan)
    df["osap_orderbacklog_chg"] = chg

    # Smoothed: 2-period rolling mean (requires 2 non-NaN)
    smooth = ratio.rolling(window=2, min_periods=2).mean()
    smooth = smooth.replace([np.inf, -np.inf], np.nan)
    df["osap_orderbacklog_smooth"] = smooth

    # Drop scratch columns
    df.drop(columns=["fund_receivables", "fund_assets"], inplace=True)

    return df
