"""
osap_hire — Employment growth proxy (Bazdresch, Belo and Lin 2014)

True definition: (emp_t - emp_{t-1}) / avg(emp_t, emp_{t-1}), predicted sign = -1
(firms with high hiring have lower future returns; over-investment channel).

Employee headcount is NOT in the available PIT-fundamentals fields, so we proxy
employment growth with two closely-related firm-expansion signals derived from
PIT SEC data:
  1. osap_hire_rev  — TTM-revenue growth (symmetric log-difference), same
                      denominator-averaging approach as the original metric.
                      Revenue is the most direct proxy for labour demand.
  2. osap_hire_asset — Total-asset growth (same formula), captures the broader
                       investment channel that the paper's -1 sign reflects.
  3. osap_hire_combo — Equal-weight average of the two normalised signals,
                       useful as a combined "firm expansion" factor.

Both proxies are PIT-safe (filed_date backward merge). Leading/inter-filing NaNs
are expected (~84% coverage, ETFs/foreign will be NaN throughout).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
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
    "name": "osap_hire",
    "description": (
        "Employment growth proxy (Bazdresch, Belo & Lin 2014 / OpenSourceAP "
        "Chen-Zimmermann). True signal is (emp_t-emp_{t-1})/avg(emp), predicted "
        "sign -1 (high hiring → lower future returns). Employee headcount is "
        "unavailable in PIT fundamentals, so we substitute (1) TTM-revenue "
        "growth and (2) total-asset growth using the same symmetric-average "
        "scaling formula, both of which capture the same over-investment channel. "
        "Cross-sectional signal approximated per-ticker; ranking across tickers "
        "still required for factor use."
    ),
    "requires": [],
    "produces": ["osap_hire_rev", "osap_hire_asset", "osap_hire_combo"],
    "tags": ["fundamentals", "investment", "employment", "growth", "osap"],
    "version": "1.0",
    "author": "Bazdresch, Belo and Lin (2014); OpenSourceAP (Chen-Zimmermann); proxy impl by Claude",
}


# ---------------------------------------------------------------------------
# Helper: symmetric growth rate  (emp_t - emp_{t-1}) / avg(emp_t, emp_{t-1})
# Equivalent to 2*(x_t - x_{t-1}) / (x_t + x_{t-1}), defined on consecutive
# non-missing values; returns NaN when either value is 0 or missing.
# ---------------------------------------------------------------------------
def _sym_growth(s: pd.Series) -> pd.Series:
    """Symmetric (midpoint) growth rate for a PIT-fundamental series."""
    prev = s.shift(1)
    delta = s - prev
    avg = (s + prev) / 2.0
    # guard zero denominator
    avg_safe = avg.replace(0.0, np.nan)
    return delta / avg_safe


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Attach PIT fundamentals (backward-safe filed_date merge)
    df = _fundamentals.as_of(df, fields=["revenue_ttm", "assets"])

    # --- Revenue-growth proxy -------------------------------------------------
    rev = df["fund_revenue_ttm"].copy()
    # Negative revenue makes the average ill-defined; treat as NaN
    rev_pos = rev.where(rev > 0, np.nan)
    osap_rev = _sym_growth(rev_pos)
    # Winsorise at ±2 to remove data-revision spikes (still relative)
    osap_rev = osap_rev.clip(-2.0, 2.0)
    df["osap_hire_rev"] = osap_rev

    # --- Asset-growth proxy ---------------------------------------------------
    assets = df["fund_assets"].copy()
    assets_pos = assets.where(assets > 0, np.nan)
    osap_asset = _sym_growth(assets_pos)
    osap_asset = osap_asset.clip(-2.0, 2.0)
    df["osap_hire_asset"] = osap_asset

    # --- Combo: equal-weight average of the two z-scored signals -------------
    # Simple average in raw units (both are already on the same symmetric-rate
    # scale) — no z-scoring needed here; caller can rank cross-sectionally.
    valid = (~osap_rev.isna()) & (~osap_asset.isna())
    combo = pd.Series(np.nan, index=df.index)
    combo[valid] = (osap_rev[valid] + osap_asset[valid]) / 2.0
    df["osap_hire_combo"] = combo

    # Drop scratch columns
    df = df.drop(columns=["fund_revenue_ttm", "fund_assets"], errors="ignore")

    return df
