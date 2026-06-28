"""
ext3_profitability_accel — Profitability acceleration (2nd derivative)

Captures inflecting profitability by computing the 2nd derivative (acceleration)
of gross profitability (gross_profit_ttm / assets) and operating_margin over
rolling 252-day windows. A positive acceleration means the rate of improvement
is itself improving — an orthogonal signal vs. level or 1st-derivative features.

Per-ticker proxy: uses PIT fundamentals (filed_date backward merge); cross-sectional
ranks are NOT computed — this is a pure time-series acceleration per stock.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── Load PIT fundamentals helper ──────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ── Metadata ──────────────────────────────────────────────────────────────────
METADATA = {
    "name": "ext3_profitability_accel",
    "description": (
        "Profitability acceleration (2nd derivative). "
        "gross_profitability = gross_profit_ttm / assets; "
        "operating_margin from fundamentals. "
        "Produces the 252-day change-of-change (acceleration) for each ratio. "
        "Captures inflecting profitability — orthogonal to level and 1st-derivative. "
        "Per-ticker time-series only (no cross-sectional ranks). "
        "Missing fundamentals coverage (~16% of rows) yields NaN, as expected."
    ),
    "requires": [],   # only OHLCV columns needed structurally; fundamentals via helper
    "produces": [
        "ext3_profitability_accel_gp_accel",   # 2nd derivative of gross profitability
        "ext3_profitability_accel_om_accel",   # 2nd derivative of operating margin
        "ext3_profitability_accel_combo",      # equal-weight average of both accelerations
    ],
    "tags": ["fundamentals", "profitability", "acceleration", "second_derivative"],
    "version": "1.0.0",
    "author": "Spec: Round-4 expansion (osap_orgcap); relates to OSAP / Organizational Capital literature",
}

# ── Compute ───────────────────────────────────────────────────────────────────
_WINDOW = 252  # trading days per year


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 2nd-derivative (acceleration) of gross profitability and operating margin.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    # Fetch PIT fundamentals (backward merge on filed_date — no lookahead)
    df = _fundamentals.as_of(
        df,
        fields=["gross_profit_ttm", "assets", "operating_margin"],
    )

    # ── Gross profitability = gross_profit_ttm / assets ────────────────────
    denom_gp = df["fund_assets"].replace(0, np.nan)
    gross_prof = df["fund_gross_profit_ttm"] / denom_gp  # scalar ratio per row

    # 1st derivative: 252-day change
    gp_d1 = gross_prof.diff(_WINDOW)
    # 2nd derivative: change in the 1st derivative over another 252 days
    gp_d2 = gp_d1.diff(_WINDOW)

    # ── Operating margin (already a ratio from fundamentals) ───────────────
    op_margin = df["fund_operating_margin"]

    om_d1 = op_margin.diff(_WINDOW)
    om_d2 = om_d1.diff(_WINDOW)

    # ── Assign produced columns ────────────────────────────────────────────
    df["ext3_profitability_accel_gp_accel"] = gp_d2
    df["ext3_profitability_accel_om_accel"] = om_d2

    # Combo: equal-weight mean; NaN if both are NaN
    combo = (gp_d2 + om_d2) / 2.0
    # Where one side is NaN but the other is not, use the available one
    both_nan = gp_d2.isna() & om_d2.isna()
    only_gp = gp_d2.notna() & om_d2.isna()
    only_om = gp_d2.isna() & om_d2.notna()
    combo = combo.copy()
    combo[only_gp] = gp_d2[only_gp]
    combo[only_om] = om_d2[only_om]
    combo[both_nan] = np.nan
    df["ext3_profitability_accel_combo"] = combo

    # ── Drop scratch fund_ columns not in produces ─────────────────────────
    for col in ["fund_gross_profit_ttm", "fund_assets", "fund_operating_margin"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
