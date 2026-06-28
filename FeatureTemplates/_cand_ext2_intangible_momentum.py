"""
_cand_ext2_intangible_momentum.py
Candidate feature block: Intangible-investment ramp (momentum of intensity).
UNPROVEN -- prefixed with underscore to keep it hidden from auto-discovery.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by path to avoid package import)
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
    "name": "ext2_intangible_momentum",
    "description": (
        "Intangible-investment intensity ramp derived from PIT SEC fundamentals. "
        "Intangible intensity = (SG&A_ttm + R&D_ttm) / revenue_ttm, where "
        "SG&A_ttm is approximated as gross_profit_ttm - operating_income_ttm. "
        "Produces: 126-day change in intensity (6-month ramp), 252-day change "
        "(annual ramp), and the acceleration (126d change minus prior-126d change). "
        "Captures firms ramping intangible investment -- a signal orthogonal to "
        "price momentum and valuation multiples. Per-ticker proxy using PIT "
        "filings; cross-sectional ranking not applied here (cross-sectional "
        "overlay is done downstream). Coverage ~84%; ETFs/foreign yield NaN."
    ),
    "requires": [],  # OHLCV not directly needed; fundamentals only
    "produces": [
        "ext2_intangible_momentum_ramp126",
        "ext2_intangible_momentum_ramp252",
        "ext2_intangible_momentum_accel",
    ],
    "tags": ["fundamentals", "intangibles", "momentum", "rnd", "sga"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of a rich winner vein (osap_orgcap)",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: one ticker, ascending Date, columns [Date, Ticker, Open, High, Low, Close, Volume].
    Adds ext2_intangible_momentum_ramp126, _ramp252, _accel.
    """
    # Pull PIT fundamentals (backward merge on filed_date -- no lookahead)
    df = _fundamentals.as_of(
        df,
        fields=[
            "gross_profit_ttm",
            "operating_income_ttm",
            "rnd_expense_ttm",
            "revenue_ttm",
        ],
    )

    # --- Intangible intensity --------------------------------------------------
    # SG&A_ttm proxy = gross_profit_ttm - operating_income_ttm
    sga_ttm = df["fund_gross_profit_ttm"] - df["fund_operating_income_ttm"]
    rnd_ttm = df["fund_rnd_expense_ttm"].fillna(0.0)  # R&D often missing → treat as 0
    rev_ttm = df["fund_revenue_ttm"]

    # Guard: zero / negative revenue → NaN
    rev_safe = rev_ttm.where(rev_ttm > 0, np.nan)

    # Intensity: (SG&A + R&D) / Revenue  (dimensionless ratio)
    intensity = (sga_ttm + rnd_ttm) / rev_safe

    # --- Momentum of intensity -------------------------------------------------
    # 126-day ramp (approx 6 months of calendar days, quarterly updates visible)
    ramp126 = intensity - intensity.shift(126)

    # 252-day ramp (approx 1 year)
    ramp252 = intensity - intensity.shift(252)

    # Acceleration: change in the 126d ramp vs the prior 126d ramp
    # = current 126d ramp  -  ramp126 lagged 126 bars
    accel = ramp126 - ramp126.shift(126)

    # --- Assign produced columns -----------------------------------------------
    df["ext2_intangible_momentum_ramp126"] = ramp126
    df["ext2_intangible_momentum_ramp252"] = ramp252
    df["ext2_intangible_momentum_accel"] = accel

    # --- Drop scratch fund_ columns not in produces ----------------------------
    fund_cols = [
        "fund_gross_profit_ttm",
        "fund_operating_income_ttm",
        "fund_rnd_expense_ttm",
        "fund_revenue_ttm",
    ]
    df = df.drop(columns=[c for c in fund_cols if c in df.columns])

    return df
