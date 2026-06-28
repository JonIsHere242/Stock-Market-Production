"""
Intangible-adjusted gross profitability (Novy-Marx).

Augments the Novy-Marx GP/Assets ratio by adding capitalised organisational
capital (OC, from SG&A) and R&D capital (RDC) to the asset base.  The
perpetual-inventory stocks are built from PIT fundamentals so no lookahead
occurs.  Produces:
  ext2_gp_intan_adj_adj  - GP_TTM / (Assets + OC + RDC)
  ext2_gp_intan_adj_raw  - GP_TTM / Assets  (unadjusted Novy-Marx)
  ext2_gp_intan_adj_wedge - adj - raw  (incremental signal from intangibles)
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fundamentals helper (PIT, backward merge_asof)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_gp_intan_adj",
    "description": (
        "Intangible-adjusted gross profitability (Novy-Marx): GP_TTM divided "
        "by (Assets + OC + RDC), where OC = capitalised SG&A via perpetual "
        "inventory (30 %/yr depreciation) and RDC = capitalised R&D via "
        "perpetual inventory (20 %/yr depreciation).  Quarterly increments are "
        "derived from PIT fundamentals (gross_profit_ttm, operating_income_ttm, "
        "rnd_expense_ttm, assets).  Also produces the unadjusted GP/Assets and "
        "the wedge between the two.  Per-ticker proxy -- no cross-sectional "
        "ranking."
    ),
    "requires": [],
    "produces": [
        "ext2_gp_intan_adj_adj",
        "ext2_gp_intan_adj_raw",
        "ext2_gp_intan_adj_wedge",
    ],
    "tags": ["fundamentals", "profitability", "intangibles", "novy-marx", "pit"],
    "version": "1.0",
    "author": (
        "Spec: Round-3 deep exploration of the osap_orgcap winner vein; "
        "Novy-Marx (2013) gross profitability; Peters & Taylor (2017) intangible "
        "capital perpetual-inventory method."
    ),
}

# Perpetual-inventory parameters
_OC_DEPR_ANNUAL = 0.30          # 30 % / yr for organisational capital
_RD_DEPR_ANNUAL = 0.20          # 20 % / yr for R&D capital
# Quarterly survival rates
_OC_SURV_Q = (1 - _OC_DEPR_ANNUAL) ** 0.25
_RD_SURV_Q = (1 - _RD_DEPR_ANNUAL) ** 0.25
# Quarterly depreciation rates
_OC_DELTA_Q = 1.0 - _OC_SURV_Q
_RD_DELTA_Q = 1.0 - _RD_SURV_Q
# Seed divisor  (seed = first_flow / (g + delta_q), g ~ 2.5 %/yr / 4 = 0.00625/q)
_G_Q = 0.025 / 4.0


def _build_perpetual_inv(quarterly_flow: np.ndarray, delta_q: float) -> np.ndarray:
    """
    Forward perpetual-inventory accumulation (causal).

    stock[0] = flow[0] / (g + delta_q)   (seed)
    stock[t] = (1-delta_q)*stock[t-1] + flow[t]

    Returns array of the same length as `quarterly_flow`.
    NaN flows propagate as NaN (carry forward last good stock).
    """
    n = len(quarterly_flow)
    stock = np.full(n, np.nan)
    prev = np.nan
    for i in range(n):
        fl = quarterly_flow[i]
        if np.isnan(fl):
            # carry previous stock forward unchanged
            stock[i] = prev
        else:
            if np.isnan(prev):
                # seed
                denom = _G_Q + delta_q
                prev = fl / denom if denom != 0.0 else np.nan
            else:
                prev = (1.0 - delta_q) * prev + fl
            stock[i] = prev
    return stock


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals (backward-safe)
    # ------------------------------------------------------------------
    fields = [
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
        "assets",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=fields)

    gp = df["fund_gross_profit_ttm"].to_numpy(dtype=float)
    oi = df["fund_operating_income_ttm"].to_numpy(dtype=float)
    rnd = df["fund_rnd_expense_ttm"].to_numpy(dtype=float)
    assets = df["fund_assets"].to_numpy(dtype=float)

    n = len(df)

    # ------------------------------------------------------------------
    # 2. Quarterly SG&A proxy  =  GP_ttm - OI_ttm - RND_ttm, clipped >= 0
    #    Divided by 4 to get a quarterly increment.
    # ------------------------------------------------------------------
    sga_ttm = gp - oi - np.where(np.isnan(rnd), 0.0, rnd)
    sga_ttm = np.where(np.isnan(sga_ttm), np.nan, np.clip(sga_ttm, 0.0, None))
    sga_q = sga_ttm / 4.0

    # Quarterly R&D increment
    rnd_safe = np.where(np.isnan(rnd), np.nan, np.clip(rnd, 0.0, None))
    rnd_q = rnd_safe / 4.0

    # ------------------------------------------------------------------
    # 3. Build perpetual-inventory stocks
    # ------------------------------------------------------------------
    oc = _build_perpetual_inv(sga_q, _OC_DELTA_Q)
    rdc = _build_perpetual_inv(rnd_q, _RD_DELTA_Q)

    # ------------------------------------------------------------------
    # 4. Compute ratios
    # ------------------------------------------------------------------
    aug_assets = assets + oc + rdc

    with np.errstate(divide="ignore", invalid="ignore"):
        gp_adj = np.where(aug_assets > 0, gp / aug_assets, np.nan)
        gp_raw = np.where(assets > 0, gp / assets, np.nan)

    # Replace inf with nan (guard)
    gp_adj = np.where(np.isfinite(gp_adj), gp_adj, np.nan)
    gp_raw = np.where(np.isfinite(gp_raw), gp_raw, np.nan)
    wedge = gp_adj - gp_raw

    # ------------------------------------------------------------------
    # 5. Assign produced columns; drop scratch fund_ columns
    # ------------------------------------------------------------------
    df["ext2_gp_intan_adj_adj"] = gp_adj
    df["ext2_gp_intan_adj_raw"] = gp_raw
    df["ext2_gp_intan_adj_wedge"] = wedge

    # Drop scratch fundamentals columns
    scratch = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch)

    return df
