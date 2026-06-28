"""
ext2_intan_adj_roa — Intangible-adjusted return on capital (Peters-Taylor style).

Capitalises SG&A (org capital) and R&D (R&D capital) via perpetual-inventory
models, then computes operating income / (assets + intangible capital) vs the
plain reported ROA and their wedge. Per-ticker, PIT via _fundamentals.as_of.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (import by file path — no package dependency)
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
    "name": "ext2_intan_adj_roa",
    "description": (
        "Intangible-adjusted return on capital (Peters-Taylor). "
        "Capitalises SG&A via a perpetual-inventory model (30 %/yr depreciation) "
        "as organisational capital (OC), and R&D at 20 %/yr as R&D capital (RDC). "
        "Intangible capital IC = OC + RDC. Produces: "
        "(1) ext2_intan_adj_roa_ic — operating_income_ttm / (assets + IC), the "
        "intangible-adjusted ROA; "
        "(2) ext2_intan_adj_roa_plain — operating_income_ttm / assets, standard ROA; "
        "(3) ext2_intan_adj_roa_wedge — difference (adj – plain), capturing how much "
        "off-balance-sheet intangibles depress reported profitability. "
        "Per-ticker proxy; cross-sectional ranking not applied here."
    ),
    "requires": [],  # all inputs come from _fundamentals (PIT)
    "produces": [
        "ext2_intan_adj_roa_ic",
        "ext2_intan_adj_roa_plain",
        "ext2_intan_adj_roa_wedge",
    ],
    "tags": ["fundamentals", "profitability", "intangibles", "r&d", "organizational_capital"],
    "version": "1.0.0",
    "author": "Round-3 deep exploration of a rich winner vein (osap_orgcap); Peters & Taylor (2017) intangible capital methodology",
}

# ---------------------------------------------------------------------------
# Perpetual-inventory helpers (vectorised over a pandas Series)
# ---------------------------------------------------------------------------
_DELTA_OC = 1.0 - (0.70 ** 0.25)   # quarterly depreciation rate for org capital   (~8.1%)
_DELTA_RD = 1.0 - (0.80 ** 0.25)   # quarterly depreciation rate for R&D capital   (~5.6%)
_GROWTH_SEED = 0.025                  # assumed long-run growth rate for seed OC

def _perpetual_inventory(quarterly_add: np.ndarray, delta: float) -> np.ndarray:
    """
    Forward-recursive perpetual-inventory accumulation (per-ticker series).

    OC_t = (1 - delta) * OC_{t-1} + quarterly_add_t
    Seed: OC_0 = quarterly_add_0 / (growth + delta), where growth = 0.025.
    NaN add values propagate the previous stock (no depreciation of NaN bars
    would be wrong; instead we carry the stock and add 0 for that period).
    Returns float64 array same length as quarterly_add.
    """
    n = len(quarterly_add)
    stock = np.empty(n, dtype=np.float64)
    retain = 1.0 - delta
    seed_denom = _GROWTH_SEED + delta

    # Find first non-NaN observation for seeding
    first_valid = -1
    for i in range(n):
        if not np.isnan(quarterly_add[i]):
            first_valid = i
            break

    if first_valid == -1:
        stock[:] = np.nan
        return stock

    # rows before first valid → NaN
    stock[:first_valid] = np.nan

    seed_val = quarterly_add[first_valid]
    stock[first_valid] = seed_val / seed_denom if seed_denom != 0.0 else np.nan

    for t in range(first_valid + 1, n):
        add = quarterly_add[t]
        prev = stock[t - 1]
        if np.isnan(prev):
            # restart
            if not np.isnan(add):
                stock[t] = add / seed_denom if seed_denom != 0.0 else np.nan
            else:
                stock[t] = np.nan
        else:
            actual_add = 0.0 if np.isnan(add) else add
            stock[t] = retain * prev + actual_add

    return stock


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals; fields are appended as fund_<name>
    needed = [
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
        "assets",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=needed)

    # ------------------------------------------------------------------
    # Quarterly SG&A proxy:
    #   reported SGA_ttm ≈ gross_profit_ttm − operating_income_ttm − rnd_expense_ttm
    #   divide by 4 → quarterly addition to org-capital stock
    # ------------------------------------------------------------------
    gp   = df["fund_gross_profit_ttm"].values.astype(np.float64)
    oi   = df["fund_operating_income_ttm"].values.astype(np.float64)
    rnd  = df["fund_rnd_expense_ttm"].values.astype(np.float64)
    ast  = df["fund_assets"].values.astype(np.float64)

    # SGA_ttm (clip at 0 — cannot be negative by construction)
    rnd_safe = np.where(np.isnan(rnd), 0.0, rnd)
    sga_ttm = gp - oi - rnd_safe
    sga_ttm = np.where(np.isnan(sga_ttm), np.nan, np.clip(sga_ttm, 0.0, None))

    quarterly_sga = sga_ttm / 4.0
    quarterly_rnd = rnd / 4.0   # NaN stays NaN

    # ------------------------------------------------------------------
    # Perpetual-inventory accumulation
    # ------------------------------------------------------------------
    oc  = _perpetual_inventory(quarterly_sga, _DELTA_OC)
    rdc = _perpetual_inventory(quarterly_rnd, _DELTA_RD)

    ic = oc + rdc   # total intangible capital

    # ------------------------------------------------------------------
    # Profitability metrics
    # ------------------------------------------------------------------
    # adj ROA = operating_income_ttm / (assets + IC)
    denom_adj = ast + ic
    denom_adj = np.where(denom_adj == 0.0, np.nan, denom_adj)
    adj_roa = np.where(np.isnan(denom_adj), np.nan, oi / denom_adj)

    # plain ROA = operating_income_ttm / assets
    ast_safe = np.where(ast == 0.0, np.nan, ast)
    plain_roa = np.where(np.isnan(ast_safe), np.nan, oi / ast_safe)

    # wedge = adj − plain  (how much intangibles depress reported ROA;
    #   adj_roa < plain_roa when oi > 0 because the denominator is larger,
    #   so wedge is negative for profitable firms — size of negative wedge
    #   measures the intangible-capital blind-spot)
    wedge = adj_roa - plain_roa

    # Guard against inf (e.g. from pathological near-zero denominators already
    # handled, but belt-and-suspenders)
    def _safe(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    df["ext2_intan_adj_roa_ic"]    = _safe(adj_roa)
    df["ext2_intan_adj_roa_plain"] = _safe(plain_roa)
    df["ext2_intan_adj_roa_wedge"] = _safe(wedge)

    # Drop scratch fund_ columns that are NOT in produces
    scratch = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch)

    return df
