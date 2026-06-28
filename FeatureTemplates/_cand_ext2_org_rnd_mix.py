"""
ext2_org_rnd_mix -- Intangible composition: organizational vs knowledge capital

Organizational capital (OC) is capitalized SG&A via perpetual inventory at 30%/yr.
R&D capital (RDC) is capitalized R&D spend via perpetual inventory at 20%/yr.
Produces OC/(OC+RDC): near 1 = process/organizational-heavy firm; near 0 = knowledge/R&D-heavy firm.
Also produces 252-day change in that ratio as a dynamic axis.
Orthogonal to the *level* of intangible capital (parent: osap_orgcap).

Proxy note: computation is per-ticker using PIT fundamentals (filed_date merge); SG&A is
estimated as gross_profit_ttm - operating_income_ttm - rnd_expense_ttm (clipped >= 0).
Cross-sectional ranking is not performed here; the raw ratio is the signal.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load PIT fundamentals helper
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "ext2_org_rnd_mix",
    "description": (
        "Intangible capital composition: OC/(OC+RDC). "
        "Organizational capital (OC) capitalized from SG&A at 30%/yr perpetual inventory; "
        "R&D capital (RDC) capitalized from rnd_expense_ttm at 20%/yr perpetual inventory. "
        "Ratio near 1 = process-heavy firm; near 0 = knowledge/R&D-heavy firm. "
        "252-day change captures transitions in firm type. "
        "Per-ticker PIT proxy; no cross-sectional ranking."
    ),
    "requires": ["Close"],  # Close used only to anchor merge; fundamentals drive the signal
    "produces": [
        "ext2_org_rnd_mix_ratio",   # OC / (OC + RDC)
        "ext2_org_rnd_mix_chg252",  # 252-day change in the ratio
    ],
    "tags": ["fundamentals", "intangibles", "organizational_capital", "rnd_capital", "quality"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of osap_orgcap vein (ext2_org_rnd_mix)",
}

# Annual decay rates
_OC_ANNUAL_DECAY = 0.30   # 30% per year for organizational capital
_RDC_ANNUAL_DECAY = 0.20  # 20% per year for R&D capital

# Quarterly retention factors: (1 - annual_decay)^0.25
_OC_RETAIN_Q = (1.0 - _OC_ANNUAL_DECAY) ** 0.25
_RDC_RETAIN_Q = (1.0 - _RDC_ANNUAL_DECAY) ** 0.25

# Quarterly depreciation deltas
_OC_DELTA_Q = 1.0 - _OC_RETAIN_Q
_RDC_DELTA_Q = 1.0 - _RDC_RETAIN_Q


def _perpetual_inventory(quarterly_additions: np.ndarray, delta: float) -> np.ndarray:
    """
    Perpetual inventory model over an array of quarterly additions.
    Seed: stock_0 = add_0 / (0.025 + delta)  (Gordon-growth style).
    Recurrence: stock_t = (1 - delta) * stock_{t-1} + add_t
    Returns array of same length as quarterly_additions.
    """
    n = len(quarterly_additions)
    stock = np.empty(n, dtype=np.float64)
    if n == 0:
        return stock

    add0 = quarterly_additions[0]
    # Seed with Gordon-growth denominator; if add0 is NaN treat seed as 0
    if np.isnan(add0) or add0 <= 0.0:
        stock[0] = 0.0
    else:
        stock[0] = add0 / (0.025 + delta)

    retain = 1.0 - delta
    for i in range(1, n):
        add_i = quarterly_additions[i]
        if np.isnan(add_i):
            # Carry forward with depreciation only
            prev = stock[i - 1]
            stock[i] = retain * prev if not np.isnan(prev) else np.nan
        else:
            prev = stock[i - 1] if not np.isnan(stock[i - 1]) else 0.0
            stock[i] = retain * prev + add_i

    return stock


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals; only fields we need
    df = _fundamentals.as_of(
        df,
        fields=["gross_profit_ttm", "operating_income_ttm", "rnd_expense_ttm"],
    )

    gp = df["fund_gross_profit_ttm"].to_numpy(dtype=float)
    oi = df["fund_operating_income_ttm"].to_numpy(dtype=float)
    rnd_ttm = df["fund_rnd_expense_ttm"].to_numpy(dtype=float)
    n = len(df)

    # --- Quarterly SG&A proxy ---
    # SGA_quarterly = clip(gross_profit_ttm - operating_income_ttm - rnd_expense_ttm, 0, None) / 4
    with np.errstate(invalid="ignore"):
        sga_annual = gp - oi - rnd_ttm
        sga_annual = np.where(np.isfinite(sga_annual), sga_annual, np.nan)
        sga_annual = np.clip(sga_annual, 0.0, None)   # must be >= 0
    sga_q = sga_annual / 4.0

    # --- Quarterly R&D addition ---
    with np.errstate(invalid="ignore"):
        rnd_q = np.where(np.isfinite(rnd_ttm), rnd_ttm / 4.0, np.nan)
        rnd_q = np.clip(rnd_q, 0.0, None)

    # --- Perpetual inventory for OC and RDC ---
    oc_stock = _perpetual_inventory(sga_q, _OC_DELTA_Q)
    rdc_stock = _perpetual_inventory(rnd_q, _RDC_DELTA_Q)

    # --- Composition ratio: OC / (OC + RDC) ---
    with np.errstate(invalid="ignore", divide="ignore"):
        total = oc_stock + rdc_stock
        ratio = np.where(total > 0.0, oc_stock / total, np.nan)
        # If both are zero (e.g. no intangibles at all) emit NaN -- not meaningful
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    ratio_s = pd.Series(ratio, index=df.index)

    # --- 252-day change in the ratio ---
    chg252 = ratio_s - ratio_s.shift(252)

    df["ext2_org_rnd_mix_ratio"] = ratio_s
    df["ext2_org_rnd_mix_chg252"] = chg252

    # Drop scratch fundamentals columns
    for col in ["fund_gross_profit_ttm", "fund_operating_income_ttm", "fund_rnd_expense_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
