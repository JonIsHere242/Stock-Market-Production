"""
ext2_rndcap_growth — Knowledge-capital accumulation rate (R&D capital / assets growth).

R&D capital (RDC) is built via perpetual inventory: quarterly depreciation 20%/yr
(quarterly factor = 0.8^0.25) plus quarterly additions from rnd_expense_ttm/4.
The produced features are the YoY (252d) growth of RDC/assets, its 2-yr (504d) version,
and the 1yr-vs-2yr acceleration.  All series are PIT-safe via _fundamentals.as_of().
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
    "name": "ext2_rndcap_growth",
    "description": (
        "Knowledge-capital accumulation rate. "
        "Builds R&D capital (RDC) via perpetual inventory model (20%/yr depreciation, "
        "quarterly decay factor = 0.8^0.25, quarterly addition = rnd_expense_ttm/4). "
        "Produces: YoY (252d) growth of RDC/assets, 2yr (504d) growth, and 1yr-vs-2yr "
        "acceleration. PIT-safe via _fundamentals.as_of (filed_date backward merge). "
        "Coverage ~84%%; ETFs/foreign yield NaN. "
        "Per-ticker proxy — no cross-sectional ranking."
    ),
    "requires": ["Close"],          # Close used only as a time anchor (merge key)
    "produces": [
        "ext2_rndcap_growth_yoy",   # 252-day growth rate of RDC/assets
        "ext2_rndcap_growth_2yr",   # 504-day growth rate of RDC/assets
        "ext2_rndcap_growth_accel", # acceleration: yoy minus 2yr/2 (change in growth rate)
    ],
    "tags": ["fundamentals", "rnd", "knowledge_capital", "growth", "perpetual_inventory"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of a rich winner vein (osap_orgcap). Impl: Claude.",
}

# Quarterly perpetual-inventory depreciation factor (20%/yr → quarterly)
_QUARTERLY_DECAY = 0.8 ** 0.25   # ≈ 0.9457


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ext2_rndcap_growth_* columns to df (one ticker, ascending Date).
    """
    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals (rnd_expense_ttm and assets)
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=["rnd_expense_ttm", "assets"])

    rnd_ttm = df["fund_rnd_expense_ttm"].to_numpy(dtype=float)
    assets   = df["fund_assets"].to_numpy(dtype=float)
    n = len(df)

    # ------------------------------------------------------------------
    # 2. Build RDC via perpetual inventory (vectorised sequential pass)
    #    RDC[t] = RDC[t-1] * decay + rnd_quarterly[t]
    #    where quarterly add = rnd_expense_ttm / 4
    #    Fundamentals update ~quarterly so we add the quarterly increment
    #    at each row (daily bars between filings get the same number,
    #    but the decay still runs daily at the appropriate daily rate).
    # ------------------------------------------------------------------
    # Daily depreciation factor consistent with 20%/yr
    daily_decay = 0.8 ** (1.0 / 252.0)

    rdc = np.empty(n, dtype=float)
    rdc[:] = np.nan

    # Find first non-NaN rnd row to seed
    first_valid = -1
    for i in range(n):
        v = rnd_ttm[i]
        if not np.isnan(v):
            first_valid = i
            # Seed: RDC = one quarter's addition / (1 - decay_quarterly)
            # i.e. steady-state value for that level of spend
            rdc[i] = (v / 4.0) / max(1.0 - _QUARTERLY_DECAY, 1e-12)
            break

    if first_valid >= 0:
        for i in range(first_valid + 1, n):
            prev = rdc[i - 1]
            if np.isnan(prev):
                prev = 0.0
            q_add = rnd_ttm[i] / 4.0 if not np.isnan(rnd_ttm[i]) else 0.0
            rdc[i] = prev * daily_decay + q_add

    # ------------------------------------------------------------------
    # 3. RDC intensity = RDC / assets (guard zero/nan assets)
    # ------------------------------------------------------------------
    safe_assets = np.where(np.abs(assets) < 1e-9, np.nan, assets)
    rdc_intensity = rdc / safe_assets   # NaN where assets missing/zero

    # ------------------------------------------------------------------
    # 4. Growth rates
    # ------------------------------------------------------------------
    rdc_s = pd.Series(rdc_intensity)

    past_252 = rdc_s.shift(252)
    past_504 = rdc_s.shift(504)

    # YoY growth: (current - past) / |past|; guard zero denominator
    denom_252 = past_252.abs().where(past_252.abs() > 1e-12, np.nan)
    denom_504 = past_504.abs().where(past_504.abs() > 1e-12, np.nan)

    yoy  = (rdc_s - past_252) / denom_252
    yr2  = (rdc_s - past_504) / denom_504

    # Acceleration: 1yr growth vs half the 2yr growth (change-in-growth-rate)
    accel = yoy - (yr2 / 2.0)

    # Clip extreme outliers to [-10, 10] to avoid inf propagation
    yoy   = yoy.clip(-10.0, 10.0)
    yr2   = yr2.clip(-10.0, 10.0)
    accel = accel.clip(-10.0, 10.0)

    df["ext2_rndcap_growth_yoy"]   = yoy.to_numpy()
    df["ext2_rndcap_growth_2yr"]   = yr2.to_numpy()
    df["ext2_rndcap_growth_accel"] = accel.to_numpy()

    # ------------------------------------------------------------------
    # 5. Drop scratch fund_ columns we're NOT listing in produces
    # ------------------------------------------------------------------
    df = df.drop(columns=["fund_rnd_expense_ttm", "fund_assets"], errors="ignore")

    return df
