"""
Intangible investment rate (Peters-Taylor iq).

Organizational capital (OC) is built via perpetual inventory on capitalized SG&A
at 30%/yr depreciation. R&D capital (RDC) is built via perpetual inventory on
capitalized R&D at 20%/yr. The intangible investment rate = current-period
intangible investment / lagged intangible capital (252-day lag). A YoY change
variant is also produced.

Per-ticker PIT fundamentals proxy using _fundamentals.as_of().
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
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_intan_investment_rate",
    "description": (
        "Intangible investment rate (Peters-Taylor iq). "
        "OC = capitalized SG&A via perpetual inventory at 30%/yr (quarterly delta = "
        "1 - 0.7^0.25). RDC = capitalized R&D at 20%/yr (quarterly delta = 1 - 0.8^0.25). "
        "intangible_investment = SGA_flow + RD_flow each period. "
        "Produces: (1) investment_rate = intangible_investment / lag-252 intangible_capital, "
        "(2) yoy change in investment_rate (252-day diff). "
        "All per-ticker using PIT fundamentals; coverage ~84% (ETFs/foreign -> NaN)."
    ),
    "requires": [],
    "produces": [
        "ext2_intan_investment_rate_ir",
        "ext2_intan_investment_rate_ir_yoy",
        "ext2_intan_investment_rate_cap",
    ],
    "tags": ["fundamentals", "intangible", "investment", "organizational_capital", "rnd"],
    "version": "1.0",
    "author": "Round-3 deep exploration of a rich winner vein (osap_orgcap); Peters-Taylor (2017)",
}

# ---------------------------------------------------------------------------
# Perpetual-inventory helpers (vectorized over a pandas Series)
# ---------------------------------------------------------------------------

def _perpetual_inventory(flow_series: pd.Series, delta: float, seed_discount: float) -> pd.Series:
    """
    Given a Series of per-period *additions* (already in the correct periodicity),
    build the perpetual-inventory capital stock:
        K_t = (1 - delta) * K_{t-1} + flow_t
    Seed: K_0 = flow_0 / (seed_discount + delta)  (standard Hall-Jorgenson seed).
    NaN flows are forward-filled from the previous non-NaN capital value (fundamentals
    update infrequently so we carry the last known capital forward).
    Returns a Series aligned to flow_series.index.
    """
    values = flow_series.to_numpy(dtype=np.float64, na_value=np.nan)
    n = len(values)
    capital = np.full(n, np.nan)

    carry = np.nan  # current capital stock

    for i in range(n):
        v = values[i]
        if np.isnan(v):
            # No new data: depreciate whatever we have (if any), no new addition
            if not np.isnan(carry):
                carry = (1.0 - delta) * carry
            capital[i] = carry
        else:
            if np.isnan(carry):
                # Seed the inventory
                denom = seed_discount + delta
                carry = v / denom if denom != 0.0 else np.nan
            else:
                carry = (1.0 - delta) * carry + v
            capital[i] = carry

    return pd.Series(capital, index=flow_series.index)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Perpetual-inventory deltas
    delta_oc = 1.0 - (0.7 ** 0.25)   # ~8.0% per quarter
    delta_rdc = 1.0 - (0.8 ** 0.25)  # ~5.4% per quarter
    seed_g = 0.025  # assumed long-run real growth rate for seed

    # Pull PIT fundamentals (backward-merged, lookahead-safe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=[
                "gross_profit_ttm",
                "operating_income_ttm",
                "rnd_expense_ttm",
            ],
        )

    gp = df["fund_gross_profit_ttm"]
    oi = df["fund_operating_income_ttm"]
    rnd = df["fund_rnd_expense_ttm"]

    # -----------------------------------------------------------------
    # Derive SG&A proxy: GP - EBIT - R&D, clipped >= 0
    # Note: this is only an approximation when D&A is embedded in COGS.
    # -----------------------------------------------------------------
    sga_annual = (gp - oi - rnd.fillna(0.0)).clip(lower=0.0)
    rnd_annual = rnd.clip(lower=0.0)

    # Convert TTM annual flows -> quarterly additions (fundamentals update ~quarterly)
    sga_q = sga_annual / 4.0
    rnd_q = rnd_annual / 4.0

    # -----------------------------------------------------------------
    # Build perpetual-inventory capital stocks (per-ticker series)
    # -----------------------------------------------------------------
    oc = _perpetual_inventory(sga_q, delta=delta_oc, seed_discount=seed_g)
    rdc = _perpetual_inventory(rnd_q, delta=delta_rdc, seed_discount=seed_g)

    intangible_capital = oc + rdc  # total intangible capital stock

    # -----------------------------------------------------------------
    # Intangible *investment* this period = quarterly flow added
    # (SGA_q + RND_q, using the same period's TTM-derived flow)
    # -----------------------------------------------------------------
    intangible_investment = sga_q.fillna(0.0) + rnd_q.fillna(0.0)

    # -----------------------------------------------------------------
    # Investment rate = investment_t / capital_{t-252}
    # (252 trading days ~ 1 year lag on the denominator)
    # -----------------------------------------------------------------
    cap_lag252 = intangible_capital.shift(252)
    cap_lag252_safe = cap_lag252.replace(0.0, np.nan)

    ir = intangible_investment / cap_lag252_safe
    # Clip extreme outliers (startup with tiny old capital can explode)
    ir = ir.clip(lower=-10.0, upper=10.0)

    # YoY change in investment rate (252-day diff)
    ir_yoy = ir - ir.shift(252)

    # -----------------------------------------------------------------
    # Assign produced columns
    # -----------------------------------------------------------------
    df["ext2_intan_investment_rate_ir"] = ir.values
    df["ext2_intan_investment_rate_ir_yoy"] = ir_yoy.values
    df["ext2_intan_investment_rate_cap"] = intangible_capital.values

    # Drop scratch fund_ columns we are not listing in produces
    for col in ["fund_gross_profit_ttm", "fund_operating_income_ttm", "fund_rnd_expense_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
