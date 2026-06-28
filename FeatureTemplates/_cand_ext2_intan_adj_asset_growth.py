"""
ext2_intan_adj_asset_growth
Intangible-adjusted asset growth (investment anomaly, sign -1).

Extends the osap_orgcap organizational-capital idea by building a *total capital*
base (book assets + capitalized SG&A + capitalized R&D) and recomputing the
asset-growth anomaly on that broader base.

Source: Round-3 deep exploration of a rich winner vein (osap_orgcap)
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_intan_adj_asset_growth",
    "description": (
        "Intangible-adjusted asset growth (investment anomaly, sign -1). "
        "Builds Total Capital = book assets + OC (perpetual-inventory of SG&A at 30%/yr decay) "
        "+ RDC (perpetual-inventory of R&D at 20%/yr decay). "
        "Produces: YoY growth of TC (252-day lag), the spread between TC-growth and plain "
        "asset growth, and 2-yr TC growth (504-day lag). "
        "Expected sign -1: high total-capital growth -> lower forward returns. "
        "Per-ticker PIT fundamentals proxy (cross-sectional rank not available here)."
    ),
    "requires": [],
    "produces": [
        "ext2_intan_adj_asset_growth_tc_yoy",      # YoY growth of Total Capital
        "ext2_intan_adj_asset_growth_spread",       # TC_growth - plain asset_growth
        "ext2_intan_adj_asset_growth_tc_2yr",       # 2-yr growth of Total Capital
    ],
    "tags": ["fundamentals", "investment", "anomaly", "intangibles", "asset_growth"],
    "version": "1.0",
    "author": "Round-3 deep exploration of a rich winner vein (osap_orgcap)",
}


# ---------------------------------------------------------------------------
# perpetual-inventory helper (vectorised over a pandas Series)
# ---------------------------------------------------------------------------
def _perp_inv(quarterly_flow: pd.Series, annual_decay_rate: float) -> pd.Series:
    """
    Perpetual-inventory capital stock from a quarterly flow series.

    delta = 1 - (1 - annual_decay_rate)^0.25
    OC_t = (1-delta)*OC_{t-1} + flow_t
    seed = flow_0 / (0.025 + delta)

    Works on a DatetimeIndex-agnostic 1-D series; NaN propagates cleanly.
    """
    delta = 1.0 - (1.0 - annual_decay_rate) ** 0.25
    surv = 1.0 - delta

    vals = quarterly_flow.to_numpy(dtype=float)
    n = len(vals)
    stock = np.full(n, np.nan)

    # find first non-NaN to seed
    first_valid = None
    for i in range(n):
        if not np.isnan(vals[i]):
            first_valid = i
            break

    if first_valid is None:
        return pd.Series(stock, index=quarterly_flow.index)

    stock[first_valid] = vals[first_valid] / (0.025 + delta)

    for i in range(first_valid + 1, n):
        if np.isnan(vals[i]):
            # carry forward existing stock with decay; if stock already NaN keep NaN
            if not np.isnan(stock[i - 1]):
                stock[i] = surv * stock[i - 1]
        else:
            if np.isnan(stock[i - 1]):
                stock[i] = vals[i] / (0.025 + delta)
            else:
                stock[i] = surv * stock[i - 1] + vals[i]

    return pd.Series(stock, index=quarterly_flow.index)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals (backward merge_asof by filed_date inside as_of)
    df = _fundamentals.as_of(
        df,
        fields=[
            "assets",
            "gross_profit_ttm",
            "operating_income_ttm",
            "rnd_expense_ttm",
        ],
    )

    a = df["fund_assets"].to_numpy(dtype=float)
    gp = df["fund_gross_profit_ttm"].to_numpy(dtype=float)
    oi = df["fund_operating_income_ttm"].to_numpy(dtype=float)
    rnd = df["fund_rnd_expense_ttm"].to_numpy(dtype=float)

    # Quarterly SG&A proxy: (gross_profit - operating_income - R&D) / 4
    # clipped to >= 0
    sga_q = np.where(
        np.isnan(gp) | np.isnan(oi) | np.isnan(rnd),
        np.nan,
        np.clip(gp - oi - rnd, 0.0, None) / 4.0,
    )

    # Quarterly R&D add: rnd_ttm / 4
    rnd_q = np.where(np.isnan(rnd), np.nan, rnd / 4.0)

    sga_q_s = pd.Series(sga_q, index=df.index)
    rnd_q_s = pd.Series(rnd_q, index=df.index)

    # Perpetual-inventory stocks
    oc = _perp_inv(sga_q_s, annual_decay_rate=0.30).to_numpy()
    rdc = _perp_inv(rnd_q_s, annual_decay_rate=0.20).to_numpy()

    # Total Capital = assets + OC + RDC
    tc = np.where(np.isnan(a), np.nan, a + np.nan_to_num(oc, nan=0.0) + np.nan_to_num(rdc, nan=0.0))
    # If assets is NaN, TC is NaN; OC/RDC missing => treat as 0 (conservative)

    tc_s = pd.Series(tc, index=df.index)
    a_s = pd.Series(a, index=df.index)

    # YoY growth (252 trading-day lag)
    LAG1 = 252
    LAG2 = 504  # ~2 years

    tc_lag1 = tc_s.shift(LAG1)
    tc_lag2 = tc_s.shift(LAG2)
    a_lag1 = a_s.shift(LAG1)

    # growth = (current - lag) / abs(lag); guard zero/nan denom
    def _safe_growth(curr: pd.Series, lag: pd.Series) -> pd.Series:
        denom = lag.abs().replace(0.0, np.nan)
        return (curr - lag) / denom

    tc_yoy = _safe_growth(tc_s, tc_lag1)
    tc_2yr = _safe_growth(tc_s, tc_lag2)
    asset_yoy = _safe_growth(a_s, a_lag1)

    spread = tc_yoy - asset_yoy

    # Replace inf/-inf with NaN
    def _clean(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan)

    df["ext2_intan_adj_asset_growth_tc_yoy"] = _clean(tc_yoy).values
    df["ext2_intan_adj_asset_growth_spread"] = _clean(spread).values
    df["ext2_intan_adj_asset_growth_tc_2yr"] = _clean(tc_2yr).values

    # Drop scratch fund_ columns
    scratch = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch)

    return df
