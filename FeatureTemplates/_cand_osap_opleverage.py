"""
Operating Leverage (Novy-Marx 2011) via PIT fundamentals.

(Cost of Goods Sold + SG&A) / Total Assets, point-in-time via SEC filings.
Cross-sectional in the original paper (long high / short low); implemented here
as a per-ticker time-series of the ratio so it can be used as a feature.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── PIT fundamentals helper ──────────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ────────────────────────────────────────────────────────────────────────────

METADATA = {
    "name": "osap_opleverage",
    "description": (
        "Operating leverage (Novy-Marx 2011): (COGS + SG&A) / Total Assets, "
        "using point-in-time SEC fundamentals (TTM flows, period-end assets). "
        "xsga (SG&A) is treated as 0 when missing, per the original definition. "
        "Produces: level ratio, a 4-quarter rolling change (momentum of the ratio), "
        "and a 2-quarter slope. Per-ticker time-series proxy; cross-sectional rank "
        "interpretation applies at inference time."
    ),
    "requires": [],
    "produces": [
        "osap_opleverage_ratio",
        "osap_opleverage_chg4q",
        "osap_opleverage_slope2q",
    ],
    "tags": ["fundamental", "leverage", "accounting", "novy-marx"],
    "version": "1.0.0",
    "author": "Novy-Marx 2011; OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull needed fundamentals (TTM cost flows + period-end assets)
    df = _fundamentals.as_of(
        df,
        fields=[
            "cost_of_revenue_ttm",   # COGS proxy
            "rnd_expense_ttm",       # part of opex; xsga not directly in schema
            "assets",                # total assets (point-in-time)
            # gross_profit_ttm = revenue_ttm - cogs_ttm, so we can back out cogs
            "gross_profit_ttm",
            "revenue_ttm",
            "operating_income_ttm",
            "gross_margin",
            "operating_margin",
        ],
    )

    # ── Build COGS (cost_of_revenue_ttm is the direct mapping) ──────────────
    cogs = df["fund_cost_of_revenue_ttm"]

    # ── Build SG&A proxy ────────────────────────────────────────────────────
    # The schema has no xsga column directly. Best available proxy:
    #   SG&A ≈ Gross Profit - Operating Income  (both TTM, same period)
    # If that is negative (unusual items), clamp to 0 per Novy-Marx convention.
    gross_profit = df["fund_gross_profit_ttm"]
    op_income = df["fund_operating_income_ttm"]
    sga_proxy = (gross_profit - op_income).clip(lower=0)
    # Where both are NaN, sga_proxy is NaN; treat NaN xsga as 0 per spec
    sga_proxy = sga_proxy.fillna(0.0)

    # ── Total assets ────────────────────────────────────────────────────────
    assets = df["fund_assets"]

    # ── Operating leverage ratio ─────────────────────────────────────────────
    numerator = cogs + sga_proxy
    denom = assets.replace(0, np.nan)
    ol_ratio = numerator / denom

    df["osap_opleverage_ratio"] = ol_ratio

    # ── 4-quarter change (~1-year momentum of the ratio) ─────────────────────
    # Fundamentals update ~quarterly; 63 trading days ≈ 1 quarter → 252 ≈ 4Q
    # Use forward-fill to align sparse filing dates, then diff
    ol_ffill = ol_ratio.ffill()
    df["osap_opleverage_chg4q"] = ol_ffill.diff(252)

    # ── 2-quarter slope (linear slope over ~126 trading days) ────────────────
    window = 126
    half = window // 2
    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    ss_x = (x_c ** 2).sum()

    ol_vals = ol_ffill.to_numpy(dtype=float)
    slope = np.full(len(ol_vals), np.nan)
    for i in range(window - 1, len(ol_vals)):
        seg = ol_vals[i - window + 1 : i + 1]
        if np.isnan(seg).any():
            continue
        slope[i] = (x_c * seg).sum() / ss_x

    df["osap_opleverage_slope2q"] = slope

    # ── Drop scratch fund_* columns not in produces ──────────────────────────
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
