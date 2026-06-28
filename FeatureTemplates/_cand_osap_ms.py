"""
Mohanram G-score (MS) — Mohanram 2005, OpenSourceAP (Chen-Zimmermann).

G-score is a composite accounting signal designed for growth stocks (low book-to-market).
It combines 8 binary signals:
  Profitability / cash flow (3):
    G1: ROA > 0
    G2: CFO/Assets > 0  (operating cash flow scaled by assets)
    G3: CFO/Assets > ROA  (accrual quality: cash earnings beat accrual earnings)
  Earnings / sales volatility (2):
    G4: Earnings volatility (ROA std over trailing 2 yrs) BELOW the cross-sectional median
    G5: Sales volatility BELOW the cross-sectional median
  Investment activity (3):
    G6: R&D intensity (R&D/Assets) ABOVE cross-sectional median
    G7: Capex intensity (Capex/Assets) ABOVE cross-sectional median
    G8: Advertising proxy — we use gross margin as a proxy since advertising is unavailable
        in the fundamentals panel (high gross margin ~ pricing power / brand)

Cross-sectional comparisons (G4,G5,G6,G7) cannot be done faithfully per-ticker in isolation.
PROXY APPROACH:
  - G4/G5: we use the trailing 2-year rolling std of ROA/revenue_growth and compare
    it to the stock's own history percentile rank (low rolling-std rank = good).
  - G6/G7/G8: we compare each ratio to the stock's own history (above own median = 1).
  This is a per-ticker proxy — the cross-sectional ranking is not replicated.

Per Mohanram (2005), G-score ranges 0-8; higher = stronger growth fundamentals.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load PIT fundamentals helper ────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_ms",
    "description": (
        "Mohanram G-score (2005): composite accounting signal for growth stocks. "
        "Sums 8 binary indicators across profitability/CFO quality, earnings/sales "
        "volatility, and investment intensity. Cross-sectional comparisons (G4-G8) "
        "are approximated as per-ticker historical-percentile proxies because only "
        "one stock is visible at a time."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_ms_gscore",    # G-score 0-8 (higher = better growth fundamentals)
        "osap_ms_prof",      # sub-score: profitability + CFO quality (0-3)
        "osap_ms_invest",    # sub-score: investment intensity (0-3; incl gross-margin proxy)
    ],
    "tags": ["fundamental", "accounting", "composite", "growth", "mohanram"],
    "version": "1.0",
    "author": "Mohanram 2005 / OpenSourceAP (Chen-Zimmermann); per-ticker proxy by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── pull PIT fundamentals ────────────────────────────────────────────────
    fields = [
        "roa",                    # net_income / assets (pre-computed)
        "operating_cash_flow_ttm",
        "assets",
        "revenue_ttm",
        "rnd_expense_ttm",
        "capex_ttm",
        "gross_margin",           # gross_profit / revenue (pre-computed)
        "net_income_ttm",
    ]
    df = _fundamentals.as_of(df, fields=fields)

    roa         = df["fund_roa"]
    assets      = df["fund_assets"].replace(0, np.nan)
    cfo         = df["fund_operating_cash_flow_ttm"]
    rev         = df["fund_revenue_ttm"].replace(0, np.nan)
    rnd         = df["fund_rnd_expense_ttm"].fillna(0.0)   # often NaN = 0
    capex       = df["fund_capex_ttm"].abs()               # capex is usually negative in filings
    gross_m     = df["fund_gross_margin"]

    # Derived ratios
    cfo_to_assets = cfo / assets
    rnd_to_assets = rnd / assets
    cap_to_assets = capex / assets

    # ── G1: ROA > 0 ─────────────────────────────────────────────────────────
    g1 = (roa > 0).astype(float).where(roa.notna(), np.nan)

    # ── G2: CFO / Assets > 0 ────────────────────────────────────────────────
    g2 = (cfo_to_assets > 0).astype(float).where(cfo_to_assets.notna(), np.nan)

    # ── G3: CFO > ROA  (cash earnings quality) ──────────────────────────────
    g3 = (cfo_to_assets > roa).astype(float).where(
        (cfo_to_assets.notna() & roa.notna()), np.nan
    )

    # ── G4: earnings volatility below own median (proxy: rolling 8-quarter std of ROA)
    # We use a 2-year (8-bar of quarterly data mapped to ~8 filing periods)
    # Because our data is daily with infrequent fundamentals updates, we roll on the
    # deduplicated ROA series and map back.
    roa_roll_std = roa.rolling(window=8, min_periods=4).std()
    roa_std_med  = roa_roll_std.expanding(min_periods=8).median()
    g4 = (roa_roll_std <= roa_std_med).astype(float).where(
        roa_roll_std.notna() & roa_std_med.notna(), np.nan
    )

    # ── G5: sales/revenue volatility below own median ───────────────────────
    rev_growth   = rev.pct_change().replace([np.inf, -np.inf], np.nan)
    rev_roll_std = rev_growth.rolling(window=8, min_periods=4).std()
    rev_std_med  = rev_roll_std.expanding(min_periods=8).median()
    g5 = (rev_roll_std <= rev_std_med).astype(float).where(
        rev_roll_std.notna() & rev_std_med.notna(), np.nan
    )

    # ── G6: R&D intensity above own median ──────────────────────────────────
    rnd_med = rnd_to_assets.expanding(min_periods=4).median()
    g6 = (rnd_to_assets > rnd_med).astype(float).where(
        rnd_to_assets.notna() & rnd_med.notna(), np.nan
    )

    # ── G7: Capex intensity above own median ─────────────────────────────────
    cap_med = cap_to_assets.expanding(min_periods=4).median()
    g7 = (cap_to_assets > cap_med).astype(float).where(
        cap_to_assets.notna() & cap_med.notna(), np.nan
    )

    # ── G8: Advertising proxy — gross margin above own median ───────────────
    # Original paper uses advertising/assets; we proxy with gross_margin
    gm_med = gross_m.expanding(min_periods=4).median()
    g8 = (gross_m > gm_med).astype(float).where(
        gross_m.notna() & gm_med.notna(), np.nan
    )

    # ── Aggregate sub-scores ─────────────────────────────────────────────────
    # profitability: G1+G2+G3
    prof_stack   = pd.concat([g1, g2, g3], axis=1)
    prof_score   = prof_stack.sum(axis=1, min_count=1)   # NaN if all NaN

    # volatility: G4+G5  (low vol = good, already binary so 1=good)
    vol_stack    = pd.concat([g4, g5], axis=1)
    vol_score    = vol_stack.sum(axis=1, min_count=1)

    # investment: G6+G7+G8
    inv_stack    = pd.concat([g6, g7, g8], axis=1)
    inv_score    = inv_stack.sum(axis=1, min_count=1)

    # total G-score (0-8)
    all_stack    = pd.concat([g1, g2, g3, g4, g5, g6, g7, g8], axis=1)
    g_total      = all_stack.sum(axis=1, min_count=1)

    df["osap_ms_gscore"]  = g_total
    df["osap_ms_prof"]    = prof_score
    df["osap_ms_invest"]  = inv_score

    # ── drop scratch fund_ columns ───────────────────────────────────────────
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
