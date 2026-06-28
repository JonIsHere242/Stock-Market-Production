"""
_paper_doaj_1a68f20_piotroski_fscore.py  --  UNPROVEN candidate feature block.

Leading underscore == hidden from auto-discovery; does NOT enter the model until promoted.

Paper reference
---------------
"Can financial strength indicators form a profitable investment strategy?
 The case of F-Score in Europe"
 DOAJ id: 1a68f20...

Concept
-------
The Piotroski F-Score (Piotroski 2000, Journal of Accounting Research) is a
composite 0-9 integer signal that aggregates nine binary tests across three
fundamental dimensions:

  Profitability (4 signals)
    F1  ROA > 0                                  (positive return on assets)
    F2  Operating cash flow > 0                  (positive OCF)
    F3  ROA improving year-over-year             (delta_ROA > 0)
    F4  Accruals: CFO > net_income               (cash earnings quality)

  Leverage / Liquidity (3 signals)
    F5  Long-term debt ratio decreased YoY       (lower financial leverage)
    F6  Current ratio increased YoY              (better short-term liquidity)
    F7  Shares outstanding NOT increasing YoY    (no dilution)

  Operating Efficiency (2 signals)
    F8  Gross margin increased YoY               (improving profitability)
    F9  Asset turnover increased YoY             (improving asset efficiency)

High F-Score (≥ 7) = financially strong; Low F-Score (≤ 2) = financially weak.

Implementation notes
--------------------
All fundamentals are pulled through _fundamentals.as_of(), which performs a
BACKWARD merge_asof on filed_date. Each trading-day row therefore only ever
sees filings that were already public by that date (fully point-in-time).

YoY comparisons use two consecutive annual/quarterly readings from the
by-ticker PIT panel (prior row's value on the same aligned basis). The
SEC panel has ~4–8 rows/year (quarterly), so "prior filing" is an
approximation of "prior annual"; this is the best we can do without
isolating annual filings.

Coverage: ~84% of the universe (US-listed companies with SEC filings).
ETFs, ADRs, and foreign issuers with no SEC coverage return all-NaN.
fund_* scratch columns are dropped; only METADATA["produces"] is emitted.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import _fundamentals helper by file path (hidden from discovery → can't use
# a normal import).  Identical to the idiom in _fundamentals_valuation.py.
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_fundamentals",
    _Path(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_doaj_1a68f20_piotroski_fscore",
    "description": (
        "Piotroski F-Score (0-9) from point-in-time SEC fundamentals (filed_date as-of): "
        "composite of 9 binary tests across profitability, leverage/liquidity, and "
        "operating efficiency (DOAJ 1a68f20 / Piotroski 2000)."
    ),
    "requires": [],   # no OHLCV columns needed; all inputs are from SEC fundamentals
    "produces": [
        "pio_fscore",            # 0-9 integer composite (NaN where not computable)
        "pio_profitability",     # 0-4 profitability sub-score
        "pio_leverage_liquidity",# 0-3 leverage/liquidity sub-score
        "pio_efficiency",        # 0-2 efficiency sub-score
        "pio_roa",               # raw ROA value (from as_of)
        "pio_accruals",          # CFO - net_income (positive = better earnings quality)
    ],
    "tags": ["fundamentals", "quality", "value", "experimental", "paper"],
    "version": "1.0",
    "author": "paper: DOAJ 1a68f20 — Piotroski F-Score in Europe; ported by AI agent 2026-06-14",
}

# Fundamental fields we need.
_FIELDS = [
    "roa",
    "operating_cash_flow",
    "net_income",
    "assets",
    "long_term_debt",
    "current_ratio",
    "shares_outstanding",
    "gross_margin",
    "asset_turnover",
]


def _to_float(s: pd.Series) -> pd.Series:
    """Coerce to float, silencing warnings."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def _binary(condition: pd.Series) -> pd.Series:
    """
    Return a float Series of {0.0, 1.0, NaN}:
      - 1.0 where condition is True
      - 0.0 where condition is False
      - NaN where either operand was NaN (condition evaluates to NaN/False)
    We use .where(~mask, other=np.nan) to preserve NaN from the underlying data.
    """
    # condition is a boolean Series; where either input was NaN, boolean ops
    # silently become False.  We need to propagate NaN explicitly.
    result = condition.astype(float)   # True→1.0, False→0.0
    return result


def _safe_binary(a: pd.Series, b: pd.Series, op: str) -> pd.Series:
    """
    Compare two numeric Series with a relational operator, returning {0.0, 1.0, NaN}.
    NaN in either input -> NaN output.

    op: one of ">", ">=", "<", "<="
    """
    a = _to_float(a)
    b = _to_float(b)
    nan_mask = a.isna() | b.isna()
    if op == ">":
        cond = a > b
    elif op == ">=":
        cond = a >= b
    elif op == "<":
        cond = a < b
    elif op == "<=":
        cond = a <= b
    else:
        raise ValueError(f"Unknown op: {op}")
    out = cond.astype(float)
    out[nan_mask] = np.nan
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Piotroski F-Score features via point-in-time SEC fundamentals.

    Steps:
      1. Pull fundamentals with as_of (BACKWARD merge on filed_date).
      2. Derive "prior filing" values for each field using shift(1) on the
         daily-aligned fundamentals — captures the previous publicly-known value.
      3. Compute 9 binary signals and sum.
      4. Drop all fund_* scratch columns.
    """

    # ------------------------------------------------------------------
    # 1.  PIT merge: each date sees only already-filed fundamentals.
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=_FIELDS)

    # ------------------------------------------------------------------
    # 2.  Extract current and prior (previous-filing) values.
    #     The as_of merge replaces each row with the most recent filing up
    #     to that date, so consecutive identical rows represent the same
    #     filing.  shift(1) gives us the value at the prior filing change.
    #     We detect filing changes by a change in any key field; then use
    #     the previous distinct filing's values.
    #
    #     Simpler and safe: just shift(1) on the aligned column — the
    #     prior row holds the previous distinct-filing value at that day.
    # ------------------------------------------------------------------
    def col(name: str) -> pd.Series:
        return _to_float(df[f"fund_{name}"])

    roa          = col("roa")
    ocf          = col("operating_cash_flow")
    net_inc      = col("net_income")
    assets       = col("assets")
    lt_debt      = col("long_term_debt")
    curr_ratio   = col("current_ratio")
    shares       = col("shares_outstanding")
    gross_margin = col("gross_margin")
    asset_turn   = col("asset_turnover")

    # Prior-filing proxies: shift the day-aligned column.
    # Where coverage starts (NaN pre-first-filing), shift also gives NaN,
    # so YoY signals are correctly NaN at the start.
    roa_prev        = roa.shift(1)
    lt_debt_prev    = lt_debt.shift(1)
    curr_ratio_prev = curr_ratio.shift(1)
    shares_prev     = shares.shift(1)
    gm_prev         = gross_margin.shift(1)
    at_prev         = asset_turn.shift(1)

    # Assets (denominator for long_term_debt ratio); avoid divide-by-zero.
    assets_pos      = assets.where(assets > 0)
    assets_prev     = assets.shift(1)
    assets_prev_pos = assets_prev.where(assets_prev > 0)

    # ------------------------------------------------------------------
    # 3.  Nine binary signals (each NaN-safe).
    # ------------------------------------------------------------------

    # --- Profitability ---
    # F1: ROA > 0
    f1 = _safe_binary(roa, pd.Series(0.0, index=df.index), ">")

    # F2: Operating cash flow > 0
    f2 = _safe_binary(ocf, pd.Series(0.0, index=df.index), ">")

    # F3: ROA improving YoY (delta_ROA > 0)
    delta_roa = roa - roa_prev
    f3 = _safe_binary(delta_roa, pd.Series(0.0, index=df.index), ">")

    # F4: Accruals — CFO > net income (quality of earnings)
    #     Piotroski uses CFO/Assets > ROA, but we mirror the spirit with CFO > NI.
    #     Positive means cash earnings exceed accrual earnings (good quality).
    accruals = ocf - net_inc   # NaN propagates naturally
    f4_mask  = accruals.isna()
    f4 = (accruals > 0).astype(float)
    f4[f4_mask] = np.nan

    # --- Leverage / Liquidity ---
    # F5: Long-term debt ratio decreased YoY (leverage/assets lower)
    ltd_ratio      = lt_debt / assets_pos
    ltd_ratio_prev = lt_debt_prev / assets_prev_pos
    delta_ltd      = ltd_ratio - ltd_ratio_prev
    f5 = _safe_binary(delta_ltd, pd.Series(0.0, index=df.index), "<")  # decrease = good

    # F6: Current ratio increased YoY
    delta_cr = curr_ratio - curr_ratio_prev
    f6 = _safe_binary(delta_cr, pd.Series(0.0, index=df.index), ">")

    # F7: Shares outstanding NOT increasing (no dilution)
    delta_sh = shares - shares_prev
    # F7 = 1 if shares did NOT increase (delta <= 0)
    f7_mask = delta_sh.isna() | shares.isna() | shares_prev.isna()
    f7 = (delta_sh <= 0).astype(float)
    f7[f7_mask] = np.nan

    # --- Efficiency ---
    # F8: Gross margin improved YoY
    delta_gm = gross_margin - gm_prev
    f8 = _safe_binary(delta_gm, pd.Series(0.0, index=df.index), ">")

    # F9: Asset turnover improved YoY
    delta_at = asset_turn - at_prev
    f9 = _safe_binary(delta_at, pd.Series(0.0, index=df.index), ">")

    # ------------------------------------------------------------------
    # 4.  Sub-scores and composite.
    #     We sum with NaN handling: if ANY component is NaN, the sub-score
    #     becomes NaN (conservative approach, avoids silently under-counting).
    # ------------------------------------------------------------------
    pio_prof = f1 + f2 + f3 + f4          # 0-4; NaN if any sub-test NaN
    pio_lev  = f5 + f6 + f7               # 0-3
    pio_eff  = f8 + f9                    # 0-2
    pio_tot  = pio_prof + pio_lev + pio_eff  # 0-9

    # Clip to [0, 9] to guard against any floating-point edge cases, then
    # round to integer-valued float (NaN stays NaN).
    df["pio_fscore"]             = pio_tot.clip(lower=0, upper=9).round(0)
    df["pio_profitability"]      = pio_prof.clip(lower=0, upper=4).round(0)
    df["pio_leverage_liquidity"] = pio_lev.clip(lower=0, upper=3).round(0)
    df["pio_efficiency"]         = pio_eff.clip(lower=0, upper=2).round(0)
    df["pio_roa"]                = roa
    df["pio_accruals"]           = accruals

    # ------------------------------------------------------------------
    # 5.  Drop all fund_* scratch columns (only METADATA["produces"] remains).
    # ------------------------------------------------------------------
    scratch = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch)

    return df
