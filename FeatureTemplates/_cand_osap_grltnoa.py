"""
Feature block: osap_grltnoa
Growth in Long-Term Net Operating Assets (minus accruals)
Source: Fairfield, Whisenant and Yohn (2003) via OpenSourceAP (Chen-Zimmermann)
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
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
    "name": "osap_grltnoa",
    "description": (
        "Growth in long-term net operating assets (LTNOA) minus operating accruals, "
        "following Fairfield, Whisenant & Yohn (2003) / OpenSourceAP. "
        "LTNOA = (receivables + inventory + ppe_net + current_assets_other + goodwill "
        "+ other_assets - accounts_payable_proxy - current_liabilities_other "
        "- other_long_term_liabilities_proxy) / total_assets. "
        "Accruals = change in working-capital items minus depreciation (proxied from "
        "ppe_net change), scaled by average assets. "
        "Produces: (1) osap_grltnoa_ltnoa -- level of LTNOA ratio, "
        "(2) osap_grltnoa_main -- annual growth in LTNOA minus accruals (the signal; "
        "higher = more aggressive investment/accruals, per-paper predicted sign +1 "
        "meaning high LTNOA growth is NEGATIVE for returns in cross-section, but here "
        "reported as-computed so downstream model learns the sign), "
        "(3) osap_grltnoa_accruals -- the accruals component alone. "
        "NOTE: ap, aco, ao, lco, lo, dp are not individually available; proxied using "
        "assets_current, liabilities_current, goodwill, long_term_debt from PIT "
        "fundamentals. Coverage ~84%; ETFs/foreign tickers will be all-NaN."
    ),
    "requires": ["Close"],
    "produces": ["osap_grltnoa_ltnoa", "osap_grltnoa_main", "osap_grltnoa_accruals"],
    "tags": ["fundamentals", "accounting", "investment", "accruals", "operating_assets"],
    "version": "1.0",
    "author": "Fairfield, Whisenant & Yohn (2003) via OpenSourceAP / Chen-Zimmermann",
}

# ---------------------------------------------------------------------------
# Helper: safe division
# ---------------------------------------------------------------------------
def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    d = denom.replace(0, np.nan)
    return num / d


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals needed for the signal
    # We need as many balance-sheet items as possible to proxy the LTNOA formula:
    #   rect     -> receivables
    #   invt     -> inventory
    #   ppent    -> ppe_net  (net PP&E; already net of depreciation)
    #   aco      -> assets_current - receivables - inventory - cash  (approx)
    #   ao       -> assets - assets_current - goodwill - ppe_net     (approx other LT assets)
    #   ap       -> not available; estimated as liabilities_current * fraction (set 0 -- missing)
    #   lco      -> liabilities_current (best available proxy for current op. liabilities incl AP)
    #   lo       -> liabilities - liabilities_current - long_term_debt (approx)
    #   intan    -> goodwill (closest available; omits other intangibles)
    #   at       -> assets
    #   dp       -> depreciation proxied by change in ppe_net (ppent_t-1 - ppent_t, floored at 0)
    df = _fundamentals.as_of(
        df,
        fields=[
            "receivables",
            "inventory",
            "ppe_net",
            "goodwill",
            "assets",
            "assets_current",
            "cash",
            "liabilities_current",
            "liabilities",
            "long_term_debt",
        ],
    )

    # ------------------------------------------------------------------
    # Build LTNOA components
    # ------------------------------------------------------------------
    rect   = df["fund_receivables"].copy()
    invt   = df["fund_inventory"].copy()
    ppent  = df["fund_ppe_net"].copy()
    intan  = df["fund_goodwill"].copy()
    at     = df["fund_assets"].copy()
    cur_a  = df["fund_assets_current"].copy()
    cash   = df["fund_cash"].copy()
    cur_l  = df["fund_liabilities_current"].copy()
    tot_l  = df["fund_liabilities"].copy()
    ltd    = df["fund_long_term_debt"].copy()

    # aco: other current assets = current_assets - receivables - inventory - cash
    aco = (cur_a - rect - invt - cash).clip(lower=0)

    # ao: other long-term assets = total_assets - current_assets - goodwill - ppe_net
    ao = (at - cur_a - intan - ppent).clip(lower=0)

    # ap proxy: not directly available; we treat ap = 0 (conservative; absorbed into lco)
    # lco: current operating liabilities ≈ liabilities_current (includes ap)
    lco = cur_l.copy()

    # lo: other long-term liabilities = total_liabilities - current_liabilities - LTD
    lo = (tot_l - cur_l - ltd).clip(lower=0)

    # ap = 0 (absorbed into lco above; no separate field)
    ap = pd.Series(0.0, index=df.index)

    # LTNOA (raw numerator)
    noa_num = rect + invt + ppent + aco + intan + ao - ap - lco - lo
    ltnoa = _safe_div(noa_num, at)

    # ------------------------------------------------------------------
    # Annual growth in LTNOA
    # ------------------------------------------------------------------
    # Fundamentals are filed quarterly/annually; shift(1) gives prior filing row
    # We use shift(4) to approximate a year-ago value (4 filing periods ≈ 1 year)
    # If data is sparser, shift(1) is acceptable for annual filings.
    # Use shift(4) with fallback to shift(1) where shift(4) is NaN.
    ltnoa_lag4 = ltnoa.shift(4)
    ltnoa_lag1 = ltnoa.shift(1)
    ltnoa_lag = ltnoa_lag4.where(ltnoa_lag4.notna(), ltnoa_lag1)

    grltnoa = ltnoa - ltnoa_lag  # growth in LTNOA (level difference)

    # ------------------------------------------------------------------
    # Operating Accruals
    # Accruals = ( Δrect + Δinvt + Δaco - Δap - Δlco - dp ) / avg_at
    # dp ≈ max(ppent_lag - ppent, 0)  [depreciation reduces ppe_net]
    # ------------------------------------------------------------------
    rect_lag   = rect.shift(4).where(rect.shift(4).notna(), rect.shift(1))
    invt_lag   = invt.shift(4).where(invt.shift(4).notna(), invt.shift(1))
    aco_lag    = aco.shift(4).where(aco.shift(4).notna(), aco.shift(1))
    ap_lag     = ap.shift(4).where(ap.shift(4).notna(), ap.shift(1))
    lco_lag    = lco.shift(4).where(lco.shift(4).notna(), lco.shift(1))
    at_lag     = at.shift(4).where(at.shift(4).notna(), at.shift(1))
    ppent_lag  = ppent.shift(4).where(ppent.shift(4).notna(), ppent.shift(1))

    # Depreciation proxy: decline in ppe_net (gross capex not available here)
    dp = (ppent_lag - ppent).clip(lower=0)

    delta_rect = rect - rect_lag
    delta_invt = invt - invt_lag
    delta_aco  = aco  - aco_lag
    delta_ap   = ap   - ap_lag    # zero series
    delta_lco  = lco  - lco_lag

    accruals_num = delta_rect + delta_invt + delta_aco - delta_ap - delta_lco - dp
    avg_at = (at + at_lag) / 2.0
    accruals = _safe_div(accruals_num, avg_at)

    # ------------------------------------------------------------------
    # Main signal: growth in LTNOA minus accruals
    # ------------------------------------------------------------------
    grltnoa_main = grltnoa - accruals

    # ------------------------------------------------------------------
    # Replace inf/-inf with NaN; assign output columns
    # ------------------------------------------------------------------
    df["osap_grltnoa_ltnoa"]    = ltnoa.replace([np.inf, -np.inf], np.nan)
    df["osap_grltnoa_main"]     = grltnoa_main.replace([np.inf, -np.inf], np.nan)
    df["osap_grltnoa_accruals"] = accruals.replace([np.inf, -np.inf], np.nan)

    # Drop intermediate fund_ columns we are NOT producing
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
