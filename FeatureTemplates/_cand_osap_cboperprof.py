"""
Cash-based Operating Profitability (Ball et al. 2016)
SPEC ID: osap_cboperprof
SOURCE: OpenSourceAP (Chen-Zimmermann)

Revenue minus COGS minus (SGA - R&D) minus changes in receivables, inventory,
and prepaid expenses, plus changes in deferred revenue (current + LT), accounts
payable, and accrued expenses; all scaled by total assets.

Predicted sign: +1 (long high cash-based op profitability).

Per-ticker implementation: uses PIT fundamentals via _fundamentals.as_of().
The annual-change components are approximated as year-over-year deltas of the
available TTM/annual fields. Cross-sectional ranking is not performed here;
the raw ratio is the signal.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_cboperprof",
    "description": (
        "Cash-based operating profitability (Ball et al. 2016 / OpenSourceAP Chen-Zimmermann). "
        "Numerator = Revenue - COGS - (SGA - R&D) - Δreceivables - Δinventory "
        "+ Δaccounts_payable; scaled by total assets. "
        "Annual changes are approximated from TTM fundamentals lagged by ~252 trading days. "
        "Two extra columns: a 4-quarter rolling z-score (level stability) and "
        "a 1-year slope (trend). Per-ticker; no cross-sectional ranking."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_cboperprof_ratio",    # level: cash op prof / assets
        "osap_cboperprof_zscore",   # rolling 4-period z-score of ratio
        "osap_cboperprof_slope",    # 1-year signed slope of ratio
    ],
    "tags": ["profitability", "fundamentals", "accounting", "cash_based"],
    "version": "1.0",
    "author": "Ball et al. 2016 / OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # 1. Pull point-in-time fundamentals
    # -----------------------------------------------------------------------
    fields = [
        "revenue_ttm",
        "cost_of_revenue_ttm",
        "rnd_expense_ttm",
        "receivables",
        "inventory",
        "assets",
        "operating_cash_flow_ttm",  # fallback cross-check (not used in formula)
        # gross_profit_ttm = revenue - cogs (available; we'll use raw fields)
        "gross_profit_ttm",
        # operating_income_ttm ~ revenue - cogs - sga (proxy when sga unavailable)
        "operating_income_ttm",
        # accounts payable not directly available as standalone field;
        # we approximate accruals via net_income_ttm vs operating_cash_flow_ttm
        "net_income_ttm",
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # -----------------------------------------------------------------------
    # 2. Build cash-based operating profitability ratio
    #
    # Exact Ball et al. formula:
    #   CbOP = [Revt - COGS - (SGA - R&D) - ΔRect - ΔInvt + ΔAP + ΔAccrued] / AT
    #
    # Available mapping:
    #   Revt  -> fund_revenue_ttm
    #   COGS  -> fund_cost_of_revenue_ttm
    #   R&D   -> fund_rnd_expense_ttm  (replace NaN with 0 per spec)
    #   SGA   ~ gross_profit_ttm - operating_income_ttm   (i.e. SGA+other opex)
    #   Δrect -> year-over-year Δ of fund_receivables
    #   Δinvt -> year-over-year Δ of fund_inventory
    #   ΔAP + Δaccrued -> approximated via cash-vs-accrual gap:
    #            (operating_cash_flow_ttm - net_income_ttm) - (-Δrect - Δinvt)
    #            i.e. remaining working-capital cash adjustments
    #   AT    -> fund_assets
    #
    # The SGA proxy: spec says subtract (SGA - R&D). We have
    #   gross_profit_ttm = revenue - COGS
    #   operating_income_ttm = gross_profit - SGA (approx)
    #   => SGA_proxy = gross_profit_ttm - operating_income_ttm
    #   => (SGA - R&D) = gross_profit_ttm - operating_income_ttm - rnd_ttm
    #
    # ΔAP + Δaccrued + Δdeferred_rev proxy:
    #   Total accruals (working-cap) = net_income_ttm - operating_cash_flow_ttm
    #   This includes -ΔAR -ΔInv -ΔPrepaid +ΔAP +Δaccrued (roughly).
    #   So: ΔAP + Δaccrued ~ (ocf_ttm - ni_ttm) + Δrect + Δinvt
    # -----------------------------------------------------------------------

    rev   = df["fund_revenue_ttm"].copy()
    cogs  = df["fund_cost_of_revenue_ttm"].copy()
    rnd   = df["fund_rnd_expense_ttm"].fillna(0.0)
    gp    = df["fund_gross_profit_ttm"].copy()
    oi    = df["fund_operating_income_ttm"].copy()
    rect  = df["fund_receivables"].copy()
    invt  = df["fund_inventory"].fillna(0.0)
    at    = df["fund_assets"].copy()
    ocf   = df["fund_operating_cash_flow_ttm"].copy()
    ni    = df["fund_net_income_ttm"].copy()

    # SGA proxy (replace missing with 0 per spec "replace missing with 0")
    sga_proxy = (gp - oi).fillna(0.0)                 # ≥ 0 normally
    sga_minus_rnd = (sga_proxy - rnd).fillna(0.0)

    # Annual changes: shift by ~252 trading days (approx 1 year of daily rows)
    ANNUAL_LAG = 252
    rect_lag = rect.shift(ANNUAL_LAG)
    invt_lag = invt.shift(ANNUAL_LAG)

    d_rect = rect - rect_lag   # positive = receivables grew (cash drain)
    d_invt = invt - invt_lag   # positive = inventory grew  (cash drain)

    # ΔAP + Δaccrued proxy via cash-accrual gap
    # accruals ≈ NI - OCF  (negative = cash > earnings = favorable)
    # working-cap portion includes: -ΔAR -ΔInv +ΔAP +Δaccrued ...
    # => ΔAP + Δaccrued ≈ (ocf - ni) + ΔAR + ΔInv
    d_rect_safe = d_rect.fillna(0.0)
    d_invt_safe = d_invt.fillna(0.0)
    d_ap_accrued = (ocf - ni).fillna(0.0) + d_rect_safe + d_invt_safe

    # Numerator (replace each term with 0 if missing, per spec)
    rev_safe          = rev.fillna(0.0)
    cogs_safe         = cogs.fillna(0.0)
    sga_minus_rnd_s   = sga_minus_rnd  # already filled above

    numerator = (
        rev_safe
        - cogs_safe
        - sga_minus_rnd_s
        - d_rect_safe
        - d_invt_safe
        + d_ap_accrued
    )

    # Scale by total assets; guard div-by-zero
    at_safe = at.replace(0.0, np.nan)
    ratio = numerator / at_safe
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_cboperprof_ratio"] = ratio

    # -----------------------------------------------------------------------
    # 3. Rolling z-score over ~4 quarters (252 trading days window, min 63)
    # -----------------------------------------------------------------------
    roll_mean = ratio.rolling(window=252, min_periods=63).mean()
    roll_std  = ratio.rolling(window=252, min_periods=63).std()
    roll_std_safe = roll_std.replace(0.0, np.nan)
    zscore = (ratio - roll_mean) / roll_std_safe
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    df["osap_cboperprof_zscore"] = zscore

    # -----------------------------------------------------------------------
    # 4. 1-year slope of ratio (simple linear trend over 252 bars)
    # -----------------------------------------------------------------------
    # Use a vectorised rolling linear regression slope via numpy trick:
    # slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    # For equally spaced x = 0,1,...,n-1:
    #   Σx = n(n-1)/2, Σx² = n(n-1)(2n-1)/6
    SLOPE_WIN = 252
    MIN_SLOPE  = 63

    y = ratio.values
    n_rows = len(y)

    slopes = np.full(n_rows, np.nan)

    # Precompute x-stats for the full window
    n = SLOPE_WIN
    sx  = n * (n - 1) / 2.0
    sx2 = n * (n - 1) * (2 * n - 1) / 6.0
    denom_full = n * sx2 - sx * sx  # constant for full window

    for i in range(n_rows):
        start = i - SLOPE_WIN + 1
        if start < 0:
            # partial window
            w = i + 1
            if w < MIN_SLOPE:
                continue
            yy = y[0: i + 1]
            valid = ~np.isnan(yy)
            w_v = valid.sum()
            if w_v < MIN_SLOPE:
                continue
            idx = np.where(valid)[0].astype(float)
            yv = yy[valid]
            sw = w_v
            six = idx.sum()
            six2 = (idx ** 2).sum()
            siy = yv.sum()
            sixy = (idx * yv).sum()
            d = sw * six2 - six * six
            if d == 0:
                continue
            slopes[i] = (sw * sixy - six * siy) / d
        else:
            yy = y[start: i + 1]
            valid = ~np.isnan(yy)
            w_v = valid.sum()
            if w_v < MIN_SLOPE:
                continue
            if w_v == n:
                # fast path: all valid, x = 0..n-1
                siy  = yy.sum()
                sixy = np.dot(np.arange(n, dtype=float), yy)
                if denom_full == 0:
                    continue
                slopes[i] = (n * sixy - sx * siy) / denom_full
            else:
                idx = np.where(valid)[0].astype(float)
                yv = yy[valid]
                sw = w_v
                six = idx.sum()
                six2 = (idx ** 2).sum()
                siy = yv.sum()
                sixy = (idx * yv).sum()
                d = sw * six2 - six * six
                if d == 0:
                    continue
                slopes[i] = (sw * sixy - six * siy) / d

    df["osap_cboperprof_slope"] = slopes

    # -----------------------------------------------------------------------
    # 5. Drop scratch fund_* columns not in produces
    # -----------------------------------------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
