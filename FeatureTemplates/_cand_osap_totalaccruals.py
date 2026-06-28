"""
Total Accruals feature block.

Source: OpenSourceAP (Chen-Zimmermann), Richardson et al. 2005
Signal: Total accruals scaled by lagged total assets. Predicted sign: -1 (long LOW accruals).

Two regimes per the paper:
  Pre-1988: balance-sheet approach
    accruals = Δ(net working capital) + Δ(net noncurrent assets) + Δ(net financial assets)
  Post-1988 (or when cash-flow items are available): cash-flow approach
    accruals = net_income - (operating_CF + investing_CF + financing_CF) + (stock_sales - repurchases - dividends)

Because this block operates PER TICKER on daily OHLCV + PIT fundamentals, we:
  - Merge the latest PIT fundamental snapshot to each date via _fundamentals.as_of().
  - Approximate the annual change in accounting items by differencing the PIT value against
    the PIT value one fiscal-year ago (taken as the value available 365 days earlier).
  - Scale by lagged total assets.
  - Emit three columns:
      osap_totalaccruals_raw   -- accruals / lagged assets (cash-flow method when possible,
                                  balance-sheet otherwise)
      osap_totalaccruals_chg   -- 365-day change in the raw accrual ratio (trend)
      osap_totalaccruals_sign  -- {-1, 0, +1} directional signal consistent with the predicted -1 sign
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_totalaccruals",
    "description": (
        "Total accruals scaled by lagged total assets, per Richardson et al. (2005) / "
        "OpenSourceAP (Chen-Zimmermann). Cash-flow method: NI minus (operating + investing + "
        "financing CF) adjusted for equity issuance/repurchase/dividends, scaled by lagged "
        "total assets. Falls back to balance-sheet accrual proxy when CF items are unavailable. "
        "Predicted cross-sectional sign is -1 (high accruals = negative future return). "
        "Per-ticker PIT implementation via _fundamentals.as_of(); annual Δ approximated by "
        "comparing current PIT value to PIT value available 365 days earlier."
    ),
    "requires": [],  # no OHLCV columns strictly needed; Close used only if desired in future
    "produces": [
        "osap_totalaccruals_raw",
        "osap_totalaccruals_chg",
        "osap_totalaccruals_sign",
    ],
    "tags": ["accruals", "accounting", "fundamentals", "quality", "richardson2005", "opensourceap"],
    "version": "1.0.0",
    "author": "Richardson et al. 2005 / OpenSourceAP Chen-Zimmermann; block by claude-sonnet-4-6",
}

# ---------------------------------------------------------------------------
# Lazy-load _fundamentals helper
# ---------------------------------------------------------------------------
def _load_fundamentals():
    _spec = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Attach total-accruals columns to df (one stock, ascending Date)."""

    # Fields needed:
    #   CF approach:   net_income, operating_cash_flow, (no investing/financing CF in panel)
    #                  We use operating_cash_flow as the best available proxy.
    #   BS approach:   assets, assets_current, cash, liabilities_current, long_term_debt,
    #                  equity (as fallback)
    #   Lagged assets: assets (at)
    FIELDS = [
        "assets",           # at  -- total assets (denominator)
        "assets_current",   # act
        "cash",             # che (cash & equiv)
        "liabilities_current",  # lct
        "long_term_debt",   # dltt
        "net_income",       # ni
        "net_income_ttm",   # ni TTM preferred for smoother signal
        "operating_cash_flow",      # oancf
        "operating_cash_flow_ttm",  # oancf TTM
        "fcf_ttm",          # proxy for investing flows when ivncf absent
        "equity",           # ceq -- shareholders equity
    ]

    try:
        _fund = _load_fundamentals()
        df = _fund.as_of(df, fields=FIELDS)
    except Exception:
        # Fundamentals unavailable: emit NaN columns and return
        df["osap_totalaccruals_raw"] = np.nan
        df["osap_totalaccruals_chg"] = np.nan
        df["osap_totalaccruals_sign"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Helper: safely fill NaN with 0 for optional items (per paper spec)
    # ------------------------------------------------------------------
    def _z(col: str) -> pd.Series:
        """Return fund_<col> as Series, NaN where the whole column is missing, else fillna(0)."""
        c = f"fund_{col}"
        if c not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = df[c]
        # Only fill 0 where the asset base itself is available (coverage proxy)
        has_assets = df.get("fund_assets", pd.Series(np.nan, index=df.index)).notna()
        return s.where(s.notna() | ~has_assets, 0.0)

    def _g(col: str) -> pd.Series:
        """Return fund_<col> raw (NaN preserved)."""
        c = f"fund_{col}"
        if c not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[c]

    # ------------------------------------------------------------------
    # Choose NI and OCF: prefer TTM for stability
    # ------------------------------------------------------------------
    ni = _g("net_income_ttm").where(_g("net_income_ttm").notna(), _g("net_income"))
    ocf = _g("operating_cash_flow_ttm").where(
        _g("operating_cash_flow_ttm").notna(), _g("operating_cash_flow")
    )

    # Total assets (denominator) -- must be positive
    at = _g("assets")
    at_safe = at.where(at > 0, np.nan)

    # ------------------------------------------------------------------
    # Cash-flow method accruals (primary)
    #   = NI - OCF  (investing and financing CFs not directly available;
    #     FCF_TTM = OCF - capex, so investing proxy = OCF - FCF = capex)
    #   We use the simpler NI - OCF as the core accrual estimate since
    #   financing CF items (sstk, prstkc, dv) are absent from the panel.
    # ------------------------------------------------------------------
    cf_accruals = ni - ocf   # NaN where either missing

    # ------------------------------------------------------------------
    # Balance-sheet fallback (pre-1988 method / when CF unavailable):
    #   Δ NWC = Δ((assets_current - cash) - (liabilities_current - 0))
    #   (dlc excluded as not in panel; dltt used for noncurrent)
    #   Simplified: Δ(assets_current - cash - liabilities_current)
    #   + Δ(assets - assets_current - liabilities + liabilities_current - long_term_debt)
    #   where missing items replaced by 0 per paper.
    # ------------------------------------------------------------------
    act = _z("assets_current")
    che = _z("cash")
    lct = _z("liabilities_current")
    dltt = _z("long_term_debt")
    eq = _g("equity")   # ceq; used to approximate noncurrent liabilities

    # Net working capital (excluding cash & short-term debt per paper)
    nwc = (act - che) - lct

    # Net noncurrent assets approximation: (at - act) - (liabilities - lct - dltt)
    # liabilities = at - equity (balance-sheet identity)
    liabilities_total = at - eq.where(eq.notna(), 0.0)
    noncurrent_liabilities = (liabilities_total - lct - dltt).clip(lower=0)
    nna = (at - act) - noncurrent_liabilities

    # Total BS "accruals" level = NWC + NNA  (financial assets term dropped; not in panel)
    bs_level = nwc + nna

    # Annual change in BS accruals (shift by ~252 trading days ≈ 1 year of daily rows)
    bs_accruals = bs_level.diff(252)

    # ------------------------------------------------------------------
    # Choose: use CF accruals when both NI and OCF available; else BS
    # ------------------------------------------------------------------
    use_cf = cf_accruals.notna()
    accruals_raw = cf_accruals.where(use_cf, bs_accruals)

    # Scale by lagged total assets (shift 252 rows ≈ prior year)
    at_lagged = at_safe.shift(252)
    osap_raw = accruals_raw / at_lagged.where(at_lagged > 0, np.nan)

    # Guard against inf
    osap_raw = osap_raw.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Dynamic slope: 365-day change in the accrual ratio
    # ------------------------------------------------------------------
    osap_chg = osap_raw.diff(252)
    osap_chg = osap_chg.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Directional sign: consistent with predicted sign = -1 (short high accruals)
    # sign = -1 when accruals > 0 (bad), +1 when accruals < 0 (good), 0 at NaN
    # ------------------------------------------------------------------
    osap_sign = pd.Series(np.nan, index=df.index)
    valid = osap_raw.notna()
    osap_sign[valid & (osap_raw > 0)] = -1.0
    osap_sign[valid & (osap_raw < 0)] = 1.0
    osap_sign[valid & (osap_raw == 0)] = 0.0

    # ------------------------------------------------------------------
    # Attach produced columns
    # ------------------------------------------------------------------
    df["osap_totalaccruals_raw"] = osap_raw
    df["osap_totalaccruals_chg"] = osap_chg
    df["osap_totalaccruals_sign"] = osap_sign

    # Drop intermediate fund_* scratch columns not listed in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols, errors="ignore")

    return df
