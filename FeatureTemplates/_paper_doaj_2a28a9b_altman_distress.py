"""
_paper_doaj_2a28a9b_altman_distress.py  --  UNPROVEN candidate; hidden from auto-discovery.

Paper reference
---------------
"Developing Multifactor Asset Pricing Models Using Threshold Regression Approach and
Credit Risk Factor" (DOAJ ID: 2a28a9b...). The paper augments a standard asset-pricing
model with a financial-distress / credit-risk factor and shows it improves cross-sectional
return prediction. This block operationalises the credit-risk component as a per-ticker
PIT Altman Z-score proxy constructed from SEC EDGAR fundamentals via `_fundamentals.as_of`.

Altman Z-score (public-market variant, Altman 1968 / 2000 revision)
--------------------------------------------------------------------
Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

where:
  X1 = working_capital / total_assets
  X2 = retained_earnings / total_assets
  X3 = EBIT / total_assets
  X4 = market_cap / total_liabilities
  X5 = revenue / total_assets

Approximations (documented honestly)
-------------------------------------
  X1: working_capital ≈ assets_current - liabilities_current
      (standard accounting identity; available directly in SEC data)
  X2: retained_earnings ≈ equity - (assets - liabilities)
      We do NOT have a direct retained_earnings field in the panel, so we
      approximate as equity minus estimated paid-in capital.  A cleaner
      proxy is net_income_ttm cumulated, but that requires cumulation state.
      SIMPLIFICATION: we use net_income_ttm / total_assets as a *flow* proxy
      that shares direction with retained_earnings/assets. This is flagged as
      approximate; IC from this component alone will be noisy.
  X3: EBIT ≈ operating_income_ttm
      (operating_income excludes interest + tax, identical to EBIT for most
      US GAAP reporters; occasionally includes D&A — acceptable approximation)
  X4: market_cap = Close * shares_outstanding (PIT Close × PIT SEC shares)
      total_liabilities = assets - equity  (balance-sheet identity)
  X5: revenue_ttm / total_assets  (standard; revenue_ttm avoids seasonal spikes)

Interpretation (Altman thresholds for public firms)
----------------------------------------------------
  Z > 2.99  → safe zone
  1.81 < Z ≤ 2.99 → grey zone
  Z ≤ 1.81  → distress zone (alt_distress_flag = 1)

Coverage notes
--------------
  - Approximately 84 % of the traded universe has SEC EDGAR fundamentals.
  - ETFs, foreign listings, and very recent IPOs will return NaN for all columns.
  - Leading rows before the first public filing will also be NaN — expected.
  - No inf values: all divisions are guarded (zero or negative denominators -> NaN).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import _fundamentals by file path (hidden from normal module discovery)
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
    "name": "paper_doaj_2a28a9b_altman_distress",
    "description": (
        "PIT Altman Z-score and distress components from SEC fundamentals (as_of filed_date); "
        "credit-risk factor for asset pricing per DOAJ:2a28a9b. "
        "Approximations: retained_earnings proxied by net_income_ttm flow; "
        "EBIT proxied by operating_income_ttm; total_liabilities = assets - equity."
    ),
    "requires": ["Close"],
    "produces": [
        "alt_zscore",           # composite Altman Z (5-factor weighted sum)
        "alt_distress_flag",    # 1 if Z <= 1.81 (distress zone), 0 otherwise, NaN if no data
        "alt_ebit_to_assets",   # X3 component: operating_income_ttm / assets (EBIT proxy)
        "alt_equity_to_liab",   # X4 component (unscaled by 0.6): market_cap / total_liabilities
        "alt_sales_to_assets",  # X5 component: revenue_ttm / assets
        "alt_wcap_to_assets",   # X1 component: working_capital / assets
    ],
    "tags": ["fundamentals", "credit_risk", "distress", "altman", "experimental", "paper"],
    "version": "1.0",
    "author": "paper DOAJ:2a28a9b — multifactor asset pricing with credit-risk factor",
}

# Fundamentals fields we need from the PIT panel
_FIELDS = [
    "assets",
    "assets_current",
    "liabilities_current",
    "equity",
    "operating_income_ttm",  # EBIT proxy
    "net_income_ttm",        # retained-earnings flow proxy
    "revenue_ttm",
    "shares_outstanding",
]

_DISTRESS_THRESHOLD = 1.81   # Altman public-firm distress cutoff


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide two series; return NaN where denominator is zero, negative, or NaN."""
    den_safe = pd.to_numeric(den, errors="coerce")
    num_safe = pd.to_numeric(num, errors="coerce")
    # Guard: zero or negative denominator -> NaN (avoids inf and economically spurious values)
    den_safe = den_safe.where(den_safe > 0)
    return num_safe / den_safe


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PIT Altman Z-score components and composite from SEC fundamentals.

    Each produced column is prefixed `alt_`. Rows without fundamentals coverage
    (pre-first-filing, ETFs, foreign names) will be NaN — that is expected.
    No inf values are emitted; all divisions are zero-guarded.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")

    # ------------------------------------------------------------------
    # Pull point-in-time fundamentals (backward merge on filed_date)
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=_FIELDS)

    assets     = pd.to_numeric(df["fund_assets"],             errors="coerce")
    cur_assets = pd.to_numeric(df["fund_assets_current"],     errors="coerce")
    cur_liab   = pd.to_numeric(df["fund_liabilities_current"],errors="coerce")
    equity     = pd.to_numeric(df["fund_equity"],             errors="coerce")
    ebit       = pd.to_numeric(df["fund_operating_income_ttm"],errors="coerce")
    net_inc    = pd.to_numeric(df["fund_net_income_ttm"],     errors="coerce")
    revenue    = pd.to_numeric(df["fund_revenue_ttm"],        errors="coerce")
    shares     = pd.to_numeric(df["fund_shares_outstanding"], errors="coerce")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    working_capital   = cur_assets - cur_liab          # may be negative — allowed
    total_liabilities = assets - equity                 # balance-sheet identity; guard below

    # Market cap: PIT Close × PIT SEC shares (filing-date shares, forward-carried to today)
    market_cap = close * shares.where(shares > 0)

    # ------------------------------------------------------------------
    # Z-score components (guard denominators against zero / negative)
    # ------------------------------------------------------------------
    # X1: working_capital / assets
    x1 = working_capital / assets.where(assets > 0)

    # X2: net_income_ttm / assets — proxy for retained_earnings / assets (see docstring)
    x2 = net_inc / assets.where(assets > 0)

    # X3: EBIT / assets
    x3 = ebit / assets.where(assets > 0)

    # X4: market_cap / total_liabilities  (liabilities must be positive; negative equity firms
    #     may produce negative liabilities — guard to NaN to avoid spurious negatives)
    x4 = market_cap / total_liabilities.where(total_liabilities > 0)

    # X5: revenue_ttm / assets
    x5 = revenue / assets.where(assets > 0)

    # ------------------------------------------------------------------
    # Composite Z-score
    # ------------------------------------------------------------------
    zscore = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # ------------------------------------------------------------------
    # Emit produced columns
    # ------------------------------------------------------------------
    df["alt_zscore"]          = zscore
    df["alt_distress_flag"]   = (zscore <= _DISTRESS_THRESHOLD).astype(float).where(zscore.notna())
    df["alt_ebit_to_assets"]  = x3          # informative standalone: profitability / leverage
    df["alt_equity_to_liab"]  = x4          # market-cap coverage of liabilities
    df["alt_sales_to_assets"] = x5          # asset efficiency / turnover
    df["alt_wcap_to_assets"]  = x1          # short-term liquidity

    # ------------------------------------------------------------------
    # Drop ALL fund_* scratch columns — only METADATA["produces"] must remain
    # ------------------------------------------------------------------
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])

    return df
