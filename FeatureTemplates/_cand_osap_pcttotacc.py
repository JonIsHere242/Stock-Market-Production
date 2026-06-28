"""
Percent Total Accruals (Hafzalla, Lundholm, Van Winkle 2011).

Per-ticker PIT proxy using SEC fundamentals via _fundamentals.as_of().

Full formula (OSAP/Chen-Zimmermann):
  PctTotAcc = [NI - (prstkcc - sstk + dvt) - oancf - fincf - ivncf] / |NI|

where:
  NI    = net income
  dvt   = total dividends paid
  prstkcc - sstk = net stock repurchases (buybacks minus issuance)
  oancf = cash flow from operations
  fincf = cash flow from financing activities (not directly available)
  ivncf = cash flow from investing activities (not directly available)

Note: fincf + ivncf are not in the available fundamentals panel.  We approximate
using the balance-sheet/CF accrual identity:
  Total accruals (CF approach) ≈ NI - OCF       (Sloan 1996 core)
  Percent total accruals       ≈ (NI - OCF - dvt) / |NI|

This omits stock-repurchase/issuance and investing flows but captures the dominant
earnings-quality signal (operating accruals) and the equity payout channel (dividends).
The predicted sign is -1: high accruals (earnings >> cash) predict negative returns.
Coverage ~84%; ETFs/foreign will be NaN.
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
    "name": "osap_pcttotacc",
    "description": (
        "Percent Total Accruals (Hafzalla, Lundholm, Van Winkle 2011) — "
        "per-ticker PIT proxy. Core signal: (NI_ttm - OCF_ttm - dividends_paid_ttm) / |NI_ttm|. "
        "Proxy for full OSAP formula which also nets out stock buybacks/issuance and "
        "investing cash flows (not available in the fundamentals panel). "
        "Predicted sign: -1 (high accruals predict lower returns). "
        "Slope variant captures quarterly change in the ratio."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_pcttotacc_ratio",   # (NI_ttm - OCF_ttm - div_ttm) / |NI_ttm|
        "osap_pcttotacc_slope",   # 1-quarter change in ratio (momentum of accruals)
        "osap_pcttotacc_ni_sign", # sign(NI_ttm): flag for loss-firms (ratio less meaningful)
    ],
    "tags": ["accruals", "fundamentals", "earnings_quality", "osap"],
    "version": "1.0",
    "author": "Hafzalla, Lundholm, Van Winkle (2011); OSAP/Chen-Zimmermann open-source AP library",
}


# ---------------------------------------------------------------------------
# Helper: safe divide, guard zeros and very small denominators
# ---------------------------------------------------------------------------
def _safe_div(num: pd.Series, denom: pd.Series, floor: float = 1e-6) -> pd.Series:
    """Divide num/denom; set NaN where |denom| < floor."""
    denom_safe = denom.copy()
    denom_safe[denom.abs() < floor] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = num / denom_safe
    return result.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: NI ttm, OCF ttm, dividends ttm
    df = _fundamentals.as_of(
        df,
        fields=["net_income_ttm", "operating_cash_flow_ttm", "dividends_paid_ttm"],
    )

    ni_ttm   = df["fund_net_income_ttm"]
    ocf_ttm  = df["fund_operating_cash_flow_ttm"]
    # dividends_paid is typically reported as a negative number (cash outflow);
    # we want the absolute outflow, so use abs() to be sign-convention-agnostic.
    div_ttm  = df["fund_dividends_paid_ttm"].abs()

    # Core accruals numerator:
    #   NI - OCF - Dividends  (omitting buybacks/issuance and investing flows)
    accruals = ni_ttm - ocf_ttm - div_ttm

    # Scale by |NI| — Hafzalla et al. scaling
    abs_ni = ni_ttm.abs()
    ratio  = _safe_div(accruals, abs_ni)

    # Winsorise at [-10, 10] to prevent extreme values from loss firms
    ratio = ratio.clip(-10.0, 10.0)

    # Slope: change over ~63 trading days (~1 quarter) via forward diff on the ratio
    # Use ffill(limit=63) so intra-quarter NaNs propagate cleanly, then diff
    ratio_filled = ratio.ffill(limit=63)
    slope = ratio_filled.diff(63)

    # Sign of NI (1=profit, -1=loss, 0=zero): useful mask for downstream
    ni_sign = np.sign(ni_ttm).fillna(0).astype("float32")

    df["osap_pcttotacc_ratio"]   = ratio.astype("float32")
    df["osap_pcttotacc_slope"]   = slope.astype("float32")
    df["osap_pcttotacc_ni_sign"] = ni_sign

    # Drop scratch fundamentals columns
    df = df.drop(
        columns=[
            "fund_net_income_ttm",
            "fund_operating_cash_flow_ttm",
            "fund_dividends_paid_ttm",
        ],
        errors="ignore",
    )

    return df
