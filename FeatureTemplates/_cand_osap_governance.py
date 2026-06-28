"""
Governance Index proxy (Gompers, Ishii & Metrick 2003) — per-ticker OHLCV+fundamentals version.

The original G-Index counts anti-takeover provisions from proxy statements (available every 2-3
years per firm from the IRRC dataset). A HIGH G-Index = more managerial entrenchment = weaker
shareholder rights = predicted UNDERPERFORMANCE (sign = -1).

Per-ticker proxy rationale: three well-documented empirical correlates of poor governance,
all computable from PIT fundamentals without cross-sectional data:

1. osap_governance_accrual_ratio  — operating accruals / total assets (Sloan 1996 / Richardson
   et al 2005). High accruals signal earnings management, a hallmark of weak governance.
   Range: unbounded; higher = worse governance proxy (sign ~ +1 → strategy sign −1).

2. osap_governance_capex_intensity — capex / operating_cash_flow (empire-building proxy). Poor
   governance firms over-invest relative to internally generated cash (Jensen 1986 free-cash-flow
   hypothesis). Clipped at [0, 5]; higher = worse governance proxy.

3. osap_governance_composite — equal-weight rank-average of the two above (both normalised within
   a trailing 252-day rolling window of the stock's own history so the scale is [0,1]). A higher
   composite = stronger proxy for managerial entrenchment = expected to underperform.

All fundamentals are loaded via PIT-safe _fundamentals.as_of() (backward merge on filed_date),
so there is zero lookahead. Stocks with no fundamental coverage produce NaN for all columns.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: load _fundamentals by file path (PIT-safe)
# ---------------------------------------------------------------------------
_fund_spec = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_fund_spec)  # type: ignore[arg-type]
_fund_spec.loader.exec_module(_fundamentals)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_governance",
    "description": (
        "Per-ticker proxy for the Gompers-Ishii-Metrick (2003) Governance Index (G-Index). "
        "The G-Index counts anti-takeover provisions from proxy statements — higher score means "
        "more managerial entrenchment and predicted underperformance (sign = -1). "
        "PROXY (original data requires IRRC governance dataset, not available per-ticker from OHLCV): "
        "Uses PIT fundamentals to compute (1) operating accrual ratio (earnings quality proxy) and "
        "(2) capex-to-OCF ratio (empire-building / free-cash-flow waste proxy). Both correlate "
        "positively with poor governance in the literature. A composite rolling-normalised score "
        "combines both signals. Cross-sectional ranking is intentionally avoided; all rolling "
        "normalisation is within each ticker's own history."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_governance_accrual_ratio",
        "osap_governance_capex_intensity",
        "osap_governance_composite",
    ],
    "tags": ["governance", "fundamentals", "accruals", "capex", "quality", "proxy"],
    "version": "1.0",
    "author": "Proxy impl — spec: Gompers, Ishii and Metrick 2003 (OpenSourceAP / Chen-Zimmermann)",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
_WINDOW = 252  # rolling normalisation window (one year of trading days)


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division; returns NaN where denom is zero or NaN."""
    d = denom.replace(0, np.nan)
    return num / d


def _rolling_rank_norm(s: pd.Series, window: int = _WINDOW) -> pd.Series:
    """
    Rolling percentile rank of each value within the trailing `window` observations
    of this ticker's own history. Result in [0, 1] (NaN where fewer than window/4
    observations available to avoid noisy early estimates).
    """
    min_p = max(4, window // 4)

    def _rank_last(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_p:
            return np.nan
        # rank of the last element relative to the window
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        return float(np.sum(valid <= last)) / len(valid)

    return s.rolling(window=window, min_periods=min_p).apply(_rank_last, raw=True)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach governance-proxy columns to the per-ticker DataFrame.

    Requires PIT fundamentals: operating_cash_flow_ttm, net_income_ttm,
    assets, capex_ttm.

    Accrual ratio = (net_income_ttm - operating_cash_flow_ttm) / assets
        Higher → more accrual-based earnings → weaker governance proxy.

    Capex intensity = capex_ttm / operating_cash_flow_ttm
        Higher → more empire-building relative to internally generated cash.
        Clipped at [0, 5]; negative OCF → NaN (signal is unreliable).
    """
    # --- load PIT fundamentals (backward-safe) ----------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = _fundamentals.as_of(
                df,
                fields=["net_income_ttm", "operating_cash_flow_ttm", "assets", "capex_ttm"],
            )
        except Exception:
            # No fundamentals available → all NaN
            df["osap_governance_accrual_ratio"] = np.nan
            df["osap_governance_capex_intensity"] = np.nan
            df["osap_governance_composite"] = np.nan
            return df

    # --- feature 1: accrual ratio ----------------------------------------
    # accruals = net_income_ttm - operating_cash_flow_ttm  (cash vs accrual gap)
    accruals = df["fund_net_income_ttm"] - df["fund_operating_cash_flow_ttm"]
    accrual_ratio = _safe_div(accruals, df["fund_assets"].abs())

    # Winsorise at ±1 to prevent outliers from dominating the composite
    accrual_ratio = accrual_ratio.clip(-1.0, 1.0)
    df["osap_governance_accrual_ratio"] = accrual_ratio

    # --- feature 2: capex intensity --------------------------------------
    # capex is often reported as a negative number in cash-flow statements;
    # take abs so we measure magnitude of spend
    capex_abs = df["fund_capex_ttm"].abs()
    ocf = df["fund_operating_cash_flow_ttm"]

    # Only meaningful when OCF is positive (negative OCF stocks have other issues)
    ocf_pos = ocf.where(ocf > 0, np.nan)
    capex_intensity = _safe_div(capex_abs, ocf_pos).clip(0.0, 5.0)
    df["osap_governance_capex_intensity"] = capex_intensity

    # --- feature 3: composite rolling-normalised governance proxy ---------
    rank_accrual = _rolling_rank_norm(accrual_ratio)
    rank_capex = _rolling_rank_norm(capex_intensity)

    # Equal-weight average of the two percentile ranks (NaN if both are NaN)
    stack = pd.concat([rank_accrual, rank_capex], axis=1)
    composite = stack.mean(axis=1, skipna=False)  # require both; pure-NaN → NaN
    # If only one is available, fall back to the one that exists
    # (coverage varies across firms; graceful degradation)
    composite = composite.where(
        rank_accrual.notna() & rank_capex.notna(),
        stack.mean(axis=1, skipna=True),
    )
    df["osap_governance_composite"] = composite

    # --- drop scratch fund_ columns not in produces ----------------------
    scratch_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=scratch_cols, inplace=True)

    return df
