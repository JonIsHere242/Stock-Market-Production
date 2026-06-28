"""
osap_tax — Taxable income to income (Lev & Nissim 2004, OpenSourceAP / Chen-Zimmermann)

Ratio of taxes actually paid to "expected" taxes (statutory rate × net income).
High ratio = earnings are tax-consistent (high quality); predicts positive future returns (sign = +1).

Per-ticker proxy: The COMPUSTAT tax-line fields (txfo, txfed, txdi) are not available in the
fundamentals helper.  We approximate the signal via:
  implied_tax = operating_income_ttm - net_income_ttm   (≈ taxes + interest + minority interest)
  effective_tax_rate = implied_tax / operating_income_ttm  (when operating_income > 0)
  expected_tax = statutory_rate × operating_income_ttm
  osap_tax_ratio = implied_tax / expected_tax  (≈ actual/expected tax; faithful per-Lev-Nissim spirit)

We also expose:
  osap_tax_eff_rate  — smoothed (4-quarter rolling mean) effective tax rate
  osap_tax_qual      — quality dummy: 1 if ratio ∈ [0.5, 2.0] and price ≥ 5, else 0
                       (mirrors the paper's exclude-if-price<5 filter and plausibility bounds)

Statutory rates (US federal corporate):
  <1979: 0.48 | 1979-1986: 0.46 | 1987: 0.40 | 1988-1992: 0.34 | 1993+: 0.35
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

# ── statutory US corporate tax rate by year ──────────────────────────────────
def _statutory_rate(year: int) -> float:
    if year < 1979:
        return 0.48
    elif year <= 1986:
        return 0.46
    elif year == 1987:
        return 0.40
    elif year <= 1992:
        return 0.34
    else:
        return 0.35


METADATA = {
    "name": "osap_tax",
    "description": (
        "Taxable income to income ratio (Lev & Nissim 2004 / OpenSourceAP Chen-Zimmermann). "
        "Ratio of implied taxes paid to expected taxes at the statutory rate. "
        "High ratio signals earnings quality (taxes consistent with reported income). "
        "Predicted sign: +1 (long high). "
        "Per-ticker proxy: uses (operating_income_ttm - net_income_ttm) as implied tax; "
        "no COMPUSTAT txfo/txfed/txdi available so this is an approximation. "
        "Exclude if Close < 5 per paper filter."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_tax_ratio",      # implied_tax / (statutory_rate × operating_income_ttm)
        "osap_tax_eff_rate",   # rolling-smoothed effective tax rate proxy
        "osap_tax_qual",       # quality flag: ratio in plausible range & price ≥ 5
    ],
    "tags": ["fundamentals", "accounting", "earnings_quality", "tax", "osap"],
    "version": "1.0.0",
    "author": "Lev & Nissim (2004); OpenSourceAP / Chen-Zimmermann dataset; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── pull PIT fundamentals ────────────────────────────────────────────────
    df = _fundamentals.as_of(
        df,
        fields=["net_income_ttm", "operating_income_ttm"],
    )

    oi = df["fund_operating_income_ttm"]
    ni = df["fund_net_income_ttm"]

    # ── implied tax expense (operating_income - net_income proxy) ────────────
    # This approximates taxes + net interest. When operating_income > 0 this is
    # the closest available signal to taxes paid.
    implied_tax = oi - ni

    # ── statutory rate series ────────────────────────────────────────────────
    years = pd.to_datetime(df["Date"]).dt.year
    stat_rate = years.map(_statutory_rate).astype(float)

    # ── expected tax = statutory_rate × operating_income ────────────────────
    expected_tax = stat_rate * oi

    # ── ratio: implied_tax / expected_tax ───────────────────────────────────
    # Guard: set NaN when operating_income ≤ 0 or expected_tax == 0
    # Also handle Lev-Nissim edge case: if net_income < 0 and implied_tax > 0 → ratio = 1
    ratio = np.where(
        (ni < 0) & (implied_tax > 0),
        1.0,
        implied_tax / expected_tax.replace(0, np.nan),
    )
    ratio = pd.Series(ratio, index=df.index, dtype=float)

    # Force NaN where operating_income ≤ 0 (undefined effective rate)
    bad_oi = oi <= 0
    ratio = ratio.where(~bad_oi, other=np.nan)

    # Clip extreme outliers (ratio > 5 or < -2 are almost certainly data artefacts)
    ratio = ratio.clip(-2.0, 5.0)

    df["osap_tax_ratio"] = ratio

    # ── effective tax rate (rolling 4-obs mean for smoothing) ───────────────
    eff_rate_raw = implied_tax / oi.replace(0, np.nan)
    eff_rate_raw = eff_rate_raw.where(oi > 0, other=np.nan).clip(-1.0, 2.0)
    df["osap_tax_eff_rate"] = eff_rate_raw.rolling(window=4, min_periods=1).mean()

    # ── quality flag: plausible ratio & price ≥ 5 ───────────────────────────
    price_ok = df["Close"] >= 5.0
    ratio_ok = (ratio >= 0.5) & (ratio <= 2.0)
    df["osap_tax_qual"] = (price_ok & ratio_ok).astype(float)
    # NaN where fundamentals are missing
    df["osap_tax_qual"] = df["osap_tax_qual"].where(ratio.notna(), other=np.nan)

    # ── drop scratch fundamental columns ────────────────────────────────────
    df = df.drop(columns=["fund_operating_income_ttm", "fund_net_income_ttm"], errors="ignore")

    return df
