"""
fvn_valuation_yields.py  --  Price-relative VALUATION YIELDS from point-in-time SEC fundamentals.

Theme: valuation multiples expressed as YIELDS (fundamental / price), distinct from the existing
pe_ratio_ttm / ps_ratio_ttm / pb_ratio block (which are price / per-share multiples). A yield is the
reciprocal framing -- higher = cheaper -- and is bounded/better-behaved for negative or tiny
denominators than a raw P/x multiple. These raw yields are the inputs the companion blocks
(fvn_valuation_zscores, fvn_valuation_dynamics) standardize against each name's OWN history.

All fundamentals are pulled through _fundamentals.as_of (BACKWARD merge on filed_date) so every
trading day only ever sees filings already public -> lookahead-safe. Daily Close supplies the price
leg; market cap = Close * shares_outstanding (PIT share count).

Yields built here:
  fvn_earnings_yield        eps_diluted_ttm / Close          (E/P)
  fvn_sales_yield           sales_per_share / Close          (S/P)
  fvn_book_yield            book_value_per_share / Close      (B/P)
  fvn_fcf_yield             fcf_ttm / market_cap              (FCF/P)
  fvn_cfo_yield             operating_cash_flow_ttm / market_cap   (CFO/P)
  fvn_dividend_yield        -dividends_paid_ttm / market_cap  (div paid is reported negative)
  fvn_ni_yield              net_income_ttm / market_cap       (NI/P; differs from eps E/P -- uses
                                                               total NI vs diluted-share EPS)

Guards: per-share denominators (Close) and market cap must be strictly positive else NaN. Negative
numerators (loss-making earnings/fcf/ni) are KEPT with sign -- a negative yield is meaningful. All
outputs clipped to a sane finite band.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Import the underscore PIT-fundamentals helper by file path (hidden from auto-discovery).
_spec = _ilu.spec_from_file_location(
    "_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "fvn_valuation_yields",
    "description": "Price-relative valuation YIELDS (earnings/sales/book/fcf/cfo/dividend/net-income "
                   "over price) from point-in-time SEC fundamentals combined with daily Close.",
    "requires":    ["Close"],
    "produces": [
        "fvn_earnings_yield",
        "fvn_sales_yield",
        "fvn_book_yield",
        "fvn_fcf_yield",
        "fvn_cfo_yield",
        "fvn_dividend_yield",
        "fvn_ni_yield",
    ],
    "tags":    ["fundamentals", "valuation", "yield"],
    "version": "1.0",
    "author":  "feature-gen",
}

# Canonical PIT fundamentals fields the yields are built from.
_FIELDS = [
    "eps_diluted_ttm",
    "sales_per_share",
    "book_value_per_share",
    "fcf_ttm",
    "operating_cash_flow_ttm",
    "dividends_paid_ttm",
    "net_income_ttm",
    "shares_outstanding",
]

# Yields are dimensionless reciprocals; a value beyond +/-5 (i.e. 500% yield) is pathological.
_CLIP = 5.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = _num(df["Close"])
    close_pos = close.where(close > 0)  # price denominator must be strictly positive

    # Lookahead-safe PIT merge -- adds fund_* scratch columns (dropped before return).
    df = _fundamentals.as_of(df, fields=_FIELDS)

    eps  = _num(df["fund_eps_diluted_ttm"])
    sps  = _num(df["fund_sales_per_share"])
    bvps = _num(df["fund_book_value_per_share"])
    fcf  = _num(df["fund_fcf_ttm"])
    cfo  = _num(df["fund_operating_cash_flow_ttm"])
    divp = _num(df["fund_dividends_paid_ttm"])
    ni   = _num(df["fund_net_income_ttm"])
    sh   = _num(df["fund_shares_outstanding"])

    # Market cap from PIT share count; only meaningful when strictly positive.
    mcap = (close * sh)
    mcap = mcap.where(mcap > 0)

    # Per-share yields (price leg = Close). Numerator sign preserved (loss = negative yield).
    df["fvn_earnings_yield"] = (eps / close_pos)
    df["fvn_sales_yield"]    = (sps / close_pos)
    df["fvn_book_yield"]     = (bvps / close_pos)

    # Whole-firm yields (price leg = market cap).
    df["fvn_fcf_yield"] = (fcf / mcap)
    df["fvn_cfo_yield"] = (cfo / mcap)
    # dividends_paid is reported as a cash OUTFLOW (negative); flip sign so a payout -> positive yield.
    df["fvn_dividend_yield"] = (-divp / mcap)
    df["fvn_ni_yield"] = (ni / mcap)

    # Clip pathological/inf values to a finite band; keep NaN as NaN.
    for c in METADATA["produces"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-_CLIP, _CLIP)

    # Drop scratch helper columns so only METADATA["produces"] is added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
