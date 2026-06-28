"""
Enterprise Multiple (EV / Operating Income) — per-ticker proxy.

Loughran & Wellman (2011), as catalogued in Chen-Zimmermann OpenSourceAP.

Cross-sectional signal: stocks with LOW enterprise multiples tend to
outperform (predicted sign = -1, i.e. short high EM).

Per-ticker implementation: EV is approximated from PIT fundamentals
(market cap = Close * shares_outstanding; LT debt; current liabilities
debt; cash) and operating income (oibdp ≈ operating_income_ttm).
Deferred charges are not separately available in the fundamentals panel,
so they are omitted (minor item). The ratio itself is the cross-sectional
signal; we also produce a trailing 63-day z-score so the model can see
whether the stock is getting cheaper/more expensive relative to its own
recent history (time-series dynamic variant).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_entmult",
    "description": (
        "Enterprise Multiple (EV / Operating Income) per Loughran & Wellman 2011. "
        "EV = market-cap + long_term_debt + liabilities_current + 0 (deferred charges "
        "unavailable) - cash; divided by operating_income_ttm. "
        "Cross-sectional: low EM stocks outperform (predicted sign -1). "
        "Per-ticker proxy: level ratio + 63-day trailing z-score (time-series cheapness "
        "trend). NaN when book equity missing, operating income <= 0, or no fundamentals."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_entmult_level",   # EV / operating_income_ttm (raw ratio)
        "osap_entmult_zscore",  # 63-day trailing z-score of the level
        "osap_entmult_chg63",   # 63-day change in the level (cheapening signal)
    ],
    "tags": ["valuation", "fundamentals", "enterprise_value", "accounting"],
    "version": "1.0",
    "author": "Loughran & Wellman 2011 (OpenSourceAP / Chen-Zimmermann); block by claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals — backward-safe merge_asof on filed_date
    df = _fundamentals.as_of(
        df,
        fields=[
            "shares_outstanding",
            "long_term_debt",
            "liabilities_current",
            "cash",
            "operating_income_ttm",
            "equity",          # needed only to filter "missing book equity"
        ],
    )

    # Market cap
    mkt_cap = df["Close"] * df["fund_shares_outstanding"].replace(0, np.nan)

    # Enterprise value components
    lt_debt  = df["fund_long_term_debt"].fillna(0.0)        # dltt
    cur_liab = df["fund_liabilities_current"].fillna(0.0)   # dlc proxy
    cash     = df["fund_cash"].fillna(0.0)                  # che

    ev = mkt_cap + lt_debt + cur_liab - cash

    # Operating income (oibdp proxy = TTM operating income)
    op_inc = df["fund_operating_income_ttm"]

    # Missing equity → NaN (Loughran & Wellman exclusion criterion)
    bad_equity = df["fund_equity"].isna()

    # Compute raw ratio; exclude non-positive operating income per spec
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (op_inc > 0) & (~bad_equity) & ev.notna(),
            ev / op_inc,
            np.nan,
        )

    df["osap_entmult_level"] = ratio

    # 63-day trailing z-score (own-history normalisation)
    level_s = pd.Series(ratio, index=df.index)
    roll63   = level_s.rolling(63, min_periods=21)
    mu63     = roll63.mean()
    sd63     = roll63.std(ddof=1)
    df["osap_entmult_zscore"] = np.where(
        sd63 > 0,
        (level_s - mu63) / sd63,
        np.nan,
    )

    # 63-day change in the level (negative = stock became cheaper → bullish)
    df["osap_entmult_chg63"] = level_s.diff(63)

    # Drop scratch fundamentals columns not listed in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=fund_cols, inplace=True)

    return df
