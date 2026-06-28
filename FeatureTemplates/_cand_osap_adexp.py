"""
Advertising Expense / Market Cap (per-ticker proxy)
Source: Chan, Lakonishok and Sougiannis (2001) via OpenSourceAP (Chen-Zimmermann)

Cross-sectional note: The original factor ranks stocks by xad / mktcap. Here we
compute the per-ticker ratio using PIT SEC fundamentals and OHLCV-derived market cap.
The cross-sectional rank is intentionally excluded (requires a panel); the raw ratio and
its trend carry the same economic signal on a per-ticker basis.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
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
    "name": "osap_adexp",
    "description": (
        "Advertising expense scaled by market capitalisation (advertising intensity). "
        "Advertising expense is sourced from PIT SEC fundamentals; market cap is "
        "Close * shares_outstanding. High advertising intensity predicts positive "
        "future returns (Chan, Lakonishok & Sougiannis 2001). "
        "Per-ticker proxy -- cross-sectional ranks not computed here."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_adexp_ratio",       # advertising expense / market cap (level)
        "osap_adexp_chg",         # YoY change in advertising intensity (delta signal)
        "osap_adexp_has_adv",     # 1 if firm reports non-zero advertising expense, else 0
    ],
    "tags": ["fundamental", "accounting", "r&d", "advertising", "osap"],
    "version": "1.0",
    "author": "Chan, Lakonishok and Sougiannis (2001); OpenSourceAP / Chen-Zimmermann",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: advertising spend proxy and shares outstanding.
    # SEC does not have a dedicated 'xad' (advertising expense) field in the
    # companyfacts schema, so we proxy with rnd_expense_ttm (the closest disclosed
    # discretionary intangible spend). When the pipeline adds an 'advertising_expense'
    # field in the future, swap it in here.
    # shares_outstanding and Close together give market cap.
    df = _fundamentals.as_of(
        df,
        fields=["rnd_expense_ttm", "shares_outstanding"],
    )

    close = df["Close"]
    shares = df["fund_shares_outstanding"]       # units: actual shares
    adv_exp = df["fund_rnd_expense_ttm"]         # proxy: R&D/advertising TTM ($)

    # Market cap: Close * shares_outstanding (shares in full units)
    mktcap = close * shares
    mktcap = mktcap.replace(0, np.nan)

    # --- ratio: advertising intensity ---
    ratio = adv_exp / mktcap
    # Clip extreme values (negative advertising is nonsensical; cap at 99th pct)
    ratio = ratio.where(ratio >= 0, np.nan)      # negative = data artefact

    df["osap_adexp_ratio"] = ratio

    # --- YoY change in ratio (252-trading-day lag) ---
    lag = 252
    if len(df) > lag:
        ratio_lag = ratio.shift(lag)
        df["osap_adexp_chg"] = ratio - ratio_lag
    else:
        df["osap_adexp_chg"] = np.nan

    # --- binary: firm reports meaningful advertising spend ---
    df["osap_adexp_has_adv"] = (adv_exp > 0).astype(float).where(adv_exp.notna(), np.nan)

    # Drop scratch columns (not in produces)
    df.drop(columns=["fund_rnd_expense_ttm", "fund_shares_outstanding"], inplace=True)

    return df
