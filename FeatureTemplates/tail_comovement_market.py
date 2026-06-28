"""
tail_comovement_market.py — Market tail co-movement / crash dependence (Tier-2).

Lower tail dependence (e.g. Longin & Solnik 2001; Patton 2004): correlations rise
in crashes, so what matters is whether a stock's OWN worst days line up with the
market's down days. We measure, over a rolling window, the fraction of the stock's
worst-decile days that coincide with market down days:

    bad_day      = r_stock in its rolling bottom decile (own-history threshold)
    crash_overlap = P( market down | stock had a bad day )
    crash_lift    = crash_overlap - P(market down)        (excess co-crash beyond base rate)

A high crash_overlap / crash_lift name dumps exactly when the market dumps — bad
diversification, priced as risk. This is a Tier-2 shape sorter: it separates names
whose tail risk is SYSTEMATIC (co-moves with SPY) from names whose tail risk is
idiosyncratic, independent of overall beta. Both the stock decile threshold and the
market-down base rate use trailing rolling windows only -> no leakage.

Market-relative via the shared _indexes helper (SPY). Vectorized. Two horizons.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Shared index helper (underscore-prefixed -> skipped by block auto-discovery)
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

_WINDOWS = [(63, 40), (126, 80)]   # (window, min_periods)
_MKT = "SPY"
_DECILE = 0.10


METADATA = {
    "name":        "tail_comovement_market",
    "description": "Fraction of stock's worst-decile days that coincide with SPY down days + excess co-crash lift at 63d/126d (lower tail dependence proxy).",
    "requires":    ["Date", "Close"],
    "produces": [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("htail_crash_overlap", "htail_crash_lift")
    ],
    "tags":    ["market_regime", "tail", "experimental"],
    "version": "1.0",
    "author":  "Tier-2 lit build (lower tail dependence; Longin-Solnik 2001)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(df["Close"].values, index=dates)
    r = stock_close.pct_change()

    try:
        idx_close = _indexes.index_close(_MKT)
    except Exception:
        idx_close = pd.Series(dtype=float)

    df_dates = dates.values
    if idx_close is None or idx_close.empty:
        for c in METADATA["produces"]:
            df[c] = np.nan
        return df

    m_all = idx_close.pct_change()
    r_a, m_a = r.align(m_all, join="inner")   # shared trading dates
    mkt_down = (m_a < 0).astype(float)

    for w, mp in _WINDOWS:
        # own bottom-decile threshold from trailing window (no leakage)
        thresh = r_a.rolling(w, min_periods=mp).quantile(_DECILE)
        bad_day = (r_a <= thresh).astype(float)

        n_bad = bad_day.rolling(w, min_periods=mp).sum().replace(0, np.nan)
        n_bad_and_down = (bad_day * mkt_down).rolling(w, min_periods=mp).sum()

        crash_overlap = (n_bad_and_down / n_bad).clip(0.0, 1.0)
        base_down = mkt_down.rolling(w, min_periods=mp).mean()
        crash_lift = (crash_overlap - base_down).clip(-1.0, 1.0)

        df[f"htail_crash_overlap_{w}"] = crash_overlap.reindex(df_dates).values
        df[f"htail_crash_lift_{w}"] = crash_lift.reindex(df_dates).values

    return df
