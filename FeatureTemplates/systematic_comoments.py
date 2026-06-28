"""
systematic_comoments.py — Systematic coskewness & cokurtosis vs SPY (Tier-2).

Harvey & Siddique (2000, JF) "Conditional Skewness in Asset Pricing Tests".
Assets that covary with the SQUARE of market shocks (negative coskewness) command
a premium: they pay off poorly exactly when the market is most volatile. The
standardized comoments of a stock's return r with the market m are:

    coskew  = E[ e_r * e_m^2 ] / ( sd(e_r) * var(e_m) )      (3rd comoment)
    cokurt  = E[ e_r * e_m^3 ] / ( sd(e_r) * sd(e_m)^3 )      (4th comoment)

where e_r, e_m are demeaned returns over a rolling window. Negative coskew names
are RISKIER (priced -). This sorts WITHIN the cross-section by the *shape* of a
name's market exposure, not its level (beta) — a Tier-2 discriminator that
concentrates next-day winners differently than ordinary beta. We also keep the
raw cross-comoment numerators (direct comovement with m^2 / m^3) which are the
quantities the asset-pricing premium is actually defined on.

Market-relative via the shared _indexes helper (SPY). Per-ticker, vectorized,
two horizons.
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

METADATA = {
    "name":        "systematic_comoments",
    "description": "Standardized coskewness & cokurtosis of stock vs SPY returns (+raw cross-comoments) at 63d/126d, per Harvey-Siddique 2000.",
    "requires":    ["Date", "Close"],
    "produces": [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("scm_coskew", "scm_cokurt", "scm_xcomom3", "scm_xcomom4")
    ],
    "tags":    ["market_regime", "tail", "higher_moments", "experimental"],
    "version": "1.0",
    "author":  "Tier-2 lit build (Harvey-Siddique 2000)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(df["Close"].values, index=dates)
    r = np.log(stock_close / stock_close.shift(1))

    try:
        idx_close = _indexes.index_close(_MKT)
    except Exception:
        idx_close = pd.Series(dtype=float)

    df_dates = dates.values
    if idx_close is None or idx_close.empty:
        for c in METADATA["produces"]:
            df[c] = np.nan
        return df

    m_all = np.log(idx_close / idx_close.shift(1))
    r_a, m_a = r.align(m_all, join="inner")   # shared trading dates

    for w, mp in _WINDOWS:
        # rolling demeaning (trailing window means only -> no leakage)
        r_mu = r_a.rolling(w, min_periods=mp).mean()
        m_mu = m_a.rolling(w, min_periods=mp).mean()
        e_r = r_a - r_mu
        e_m = m_a - m_mu

        sd_r = e_r.rolling(w, min_periods=mp).std()
        sd_m = e_m.rolling(w, min_periods=mp).std()
        var_m = e_m.pow(2).rolling(w, min_periods=mp).mean()

        # raw cross-comoments: E[e_r * e_m^2] and E[e_r * e_m^3]
        xcomom3 = (e_r * e_m.pow(2)).rolling(w, min_periods=mp).mean()
        xcomom4 = (e_r * e_m.pow(3)).rolling(w, min_periods=mp).mean()

        denom3 = (sd_r * var_m).replace(0, np.nan)
        denom4 = (sd_r * sd_m.pow(3)).replace(0, np.nan)

        coskew = (xcomom3 / denom3).clip(-10, 10)
        cokurt = (xcomom4 / denom4).clip(-25, 25)

        df[f"scm_coskew_{w}"] = coskew.reindex(df_dates).values
        df[f"scm_cokurt_{w}"] = cokurt.reindex(df_dates).values
        # scale raw cross-comoments (tiny daily-return magnitudes) and clip
        df[f"scm_xcomom3_{w}"] = (xcomom3 * 1e4).clip(-50, 50).reindex(df_dates).values
        df[f"scm_xcomom4_{w}"] = (xcomom4 * 1e6).clip(-50, 50).reindex(df_dates).values

    return df
