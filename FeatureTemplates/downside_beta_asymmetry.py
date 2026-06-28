"""
downside_beta_asymmetry.py — Downside vs upside beta asymmetry vs SPY (Tier-2).

Ang, Chen & Xing (2006, RFS) "Downside Risk". Stocks that covary with the market
specifically when the market FALLS carry a premium that ordinary (symmetric) beta
misses. Estimate beta separately on down-market and up-market days:

    beta_minus = cov(r, m | m < mean_m) / var(m | m < mean_m)   (downside beta)
    beta_plus  = cov(r, m | m > mean_m) / var(m | m > mean_m)   (upside  beta)
    beta_asym  = beta_minus - beta_plus                          (relative downside risk)

Their finding: high beta_minus earns higher returns; the SPREAD (asymmetry) is the
Tier-2 separator — it sorts names by *when* their market exposure bites, not how
much, concentrating winners differently than a single rolling beta. We threshold
up/down by the rolling market mean (conditioning only on trailing data -> no
leakage). Two horizons.

Market-relative via the shared _indexes helper (SPY). Per-ticker, vectorized.
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


def _cond_beta(r, m, mask, w, mp):
    """Rolling beta of r on m restricted to rows where mask is True (else 0 contribution).

    Uses the masked-sum identities so it stays fully vectorized:
        cov = E[r*m | mask] - E[r|mask]*E[m|mask] ;  var = E[m^2|mask] - E[m|mask]^2
    computed via rolling sums of mask-zeroed series divided by the rolling count.
    """
    msk = mask.astype(float)
    cnt = msk.rolling(w, min_periods=mp).sum().replace(0, np.nan)
    rm = (r * m * msk).rolling(w, min_periods=mp).sum() / cnt
    rr = (r * msk).rolling(w, min_periods=mp).sum() / cnt
    mm = (m * msk).rolling(w, min_periods=mp).sum() / cnt
    m2 = (m * m * msk).rolling(w, min_periods=mp).sum() / cnt
    cov = rm - rr * mm
    var = (m2 - mm * mm).replace(0, np.nan)
    # require a minimum number of conditioning days for stability
    beta = (cov / var).where(cnt >= max(8, mp // 4))
    return beta


METADATA = {
    "name":        "downside_beta_asymmetry",
    "description": "Downside vs upside beta vs SPY and their spread (downside minus upside) at 63d/126d, per Ang-Chen-Xing 2006.",
    "requires":    ["Date", "Close"],
    "produces": [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("dtb_beta_minus", "dtb_beta_plus", "dtb_beta_asym")
    ],
    "tags":    ["market_regime", "beta", "tail", "experimental"],
    "version": "1.0",
    "author":  "Tier-2 lit build (Ang-Chen-Xing 2006)",
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
        m_mu = m_a.rolling(w, min_periods=mp).mean()   # trailing threshold (no leakage)
        down = m_a < m_mu
        up = m_a > m_mu

        beta_minus = _cond_beta(r_a, m_a, down, w, mp).clip(-10, 10)
        beta_plus = _cond_beta(r_a, m_a, up, w, mp).clip(-10, 10)
        beta_asym = (beta_minus - beta_plus).clip(-15, 15)

        df[f"dtb_beta_minus_{w}"] = beta_minus.reindex(df_dates).values
        df[f"dtb_beta_plus_{w}"] = beta_plus.reindex(df_dates).values
        df[f"dtb_beta_asym_{w}"] = beta_asym.reindex(df_dates).values

    return df
