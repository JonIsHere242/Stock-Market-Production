"""
residual_momentum.py - Market-residual (idiosyncratic) momentum (Tier-2).

Blitz, Huij & Martens (2011, JFE) "Residual Momentum". Plain price momentum is
dominated by the names that simply have high market/factor exposure (high beta) and
therefore loads heavily on volatility -- exactly the trap. Strip the market component
first: regress the stock's returns on SPY over a formation window, keep the RESIDUAL,
and rank by the residual's own information ratio (mean / std of residuals). Blitz et
al. show residual momentum earns ~double the risk-adjusted return of total-return
momentum with HALF the volatility, because it is orthogonal to the market factor and
to the volatility tail. This is a within-cross-section sorter on *idiosyncratic* drift,
not on who is most volatile -- a clean Tier-2 separator.

  resid_t   = r_t - (alpha_W + beta_W * m_t)        (trailing window regression)
  resmom_W  = sum of residuals over W               (cumulative idiosyncratic move)
  residir_W = mean(resid) / std(resid) over W       (THE Blitz-Huij-Martens signal)

Market-relative block: uses _indexes.index_close('SPY'). Per Blitz, a skip-recent
variant (skip last 21d) avoids short-term reversal contamination of the formation move.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

_WINDOWS = [126, 252]
_MKT = "SPY"

METADATA = {
    "name":        "residual_momentum",
    "description": "Market-residual (idiosyncratic) momentum vs SPY: cumulative CAPM residual return and residual information ratio at 126d/252d (+skip-21d), per Blitz-Huij-Martens 2011.",
    "requires":    ["Date", "Close"],
    "produces": (
        [f"csa_resmom_{w}" for w in _WINDOWS]
        + [f"csa_residir_{w}" for w in _WINDOWS]
        + ["csa_residir_252_sk21"]
    ),
    "tags":    ["momentum", "market_regime", "beta", "tail", "experimental"],
    "version": "1.0",
    "author":  "Tier-2 lit build (Blitz-Huij-Martens 2011)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(df["Close"].values, index=dates)
    r = np.log(stock_close / stock_close.shift(1))

    try:
        idx_close = _indexes.index_close(_MKT)
    except Exception:
        idx_close = pd.Series(dtype=float)

    if idx_close is None or idx_close.empty:
        for c in METADATA["produces"]:
            df[c] = np.nan
        return df

    m_all = np.log(idx_close / idx_close.shift(1))
    r_a, m_a = r.align(m_all, join="inner")          # shared trading dates
    df_dates = dates.values

    resid_cache = {}
    for w in _WINDOWS:
        mp = int(w * 0.7)
        cov = r_a.rolling(w, min_periods=mp).cov(m_a)
        var = m_a.rolling(w, min_periods=mp).var().replace(0, np.nan)
        beta = (cov / var).clip(-5, 5)
        alpha = r_a.rolling(w, min_periods=mp).mean() - beta * m_a.rolling(w, min_periods=mp).mean()
        resid = r_a - alpha - beta * m_a            # trailing idiosyncratic return
        resid_cache[w] = resid

        resmom = resid.rolling(w, min_periods=mp).sum()
        resir = resid.rolling(w, min_periods=mp).mean() / resid.rolling(w, min_periods=mp).std().replace(0, np.nan)
        df[f"csa_resmom_{w}"] = resmom.clip(-3, 3).reindex(df_dates).values
        df[f"csa_residir_{w}"] = resir.clip(-1, 1).reindex(df_dates).values

    # Skip-recent-21d residual info ratio over the 252d formation (classic momentum skip).
    resid252 = resid_cache[252].shift(21)
    win = 252 - 21
    sk_ir = resid252.rolling(win, min_periods=int(win * 0.7)).mean() / \
        resid252.rolling(win, min_periods=int(win * 0.7)).std().replace(0, np.nan)
    df["csa_residir_252_sk21"] = sk_ir.clip(-1, 1).reindex(df_dates).values

    return df
