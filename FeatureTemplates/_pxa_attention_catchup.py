"""
pxa_attention_catchup.py - Peng-Xiong limited-attention catch-up (Tier-2).

Peng & Xiong (2006, JFE) "Investor Attention, Overconfidence and Category Learning". With limited
attention investors process category (market) information faster than firm-specific information, so
stocks OVER-COMOVE with the market and UNDER-react to their own idiosyncratic shocks, which then
resolve later -- a catch-up drift in the high-comovement names. We combine the comovement gate with
the QUALITY (information ratio) of recent idiosyncratic drift:

  comove  = rolling 60d corr(stock return, SPY return), clipped >= 0
  eps     = stock return - beta * SPY return            (market residual)
  idio_ir = sum(eps over 21d, skipping the last 3d) / (sqrt(21) * 60d std of eps)
  catchup = comove * idio_ir

De-trapped per the review: the idiosyncratic drift is shipped as an INFORMATION RATIO (drift quality,
not magnitude -> not a vol/return-distribution proxy, not pocket A), is market-RESIDUAL (no price
level, not pocket B), and uses a 21d sum with a 3d skip to clear the 1-week reversal window. Distinct
from residual_momentum (long-horizon, no comovement gate). Market-coupled (uses the SPY helper).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

_CORR_W = 60
_SUM_W = 21
_SKIP = 3
_MKT = "SPY"

METADATA = {
    "name":        "pxa_attention_catchup",
    "description": "Peng-Xiong limited-attention catch-up: comovement-gated information ratio of recent market-residual (idiosyncratic) drift, 21d sum with a 3d skip. Drift QUALITY not magnitude, market-residual -> orthogonal to return-distribution and cost-basis pockets.",
    "requires":    ["Date", "Close"],
    "produces":    ["pxa_catchup", "pxa_idio_ir"],
    "tags":        ["behavioral", "market_regime", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 open-room build (Peng-Xiong 2006)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(df["Close"].astype(float).values, index=dates)
    r = np.log(stock_close / stock_close.shift(1))

    try:
        idx_close = _indexes.index_close(_MKT)
    except Exception:
        idx_close = pd.Series(dtype=float)

    if idx_close is None or idx_close.empty:
        df["pxa_catchup"] = np.nan
        df["pxa_idio_ir"] = np.nan
        return df

    m = np.log(idx_close / idx_close.shift(1))
    r_a, m_a = r.align(m, join="inner")
    df_dates = dates.values

    cov = r_a.rolling(_CORR_W, min_periods=40).cov(m_a)
    var = m_a.rolling(_CORR_W, min_periods=40).var().replace(0, np.nan)
    beta = (cov / var).clip(-5, 5)
    comove = r_a.rolling(_CORR_W, min_periods=40).corr(m_a).clip(lower=0)
    eps = r_a - beta * m_a

    idio_ir = (eps.shift(_SKIP).rolling(_SUM_W, min_periods=14).sum()
               / (np.sqrt(_SUM_W) * eps.rolling(_CORR_W, min_periods=40).std().replace(0, np.nan)))
    catchup = comove * idio_ir

    df["pxa_catchup"] = np.clip(catchup.clip(-2, 2).reindex(df_dates).values, -2, 2)
    df["pxa_idio_ir"] = np.clip(idio_ir.clip(-2, 2).reindex(df_dates).values, -2, 2)
    return df
