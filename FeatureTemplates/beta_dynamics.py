"""
beta_dynamics.py - Time-variation of market beta (Tier-2), NOT the beta level.

The static beta block (beta_metrics) already gives the rolling beta LEVEL. The level
is largely a volatility proxy and is not the interesting cross-sectional sorter. What
is under-exploited is the *dynamics* of beta:

  - beta TREND   : is the name's market sensitivity rising or falling right now?
                   A falling beta on a rising name signals idiosyncratic strength taking
                   over from market beta (Jostova-Philipov style beta-decay / de-risking).
  - beta-of-beta : instability of beta. Cederburg-O'Doherty (2016) and Hollstein-Prokopczuk
                   (2016) show beta INSTABILITY is priced separately from beta itself --
                   unstable-beta names carry an estimation-risk premium.
  - beta ACCEL   : recent beta minus older beta (momentum of market loading).

All three are normalized, dimensionless transforms of the rolling cov/var beta series,
so they separate *how a name's market coupling is changing* from *how volatile it is* --
orthogonal to the volatility tail. Market-relative block: uses _indexes SPY.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

_BETA_W = 63       # window for the base rolling beta
_TREND_L = 21      # slope window over the beta series
_VOL_W = 252       # window for beta instability
_MKT = "SPY"

METADATA = {
    "name":        "beta_dynamics",
    "description": "Dynamics of 63d rolling beta vs SPY (distinct from the static level): 21d beta trend/slope, 252d beta instability (beta-of-beta), and 21d beta acceleration. Cederburg-O'Doherty 2016 / Hollstein-Prokopczuk 2016 beta-instability pricing.",
    "requires":    ["Date", "Close"],
    "produces":    ["csa_beta_trend_63", "csa_beta_vol_252", "csa_beta_accel_63"],
    "tags":        ["market_regime", "beta", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Cederburg-O'Doherty 2016)",
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
    r_a, m_a = r.align(m_all, join="inner")
    df_dates = dates.values

    mp = int(_BETA_W * 0.7)
    cov = r_a.rolling(_BETA_W, min_periods=mp).cov(m_a)
    var = m_a.rolling(_BETA_W, min_periods=mp).var().replace(0, np.nan)
    beta = (cov / var).clip(-5, 5)

    # Rolling slope of the beta series via cov(beta, time) / var(time) over _TREND_L bars.
    tt = pd.Series(np.arange(len(beta), dtype=float), index=beta.index)
    slope = beta.rolling(_TREND_L, min_periods=15).cov(tt) / \
        tt.rolling(_TREND_L, min_periods=15).var().replace(0, np.nan)

    beta_vol = beta.rolling(_VOL_W, min_periods=120).std()
    beta_accel = beta - beta.shift(_TREND_L)

    df["csa_beta_trend_63"] = slope.clip(-2, 2).reindex(df_dates).values
    df["csa_beta_vol_252"] = beta_vol.clip(0, 5).reindex(df_dates).values
    df["csa_beta_accel_63"] = beta_accel.clip(-5, 5).reindex(df_dates).values

    return df
