"""
rgi_vix_momentum_interaction.py  --  Stock behaviour CONDITIONED on the VIX regime.

The same price move means different things in calm vs panicked markets.  This block
builds INTERACTION features: stock-specific quantities multiplied/gated by the VIX
risk regime.  These are the signal-orthogonal part of the theme — they encode
"momentum works in low vol / mean-reverts in high vol" style state-dependence that a
plain momentum feature cannot express.

Self-contained: it recomputes its own VIX 250d own-history high/low-risk flags
(same logic as rgi_vix_regime_state, kept local so this block has no inter-block
ordering dependency) and combines them with the stock's own 20d momentum, short-
horizon reversal, and downside semivariance.

Produces:
  - rgi_vix_mom20_x_high : 20d return * high-vix flag (momentum carried in stress)
  - rgi_vix_mom20_x_low  : 20d return * low-vix  flag (momentum carried in calm)
  - rgi_vix_rev5_x_high  : negative 5d return * high-vix flag (panic-reversal setup)
  - rgi_vix_semivar_high : 20d downside semivariance, gated to high-vix regime
  - rgi_vix_semivar_low  : 20d downside semivariance, gated to low-vix  regime
  - rgi_vix_mom_regime_spread : mom20 signed by regime (low - high contribution)

All VIX is past-only (backward merge_asof). df order untouched.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "rgi_vix_momentum_interaction",
    "description": "Stock momentum / reversal / downside-semivariance interacted with the VIX high/low risk regime.",
    "requires":    ["Date", "Close"],
    "produces":    [
        "rgi_vix_mom20_x_high",
        "rgi_vix_mom20_x_low",
        "rgi_vix_rev5_x_high",
        "rgi_vix_semivar_high",
        "rgi_vix_semivar_low",
        "rgi_vix_mom_regime_spread",
    ],
    "tags":        ["market_regime", "momentum", "interaction", "rgi"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def _aligned_vix(df: pd.DataFrame) -> pd.Series:
    vix_daily = _indexes.vix_daily_close()
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    if vix_daily.empty:
        return pd.Series(np.nan, index=df.index)
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    return pd.Series(merged["vix_close"].ffill().to_numpy(), index=df.index)


def _trailing_pctl(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _rank(arr):
        if len(arr) < 2:
            return np.nan
        return float((arr[:-1] <= arr[-1]).mean())
    return s.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    vc = _aligned_vix(df)

    # Local VIX regime flags (own-history 250d percentile; NaN until enough data).
    p250 = _trailing_pctl(vc, 250, 60)
    high_flag = np.where(p250.notna(), (p250 >= 0.80).astype(float), np.nan)
    low_flag = np.where(p250.notna(), (p250 <= 0.20).astype(float), np.nan)
    high_flag = pd.Series(high_flag, index=df.index)
    low_flag = pd.Series(low_flag, index=df.index)

    close = df["Close"]
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    # Stock 20d momentum (cumulative log return) and 5d short reversal input.
    mom20 = log_ret.rolling(20, min_periods=15).sum()
    ret5 = log_ret.rolling(5, min_periods=4).sum()

    new = {}

    # Momentum carried in each regime: zero contribution when not in that regime,
    # NaN when the regime is unknown (so the model never sees a fake 0).
    new["rgi_vix_mom20_x_high"] = (mom20 * high_flag)
    new["rgi_vix_mom20_x_low"] = (mom20 * low_flag)

    # Panic-reversal setup: magnitude of a NEGATIVE 5d move, only in high vol.
    neg_ret5 = (-ret5).clip(lower=0.0)  # positive when stock fell over 5d
    new["rgi_vix_rev5_x_high"] = (neg_ret5 * high_flag)

    # Downside semivariance over 20d: mean of squared negative daily returns.
    neg_sq = log_ret.clip(upper=0.0) ** 2
    semivar20 = neg_sq.rolling(20, min_periods=15).mean()
    new["rgi_vix_semivar_high"] = (semivar20 * high_flag)
    new["rgi_vix_semivar_low"] = (semivar20 * low_flag)

    # Signed regime spread: momentum counted positively in calm, negatively in
    # stress (captures the calm-trend / stress-reverse asymmetry in one column).
    # Defined only where at least one regime flag is known.
    known = high_flag.notna() | low_flag.notna()
    spread = mom20 * (low_flag.fillna(0.0) - high_flag.fillna(0.0))
    new["rgi_vix_mom_regime_spread"] = pd.Series(
        np.where(known, spread, np.nan), index=df.index
    )

    for k in new:
        new[k] = new[k].replace([np.inf, -np.inf], np.nan)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
