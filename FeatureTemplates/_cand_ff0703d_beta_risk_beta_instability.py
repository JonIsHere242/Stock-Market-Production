"""
_cand_ff0703d_beta_risk_beta_instability.py -- Beta INSTABILITY (time-variation of market
exposure), not the beta level itself.

METHOD
------
1. r_i  = Close.pct_change()                         (per-ticker daily return)
2. r_m  = SPY Close.pct_change(), merge_asof backward onto df['Date']  (lookahead-safe)
3. beta20_t = rolling 20-day cov(r_i, r_m) / rolling 20-day var(r_m)   (rolling OLS-beta proxy)
   guarded: var(r_m) < 1e-10 -> NaN
4. beta_instability      = rolling 120-day std of beta20_t  (how much the short beta wanders)
5. beta_instability_chg  = beta_instability(t) - beta_instability(t-40)  (regime-shift flag)
6. beta_instability_norm = beta_instability / (|rolling 120d mean of beta20_t| + 0.25)
   (a coefficient-of-variation reading of beta stability, denominator floored so it never blows up)

All stats are strictly causal rolling windows computed on cumulative history up to t; no
negative shifts, no future information. If SPY data is unavailable the whole block degrades
to NaN (structural columns are still emitted).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff0703d_beta_risk_beta_instability",
    "description": (
        "Time-variation (instability) of a stock's rolling 20-day market beta vs SPY, rather "
        "than the beta level. beta_instability = rolling-120d std of beta20_t "
        "(cov(r_i,r_m,20)/var(r_m,20)); beta_instability_chg = 20-day change of that instability "
        "flagging regime shifts in factor-loading stability; beta_instability_norm normalizes by "
        "the trailing |mean beta| (+0.25 floor) to read as a coefficient of variation. Per-ticker "
        "proxy for market-exposure/factor-loading instability using SPY as the market portfolio "
        "(closest OHLCV-only faithful implementation of the beta-instability method)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703d_beta_instability",
        "ff0703d_beta_instability_chg",
        "ff0703d_beta_instability_norm",
    ],
    "tags": ["beta_risk", "factor", "instability", "regime", "market_exposure"],
    "version": "1.0",
    "author": (
        "feature-factory codegen; faithful per-ticker implementation using SPY as the market "
        "portfolio proxy (rolling OLS-beta via rolling cov/var, all causal)."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    beta_instability = np.full(n, np.nan)
    beta_instability_chg = np.full(n, np.nan)
    beta_instability_norm = np.full(n, np.nan)

    if n == 0 or "Close" not in df.columns:
        df["ff0703d_beta_instability"] = beta_instability
        df["ff0703d_beta_instability_chg"] = beta_instability_chg
        df["ff0703d_beta_instability_norm"] = beta_instability_norm
        return df

    spy_close = _indexes.index_close("SPY")

    if spy_close.empty:
        df["ff0703d_beta_instability"] = beta_instability
        df["ff0703d_beta_instability_chg"] = beta_instability_chg
        df["ff0703d_beta_instability_norm"] = beta_instability_norm
        return df

    dates = pd.to_datetime(df["Date"])
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date")

    # df is documented ascending-by-Date already, so a plain positional merge_asof is safe.
    left = pd.DataFrame({"Date": dates}).reset_index(drop=True)
    merged = pd.merge_asof(left, spy_df, on="Date", direction="backward")
    spy_aligned = merged["spy_close"].to_numpy(dtype="float64")

    r_i = df["Close"].astype("float64").pct_change()
    r_m = pd.Series(spy_aligned, index=df.index).pct_change()

    win_beta = 20
    roll_cov = r_i.rolling(win_beta, min_periods=win_beta).cov(r_m)
    roll_var = r_m.rolling(win_beta, min_periods=win_beta).var()

    var_ok = roll_var.to_numpy(dtype="float64")
    var_ok = np.where(np.isfinite(var_ok) & (var_ok >= 1e-10), var_ok, np.nan)
    beta20 = roll_cov.to_numpy(dtype="float64") / var_ok
    beta20 = pd.Series(beta20, index=df.index)

    win_inst = 120
    min_p_inst = 60
    instability = beta20.rolling(win_inst, min_periods=min_p_inst).std()
    mean_beta = beta20.rolling(win_inst, min_periods=min_p_inst).mean()

    lag = 40
    inst_chg = instability - instability.shift(lag)

    denom = mean_beta.abs() + 0.25
    denom_arr = denom.to_numpy(dtype="float64")
    denom_arr = np.where(np.isfinite(denom_arr) & (denom_arr > 1e-10), denom_arr, np.nan)
    norm = instability.to_numpy(dtype="float64") / denom_arr

    beta_instability = instability.to_numpy(dtype="float64")
    beta_instability_chg = inst_chg.to_numpy(dtype="float64")
    beta_instability_norm = norm

    for arr in (beta_instability, beta_instability_chg, beta_instability_norm):
        bad = ~np.isfinite(arr)
        arr[bad] = np.nan

    df["ff0703d_beta_instability"] = beta_instability
    df["ff0703d_beta_instability_chg"] = beta_instability_chg
    df["ff0703d_beta_instability_norm"] = beta_instability_norm
    return df
