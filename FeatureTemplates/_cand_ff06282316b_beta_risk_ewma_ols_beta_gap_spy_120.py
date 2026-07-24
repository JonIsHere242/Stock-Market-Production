"""
Feature: ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120
Vein: beta_risk
Batch: 06282316b

Produces the gap between EWMA time-varying beta (half-life ~20 days) and a
rolling 120-day OLS beta vs SPY. A positive gap means the stock's recent
sensitivity to the market has spiked above its long-run average (momentum in
beta); a negative gap means it has faded. The EWMA beta itself is also
emitted as a stand-alone feature.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load _indexes helper ───────────────────────────────────────────────────
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120",
    "description": (
        "EWMA beta (half-life 20 d) vs SPY minus rolling-120-day OLS beta vs SPY. "
        "Captures how fast-adapting recent market sensitivity diverges from the slow "
        "long-run beta. Positive = recent beta higher than average (beta surge); "
        "negative = beta compression. Per-ticker, causal, no lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_ewma_beta",
        "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_ols_beta",
        "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_gap",
    ],
    "tags": ["beta", "risk", "ewma", "ols", "spy", "market_sensitivity"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

_HALFLIFE = 20       # EWMA half-life in days
_OLS_WIN  = 120      # rolling OLS window in days
_EPS      = 1e-10    # guard against zero-variance denominator


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out_ewma = "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_ewma_beta"
    out_ols  = "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_ols_beta"
    out_gap  = "ff06282316b_beta_risk_ewma_ols_beta_gap_spy_120_gap"

    # Initialise all produced columns to NaN so every code path is covered.
    df[out_ewma] = np.nan
    df[out_ols]  = np.nan
    df[out_gap]  = np.nan

    if len(df) < 2:
        return df

    # ── fetch SPY close and merge asof ────────────────────────────────────
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.reset_index()
    # index_close returns a Series with DatetimeIndex; reset to DataFrame
    if "Date" not in spy_df.columns:
        spy_df.columns = ["Date", "spy_close"]
    else:
        spy_df = spy_df.rename(columns={spy_df.columns[1]: "spy_close"})

    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df[["Date", "spy_close"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # ── compute daily log-returns (causal; no lookahead) ──────────────────
    ticker_ret = np.log(work["Close"] / work["Close"].shift(1))
    spy_ret    = np.log(work["spy_close"] / work["spy_close"].shift(1))

    # ── EWMA beta = EWMA_cov(ticker, spy) / EWMA_var(spy) ────────────────
    # Use pandas ewm with span derived from half-life: span = 2*hl - 1 or use halflife=
    alpha     = 1.0 - np.exp(-np.log(2) / _HALFLIFE)  # decay per day
    ewm_kwargs = dict(halflife=_HALFLIFE, min_periods=2, adjust=True)

    ewma_cov = ticker_ret.ewm(**ewm_kwargs).cov(spy_ret)
    ewma_var = spy_ret.ewm(**ewm_kwargs).var()

    ewma_beta_arr = ewma_cov / ewma_var.replace(0, np.nan).where(ewma_var.abs() > _EPS, np.nan)

    # ── rolling 120-day OLS beta = cov(ticker, spy) / var(spy) ───────────
    # Vectorised via rolling; this equals the slope of OLS with intercept
    # because cov/var = slope of the OLS regression.
    roll_cov = ticker_ret.rolling(_OLS_WIN, min_periods=max(20, _OLS_WIN // 4)).cov(spy_ret)
    roll_var = spy_ret.rolling(_OLS_WIN, min_periods=max(20, _OLS_WIN // 4)).var()

    ols_beta_arr = roll_cov / roll_var.replace(0, np.nan).where(roll_var.abs() > _EPS, np.nan)

    # ── re-align to original df index (merge_asof may have re-sorted) ────
    # work was sorted by Date; align by positional index back to df order
    # We build a Date->value map and map onto df.
    date_col = pd.to_datetime(df["Date"])

    ewma_map = pd.Series(ewma_beta_arr.values, index=pd.to_datetime(work["Date"]))
    ols_map  = pd.Series(ols_beta_arr.values,  index=pd.to_datetime(work["Date"]))

    ewma_aligned = date_col.map(ewma_map)
    ols_aligned  = date_col.map(ols_map)

    df[out_ewma] = ewma_aligned.values
    df[out_ols]  = ols_aligned.values
    df[out_gap]  = ewma_aligned.values - ols_aligned.values

    return df
