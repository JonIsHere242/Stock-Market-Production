"""
Bollerslev-Li-Patton signed semibeta decomposition vs SPY over 120 days.

Decomposes covariance by sign quadrant:
  - beta_N (concordant-down): both stock and market down  -> systematic downside co-movement
  - beta_P (concordant-up):   both stock and market up    -> systematic upside co-movement
  - beta_M (discordant-mixed): sign mismatch (stock up/mkt down OR stock down/mkt up) -> diversification regime

Per-ticker rolling 120-day window; causal/no-lookahead.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load _indexes helper by path ------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06280034b_beta_risk_semibeta_signed_120",
    "description": (
        "Bollerslev-Li-Patton signed semibeta decomposition vs SPY over a 120-day rolling window. "
        "Splits total OLS beta into three sign-quadrant components: beta_N (concordant-down: both "
        "stock and market return negative), beta_P (concordant-up: both positive), and beta_M "
        "(discordant/mixed: sign mismatch). beta_N captures crash co-movement; beta_P captures "
        "rally participation; beta_M captures diversification/hedge quality. Per-ticker proxy -- "
        "no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_beta_risk_semibeta_signed_120_beta_n",
        "ff06280034b_beta_risk_semibeta_signed_120_beta_p",
        "ff06280034b_beta_risk_semibeta_signed_120_beta_m",
    ],
    "tags": ["beta", "semibeta", "downside_risk", "market_structure", "120d"],
    "version": "1.0.0",
    "author": "feature-factory",
}

_COL_N = "ff06280034b_beta_risk_semibeta_signed_120_beta_n"
_COL_P = "ff06280034b_beta_risk_semibeta_signed_120_beta_p"
_COL_M = "ff06280034b_beta_risk_semibeta_signed_120_beta_m"

_WINDOW = 120


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs as NaN on every code path
    df[_COL_N] = np.nan
    df[_COL_P] = np.nan
    df[_COL_M] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ---- align SPY returns -------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build a SPY return series aligned to df dates
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    work = work.set_index(df.index)

    stock_ret = work["Close"].pct_change()
    mkt_ret = work["spy_close"].pct_change()

    # ----- rolling signed semibeta computation ------------------------------------
    # For each window ending at t (inclusive), separate observations into 3 quadrants:
    #   N: stock_ret < 0 AND mkt_ret < 0  (concordant-down)
    #   P: stock_ret > 0 AND mkt_ret > 0  (concordant-up)
    #   M: sign mismatch (discordant)
    #
    # Semibeta = cov(r_stock, r_mkt | quadrant) / var(r_mkt | quadrant)
    # We use a rolling vectorised approach with pandas rolling apply.

    n = len(stock_ret)
    beta_n_arr = np.full(n, np.nan)
    beta_p_arr = np.full(n, np.nan)
    beta_m_arr = np.full(n, np.nan)

    sr = stock_ret.values
    mr = mkt_ret.values

    for t in range(_WINDOW - 1, n):
        rs = sr[t - _WINDOW + 1 : t + 1]
        rm = mr[t - _WINDOW + 1 : t + 1]

        valid = np.isfinite(rs) & np.isfinite(rm)
        rs_v = rs[valid]
        rm_v = rm[valid]

        if len(rs_v) < 10:
            continue

        # Quadrant masks
        mask_n = (rs_v < 0) & (rm_v < 0)   # concordant-down
        mask_p = (rs_v > 0) & (rm_v > 0)   # concordant-up
        mask_m = ~mask_n & ~mask_p           # discordant / mixed

        # helper: beta from two aligned arrays (covariance / variance of mkt)
        def _semibeta(rs_q, rm_q):
            if len(rm_q) < 4:
                return np.nan
            var_m = np.var(rm_q, ddof=1)
            if var_m == 0 or not np.isfinite(var_m):
                return np.nan
            cov_val = np.cov(rs_q, rm_q, ddof=1)[0, 1]
            return cov_val / var_m

        beta_n_arr[t] = _semibeta(rs_v[mask_n], rm_v[mask_n])
        beta_p_arr[t] = _semibeta(rs_v[mask_p], rm_v[mask_p])
        beta_m_arr[t] = _semibeta(rs_v[mask_m], rm_v[mask_m])

    df[_COL_N] = beta_n_arr
    df[_COL_P] = beta_p_arr
    df[_COL_M] = beta_m_arr

    return df
