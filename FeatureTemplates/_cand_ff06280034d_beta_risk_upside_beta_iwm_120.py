"""
Upside/Downside Beta vs IWM (120-day rolling).

Computes rolling 120-day betas of the stock's daily returns against IWM
(Russell 2000 ETF) separately for days when IWM was up (upside beta) and
days when IWM was down (downside beta), plus a full-period beta. Produces:

  * ff06280034d_beta_risk_upside_beta_iwm_120_up   -- upside beta (IWM-up days only)
  * ff06280034d_beta_risk_upside_beta_iwm_120_asym -- downside minus upside beta (asymmetry)
  * ff06280034d_beta_risk_upside_beta_iwm_120_ratio -- short(60d)/long(120d) upside-beta ratio

Per-ticker proxy; cross-sectional rank not applied.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by path
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034d_beta_risk_upside_beta_iwm_120",
    "description": (
        "Rolling 120-day upside / downside beta vs IWM. "
        "Upside beta uses only days when IWM return > 0; downside uses IWM return < 0. "
        "Produces: upside_beta level, down-minus-up asymmetry, and 60d/120d upside-beta ratio. "
        "Per-ticker causal proxy; no cross-sectional ranking."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034d_beta_risk_upside_beta_iwm_120_up",
        "ff06280034d_beta_risk_upside_beta_iwm_120_asym",
        "ff06280034d_beta_risk_upside_beta_iwm_120_ratio",
    ],
    "tags": ["beta", "risk", "iwm", "upside_beta", "downside_beta", "asymmetry"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034d",
}

_WINDOW_LONG = 120
_WINDOW_SHORT = 60
_MIN_OBS = 20   # minimum valid observations within window for beta calc


def _rolling_ols_beta(
    y: np.ndarray, x: np.ndarray, window: int, min_obs: int
) -> np.ndarray:
    """
    Rolling OLS beta (slope of y on x) without intercept adjustment.
    Uses the formula: beta = cov(y,x) / var(x).
    Computes over a sliding window; returns array of same length as y, NaN for
    bars with insufficient data.
    """
    n = len(y)
    betas = np.full(n, np.nan)
    for i in range(window - 1, n):
        ywin = y[i - window + 1 : i + 1]
        xwin = x[i - window + 1 : i + 1]
        # mask NaNs
        mask = np.isfinite(ywin) & np.isfinite(xwin)
        if mask.sum() < min_obs:
            continue
        yw = ywin[mask]
        xw = xwin[mask]
        xvar = np.var(xw, ddof=1)
        if xvar == 0 or not np.isfinite(xvar):
            continue
        xcov = np.cov(yw, xw, ddof=1)[0, 1]
        betas[i] = xcov / xvar
    return betas


def _rolling_conditional_beta(
    stock_ret: np.ndarray,
    idx_ret: np.ndarray,
    window: int,
    min_obs: int,
    condition: str,  # "up" or "down"
) -> np.ndarray:
    """
    Rolling beta conditioned on the direction of index returns.
    condition="up"  -> only bars where idx_ret > 0
    condition="down" -> only bars where idx_ret < 0
    """
    n = len(stock_ret)
    betas = np.full(n, np.nan)
    for i in range(window - 1, n):
        ywin = stock_ret[i - window + 1 : i + 1]
        xwin = idx_ret[i - window + 1 : i + 1]
        mask_valid = np.isfinite(ywin) & np.isfinite(xwin)
        if condition == "up":
            mask_dir = xwin > 0
        else:
            mask_dir = xwin < 0
        mask = mask_valid & mask_dir
        if mask.sum() < min_obs:
            continue
        yw = ywin[mask]
        xw = xwin[mask]
        xvar = np.var(xw, ddof=1)
        if xvar == 0 or not np.isfinite(xvar):
            continue
        xcov = np.cov(yw, xw, ddof=1)[0, 1]
        betas[i] = xcov / xvar
    return betas


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-initialise all produced columns to NaN (required on all code paths)
    col_up = "ff06280034d_beta_risk_upside_beta_iwm_120_up"
    col_asym = "ff06280034d_beta_risk_upside_beta_iwm_120_asym"
    col_ratio = "ff06280034d_beta_risk_upside_beta_iwm_120_ratio"
    df[col_up] = np.nan
    df[col_asym] = np.nan
    df[col_ratio] = np.nan

    if len(df) < _WINDOW_LONG + 2:
        return df

    # Fetch IWM close series (DatetimeIndex)
    try:
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df

    if iwm_close is None or len(iwm_close) == 0:
        return df

    # Build IWM returns frame and merge_asof with df on Date (backward-safe)
    iwm_ret_df = pd.DataFrame({
        "Date": pd.to_datetime(iwm_close.index),
        "_iwm_ret": iwm_close.pct_change().values,
    }).dropna(subset=["Date"])

    # Work on a copy with DatetimeIndex-safe Date column
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date").reset_index(drop=True)

    iwm_ret_df = iwm_ret_df.sort_values("Date")
    merged = pd.merge_asof(
        work,
        iwm_ret_df,
        on="Date",
        direction="backward",
    )

    # Stock daily returns (pct_change causal -- only looks back 1 bar)
    merged["_stk_ret"] = merged["Close"].pct_change()

    stk = merged["_stk_ret"].to_numpy(dtype=float)
    iwm = merged["_iwm_ret"].to_numpy(dtype=float)

    # Upside beta (120d)
    up_beta_long = _rolling_conditional_beta(stk, iwm, _WINDOW_LONG, _MIN_OBS, "up")
    # Downside beta (120d)
    dn_beta_long = _rolling_conditional_beta(stk, iwm, _WINDOW_LONG, _MIN_OBS, "down")
    # Upside beta (60d) for ratio
    up_beta_short = _rolling_conditional_beta(stk, iwm, _WINDOW_SHORT, _MIN_OBS // 2, "up")

    # Asymmetry: downside - upside (positive = amplified downside exposure)
    asym = np.where(
        np.isfinite(dn_beta_long) & np.isfinite(up_beta_long),
        dn_beta_long - up_beta_long,
        np.nan,
    )

    # Ratio: short / long upside beta; guard zero denominator
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            np.isfinite(up_beta_short) & np.isfinite(up_beta_long) & (up_beta_long != 0),
            up_beta_short / up_beta_long,
            np.nan,
        )

    # Map back onto original df rows via Date alignment
    result_df = pd.DataFrame({
        "Date": merged["Date"],
        col_up: up_beta_long,
        col_asym: asym,
        col_ratio: ratio,
    })

    orig_dates = pd.to_datetime(df["Date"])
    result_df = result_df.set_index("Date")

    # Assign back using index lookup (preserves original df row order)
    mapped_up = orig_dates.map(result_df[col_up])
    mapped_asym = orig_dates.map(result_df[col_asym])
    mapped_ratio = orig_dates.map(result_df[col_ratio])

    df[col_up] = mapped_up.values
    df[col_asym] = mapped_asym.values
    df[col_ratio] = mapped_ratio.values

    return df
