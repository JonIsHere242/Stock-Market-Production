"""
ff06280034b_beta_risk_downside_beta_iwm_120

Downside/upside beta vs IWM over a rolling 120-day window.

Three columns:
  _beta_down   – downside beta (covar(ret, IWM_ret | IWM_ret < 0) / var(IWM_ret | IWM_ret < 0))
  _beta_up     – upside beta   (same, but IWM_ret >= 0)
  _asymmetry   – downside_beta − upside_beta  (positive = amplifies drawdowns vs rallies)

Per-ticker proxy: cross-sectional ranking is not needed; the asymmetry already
captures the economically meaningful skew in systematic risk loading.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by file path (never via package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034b_beta_risk_downside_beta_iwm_120",
    "description": (
        "Rolling 120-day downside/upside beta vs IWM small-cap index. "
        "Downside beta uses only days when IWM fell; upside beta uses only days "
        "when IWM rose. Asymmetry = downside − upside captures convexity in "
        "systematic risk exposure. Per-ticker causal proxy (no cross-sectional "
        "ranking needed — the raw betas carry the economic signal)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_beta_risk_downside_beta_iwm_120_beta_down",
        "ff06280034b_beta_risk_downside_beta_iwm_120_beta_up",
        "ff06280034b_beta_risk_downside_beta_iwm_120_asymmetry",
    ],
    "tags": ["beta", "risk", "downside", "IWM", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}

_WINDOW = 120
_COL_DOWN = "ff06280034b_beta_risk_downside_beta_iwm_120_beta_down"
_COL_UP   = "ff06280034b_beta_risk_downside_beta_iwm_120_beta_up"
_COL_ASYM = "ff06280034b_beta_risk_downside_beta_iwm_120_asymmetry"


def _rolling_cond_beta(
    stock_ret: np.ndarray,
    mkt_ret: np.ndarray,
    condition: np.ndarray,   # boolean mask selecting which bars to include
    window: int,
) -> np.ndarray:
    """
    Rolling conditional beta over `window` bars.
    At each bar t, beta = cov(stock[t-w+1..t][cond], mkt[t-w+1..t][cond])
                        / var(mkt[t-w+1..t][cond])
    where cond is the condition evaluated over the SAME window.
    Returns NaN when fewer than 5 observations pass the condition in the window.
    """
    n = len(stock_ret)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = stock_ret[t - window + 1 : t + 1]
        m = mkt_ret[t - window + 1 : t + 1]
        c = condition[t - window + 1 : t + 1]
        sc, mc = s[c], m[c]
        if len(mc) < 5:
            continue
        var_m = np.var(mc, ddof=1)
        if var_m == 0 or np.isnan(var_m):
            continue
        cov_sm = np.cov(sc, mc, ddof=1)[0, 1]
        out[t] = cov_sm / var_m
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_DOWN] = np.nan
    df[_COL_UP]   = np.nan
    df[_COL_ASYM] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # -----------------------------------------------------------------------
    # Fetch IWM close and align via merge_asof (backward = no lookahead)
    # -----------------------------------------------------------------------
    try:
        iwm_close = _indexes.index_close("IWM")
        if iwm_close is None or iwm_close.empty:
            return df
        iwm_df = iwm_close.reset_index()
        iwm_df.columns = ["Date", "iwm_close"]
    except Exception:
        return df

    # Ensure Date columns are the same dtype for merge_asof
    stock = df[["Date", "Close"]].copy()
    stock["Date"] = pd.to_datetime(stock["Date"])
    iwm_df["Date"] = pd.to_datetime(iwm_df["Date"])

    merged = pd.merge_asof(
        stock.sort_values("Date"),
        iwm_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    if merged["iwm_close"].isna().all():
        return df

    # -----------------------------------------------------------------------
    # Compute daily log returns (shift(1) = fully causal)
    # -----------------------------------------------------------------------
    stock_ret = np.log(
        merged["Close"].values / np.where(
            merged["Close"].shift(1).values == 0, np.nan, merged["Close"].shift(1).values
        )
    )
    iwm_ret = np.log(
        merged["iwm_close"].values / np.where(
            merged["iwm_close"].shift(1).values == 0, np.nan, merged["iwm_close"].shift(1).values
        )
    )

    # Condition arrays (element-wise on returns array)
    mkt_down = iwm_ret < 0          # IWM fell on that day
    mkt_up   = iwm_ret >= 0         # IWM rose (or flat) on that day

    beta_down = _rolling_cond_beta(stock_ret, iwm_ret, mkt_down, _WINDOW)
    beta_up   = _rolling_cond_beta(stock_ret, iwm_ret, mkt_up,   _WINDOW)
    asymmetry = beta_down - beta_up  # NaN propagates naturally

    # -----------------------------------------------------------------------
    # Write back into df (original row order) via Date alignment
    # -----------------------------------------------------------------------
    result = merged[["Date"]].copy()
    result[_COL_DOWN] = beta_down
    result[_COL_UP]   = beta_up
    result[_COL_ASYM] = asymmetry

    # Merge back to original df index order
    df = df.merge(result, on="Date", how="left", suffixes=("_orig", ""))

    # Drop any duplicate columns that might appear if df already had the cols
    for col in [_COL_DOWN, _COL_UP, _COL_ASYM]:
        dup = col + "_orig"
        if dup in df.columns:
            df.drop(columns=[dup], inplace=True)

    # Guard: replace inf/-inf with NaN
    for col in [_COL_DOWN, _COL_UP, _COL_ASYM]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
