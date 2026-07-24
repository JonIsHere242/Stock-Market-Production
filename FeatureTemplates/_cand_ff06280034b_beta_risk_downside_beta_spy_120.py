"""
Downside / upside beta vs SPY over a rolling 120-day window.

Produces three columns:
  ff06280034b_beta_risk_downside_beta_spy_120_down   : beta estimated on SPY-down days only
  ff06280034b_beta_risk_downside_beta_spy_120_asym   : down-beta minus up-beta (asymmetry)
  ff06280034b_beta_risk_downside_beta_spy_120_ratio  : short (30d) down-beta / long (120d) down-beta
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path – required by sandbox rules)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034b_beta_risk_downside_beta_spy_120",
    "description": (
        "Per-ticker rolling 120-day downside beta vs SPY (beta on SPY-down days), "
        "upside beta (SPY-up days), down-minus-up asymmetry, and a 30d/120d down-beta "
        "ratio that captures regime change in tail sensitivity. "
        "Proxy: all computations are per-stock; cross-sectional ranking is not applied."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_beta_risk_downside_beta_spy_120_down",
        "ff06280034b_beta_risk_downside_beta_spy_120_asym",
        "ff06280034b_beta_risk_downside_beta_spy_120_ratio",
    ],
    "tags": ["beta", "downside_risk", "spy", "rolling", "asymmetry"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LONG_WIN = 120
_SHORT_WIN = 30
_MIN_OBS = 15          # minimum SPY-conditioned observations required for a valid beta


def _rolling_cond_beta(
    stock_ret: np.ndarray,
    spy_ret: np.ndarray,
    window: int,
    condition: np.ndarray,   # boolean mask – True = include this bar
    min_obs: int,
) -> np.ndarray:
    """
    Rolling covariance/variance beta using only bars where `condition` is True.
    Uses a direct O(n*window) pass but over at most 120 bars per step, which is
    well within the <100 ms budget for ~700 rows.
    """
    n = len(stock_ret)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = stock_ret[t - window + 1 : t + 1]
        m = spy_ret[t - window + 1 : t + 1]
        c = condition[t - window + 1 : t + 1]
        s_c = s[c]
        m_c = m[c]
        if len(s_c) < min_obs:
            continue
        m_mean = m_c.mean()
        s_mean = s_c.mean()
        cov = ((s_c - s_mean) * (m_c - m_mean)).mean()
        var = ((m_c - m_mean) ** 2).mean()
        if var == 0:
            continue
        out[t] = cov / var
    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path
    col_down = "ff06280034b_beta_risk_downside_beta_spy_120_down"
    col_asym = "ff06280034b_beta_risk_downside_beta_spy_120_asym"
    col_ratio = "ff06280034b_beta_risk_downside_beta_spy_120_ratio"

    df[col_down] = np.nan
    df[col_asym] = np.nan
    df[col_ratio] = np.nan

    if len(df) < _LONG_WIN + 1:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY index closes and align via merge_asof (backward)
    # ------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_series is None or spy_series.empty:
        return df

    spy_df = spy_series.rename("spy_close").reset_index()
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
    # Restore original row order
    work = work.set_index(df.index)

    # ------------------------------------------------------------------
    # 2. Daily log-returns (current / prior) – causal, no lookahead
    # ------------------------------------------------------------------
    stock_ret = np.log(work["Close"] / work["Close"].shift(1)).to_numpy(dtype=float)
    spy_ret = np.log(work["spy_close"] / work["spy_close"].shift(1)).to_numpy(dtype=float)

    # Condition masks (based on SPY return direction)
    spy_down = spy_ret < 0          # True on SPY-down days
    spy_up = spy_ret >= 0           # True on SPY-up (flat included with up)

    # ------------------------------------------------------------------
    # 3. Rolling conditional betas
    # ------------------------------------------------------------------
    beta_down_long = _rolling_cond_beta(stock_ret, spy_ret, _LONG_WIN, spy_down, _MIN_OBS)
    beta_up_long   = _rolling_cond_beta(stock_ret, spy_ret, _LONG_WIN, spy_up,   _MIN_OBS)
    beta_down_short = _rolling_cond_beta(stock_ret, spy_ret, _SHORT_WIN, spy_down, max(5, _MIN_OBS // 3))

    # ------------------------------------------------------------------
    # 4. Derived signals
    # ------------------------------------------------------------------
    # Down-beta (level)
    df[col_down] = beta_down_long

    # Asymmetry: down-beta minus up-beta (positive = more sensitive on drawdowns)
    asym = beta_down_long - beta_up_long
    df[col_asym] = asym

    # Short/long ratio – captures whether downside tail sensitivity is
    # expanding (>1) or contracting (<1) relative to the longer-term norm
    ratio = np.where(
        np.abs(beta_down_long) > 1e-9,
        beta_down_short / beta_down_long,
        np.nan,
    )
    df[col_ratio] = ratio

    return df
