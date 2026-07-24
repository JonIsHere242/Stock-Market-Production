"""
ff06280034b_beta_risk_full_beta_spy_120
Rolling 120-day full/downside/upside beta vs SPY, plus short-vs-long ratio and
down-minus-up asymmetry.  All computations are per-ticker causal (no lookahead).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by file path (never import as package)
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
    "name": "ff06280034b_beta_risk_full_beta_spy_120",
    "description": (
        "Rolling 120-day full OLS beta of stock daily returns vs SPY returns, "
        "plus short-window (30d) vs long-window (120d) beta ratio as a trend "
        "indicator, and downside-minus-upside beta asymmetry. "
        "Downside beta uses only days when SPY return < 0; upside uses SPY > 0. "
        "Per-ticker proxy; merged with SPY via merge_asof (backward, no lookahead)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_beta_risk_full_beta_spy_120_full",
        "ff06280034b_beta_risk_full_beta_spy_120_short_long_ratio",
        "ff06280034b_beta_risk_full_beta_spy_120_asym",
    ],
    "tags": ["beta", "risk", "spy", "downside", "upside", "asymmetry"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}

_WINDOW_LONG = 120
_WINDOW_SHORT = 30
_MIN_OBS = 20   # minimum valid observations inside a window


def _rolling_beta(stock_ret: np.ndarray, market_ret: np.ndarray,
                  window: int, min_obs: int) -> np.ndarray:
    """
    Vectorised rolling OLS beta via the cov/var formula.
    beta = cov(r_stock, r_market) / var(r_market)
    Uses a two-pass sliding window with numpy stride tricks for speed.
    """
    n = len(stock_ret)
    out = np.full(n, np.nan)
    if n < min_obs:
        return out

    for i in range(window - 1, n):
        s = stock_ret[i - window + 1: i + 1]
        m = market_ret[i - window + 1: i + 1]
        valid = np.isfinite(s) & np.isfinite(m)
        if valid.sum() < min_obs:
            continue
        sv, mv = s[valid], m[valid]
        mv_mean = mv.mean()
        sv_mean = sv.mean()
        denom = np.sum((mv - mv_mean) ** 2)
        if denom == 0.0 or not np.isfinite(denom):
            continue
        numer = np.sum((sv - sv_mean) * (mv - mv_mean))
        out[i] = numer / denom

    return out


def _rolling_conditional_beta(stock_ret: np.ndarray, market_ret: np.ndarray,
                               window: int, min_obs: int,
                               side: str) -> np.ndarray:
    """
    Rolling conditional beta.
    side='down' → use only bars where market_ret < 0
    side='up'   → use only bars where market_ret > 0
    """
    n = len(stock_ret)
    out = np.full(n, np.nan)
    if n < min_obs:
        return out

    for i in range(window - 1, n):
        s = stock_ret[i - window + 1: i + 1]
        m = market_ret[i - window + 1: i + 1]
        valid = np.isfinite(s) & np.isfinite(m)
        if side == "down":
            mask = valid & (m < 0.0)
        else:
            mask = valid & (m > 0.0)
        if mask.sum() < min_obs:
            continue
        sv, mv = s[mask], m[mask]
        mv_mean = mv.mean()
        sv_mean = sv.mean()
        denom = np.sum((mv - mv_mean) ** 2)
        if denom == 0.0 or not np.isfinite(denom):
            continue
        numer = np.sum((sv - sv_mean) * (mv - mv_mean))
        out[i] = numer / denom

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_full = "ff06280034b_beta_risk_full_beta_spy_120_full"
    col_ratio = "ff06280034b_beta_risk_full_beta_spy_120_short_long_ratio"
    col_asym = "ff06280034b_beta_risk_full_beta_spy_120_asym"

    # Initialise outputs to NaN (required on every code path)
    df[col_full] = np.nan
    df[col_ratio] = np.nan
    df[col_asym] = np.nan

    if len(df) < _MIN_OBS:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY closes and align via merge_asof (backward = no lookahead)
    # ------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_series is None or len(spy_series) == 0:
        return df

    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date").reset_index(drop=False)  # keep original index

    merged = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Daily log-returns (causal: shift(1) looks back)
    merged["stock_ret"] = np.log(
        merged["Close"] / merged["Close"].shift(1)
    )
    merged["spy_ret"] = np.log(
        merged["spy_close"] / merged["spy_close"].shift(1)
    )

    stock_ret = merged["stock_ret"].to_numpy(dtype=float)
    spy_ret = merged["spy_ret"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Rolling betas
    # ------------------------------------------------------------------
    beta_long = _rolling_beta(stock_ret, spy_ret, _WINDOW_LONG, _MIN_OBS)
    beta_short = _rolling_beta(stock_ret, spy_ret, _WINDOW_SHORT, _MIN_OBS)
    beta_down = _rolling_conditional_beta(stock_ret, spy_ret, _WINDOW_LONG, _MIN_OBS, "down")
    beta_up = _rolling_conditional_beta(stock_ret, spy_ret, _WINDOW_LONG, _MIN_OBS, "up")

    # short-vs-long ratio: momentum of beta (>1 means beta rising)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (beta_long != 0) & np.isfinite(beta_long) & np.isfinite(beta_short),
            beta_short / beta_long,
            np.nan,
        )

    # downside minus upside asymmetry
    asym = np.where(
        np.isfinite(beta_down) & np.isfinite(beta_up),
        beta_down - beta_up,
        np.nan,
    )

    # Map results back to original df index via the preserved original index
    orig_index = merged["index"].to_numpy()
    df.loc[orig_index, col_full] = beta_long
    df.loc[orig_index, col_ratio] = ratio
    df.loc[orig_index, col_asym] = asym

    # Replace any inf values with NaN
    for col in [col_full, col_ratio, col_asym]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
