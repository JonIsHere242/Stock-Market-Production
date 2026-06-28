"""
_paper_2605_23962_index_relative_transfer.py
============================================

Inspired by: "From Index to Equity: Pre-Training Transformers for Stock Return
Prediction" (arXiv 2605.23962).

The paper's core insight: the market index carries dynamics that TRANSFER to
individual equities.  This block operationalises that idea with a per-ticker
OHLCV + SPY proxy using only standard statistics (no transformer, no pre-training
overhead).

SIGNALS COMPUTED
----------------
ixr_lead_lag_corr_21d  : corr(stock_ret[t], spy_ret[t-1]) over 21-day window.
                         Positive = SPY's yesterday predicts today's stock return
                         (index leads the equity).
ixr_beta_21d           : Short-window rolling beta of stock vs SPY (OLS slope),
                         21-day window.  More reactive than the 60-day block in
                         beta_metrics.py.
ixr_beta_change_21v63  : ixr_beta_21d minus its own 63-day trailing mean — reveals
                         recent beta drift, a transfer-intensity signal.
ixr_transfer_drift     : SPY trailing 21-day return × ixr_beta_21d = "expected
                         drift transferred" from the index to this stock.
ixr_rel_strength_21d   : Stock cumulative log return minus SPY cumulative log
                         return over the last 21 days (relative strength).
ixr_rel_strength_trend : 5-day change in ixr_rel_strength_21d — is the stock
                         gaining or losing ground vs the index?
ixr_beta_asym          : Downside beta (SPY-down days, 63-day window) minus
                         upside beta (SPY-up days, 63-day window).  Positive = stock
                         falls harder than it rises vs SPY.

IMPORTANT NOTES
---------------
- UNPROVEN candidate — leading _ keeps this out of auto-discovery.  Promote by
  renaming to remove the leading underscore after multi-seed validation.
- Honest "per-ticker index-relative transfer proxy" — these are structural OHLCV
  proxies, NOT a transferred transformer embedding.
- All signals use only on-or-before-t data (causality-safe, backward merge_asof).
- If SPY is unavailable the block falls back to the first available symbol via
  _indexes.available().  If no symbols are available, all produced columns are NaN.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared index helper by path (auto-discovery skips it).
# Mirrors the idiom in vix_features.py and beta_metrics.py exactly.
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2605_23962_index_relative_transfer",
    "description": (
        "Per-ticker index-relative transfer proxy inspired by arXiv 2605.23962: "
        "lead-lag cross-correlation, short-window rolling beta + drift, index-momentum "
        "pass-through, relative strength vs SPY, and upside/downside beta asymmetry."
    ),
    "requires": ["Date", "Close"],
    "produces": [
        "ixr_lead_lag_corr_21d",
        "ixr_beta_21d",
        "ixr_beta_change_21v63",
        "ixr_transfer_drift",
        "ixr_rel_strength_21d",
        "ixr_rel_strength_trend",
        "ixr_beta_asym",
    ],
    "tags": ["market_regime", "momentum", "beta", "experimental"],
    "version": "1.0",
    "author": "paper arXiv 2605.23962 — index-to-equity transfer proxy",
}

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_SHORT_WIN = 21    # "short" rolling window (days)
_LONG_WIN  = 63    # "long" rolling window used for beta drift and asym betas
_TREND_WIN = 5     # trend lookback for relative-strength trend
_MIN_PER   = 15    # minimum observations to emit a rolling value


def _rolling_beta(stock_ret: pd.Series, idx_ret: pd.Series,
                  window: int, min_periods: int) -> pd.Series:
    """OLS beta (cov/var) on a rolling window.  Both series must be aligned."""
    cov = stock_ret.rolling(window, min_periods=min_periods).cov(idx_ret)
    var = idx_ret.rolling(window, min_periods=min_periods).var()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beta = cov / var
    # Clip extreme outliers
    return beta.clip(-5.0, 5.0)


def _rolling_beta_conditional(stock_ret: np.ndarray, idx_ret: np.ndarray,
                               window: int, min_periods: int,
                               up: bool) -> np.ndarray:
    """
    Beta estimated only on days where SPY moved up (up=True) or down (up=False).
    Rolling, computed manually so we stay within the look-ahead safe window.
    Returns a float array aligned to the input length.
    """
    n = len(stock_ret)
    result = np.full(n, np.nan)
    mask = (idx_ret > 0) if up else (idx_ret < 0)

    for t in range(window - 1, n):
        s_win = stock_ret[t - window + 1: t + 1]
        i_win = idx_ret[t - window + 1: t + 1]
        m_win = mask[t - window + 1: t + 1]

        s_cond = s_win[m_win]
        i_cond = i_win[m_win]

        if len(s_cond) < min_periods:
            continue

        var_i = np.var(i_cond, ddof=1)
        if var_i <= 0 or not np.isfinite(var_i):
            continue

        cov_si = np.cov(s_cond, i_cond, ddof=1)[0, 1]
        beta_val = cov_si / var_i
        result[t] = np.clip(beta_val, -5.0, 5.0)

    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add index-relative transfer features to a single-ticker OHLCV frame.

    df is per-ticker, ascending by Date, ~700 rows.
    Only the columns in METADATA['produces'] are added; nothing else is touched.
    """
    n = len(df)

    # -----------------------------------------------------------------------
    # 1.  Resolve the index symbol and build a look-ahead-safe merge
    # -----------------------------------------------------------------------
    # Prefer SPY; fall back to first available symbol
    avail = _indexes.available()
    if not avail:
        # No index data at all — emit all-NaN columns and return
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    symbol = "SPY" if "SPY" in avail else avail[0]
    spy_close_raw = _indexes.index_close(symbol)   # pd.Series, DatetimeIndex

    if spy_close_raw.empty:
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    # Prepare a two-column lookup table ready for merge_asof
    spy_lookup = (
        spy_close_raw
        .reset_index()
        .rename(columns={"Date": "Date", "Close": "_spy_close"})
    )
    spy_lookup["Date"] = pd.to_datetime(spy_lookup["Date"])
    spy_lookup = spy_lookup.sort_values("Date").reset_index(drop=True)

    # Backward merge onto the per-ticker frame (never forward = no lookahead)
    ticker_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(
        ticker_dates,
        spy_lookup[["Date", "_spy_close"]],
        on="Date",
        direction="backward",
    )
    # _spy_close is now 1-to-1 with df rows, aligned on row position
    spy_close = merged["_spy_close"].to_numpy(dtype=np.float64)

    # -----------------------------------------------------------------------
    # 2.  Log returns — both series aligned to df row positions
    # -----------------------------------------------------------------------
    stock_close_np = df["Close"].to_numpy(dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stock_ret_np = np.empty(n)
        stock_ret_np[0] = np.nan
        stock_ret_np[1:] = np.log(stock_close_np[1:] / stock_close_np[:-1])

        spy_ret_np = np.empty(n)
        spy_ret_np[0] = np.nan
        spy_ret_np[1:] = np.log(spy_close[1:] / spy_close[:-1])

    # Replace inf with nan (safety guard — shouldn't happen but costs nothing)
    stock_ret_np = np.where(np.isfinite(stock_ret_np), stock_ret_np, np.nan)
    spy_ret_np   = np.where(np.isfinite(spy_ret_np),   spy_ret_np,   np.nan)

    # Convert to pandas Series (preserving df.index for concat-back)
    stock_ret = pd.Series(stock_ret_np, index=df.index)
    spy_ret   = pd.Series(spy_ret_np,   index=df.index)

    # -----------------------------------------------------------------------
    # 3.  Feature: lead-lag cross-correlation
    #     corr(stock_ret[t], spy_ret[t-1]) over SHORT_WIN rolling window
    #     SPY return is shifted by 1 so its yesterday is correlated with
    #     today's stock return — measures index-to-equity lead.
    # -----------------------------------------------------------------------
    spy_ret_lagged = spy_ret.shift(1)
    lead_lag_corr = stock_ret.rolling(_SHORT_WIN, min_periods=_MIN_PER).corr(spy_ret_lagged)
    lead_lag_corr = lead_lag_corr.clip(-1.0, 1.0)

    # -----------------------------------------------------------------------
    # 4.  Feature: short-window rolling beta (21d)
    #     Distinct from beta_metrics.py's static 60-day beta.
    # -----------------------------------------------------------------------
    beta_21 = _rolling_beta(stock_ret, spy_ret, _SHORT_WIN, _MIN_PER)

    # -----------------------------------------------------------------------
    # 5.  Feature: beta change (21d beta minus its 63d trailing mean)
    #     Captures recent acceleration/deceleration of beta — a proxy for
    #     how strongly the index transfer has been changing.
    # -----------------------------------------------------------------------
    beta_21_mean_63 = beta_21.shift(1).rolling(_LONG_WIN, min_periods=_MIN_PER).mean()
    beta_change = beta_21 - beta_21_mean_63

    # -----------------------------------------------------------------------
    # 6.  Feature: index-momentum transfer drift
    #     SPY trailing 21-day log return × current rolling beta → expected
    #     drift that SPY momentum "transfers" to this stock.
    # -----------------------------------------------------------------------
    spy_ret_21d = spy_ret.rolling(_SHORT_WIN, min_periods=_MIN_PER).sum()
    transfer_drift = spy_ret_21d * beta_21

    # -----------------------------------------------------------------------
    # 7.  Feature: relative strength vs SPY (21-day)
    #     Cumulative log return of stock minus cumulative log return of SPY
    #     over the last 21 trading days.
    # -----------------------------------------------------------------------
    stock_cum_21 = stock_ret.rolling(_SHORT_WIN, min_periods=_MIN_PER).sum()
    spy_cum_21   = spy_ret.rolling(_SHORT_WIN,   min_periods=_MIN_PER).sum()
    rel_strength_21 = stock_cum_21 - spy_cum_21

    # -----------------------------------------------------------------------
    # 8.  Feature: trend of relative strength (5-day change)
    # -----------------------------------------------------------------------
    rel_strength_trend = rel_strength_21.diff(_TREND_WIN)

    # -----------------------------------------------------------------------
    # 9.  Feature: beta asymmetry (downside beta − upside beta, 63-day window)
    #     Uses conditional_beta helper.  Positive = stock amplifies index drops
    #     more than it participates in index rallies.
    # -----------------------------------------------------------------------
    # Use numpy arrays (already aligned to df row positions)
    sr_np  = stock_ret_np
    idx_np = spy_ret_np

    up_beta   = _rolling_beta_conditional(sr_np, idx_np, _LONG_WIN, _MIN_PER, up=True)
    down_beta = _rolling_beta_conditional(sr_np, idx_np, _LONG_WIN, _MIN_PER, up=False)

    beta_asym_np = np.where(
        np.isfinite(down_beta) & np.isfinite(up_beta),
        down_beta - up_beta,
        np.nan,
    )
    beta_asym = pd.Series(beta_asym_np, index=df.index)

    # -----------------------------------------------------------------------
    # 10.  Assemble — only ADD produced columns, never touch existing ones
    # -----------------------------------------------------------------------
    new_cols = pd.DataFrame(
        {
            "ixr_lead_lag_corr_21d":  lead_lag_corr,
            "ixr_beta_21d":           beta_21,
            "ixr_beta_change_21v63":  beta_change,
            "ixr_transfer_drift":     transfer_drift,
            "ixr_rel_strength_21d":   rel_strength_21,
            "ixr_rel_strength_trend": rel_strength_trend,
            "ixr_beta_asym":          beta_asym,
        },
        index=df.index,
    )

    # Final guard: replace any inf that slipped through with nan
    new_cols = new_cols.replace([np.inf, -np.inf], np.nan)

    return pd.concat([df, new_cols], axis=1)
