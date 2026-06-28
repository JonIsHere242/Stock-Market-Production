"""
beta_metrics.py — Rolling beta, correlation, and alpha vs SPY, QQQ, IWM, DIA, VIX.

Window: 60 bars, min_periods: 30.  Log returns.  Inner-join alignment on Date.
beta clipped [-5, 5]; corr clipped [-1, 1].
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared index helper (skipped by framework auto-discovery)
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
    "name":        "beta_metrics",
    "description": (
        "Rolling 60-day beta, Pearson correlation, and alpha vs SPY, QQQ, IWM, DIA, VIX "
        "using log returns and inner-date alignment."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "beta_spy", "corr_spy", "alpha_spy",
        "beta_qqq", "corr_qqq", "alpha_qqq",
        "beta_iwm", "corr_iwm", "alpha_iwm",
        "beta_dia", "corr_dia", "alpha_dia",
        "beta_vix", "corr_vix", "alpha_vix",
    ],
    "tags":    ["market_regime", "beta"],
    "version": "1.0",
}

_WINDOW      = 60
_MIN_PERIODS = 30
_SYMBOLS     = ["SPY", "QQQ", "IWM", "DIA", "VIX"]

# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling beta / corr / alpha columns for each index in _SYMBOLS.

    df is one ticker, ascending by Date.  Only the produces columns are added;
    existing columns are never touched.
    """

    # Build a Date-indexed Series of stock log returns — temporary, never stored.
    stock_close = pd.Series(
        df["Close"].values,
        index=pd.to_datetime(df["Date"]),
    )
    stock_ret = np.log(stock_close / stock_close.shift(1))

    # Original df Date index used for reindex-back
    df_dates = pd.to_datetime(df["Date"]).values

    for sym in _SYMBOLS:
        suffix = sym.lower()
        try:
            idx_close = _indexes.index_close(sym)   # Series, DatetimeIndex
            if idx_close.empty:
                continue

            idx_ret = np.log(idx_close / idx_close.shift(1))

            # Inner-join alignment so we only work on shared trading dates
            aligned_stock, aligned_index = stock_ret.align(idx_ret, join="inner")

            if len(aligned_stock) < _MIN_PERIODS:
                continue

            # Rolling covariance / variance -> beta
            rolling_cov = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).cov(aligned_index)
            rolling_var = aligned_index.rolling(_WINDOW, min_periods=_MIN_PERIODS).var()
            beta = rolling_cov / rolling_var

            # Rolling Pearson correlation
            corr = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).corr(aligned_index)

            # Rolling alpha = mean(stock) - beta * mean(index)
            stock_mean = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).mean()
            index_mean = aligned_index.rolling(_WINDOW, min_periods=_MIN_PERIODS).mean()
            alpha = stock_mean - beta * index_mean

            # Clip outliers
            beta = beta.clip(-5, 5)
            corr = corr.clip(-1, 1)

            # Reindex back to the original df row order and assign via .values
            # (preserves df row order — no sort / reindex on df itself)
            df[f"beta_{suffix}"] = beta.reindex(df_dates).values
            df[f"corr_{suffix}"] = corr.reindex(df_dates).values
            df[f"alpha_{suffix}"] = alpha.reindex(df_dates).values

        except Exception:
            # If an index is unavailable or errors, skip its three columns silently.
            continue

    return df
