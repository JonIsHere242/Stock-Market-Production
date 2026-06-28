"""
orig_orig_market_beta.py -- Faithful AlphaSensitivity port of rolling beta / corr / alpha.

Ports calculate_beta (backup L944-1035). Rolling 60-day (min_periods 30) beta, Pearson
correlation, and alpha vs SPY, QQQ, IWM, DIA, VIX using log returns and inner-date alignment.
beta clipped [-5, 5]; corr clipped [-1, 1]; alpha NOT clipped.

Emits the EXACT original AlphaSensitivity column names (CapCase symbol suffixes):
    alpha_<SYM>, beta_<SYM>, corr_<SYM>  for SYM in {SPY, QQQ, IWM, DIA, VIX}.

Index data is served by the shared FeatureTemplates/_indexes.py helper, which reads the
same Data/Indexes/{SYM}.parquet files the monolith loaded into GLOBAL_INDEX_DATA.
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
    "name":        "orig_orig_market_beta",
    "description": (
        "Faithful AlphaSensitivity port: rolling 60-day beta, Pearson correlation, and "
        "alpha vs SPY, QQQ, IWM, DIA, VIX using log returns and inner-date alignment."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "alpha_SPY", "alpha_QQQ", "alpha_IWM", "alpha_DIA", "alpha_VIX",
        "beta_SPY",  "beta_QQQ",  "beta_IWM",  "beta_DIA",  "beta_VIX",
        "corr_SPY",  "corr_QQQ",  "corr_IWM",  "corr_DIA",  "corr_VIX",
    ],
    "tags":    ["market_regime", "beta"],
    "version": "1.0",
    "author":  "alphasens port",
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

    df is one ticker, ascending by Date. Only the produces columns are added;
    existing columns are never touched, sorted, or reindexed.
    """

    # Stock log returns on a Date index -- temporary, never stored.
    stock_close = pd.Series(
        df["Close"].values,
        index=pd.to_datetime(df["Date"]),
    )
    stock_returns = np.log(stock_close / stock_close.shift(1))

    # Original df Date index used for reindex-back (preserves df row order).
    df_dates = pd.to_datetime(df["Date"]).values

    for sym in _SYMBOLS:
        try:
            index_close = _indexes.index_close(sym)   # Series, DatetimeIndex
            if index_close.empty:
                continue

            index_returns = np.log(index_close / index_close.shift(1))

            # Align on dates (inner join -> only shared trading dates).
            aligned_stock, aligned_index = stock_returns.align(index_returns, join="inner")

            if len(aligned_stock) < _MIN_PERIODS:
                continue

            # Rolling covariance / variance -> beta.
            rolling_cov = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).cov(aligned_index)
            rolling_var = aligned_index.rolling(_WINDOW, min_periods=_MIN_PERIODS).var()
            beta = rolling_cov / rolling_var

            # Rolling Pearson correlation.
            corr = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).corr(aligned_index)

            # Rolling alpha = mean(stock) - beta * mean(index).
            stock_mean = aligned_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).mean()
            index_mean = aligned_index.rolling(_WINDOW, min_periods=_MIN_PERIODS).mean()
            alpha = stock_mean - (beta * index_mean)

            # Clip outliers (alpha NOT clipped).
            beta = beta.clip(-5, 5)
            corr = corr.clip(-1, 1)

            # Reindex back to original df row order, assign via .values.
            df[f"beta_{sym}"]  = beta.reindex(df_dates).values
            df[f"corr_{sym}"]  = corr.reindex(df_dates).values
            df[f"alpha_{sym}"] = alpha.reindex(df_dates).values

        except Exception:
            # If an index is unavailable or errors, skip its three columns silently.
            continue

    return df
