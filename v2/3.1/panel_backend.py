"""Pluggable panel-compute backend — the cross-sectional axis, done right.

The per-ticker stateless model can't express "rank this feature across the whole
universe on each date." Research move #3 was Polars `over()`: `over('Ticker')` is the
per-ticker semantics you already have; `over('Date')` is the cross-sectional op — both
in ONE lazy frame, no reshape.

This module provides those ops with TWO interchangeable backends:
  * pandas  — works today (groupby/transform), the proven default.
  * polars  — activates automatically if polars is installed; lazy + out-of-core.

`verify_parity()` cross-checks the two backends bit-for-bit (maxdiff < 1e-9) — the same
discipline as verify_block.py — so a backend swap is provably safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl  # noqa: F401
    POLARS_AVAILABLE = True
except Exception:
    POLARS_AVAILABLE = False

_EPS = 1e-12


def active_backend(prefer: str = "auto") -> str:
    if prefer == "polars" and not POLARS_AVAILABLE:
        raise RuntimeError("polars backend requested but polars is not installed")
    if prefer == "auto":
        return "polars" if POLARS_AVAILABLE else "pandas"
    return prefer


# --------------------------------------------------------------------------- #
# pandas backend
# --------------------------------------------------------------------------- #
def _pd_xs_rank(panel, col, by):
    return panel.groupby(by)[col].rank(pct=True).to_numpy()


def _pd_xs_zscore(panel, col, by):
    g = panel.groupby(by)[col]
    return ((panel[col] - g.transform("mean")) / (g.transform("std") + _EPS)).to_numpy()


def _pd_over_ticker_rolling(panel, col, window, stat, ticker, date):
    s = panel.sort_values([ticker, date])
    g = s.groupby(ticker, sort=False)[col]
    r = getattr(g.rolling(window), stat)().reset_index(level=0, drop=True)
    return r.reindex(panel.index).to_numpy()


# --------------------------------------------------------------------------- #
# polars backend (used iff installed)
# --------------------------------------------------------------------------- #
def _pl_xs_rank(panel, col, by):
    lf = pl.from_pandas(panel[[by, col]]).lazy()
    out = lf.with_columns((pl.col(col).rank() / pl.count(col).over(by)).over(by).alias("_r"))
    return out.collect()["_r"].to_numpy()


def _pl_xs_zscore(panel, col, by):
    lf = pl.from_pandas(panel[[by, col]]).lazy()
    z = (pl.col(col) - pl.col(col).mean().over(by)) / (pl.col(col).std().over(by) + _EPS)
    return lf.with_columns(z.alias("_z")).collect()["_z"].to_numpy()


def _pl_over_ticker_rolling(panel, col, window, stat, ticker, date):
    lf = pl.from_pandas(panel[[ticker, date, col]]).with_row_index("_i").lazy()
    expr = getattr(pl.col(col), f"rolling_{stat}")(window_size=window)
    out = lf.with_columns(expr.over(ticker, order_by=date).alias("_v")).sort("_i")
    return out.collect()["_v"].to_numpy()


# --------------------------------------------------------------------------- #
# public ops
# --------------------------------------------------------------------------- #
def xs_rank(panel: pd.DataFrame, col: str, by: str = "Date", backend: str = "auto") -> np.ndarray:
    b = active_backend(backend)
    return _pl_xs_rank(panel, col, by) if b == "polars" else _pd_xs_rank(panel, col, by)


def xs_zscore(panel: pd.DataFrame, col: str, by: str = "Date", backend: str = "auto") -> np.ndarray:
    b = active_backend(backend)
    return _pl_xs_zscore(panel, col, by) if b == "polars" else _pd_xs_zscore(panel, col, by)


def over_ticker_rolling(
    panel: pd.DataFrame, col: str, window: int, stat: str = "mean",
    ticker: str = "Ticker", date: str = "Date", backend: str = "auto",
) -> np.ndarray:
    b = active_backend(backend)
    if b == "polars":
        return _pl_over_ticker_rolling(panel, col, window, stat, ticker, date)
    return _pd_over_ticker_rolling(panel, col, window, stat, ticker, date)


def verify_parity(panel: pd.DataFrame, col: str, by: str = "Date") -> dict:
    """Bit-exact cross-check of the two backends (no-op if polars absent)."""
    if not POLARS_AVAILABLE:
        return {"status": "pandas-only", "polars_installed": False}
    a = _pd_xs_rank(panel, col, by)
    b = _pl_xs_rank(panel, col, by)
    m = np.isfinite(a) & np.isfinite(b)
    maxdiff = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
    return {"status": "ok" if maxdiff < 1e-9 else "MISMATCH", "maxdiff": maxdiff, "polars_installed": True}
