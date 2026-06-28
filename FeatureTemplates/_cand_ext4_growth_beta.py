"""
Growth/tech-factor beta (QQQ - SPY) feature block.

Per-ticker rolling 2-factor OLS: regresses stock returns on
  (1) SPY return (market factor)
  (2) growth factor = QQQ_return - SPY_return  (tech/growth tilt)
over a 120-day rolling window.

Produces: ext4_growth_beta_beta (growth-factor loading),
          ext4_growth_beta_mkt  (market-factor loading),
          ext4_growth_beta_r2   (in-sample R^2 of the 2-factor model).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path; sandbox rule)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_indexes)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_growth_beta",
    "description": (
        "Rolling 120-day 2-factor OLS: stock returns ~ SPY_return + growth_factor, "
        "where growth_factor = QQQ_return - SPY_return (tech/growth-minus-market). "
        "Produces the growth-factor beta, market beta, and in-sample R^2. "
        "Pure per-ticker proxy — no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_growth_beta_beta",   # loading on growth factor (QQQ - SPY)
        "ext4_growth_beta_mkt",    # loading on SPY (market)
        "ext4_growth_beta_r2",     # 2-factor R^2
    ],
    "tags": ["beta", "factor", "growth", "tech", "rolling", "ols"],
    "version": "1.0.0",
    "author": "Round-5 expansion spec (NEW: multi-index factor)",
}

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
_WINDOW = 120  # trading days


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add growth-factor beta columns to df (one ticker, ascending Date)."""

    n = len(df)
    out_beta = np.full(n, np.nan)
    out_mkt  = np.full(n, np.nan)
    out_r2   = np.full(n, np.nan)

    if n < _WINDOW + 1:
        df["ext4_growth_beta_beta"] = np.nan
        df["ext4_growth_beta_mkt"]  = np.nan
        df["ext4_growth_beta_r2"]   = np.nan
        return df

    # ------------------------------------------------------------------
    # Fetch index series and align to df via merge_asof (backward)
    # ------------------------------------------------------------------
    dates_df = df[["Date"]].copy()
    # Ensure datetime
    dates_df["Date"] = pd.to_datetime(dates_df["Date"])

    def _get_index_returns(symbol: str) -> np.ndarray | None:
        """Return log-returns aligned to df rows; None on failure."""
        try:
            s = _indexes.index_close(symbol)  # pd.Series, DatetimeIndex
        except Exception:
            return None
        if s is None or len(s) == 0:
            return None
        idx_df = s.rename("idx_close").reset_index()
        idx_df.columns = ["Date", "idx_close"]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])
        merged = pd.merge_asof(
            dates_df.sort_values("Date"),
            idx_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original row order
        merged = merged.set_index(dates_df.sort_values("Date").index).reindex(df.index)
        closes = merged["idx_close"].to_numpy(dtype=float)
        rets = np.empty(len(closes))
        rets[0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev = closes[:-1]
            prev_safe = np.where(prev == 0, np.nan, prev)
            rets[1:] = np.log(np.where(closes[1:] <= 0, np.nan, closes[1:]) / prev_safe)
        return rets

    spy_ret = _get_index_returns("SPY")
    qqq_ret = _get_index_returns("QQQ")

    # If either index is unavailable, emit NaNs and return
    if spy_ret is None or qqq_ret is None:
        df["ext4_growth_beta_beta"] = np.nan
        df["ext4_growth_beta_mkt"]  = np.nan
        df["ext4_growth_beta_r2"]   = np.nan
        return df

    # Growth factor = QQQ return - SPY return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        growth = qqq_ret - spy_ret   # shape (n,)

    # Stock log-returns
    stock_close = df["Close"].to_numpy(dtype=float)
    stock_ret = np.empty(n, dtype=float)
    stock_ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev_s = stock_close[:-1]
        prev_s_safe = np.where(prev_s == 0, np.nan, prev_s)
        stock_ret[1:] = np.log(
            np.where(stock_close[1:] <= 0, np.nan, stock_close[1:]) / prev_s_safe
        )

    # ------------------------------------------------------------------
    # Rolling 120-day 2-factor OLS via numpy sliding windows
    # y = a + b1*SPY + b2*GROWTH
    # We use the closed-form OLS solution on each window.
    # ------------------------------------------------------------------
    for i in range(_WINDOW, n):
        sl = slice(i - _WINDOW + 1, i + 1)
        y  = stock_ret[sl]
        x1 = spy_ret[sl]
        x2 = growth[sl]

        # Drop any NaN rows within the window
        mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        if mask.sum() < _WINDOW // 2:
            continue  # not enough valid obs; leave NaN

        y_  = y[mask]
        x1_ = x1[mask]
        x2_ = x2[mask]

        # Design matrix with intercept
        X = np.column_stack([np.ones(len(y_)), x1_, x2_])

        try:
            # Normal equations: (X'X) beta = X'y
            XtX = X.T @ X
            Xty = X.T @ y_
            # Regularise tiny diagonal for numerical safety
            XtX += np.eye(3) * 1e-12
            coeffs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        b_mkt    = coeffs[1]
        b_growth = coeffs[2]

        # R^2
        y_hat = X @ coeffs
        ss_res = np.sum((y_ - y_hat) ** 2)
        ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan

        out_beta[i] = b_growth
        out_mkt[i]  = b_mkt
        out_r2[i]   = r2

    df["ext4_growth_beta_beta"] = out_beta
    df["ext4_growth_beta_mkt"]  = out_mkt
    df["ext4_growth_beta_r2"]   = out_r2
    return df
