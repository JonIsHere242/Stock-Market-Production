"""
ff06280034b_drawdown_drawdown_beta_120
--------------------------------------
Rolling 120-day drawdown-beta: OLS slope and Pearson correlation of the
stock's running drawdown series on SPY's running drawdown series.

Drawdown at bar t = (price_t - rolling_max_to_t) / rolling_max_to_t  (≤ 0).

The slope (drawdown-beta) measures how much of SPY's drawdown the stock
amplifies or attenuates. A beta > 1 means the stock falls harder than SPY
in drawdowns; < 1 = more resilient; < 0 = counter-cyclical. The correlation
captures how co-directional drawdown paths are, independent of magnitude.

These are per-ticker time-series proxies for a naturally cross-sectional
concept (relative drawdown sensitivity vs. a benchmark), implemented
fully causally via expanding-window rolling regressions aligned backward.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
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
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034b_drawdown_drawdown_beta_120",
    "description": (
        "Rolling 120-bar OLS slope (drawdown-beta) and Pearson correlation of "
        "per-ticker drawdown on SPY drawdown. Drawdown = (close - expanding_max) / "
        "expanding_max. Beta > 1 => amplifies SPY drawdowns; < 1 => resilient. "
        "Per-ticker causal proxy for cross-sectional drawdown sensitivity."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_drawdown_drawdown_beta_120_beta",   # rolling OLS slope
        "ff06280034b_drawdown_drawdown_beta_120_corr",   # rolling Pearson corr
        "ff06280034b_drawdown_drawdown_beta_120_spread", # stock_dd - beta*spy_dd (residual)
    ],
    "tags": ["drawdown", "beta", "market", "correlation", "risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}

_WINDOW = 120


def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorised rolling OLS slope over window w using numpy stride tricks.
    Returns array same length as input, NaN for bars < w.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < w:
        return out

    # Use cumsum trick for rolling sums (avoids O(n*w) loop)
    # E[x], E[y], E[x^2], E[xy] over window
    x2 = x * x
    xy = x * y

    # Prefix sums
    cx = np.concatenate(([0.0], np.nancumsum(x)))
    cy = np.concatenate(([0.0], np.nancumsum(y)))
    cx2 = np.concatenate(([0.0], np.nancumsum(x2)))
    cxy = np.concatenate(([0.0], np.nancumsum(xy)))

    for i in range(w - 1, n):
        sx = cx[i + 1] - cx[i + 1 - w]
        sy = cy[i + 1] - cy[i + 1 - w]
        sx2 = cx2[i + 1] - cx2[i + 1 - w]
        sxy = cxy[i + 1] - cxy[i + 1 - w]

        denom = w * sx2 - sx * sx
        if denom == 0.0 or np.isnan(denom):
            out[i] = np.nan
        else:
            out[i] = (w * sxy - sx * sy) / denom

    return out


def _rolling_corr(y: np.ndarray, x: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorised rolling Pearson correlation over window w.
    Returns array same length as input, NaN for bars < w.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < w:
        return out

    x2 = x * x
    y2 = y * y
    xy = x * y

    cx = np.concatenate(([0.0], np.nancumsum(x)))
    cy = np.concatenate(([0.0], np.nancumsum(y)))
    cx2 = np.concatenate(([0.0], np.nancumsum(x2)))
    cy2 = np.concatenate(([0.0], np.nancumsum(y2)))
    cxy = np.concatenate(([0.0], np.nancumsum(xy)))

    for i in range(w - 1, n):
        sx = cx[i + 1] - cx[i + 1 - w]
        sy = cy[i + 1] - cy[i + 1 - w]
        sx2 = cx2[i + 1] - cx2[i + 1 - w]
        sy2 = cy2[i + 1] - cy2[i + 1 - w]
        sxy = cxy[i + 1] - cxy[i + 1 - w]

        num = w * sxy - sx * sy
        denom = np.sqrt((w * sx2 - sx * sx) * (w * sy2 - sy * sy))
        if denom == 0.0 or np.isnan(denom):
            out[i] = np.nan
        else:
            out[i] = np.clip(num / denom, -1.0, 1.0)

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (covers early-return paths)
    col_beta = "ff06280034b_drawdown_drawdown_beta_120_beta"
    col_corr = "ff06280034b_drawdown_drawdown_beta_120_corr"
    col_spread = "ff06280034b_drawdown_drawdown_beta_120_spread"

    df[col_beta] = np.nan
    df[col_corr] = np.nan
    df[col_spread] = np.nan

    if len(df) < 2:
        return df

    # ------------------------------------------------------------------
    # 1. Stock drawdown series (causal expanding max)
    # ------------------------------------------------------------------
    close = df["Close"].to_numpy(dtype=float)
    exp_max = np.maximum.accumulate(np.where(np.isnan(close), -np.inf, close))
    # If expanding max is 0 or nan, guard
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_dd = np.where(
            (exp_max == 0) | np.isnan(exp_max) | np.isnan(close),
            np.nan,
            (close - exp_max) / exp_max,
        )

    # ------------------------------------------------------------------
    # 2. SPY drawdown series aligned to df dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close is None or len(spy_close) == 0:
            return df

        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "SPY_Close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        # Compute SPY drawdown on its own full series first (causal)
        spy_close_vals = spy_df["SPY_Close"].to_numpy(dtype=float)
        spy_exp_max = np.maximum.accumulate(
            np.where(np.isnan(spy_close_vals), -np.inf, spy_close_vals)
        )
        spy_dd_vals = np.where(
            (spy_exp_max == 0) | np.isnan(spy_exp_max) | np.isnan(spy_close_vals),
            np.nan,
            (spy_close_vals - spy_exp_max) / spy_exp_max,
        )
        spy_df["SPY_dd"] = spy_dd_vals

        # Align SPY drawdown to df via merge_asof (backward = no lookahead)
        stock_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
        merged = pd.merge_asof(
            stock_dates.sort_values("Date"),
            spy_df[["Date", "SPY_dd"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original order
        merged.index = stock_dates.sort_values("Date").index
        merged = merged.reindex(df.index)
        spy_dd = merged["SPY_dd"].to_numpy(dtype=float)

    except Exception:
        return df

    # ------------------------------------------------------------------
    # 3. Rolling 120-bar OLS slope and correlation
    # ------------------------------------------------------------------
    valid = ~(np.isnan(stock_dd) | np.isnan(spy_dd))

    # We need aligned valid arrays; fill invalid with 0 temporarily only
    # for the rolling functions but mask results where either is NaN.
    # Instead: pass raw arrays (nancumsum handles NaN accumulation but
    # NaN in a window corrupts that window's sum). Use masked approach:
    # replace NaN with 0 and track a count; if any NaN in window => NaN result.
    # For simplicity and correctness, use pandas rolling (slower but safe).
    s_dd = pd.Series(stock_dd)
    x_dd = pd.Series(spy_dd)

    # Rolling OLS slope via corr * std_ratio
    roll_corr = s_dd.rolling(_WINDOW, min_periods=_WINDOW).corr(x_dd).to_numpy()
    roll_std_s = s_dd.rolling(_WINDOW, min_periods=_WINDOW).std().to_numpy()
    roll_std_x = x_dd.rolling(_WINDOW, min_periods=_WINDOW).std().to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(
            roll_std_x == 0,
            np.nan,
            roll_corr * roll_std_s / roll_std_x,
        )

    # Clip to reasonable range to avoid extreme outliers from near-zero SPY std
    beta = np.where(np.isfinite(beta), np.clip(beta, -10.0, 10.0), np.nan)
    corr = np.where(np.isfinite(roll_corr), roll_corr, np.nan)

    # Spread = residual: stock_dd - beta * spy_dd (how much excess/deficit vs model)
    spread = np.where(
        np.isnan(beta) | np.isnan(stock_dd) | np.isnan(spy_dd),
        np.nan,
        stock_dd - beta * spy_dd,
    )

    df[col_beta] = beta
    df[col_corr] = corr
    df[col_spread] = spread

    return df
