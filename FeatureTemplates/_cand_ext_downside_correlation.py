"""
Downside correlation & correlation asymmetry (per-ticker proxy).

Computes rolling 120-day correlation of stock returns vs SPY returns,
conditioned on market-DOWN days vs market-UP days, and their asymmetry.
Correlation isolates co-movement strength from volatility scaling,
making this orthogonal to xdom2_downside_beta (which captures beta/slope).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path -- never via package import)
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
    "name": "ext_downside_correlation",
    "description": (
        "Rolling 120-day stock-vs-SPY return correlation conditioned on "
        "market-DOWN days (ext_downside_correlation_down), market-UP days "
        "(ext_downside_correlation_up), and their asymmetry "
        "(ext_downside_correlation_asym = down_corr - up_corr). "
        "Captures contagion loading vs decoupling. Orthogonal to "
        "xdom2_downside_beta (beta/slope); this uses correlation which "
        "normalises out individual volatility scaling. Per-ticker proxy: "
        "SPY used as the market proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_downside_correlation_down",
        "ext_downside_correlation_up",
        "ext_downside_correlation_asym",
    ],
    "tags": ["correlation", "downside", "market_regime", "risk", "contagion"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner xdom2_downside_beta; "
        "downside correlation asymmetry / contagion loading concept."
    ),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120   # rolling window in trading days
_MIN_OBS = 30   # minimum down/up observations required per window


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: one stock, ascending Date, columns include Date + OHLCV.
    Adds ext_downside_correlation_down, ext_downside_correlation_up,
    ext_downside_correlation_asym.
    """
    n = len(df)

    # Initialise output columns as NaN
    down_corr = np.full(n, np.nan)
    up_corr = np.full(n, np.nan)
    asym_corr = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Fetch SPY close, align to stock dates
    # ------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        # If index data unavailable, leave all NaN
        df["ext_downside_correlation_down"] = np.nan
        df["ext_downside_correlation_up"] = np.nan
        df["ext_downside_correlation_asym"] = np.nan
        return df

    # Build a DataFrame for merging
    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    # Align stock dates
    stock_dates = pd.to_datetime(df["Date"].values)
    merged = pd.merge_asof(
        pd.DataFrame({"Date": stock_dates}),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    spy_close_aligned = merged["spy_close"].values  # length == n

    # ------------------------------------------------------------------
    # Compute 1-day returns (log-returns; NaN at index 0)
    # ------------------------------------------------------------------
    stock_close = df["Close"].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret = np.empty(n)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(
            np.where(stock_close[:-1] > 0, stock_close[1:] / stock_close[:-1], np.nan)
        )

        spy_ret = np.empty(n)
        spy_ret[0] = np.nan
        spy_aligned = spy_close_aligned.astype(float)
        spy_ret[1:] = np.log(
            np.where(spy_aligned[:-1] > 0, spy_aligned[1:] / spy_aligned[:-1], np.nan)
        )

    # ------------------------------------------------------------------
    # Rolling conditional correlation
    # For each bar t, use the window [t-WINDOW+1 .. t] (past inclusive).
    # Split into down-market (spy_ret < 0) and up-market (spy_ret >= 0).
    # Compute Pearson correlation of stock_ret vs spy_ret within each group.
    # ------------------------------------------------------------------
    def _rolling_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray, t: int) -> float:
        """Pearson corr of x[window] vs y[window] where mask is True."""
        start = max(0, t - _WINDOW + 1)
        xs = x[start : t + 1]
        ys = y[start : t + 1]
        ms = mask[start : t + 1]
        # valid (non-NaN and mask)
        valid = ms & np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < _MIN_OBS:
            return np.nan
        xv = xs[valid]
        yv = ys[valid]
        xm = xv - xv.mean()
        ym = yv - yv.mean()
        denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        if denom == 0.0:
            return np.nan
        return float(np.dot(xm, ym) / denom)

    # Pre-compute masks (avoid recomputing inside loop)
    spy_down = spy_ret < 0       # market-down days
    spy_up = spy_ret >= 0        # market-up days (includes exactly-zero)

    # We need at least _WINDOW bars before producing a value.
    # Vectorising a conditional-split rolling correlation is non-trivial;
    # we use a Python loop over bars but operate on slices (O(n * WINDOW)).
    # For n~700 and WINDOW=120 this is ~84k operations -- well within 100ms.
    for t in range(_WINDOW - 1, n):
        down_corr[t] = _rolling_corr(stock_ret, spy_ret, spy_down, t)
        up_corr[t] = _rolling_corr(stock_ret, spy_ret, spy_up, t)
        if np.isfinite(down_corr[t]) and np.isfinite(up_corr[t]):
            asym_corr[t] = down_corr[t] - up_corr[t]

    df["ext_downside_correlation_down"] = down_corr
    df["ext_downside_correlation_up"] = up_corr
    df["ext_downside_correlation_asym"] = asym_corr

    return df
