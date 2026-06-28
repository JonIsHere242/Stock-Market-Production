"""
ext3_beta_instability — Beta instability (time-varying beta dispersion)

Rolling 40-day OLS beta vs SPY is computed at every bar, then the
120-day std and range (max-min) of that beta series are produced.
Unstable beta implies an unreliable risk loading; the signal is
orthogonal to the level of beta (e.g. xdom2_downside_beta) because
it measures *variance* rather than direction.

Proxy note: exact XS-rank instability is impossible per-ticker; this
is a faithful per-stock time-series proxy using only OHLCV + SPY index.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
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
    "name": "ext3_beta_instability",
    "description": (
        "Time-varying beta dispersion: rolling 40-day OLS beta vs SPY is "
        "computed at every bar; then the trailing 120-day standard deviation "
        "(ext3_beta_instability_std) and range max-min "
        "(ext3_beta_instability_range) of that beta series are produced. "
        "High values indicate an unstable risk loading. "
        "Per-ticker proxy — cross-sectional rank of instability not available "
        "in single-stock mode."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_beta_instability_beta40",   # rolling 40d beta level (anchor)
        "ext3_beta_instability_std",      # 120d std of the rolling beta
        "ext3_beta_instability_range",    # 120d range (max-min) of rolling beta
    ],
    "tags": ["beta", "risk", "instability", "market", "spy", "time-varying"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom2_downside_beta); spec: ext3_beta_instability",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BETA_WIN = 40     # window for rolling OLS beta vs SPY
_INSTAB_WIN = 120  # window over which to measure beta dispersion
_MIN_OBS = 20      # minimum observations inside the beta window


def compute(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """
    Parameters
    ----------
    df : DataFrame with columns [Date, Ticker, Open, High, Low, Close, Volume],
         one stock, ascending by Date.

    Returns
    -------
    df with three new columns appended (NaN where data is insufficient).
    """
    n = len(df)
    beta40 = np.full(n, np.nan)
    beta_std = np.full(n, np.nan)
    beta_range = np.full(n, np.nan)

    # --- Fetch SPY close and align to stock dates ---------------------------
    try:
        spy_close: pd.Series = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    if spy_close is None or len(spy_close) == 0:
        df["ext3_beta_instability_beta40"] = np.nan
        df["ext3_beta_instability_std"] = np.nan
        df["ext3_beta_instability_range"] = np.nan
        return df

    # Build aligned SPY returns (same length as df, NaN where missing)
    stock_dates = pd.to_datetime(df["Date"].values)
    spy_close_df = spy_close.reset_index()
    spy_close_df.columns = ["Date", "spy_close"]
    spy_close_df["Date"] = pd.to_datetime(spy_close_df["Date"])

    stock_df_tmp = pd.DataFrame({"Date": stock_dates})
    merged = pd.merge_asof(
        stock_df_tmp.sort_values("Date"),
        spy_close_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original row order
    merged = merged.set_index("Date").reindex(stock_dates).reset_index()
    spy_vals = merged["spy_close"].values.astype(float)

    # --- Daily log returns --------------------------------------------------
    stock_close = df["Close"].values.astype(float)

    # stock returns: log(c[t]/c[t-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stock_ret = np.empty(n)
        stock_ret[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            stock_ret[1:] = np.where(
                stock_close[:-1] > 0,
                np.log(stock_close[1:] / stock_close[:-1]),
                np.nan,
            )

        spy_ret = np.empty(n)
        spy_ret[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            spy_ret[1:] = np.where(
                spy_vals[:-1] > 0,
                np.log(spy_vals[1:] / spy_vals[:-1]),
                np.nan,
            )

    # --- Rolling 40-day OLS beta -------------------------------------------
    # For each bar t, regress stock_ret[t-W+1..t] on spy_ret[t-W+1..t]
    # beta = cov(y,x) / var(x)   (no intercept variant, but we demean)
    W = _BETA_WIN
    for t in range(W - 1, n):
        y = stock_ret[t - W + 1: t + 1]
        x = spy_ret[t - W + 1: t + 1]
        mask = np.isfinite(y) & np.isfinite(x)
        obs = mask.sum()
        if obs < _MIN_OBS:
            continue
        y_m = y[mask]
        x_m = x[mask]
        x_bar = x_m.mean()
        y_bar = y_m.mean()
        xc = x_m - x_bar
        denom = float(np.dot(xc, xc))
        if denom == 0.0 or not np.isfinite(denom):
            continue
        beta40[t] = float(np.dot(xc, y_m - y_bar)) / denom

    # --- 120-day std and range of the rolling beta -------------------------
    IW = _INSTAB_WIN
    for t in range(IW - 1, n):
        window = beta40[t - IW + 1: t + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < 10:
            continue
        beta_std[t] = float(np.std(finite, ddof=1))
        beta_range[t] = float(np.max(finite) - np.min(finite))

    # --- Assign to df -------------------------------------------------------
    df["ext3_beta_instability_beta40"] = beta40
    df["ext3_beta_instability_std"] = beta_std
    df["ext3_beta_instability_range"] = beta_range

    return df
