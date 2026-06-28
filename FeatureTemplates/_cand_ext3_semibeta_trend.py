"""
Candidate feature block: ext3_semibeta_trend
Downside-semibeta momentum — rolling 120d downside semibeta vs SPY,
its 60d change, and its ratio to 252d average.
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
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_semibeta_trend",
    "description": (
        "Per-ticker downside semibeta vs SPY computed with a 120-day rolling "
        "window (cov(stock_ret, spy_ret | spy_ret<0) / var(spy_ret | spy_ret<0)). "
        "Produces: (1) the level (ext3_semibeta_trend_lvl), (2) the 60-day change "
        "in semibeta (semibeta momentum, ext3_semibeta_trend_mom), and (3) the "
        "ratio of current semibeta to its trailing 252-day mean "
        "(ext3_semibeta_trend_ratio). Captures how a stock's tail-risk loading "
        "on bad-market days is trending — an orthogonal axis to upside beta and "
        "plain rolling beta. Proxy: per-ticker time-series only (no cross-sectional "
        "ranking). Uses _indexes(SPY)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_semibeta_trend_lvl",
        "ext3_semibeta_trend_mom",
        "ext3_semibeta_trend_ratio",
    ],
    "tags": ["beta", "downside", "semibeta", "momentum", "market_sensitivity"],
    "version": "1.0",
    "author": "Round-4 expansion (xdom2_downside_beta), extends xdom2_downside_beta spec",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WIN_SEMI = 120    # days for rolling downside semibeta
_WIN_MOM = 60     # lag for semibeta momentum (change)
_WIN_AVG = 252    # days for long-run semibeta average


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute downside-semibeta level, momentum, and ratio vs long-run average."""
    # -----------------------------------------------------------------------
    # 1. Fetch SPY daily returns aligned to df dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        spy_close = None

    n = len(df)
    lvl = np.full(n, np.nan)
    mom = np.full(n, np.nan)
    ratio = np.full(n, np.nan)

    if spy_close is None or spy_close.empty or n < 2:
        df["ext3_semibeta_trend_lvl"] = lvl
        df["ext3_semibeta_trend_mom"] = mom
        df["ext3_semibeta_trend_ratio"] = ratio
        return df

    # -----------------------------------------------------------------------
    # 2. Compute daily log-returns for this ticker
    # -----------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret = np.empty(n)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(close[1:] / close[:-1])

    # -----------------------------------------------------------------------
    # 3. Align SPY returns to our dates (backward merge_asof)
    # -----------------------------------------------------------------------
    # Build a SPY return series
    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df = spy_df.sort_values("Date").copy()
    spy_df["spy_ret"] = np.log(
        spy_df["spy_close"] / spy_df["spy_close"].shift(1)
    )

    # Align ticker dates to SPY dates via merge_asof
    dates_df = df[["Date"]].copy()
    dates_df["Date"] = pd.to_datetime(dates_df["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        dates_df.sort_values("Date").reset_index(drop=False),
        spy_df[["Date", "spy_ret"]].dropna(),
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.sort_values("index").reset_index(drop=True)
    spy_r = merged["spy_ret"].values.astype(np.float64)

    # -----------------------------------------------------------------------
    # 4. Rolling downside semibeta (vectorised with stride tricks)
    # -----------------------------------------------------------------------
    # For each bar t, restrict to the window [t-WIN_SEMI+1 .. t] where spy_ret < 0,
    # then semibeta = cov(stock, spy | spy<0) / var(spy | spy<0).
    # We need at least some down days; guard with min_periods logic.
    #
    # Use a Python loop over windows — but each iteration is O(W) numpy, so
    # total cost is O(n * W / avg_downdays_per_window) which is acceptable for
    # n≈700, W=120.
    for t in range(_WIN_SEMI - 1, n):
        s_w = stock_ret[t - _WIN_SEMI + 1 : t + 1]
        r_w = spy_r[t - _WIN_SEMI + 1 : t + 1]
        # Mask: both valid AND spy down
        valid = (~np.isnan(s_w)) & (~np.isnan(r_w))
        down = valid & (r_w < 0.0)
        n_down = int(np.sum(down))
        if n_down < 5:
            continue
        s_d = s_w[down]
        r_d = r_w[down]
        spy_var = np.var(r_d, ddof=1)
        if spy_var < 1e-12:
            continue
        # cov(s, r) / var(r) on down-market subset
        semi_b = np.cov(s_d, r_d, ddof=1)[0, 1] / spy_var
        lvl[t] = semi_b

    # -----------------------------------------------------------------------
    # 5. Semibeta momentum = change over 60 bars
    # -----------------------------------------------------------------------
    for t in range(_WIN_MOM, n):
        if not np.isnan(lvl[t]) and not np.isnan(lvl[t - _WIN_MOM]):
            mom[t] = lvl[t] - lvl[t - _WIN_MOM]

    # -----------------------------------------------------------------------
    # 6. Ratio of current semibeta to its trailing 252-day average
    # -----------------------------------------------------------------------
    lvl_s = pd.Series(lvl)
    rolling_mean = lvl_s.rolling(window=_WIN_AVG, min_periods=30).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            np.abs(rolling_mean) > 1e-9,
            lvl / rolling_mean,
            np.nan,
        )
    # Guard inf
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    df["ext3_semibeta_trend_lvl"] = lvl
    df["ext3_semibeta_trend_mom"] = mom
    df["ext3_semibeta_trend_ratio"] = ratio
    return df
