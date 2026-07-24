"""
Idiosyncratic Volatility Share vs SPY (120-day rolling OLS).

Regresses ticker daily returns against SPY returns over a trailing 120-day window.
Feature = residual variance / total ticker variance  (= 1 - R²).

Higher values mean the stock's moves are more stock-specific (less driven by the
market factor). This is a per-ticker proxy for the cross-sectional concept of
idiosyncratic risk share.

Produces:
  ff06282316b_factor_idio_vol_share_spy_120       -- idio-var / total-var (1 - R²), 120d
  ff06282316b_factor_idio_vol_share_spy_120_slope -- 20-day rate-of-change of the level
  ff06282316b_factor_idio_vol_share_spy_120_beta  -- trailing 120-day SPY beta (OLS slope)
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------- load _indexes helper ----------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_factor_idio_vol_share_spy_120",
    "description": (
        "Idiosyncratic volatility share: 1 - R² from a rolling 120-day OLS of "
        "ticker returns on SPY returns.  Higher = more stock-specific risk.  "
        "Per-ticker proxy for cross-sectional idio-vol share; lookahead-free."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_idio_vol_share_spy_120",
        "ff06282316b_factor_idio_vol_share_spy_120_slope",
        "ff06282316b_factor_idio_vol_share_spy_120_beta",
    ],
    "tags": ["factor", "risk", "beta", "idiosyncratic", "spy", "rolling-ols"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

_WIN = 120
_SLOPE_WIN = 20
_MIN_OBS = 30  # minimum valid bars inside window
_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN on every code path
    for col in METADATA["produces"]:
        df[col] = np.nan

    if len(df) < _MIN_OBS + 2:
        return df

    # ---- SPY index close ------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build SPY return series aligned to df dates
    spy_ret_series = spy_close.pct_change()

    # Merge SPY returns onto df using merge_asof (backward = lookahead-safe)
    df_dates = df[["Date"]].copy()
    df_dates["Date"] = pd.to_datetime(df_dates["Date"])
    spy_df = spy_ret_series.rename("_spy_ret").reset_index()
    spy_df.columns = ["Date", "_spy_ret"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date")
    df_dates = df_dates.sort_values("Date")

    merged = pd.merge_asof(df_dates, spy_df, on="Date", direction="backward")
    # Re-align to original df index order
    merged = merged.set_index(df_dates.index)

    spy_ret = merged["_spy_ret"].values  # length = len(df)

    # Ticker daily returns
    close = df["Close"].values.astype(np.float64)
    tkr_ret = np.empty(len(close))
    tkr_ret[0] = np.nan
    tkr_ret[1:] = np.diff(close) / np.where(close[:-1] == 0, np.nan, close[:-1])

    n = len(df)
    idio_share = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)

    for t in range(_WIN - 1, n):
        y = tkr_ret[t - _WIN + 1: t + 1]   # shape (_WIN,)
        x = spy_ret[t - _WIN + 1: t + 1]

        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < _MIN_OBS:
            continue

        y_m = y[mask]
        x_m = x[mask]

        # OLS: beta = cov(x,y)/var(x)
        x_mean = x_m.mean()
        y_mean = y_m.mean()
        x_dm = x_m - x_mean
        y_dm = y_m - y_mean

        var_x = (x_dm ** 2).mean()
        if var_x < _EPS:
            continue

        beta = (x_dm * y_dm).mean() / var_x

        # residuals
        resid = y_dm - beta * x_dm
        var_resid = (resid ** 2).mean()
        var_y = (y_dm ** 2).mean()

        if var_y < _EPS:
            idio_share[t] = np.nan
        else:
            idio_share[t] = var_resid / var_y  # = 1 - R²

        beta_arr[t] = beta

    # Restore to df index order (df may not be sorted by Date, but contract says ascending)
    df["ff06282316b_factor_idio_vol_share_spy_120"] = idio_share
    df["ff06282316b_factor_idio_vol_share_spy_120_beta"] = beta_arr

    # Slope: rate of change of idio_share over _SLOPE_WIN bars
    idio_series = pd.Series(idio_share)
    slope = idio_series.diff(_SLOPE_WIN) / _SLOPE_WIN
    df["ff06282316b_factor_idio_vol_share_spy_120_slope"] = slope.values

    return df
