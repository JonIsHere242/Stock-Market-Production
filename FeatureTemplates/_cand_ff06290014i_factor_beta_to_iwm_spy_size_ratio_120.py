"""
Feature block: ff06290014i_factor_beta_to_iwm_spy_size_ratio_120
Beta to the pure size factor (IWM/SPY ratio) via 2-variable OLS, controlling for
broad market return. Captures incremental small-cap tilt orthogonal to SPY.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load index helper -----------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120",
    "description": (
        "Per-ticker 120-day rolling 2-variable OLS: ticker_ret ~ beta_mkt*SPY_ret + "
        "beta_size*size_factor_ret, where size_factor_ret = IWM_ret - SPY_ret. "
        "beta_size captures incremental small-cap tilt orthogonal to broad market. "
        "Also emits rolling 20-day change in beta_size (momentum/drift of size tilt)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_size",
        "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_mkt",
        "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_size_delta20",
    ],
    "tags": ["factor", "beta", "size", "small_cap", "ols", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06290014i",
}

_WINDOW = 120
_DELTA_WINDOW = 20
_MIN_PERIODS = 30  # minimum observations to fit OLS
_DET_FLOOR = 1e-14  # guard against singular 2x2 system


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (required on every code path)
    col_beta_size = "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_size"
    col_beta_mkt = "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_mkt"
    col_delta20 = "ff06290014i_factor_beta_to_iwm_spy_size_ratio_120_beta_size_delta20"
    df[col_beta_size] = np.nan
    df[col_beta_mkt] = np.nan
    df[col_delta20] = np.nan

    if len(df) < _MIN_PERIODS + 1:
        return df

    # Pull index close series (DatetimeIndex)
    try:
        spy_close = _indexes.index_close("SPY")
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df

    if spy_close is None or iwm_close is None or spy_close.empty or iwm_close.empty:
        return df

    # Align dates: convert df Date to DatetimeIndex for merge
    dates = pd.to_datetime(df["Date"])

    # Build SPY and IWM returns aligned to ticker dates via merge_asof
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    iwm_df = iwm_close.rename("iwm_close").reset_index()
    iwm_df.columns = ["Date", "iwm_close"]
    iwm_df["Date"] = pd.to_datetime(iwm_df["Date"])

    work = pd.DataFrame({"Date": dates}).reset_index(drop=True)
    work = pd.merge_asof(work.sort_values("Date"), spy_df.sort_values("Date"),
                         on="Date", direction="backward")
    work = pd.merge_asof(work, iwm_df.sort_values("Date"), on="Date", direction="backward")

    # Restore original order (ascending by date as given)
    work = work.sort_values("Date").reset_index(drop=True)

    spy_c = work["spy_close"].values
    iwm_c = work["iwm_close"].values

    # Ticker log returns (shift 1 = yesterday's close -> no lookahead)
    ticker_close = df["Close"].values.astype(np.float64)
    ticker_ret = np.empty(len(ticker_close))
    ticker_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ticker_ret[1:] = np.log(ticker_close[1:] / ticker_close[:-1])

    spy_ret = np.empty(len(spy_c))
    spy_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        spy_ret[1:] = np.log(np.where(spy_c[:-1] > 0, spy_c[1:] / spy_c[:-1], np.nan))

    iwm_ret = np.empty(len(iwm_c))
    iwm_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        iwm_ret[1:] = np.log(np.where(iwm_c[:-1] > 0, iwm_c[1:] / iwm_c[:-1], np.nan))

    # size_factor = IWM_ret - SPY_ret (pure size premium, market-neutral)
    size_ret = iwm_ret - spy_ret

    n = len(df)
    beta_size_arr = np.full(n, np.nan)
    beta_mkt_arr = np.full(n, np.nan)

    # Rolling 2-variable OLS via closed-form 2x2 normal equations
    # y = b0 + b1*x1 + b2*x2  (with intercept demean trick: just use demeaned)
    # For each window [i-W+1 .. i], solve:
    #   [X'X] [b1, b2]' = X'y  where X columns are x1=SPY_ret, x2=size_ret
    # with intercept by demeaning inside window.

    for i in range(_WINDOW - 1, n):
        start = i - _WINDOW + 1
        y = ticker_ret[start:i + 1]
        x1 = spy_ret[start:i + 1]
        x2 = size_ret[start:i + 1]

        # Mask out NaN rows
        valid = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        cnt = valid.sum()
        if cnt < _MIN_PERIODS:
            continue

        yv = y[valid]
        x1v = x1[valid]
        x2v = x2[valid]

        # Demean (equivalent to including intercept)
        ym = yv - yv.mean()
        x1m = x1v - x1v.mean()
        x2m = x2v - x2v.mean()

        # 2x2 XtX
        a11 = np.dot(x1m, x1m)
        a12 = np.dot(x1m, x2m)
        a22 = np.dot(x2m, x2m)
        det = a11 * a22 - a12 * a12

        if abs(det) < _DET_FLOOR:
            continue

        # XtY
        b1_rhs = np.dot(x1m, ym)
        b2_rhs = np.dot(x2m, ym)

        # Cramer's rule
        b1 = (a22 * b1_rhs - a12 * b2_rhs) / det   # beta_mkt
        b2 = (a11 * b2_rhs - a12 * b1_rhs) / det   # beta_size

        beta_mkt_arr[i] = b1
        beta_size_arr[i] = b2

    df[col_beta_size] = beta_size_arr
    df[col_beta_mkt] = beta_mkt_arr

    # 20-day change in beta_size (drift / momentum of size tilt)
    bs = pd.Series(beta_size_arr)
    df[col_delta20] = (bs - bs.shift(_DELTA_WINDOW)).values

    # Guard: replace inf/-inf with NaN
    for col in [col_beta_size, col_beta_mkt, col_delta20]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
