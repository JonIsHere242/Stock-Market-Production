"""
_cand_ff0703c_cross_asset_dia_change_corr_120.py -- Correlation of ticker daily log-returns
to DIA (Dow-30 blue-chip proxy) daily log-returns, rolling 120-bar Pearson corr.

METHOD (per spec ff0703c_cross_asset_dia_change_corr_120):
  Over a trailing 120-bar window, Pearson correlation between the ticker's daily
  log-return and DIA's daily log-return. Higher = more "Dow / blue-chip cyclical"
  co-movement; lower/negative = defensive or idiosyncratic behavior relative to
  large-cap industrials/value names. Guards both series' rolling std > 1e-9.
  NaN for the first 120 valid bars (post index-join) -- expected leading NaN.

  A secondary short-window (20-bar) corr and its delta vs the 120-bar corr are
  included to capture regime dynamics (is the stock's Dow-beta currently
  drifting away from its longer-run baseline -- e.g. rotating into/out of a
  cyclical regime) without duplicating any existing single-window beta/corr
  feature already in the framework.

LEAKAGE / CAUSALITY:
  DIA close is joined via merge_asof(..., direction="backward") on Date, so only
  DIA data known as of (or before) each bar's Date is used. Log-returns use only
  current and past closes (diff of log price, no negative shift). Rolling corr at
  bar t uses bars [t-window+1, t] only -- strictly causal.

FEASIBILITY NOTE: fully faithful to the spec (per-ticker rolling corr vs DIA daily
change) -- no proxy needed since DIA is directly available via the _indexes helper.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff0703c_cross_asset_dia_change_corr_120",
    "description": (
        "Rolling 120-bar Pearson correlation between the ticker's daily log-return and "
        "DIA's (Dow-30) daily log-return -- a proxy for blue-chip/cyclical co-movement. "
        "Also includes a 20-bar fast-window corr and its delta vs the 120-bar baseline to "
        "capture regime drift in Dow-beta. Fully faithful implementation (DIA is directly "
        "available via the shared _indexes helper); no proxying required."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703c_dia_corr120",
        "ff0703c_dia_corr20",
        "ff0703c_dia_corr_delta",
    ],
    "tags": ["cross_asset", "correlation", "dia", "regime", "candidate"],
    "version": "1.0",
    "author": "feature-factory (ff0703c batch, cross_asset vein)",
}

_WIN_LONG = 120
_WIN_SHORT = 20
_MIN_STD = 1e-9


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df["ff0703c_dia_corr120"] = np.nan
    df["ff0703c_dia_corr20"] = np.nan
    df["ff0703c_dia_corr_delta"] = np.nan

    n = len(df)
    if n == 0 or "Close" not in df.columns or "Date" not in df.columns:
        return df

    try:
        dia_close = _indexes.index_close("DIA")
    except Exception:
        dia_close = pd.Series(dtype="float64")

    if dia_close.empty:
        return df

    dia_df = dia_close.rename("dia_close").reset_index()
    dia_df.columns = ["Date", "dia_close"]
    dia_df["Date"] = pd.to_datetime(dia_df["Date"])
    dia_df = dia_df.sort_values("Date").reset_index(drop=True)

    left = df[["Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_ord"] = np.arange(n)
    left = left.sort_values("Date")

    merged = pd.merge_asof(left, dia_df, on="Date", direction="backward")
    merged = merged.sort_values("_ord")
    dia_aligned = merged["dia_close"].to_numpy(dtype="float64")

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype="float64")

    # Causal log-returns (no negative shift): ret[t] = log(close[t] / close[t-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ticker_ret = np.full(n, np.nan, dtype="float64")
        valid_c = (close[1:] > 0) & (close[:-1] > 0)
        ticker_ret[1:] = np.where(valid_c, np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan)), np.nan)

        dia_ret = np.full(n, np.nan, dtype="float64")
        valid_d = (dia_aligned[1:] > 0) & (dia_aligned[:-1] > 0)
        dia_ret[1:] = np.where(
            valid_d,
            np.log(dia_aligned[1:] / np.where(dia_aligned[:-1] > 0, dia_aligned[:-1], np.nan)),
            np.nan,
        )

    tr = pd.Series(ticker_ret, index=df.index)
    dr = pd.Series(dia_ret, index=df.index)

    def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
        std_a = a.rolling(window, min_periods=window).std()
        std_b = b.rolling(window, min_periods=window).std()
        corr = a.rolling(window, min_periods=window).corr(b)
        bad = (std_a < _MIN_STD) | (std_b < _MIN_STD) | std_a.isna() | std_b.isna()
        corr = corr.where(~bad, np.nan)
        return corr.clip(-1.0, 1.0)

    corr120 = _rolling_corr(tr, dr, _WIN_LONG)
    corr20 = _rolling_corr(tr, dr, _WIN_SHORT)

    df["ff0703c_dia_corr120"] = corr120.to_numpy()
    df["ff0703c_dia_corr20"] = corr20.to_numpy()
    df["ff0703c_dia_corr_delta"] = (corr20 - corr120).to_numpy()

    return df
