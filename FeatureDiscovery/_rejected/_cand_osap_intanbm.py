"""
Intangible return using BM (Daniel & Titman 2006)
Per-ticker proxy: 5-year return residual after removing the component
explained by the change in book-to-market ratio.

The original is cross-sectional: monthly CS regression of 5yr return on
lagged BM and (delta_BM + 5yr_ret). The residual = IntanBM (predicted sign -1:
high intangible return stocks revert). This block implements the closest
faithful per-ticker version using PIT fundamentals for book value.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_intanbm",
    "description": (
        "Intangible return via book-to-market decomposition (Daniel & Titman 2006). "
        "Original method: monthly cross-sectional regression of 5-year stock return "
        "on 5-year-lagged BM and (delta_BM + 5yr_ret); residual = IntanBM. "
        "Per-ticker proxy: we compute the 5-year total return, the change in B/M "
        "over the same window (using PIT book_value_per_share / Close as B/M proxy), "
        "and back out the 'intangible' portion of the return as the return that was "
        "NOT accompanied by a corresponding improvement in B/M -- i.e., "
        "osap_intanbm_level = 5yr_ret - beta_ols * delta_bm, "
        "where beta_ols is a trailing 60-month OLS coefficient estimated per-ticker. "
        "Predicted sign -1 (high intangible return = future underperformance). "
        "PROXY NOTE: cross-sectional variation is lost; only time-series variation "
        "within a single ticker is captured."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_intanbm_level",    # intangible return (5yr return minus BM-explained component)
        "osap_intanbm_delta_bm", # 5yr change in B/M ratio (tangible component driver)
        "osap_intanbm_ret5y",    # raw 5yr total return (input signal)
    ],
    "tags": ["reversal", "long_term_reversal", "fundamentals", "intangibles", "accounting"],
    "version": "1.0",
    "author": "Daniel and Titman (2006); OpenSourceAP / Chen-Zimmermann; per-ticker proxy implementation",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_1Y = 252
_TRADING_DAYS_5Y = 252 * 5          # ~1260 bars for 5-year return
_TRADING_DAYS_5Y_LAG = 252 * 5      # lookback window
_MIN_OBS = 200                       # minimum bars for OLS slope estimate
_OLS_WINDOW = _TRADING_DAYS_5Y      # rolling OLS window for beta estimation


def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """
    Rolling OLS slope beta of y ~ x over a trailing window.
    Returns array of same length as y, with leading NaNs.
    Uses a numerically stable accumulation approach.
    """
    n = len(y)
    slopes = np.full(n, np.nan)
    for i in range(window - 1, n):
        yi = y[i - window + 1 : i + 1]
        xi = x[i - window + 1 : i + 1]
        mask = np.isfinite(yi) & np.isfinite(xi)
        if mask.sum() < min_obs:
            continue
        ym, xm = yi[mask] - yi[mask].mean(), xi[mask] - xi[mask].mean()
        denom = (xm * xm).sum()
        if denom == 0.0 or not np.isfinite(denom):
            continue
        slopes[i] = (xm * ym).sum() / denom
    return slopes


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # 1. Pull PIT book value per share
    # -----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["book_value_per_share"])

    bvps = df["fund_book_value_per_share"].values.astype(float)
    close = df["Close"].values.astype(float)

    # -----------------------------------------------------------------------
    # 2. B/M ratio (book per share / price); guard zero price
    # -----------------------------------------------------------------------
    bm = np.where(close > 0, bvps / close, np.nan)

    # -----------------------------------------------------------------------
    # 3. 5-year lagged quantities (shift forward by 5yr trading days)
    # -----------------------------------------------------------------------
    lag = _TRADING_DAYS_5Y

    # 5-year return: (Close_t / Close_{t-lag}) - 1
    close_lag = np.full_like(close, np.nan)
    close_lag[lag:] = close[:-lag]
    ret5y = np.where(
        np.isfinite(close_lag) & (close_lag > 0),
        close / close_lag - 1.0,
        np.nan,
    )

    # 5-year change in B/M: bm_t - bm_{t-lag}
    bm_lag = np.full_like(bm, np.nan)
    bm_lag[lag:] = bm[:-lag]
    delta_bm = np.where(
        np.isfinite(bm) & np.isfinite(bm_lag),
        bm - bm_lag,
        np.nan,
    )

    # -----------------------------------------------------------------------
    # 4. Rolling OLS: slope of ret5y ~ delta_bm
    #    This proxies the cross-sectional regression coefficient that
    #    determines how much of the return is "explained" by BM change.
    # -----------------------------------------------------------------------
    # Use a window of 5 years of DAILY observations (re-rolling so we have
    # a stable beta estimate at each date without lookahead).
    window = min(_OLS_WINDOW, len(df))
    beta = _rolling_ols_slope(ret5y, delta_bm, window=window, min_obs=_MIN_OBS)

    # -----------------------------------------------------------------------
    # 5. Intangible return = ret5y - beta * delta_bm
    #    (the return component not explained by the BM change)
    # -----------------------------------------------------------------------
    intanbm = np.where(
        np.isfinite(ret5y) & np.isfinite(delta_bm) & np.isfinite(beta),
        ret5y - beta * delta_bm,
        np.nan,
    )

    # Replace any inf values with NaN
    intanbm = np.where(np.isfinite(intanbm), intanbm, np.nan)
    ret5y_out = np.where(np.isfinite(ret5y), ret5y, np.nan)
    delta_bm_out = np.where(np.isfinite(delta_bm), delta_bm, np.nan)

    # -----------------------------------------------------------------------
    # 6. Assign produced columns; drop scratch fundamental column
    # -----------------------------------------------------------------------
    df["osap_intanbm_level"] = intanbm
    df["osap_intanbm_delta_bm"] = delta_bm_out
    df["osap_intanbm_ret5y"] = ret5y_out

    # Drop scratch fund_ column not listed in produces
    if "fund_book_value_per_share" in df.columns:
        df = df.drop(columns=["fund_book_value_per_share"])

    return df
