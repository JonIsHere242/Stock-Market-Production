"""
ext3_earnings_persistence — Earnings persistence / stability.

From PIT net_income_ttm, produces:
  1. ext3_earnings_persistence_cv      : coefficient of variation of net_income_ttm over ~3 yr (756 trading days).
                                         Low CV = more stable earnings.  Inverted so high = more persistent.
  2. ext3_earnings_persistence_stab    : rolling std of YoY (252-day) changes in net_income_ttm, inverted
                                         (1 / (1 + std)) so high = more persistent / less volatile earnings.
  3. ext3_earnings_persistence_trend   : sign * magnitude of the 252-day slope of net_income_ttm (linear
                                         coefficient from rolling OLS proxy via polyfit on evenly-spaced index),
                                         normalised by the mean absolute level, so it is scale-free.

All columns are NaN where fundamentals coverage is absent or rolling windows are too short.
Scratch fund_* columns are dropped before return.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_earnings_persistence",
    "description": (
        "Earnings persistence / stability from PIT net_income_ttm. "
        "Produces (1) inverted coefficient of variation over ~3yr (higher = more stable), "
        "(2) inverted rolling std of YoY NI changes (higher = more persistent), "
        "(3) scale-free 252-day slope of net_income_ttm (signed momentum). "
        "Per-ticker OHLCV proxy using PIT SEC fundamentals; "
        "NaN for tickers without fundamentals coverage (~16%% of universe). "
        "Cross-sectional ranking should be applied downstream."
    ),
    "requires": [],
    "produces": [
        "ext3_earnings_persistence_cv",
        "ext3_earnings_persistence_stab",
        "ext3_earnings_persistence_trend",
    ],
    "tags": ["fundamentals", "earnings", "persistence", "stability", "quality"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_orgcap) — ext3_earnings_persistence spec",
}

# ---------------------------------------------------------------------------
# Rolling linear slope helper (vectorised via sliding_window_view)
# ---------------------------------------------------------------------------
_SLOPE_WIN = 252  # ~1yr of trading days


def _rolling_slope_normalised(series: pd.Series, window: int) -> pd.Series:
    """
    Return a scale-free rolling linear slope: beta / mean_abs(y).
    beta is the OLS slope coefficient (normalised x in [0, window-1]).
    Result is NaN for leading rows and where mean_abs == 0.
    """
    vals = series.to_numpy(dtype=float, na_value=np.nan)
    n = len(vals)
    out = np.full(n, np.nan)

    if n < window:
        return pd.Series(out, index=series.index)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_denom = np.sum((x - x_mean) ** 2)

    if x_denom == 0:
        return pd.Series(out, index=series.index)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(vals, window)  # shape (n-window+1, window)
    except Exception:
        return pd.Series(out, index=series.index)

    y = windows  # (M, window)
    y_mean = np.nanmean(y, axis=1)  # (M,)
    cov = np.nansum((y - y_mean[:, None]) * (x - x_mean)[None, :], axis=1)
    beta = cov / x_denom  # (M,)

    abs_mean = np.abs(y_mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(abs_mean > 0, beta / abs_mean, np.nan)

    out[window - 1:] = normalised
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT net_income_ttm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["net_income_ttm"])

    ni = df["fund_net_income_ttm"].copy()  # may be all-NaN for ETFs

    # -----------------------------------------------------------------------
    # 1. Coefficient of variation over ~3yr (~756 trading days), inverted
    #    persistence_cv = 1 / (1 + |std / mean|)   so high = more stable
    # -----------------------------------------------------------------------
    _CV_WIN = 756
    roll_mean = ni.rolling(_CV_WIN, min_periods=63).mean()
    roll_std = ni.rolling(_CV_WIN, min_periods=63).std()

    with np.errstate(invalid="ignore", divide="ignore"):
        raw_cv = np.where(
            roll_mean.abs() > 0,
            roll_std.abs() / roll_mean.abs(),
            np.nan,
        )
    # invert: low CV (stable) -> high score
    persistence_cv = 1.0 / (1.0 + np.where(np.isfinite(raw_cv), raw_cv, np.nan))
    df["ext3_earnings_persistence_cv"] = persistence_cv

    # -----------------------------------------------------------------------
    # 2. Rolling std of YoY (252-day) changes, inverted
    #    persistence_stab = 1 / (1 + rolling_std(delta_ni))
    # -----------------------------------------------------------------------
    delta_ni = ni.diff(252)  # YoY change in TTM NI
    _STAB_WIN = 504  # ~2yr of YoY deltas
    roll_delta_std = delta_ni.rolling(_STAB_WIN, min_periods=63).std()

    with np.errstate(invalid="ignore", divide="ignore"):
        inv_stab = 1.0 / (1.0 + np.where(
            np.isfinite(roll_delta_std.values),
            roll_delta_std.values,
            np.nan,
        ))
    df["ext3_earnings_persistence_stab"] = inv_stab

    # -----------------------------------------------------------------------
    # 3. Scale-free rolling slope of net_income_ttm over 252 days
    # -----------------------------------------------------------------------
    df["ext3_earnings_persistence_trend"] = _rolling_slope_normalised(ni, _SLOPE_WIN)

    # -----------------------------------------------------------------------
    # Drop scratch fund_ columns
    # -----------------------------------------------------------------------
    df = df.drop(columns=["fund_net_income_ttm"], errors="ignore")

    return df
