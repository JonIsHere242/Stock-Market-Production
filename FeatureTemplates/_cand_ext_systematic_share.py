"""
Systematic variance share (market R^2) and trend.

Rolling 120-day R^2 of the market-model regression (stock return ~ SPY return).
Produces:
  ext_systematic_share_r2    - fraction of variance explained by the market (systematic share)
  ext_systematic_share_idio  - 1 - R^2 (idiosyncratic share; complement of systematic)
  ext_systematic_share_delta - 60-day change in R^2 (rising co-movement / regime integration)

Per-ticker proxy: the cross-sectional R^2 distribution cannot be computed here, but the
rolling bivariate regression on SPY captures the same economic signal (how much of this
stock's daily variance is driven by systematic/market forces vs. stock-specific noise).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY)
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
    "name": "ext_systematic_share",
    "description": (
        "Rolling 120-day market-model R^2 (systematic variance share), its complement "
        "(idiosyncratic share = 1 - R^2), and the 60-day change in R^2 (rising "
        "co-movement / regime integration). Extends the osap_idiovolaht family by "
        "capturing the FRACTION of variance explained rather than the level of "
        "idiosyncratic volatility itself -- a different signal axis. "
        "Per-ticker proxy using SPY as the market factor (lookahead-safe via merge_asof). "
        "Based on: Extension/exploration of gate-validated winner osap_idiovolaht."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_systematic_share_r2",
        "ext_systematic_share_idio",
        "ext_systematic_share_delta",
    ],
    "tags": ["market_model", "r2", "systematic_risk", "idiosyncratic", "regime"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner osap_idiovolaht",
}

# ---------------------------------------------------------------------------
# Rolling R^2 helper (vectorised via sliding_window_view)
# ---------------------------------------------------------------------------
_WINDOW = 120
_DELTA_LAG = 60
_MIN_OBS = 30  # minimum observations needed for a valid R^2


def _rolling_r2(x: np.ndarray, y: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """
    Rolling OLS R^2 of y ~ x (with intercept) using Pearson correlation:
    R^2 = corr(x, y)^2 for a bivariate regression with intercept.
    Vectorised via rolling correlation computed in a single pass.
    """
    n = len(x)
    r2 = np.full(n, np.nan)

    # Use numpy sliding_window_view for efficiency
    if n < window:
        return r2

    from numpy.lib.stride_tricks import sliding_window_view

    xw = sliding_window_view(x, window_shape=window)  # shape (n-window+1, window)
    yw = sliding_window_view(y, window_shape=window)

    # Pearson r across the window axis (axis=1)
    xm = xw.mean(axis=1)
    ym = yw.mean(axis=1)

    xd = xw - xm[:, None]
    yd = yw - ym[:, None]

    num = (xd * yd).sum(axis=1)
    denom = np.sqrt((xd**2).sum(axis=1) * (yd**2).sum(axis=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, num / denom, np.nan)

    r2_vals = corr**2  # R^2 = r^2 for bivariate OLS with intercept

    # Count valid pairs and mask windows with too few observations
    valid_counts = (~(np.isnan(xw) | np.isnan(yw))).sum(axis=1)
    r2_vals = np.where(valid_counts >= min_obs, r2_vals, np.nan)

    # Assign to the last position of each window (index window-1 onwards)
    r2[window - 1:] = r2_vals

    return r2


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext_systematic_share_r2, ext_systematic_share_idio, ext_systematic_share_delta."""

    n = len(df)
    nan_col = np.full(n, np.nan)

    # --- Load SPY close series ---
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series indexed by DatetimeIndex Date
    except Exception:
        spy_close = None

    if spy_close is None or len(spy_close) == 0:
        df["ext_systematic_share_r2"] = nan_col.copy()
        df["ext_systematic_share_idio"] = nan_col.copy()
        df["ext_systematic_share_delta"] = nan_col.copy()
        return df

    # --- Align SPY to stock dates via merge_asof (lookahead-safe) ---
    stock_dates = pd.to_datetime(df["Date"])

    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        pd.DataFrame({"Date": stock_dates}).sort_values("Date"),
        spy_df,
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.set_index(stock_dates.sort_values().index)
    merged = merged.loc[df.index] if df.index.is_monotonic_increasing else merged

    spy_aligned = merged["spy_close"].values.astype(float)

    # --- Daily log-returns ---
    stock_price = df["Close"].values.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        stock_ret = np.where(
            stock_price[:-1] > 0,
            np.log(stock_price[1:] / stock_price[:-1]),
            np.nan,
        )
        spy_ret = np.where(
            spy_aligned[:-1] > 0,
            np.log(spy_aligned[1:] / spy_aligned[:-1]),
            np.nan,
        )

    # Prepend NaN for day 0 (no prior bar)
    stock_ret = np.concatenate([[np.nan], stock_ret])
    spy_ret = np.concatenate([[np.nan], spy_ret])

    # --- Rolling R^2 ---
    r2 = _rolling_r2(spy_ret, stock_ret, window=_WINDOW, min_obs=_MIN_OBS)

    # Clamp to [0, 1] (numerical safety)
    r2 = np.clip(r2, 0.0, 1.0)

    idio = np.where(np.isnan(r2), np.nan, 1.0 - r2)

    # --- 60-day change in R^2 ---
    r2_series = pd.Series(r2)
    delta = (r2_series - r2_series.shift(_DELTA_LAG)).values

    df["ext_systematic_share_r2"] = r2
    df["ext_systematic_share_idio"] = idio
    df["ext_systematic_share_delta"] = delta

    return df
