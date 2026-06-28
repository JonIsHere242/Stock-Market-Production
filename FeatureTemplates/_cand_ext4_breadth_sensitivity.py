"""
Market-breadth sensitivity feature block.

Computes rolling beta of a stock's daily return on the daily change in
log(IWM/SPY) — a breadth proxy (equal-weight vs cap-weight). Rising
IWM/SPY = broadening market participation. Beta captures how sensitive
this stock is to breadth regimes.

Produces:
  ext4_breadth_sensitivity_beta  : 120-day rolling OLS beta of stock
                                    return on d_log_breadth
  ext4_breadth_sensitivity_chg   : 20-day change in the beta (momentum
                                    of sensitivity shift)

Per-ticker proxy: cross-sectional breadth ranks are not available in
single-stock mode; instead we regress each ticker's return against the
same market-wide IWM/SPY signal, which faithfully captures the economic
signal at the ticker level.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Load _indexes helper by file path (required pattern)
# --------------------------------------------------------------------------- #
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
METADATA = {
    "name": "ext4_breadth_sensitivity",
    "description": (
        "Rolling 120-day beta of the stock's daily return on d_log(IWM/SPY), "
        "a market-breadth proxy (equal-weight vs cap-weight). High beta = "
        "stock moves with broadening participation; 20-day change (chg) "
        "captures sensitivity shifts. Per-ticker proxy for cross-sectional "
        "breadth factor. Lookahead-free via merge_asof backward join."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_breadth_sensitivity_beta",
        "ext4_breadth_sensitivity_chg",
    ],
    "tags": ["market_breadth", "beta", "regime", "multi_index", "macro"],
    "version": "1.0",
    "author": "Round-5 expansion (NEW: multi-index factor)",
}

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_BETA_WINDOW = 120   # rolling window for OLS beta
_CHG_WINDOW  = 20    # lookback for change in beta


def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling OLS beta of y on x (no intercept-less; full OLS slope).
    Both arrays must be aligned and same length. Returns array of same length,
    with leading NaNs where the window is incomplete.
    Vectorised via cumulative sums (O(n), no Python row-loop).
    """
    n = len(y)
    out = np.full(n, np.nan)

    # Precompute prefix sums for: x, y, x^2, x*y, count
    # Use numpy cumsum trick: window sum = cs[t] - cs[t-window]
    # NaN-aware: mask positions where either series is NaN
    valid = (~np.isnan(x)) & (~np.isnan(y))
    xc = np.where(valid, x, 0.0)
    yc = np.where(valid, y, 0.0)
    vc = valid.astype(np.float64)          # 1 if valid, else 0
    x2c = np.where(valid, x * x, 0.0)
    xyc = np.where(valid, x * y, 0.0)

    cx  = np.cumsum(xc)
    cy  = np.cumsum(yc)
    cv  = np.cumsum(vc)
    cx2 = np.cumsum(x2c)
    cxy = np.cumsum(xyc)

    for t in range(window - 1, n):
        lo = t - window          # index before window start (or -1)
        if lo >= 0:
            sx  = cx[t]  - cx[lo]
            sy  = cy[t]  - cy[lo]
            sv  = cv[t]  - cv[lo]
            sx2 = cx2[t] - cx2[lo]
            sxy = cxy[t] - cxy[lo]
        else:
            sx  = cx[t]
            sy  = cy[t]
            sv  = cv[t]
            sx2 = cx2[t]
            sxy = cxy[t]

        if sv < 10:           # require at least 10 valid observations
            continue
        # OLS beta: (n*Sxy - Sx*Sy) / (n*Sx2 - Sx^2)
        denom = sv * sx2 - sx * sx
        if denom == 0.0:
            continue
        out[t] = (sv * sxy - sx * sy) / denom

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-breadth sensitivity features."""
    # Initialise output columns to NaN
    df["ext4_breadth_sensitivity_beta"] = np.nan
    df["ext4_breadth_sensitivity_chg"]  = np.nan

    if len(df) < _BETA_WINDOW + _CHG_WINDOW + 5:
        return df

    # ------------------------------------------------------------------ #
    # 1. Pull IWM and SPY close series; align to df via merge_asof
    # ------------------------------------------------------------------ #
    try:
        iwm_close = _indexes.index_close("IWM")
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if iwm_close is None or spy_close is None:
        return df

    iwm_df = iwm_close.rename("iwm").reset_index().rename(columns={"index": "Date"})
    spy_df = spy_close.rename("spy").reset_index().rename(columns={"index": "Date"})

    # Ensure DatetimeIndex columns are named "Date"
    if "Date" not in iwm_df.columns:
        iwm_df.columns = ["Date", "iwm"]
    if "Date" not in spy_df.columns:
        spy_df.columns = ["Date", "spy"]

    iwm_df["Date"] = pd.to_datetime(iwm_df["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    df_work = df[["Date", "Close"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = df_work.sort_values("Date").reset_index(drop=True)

    # Merge IWM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_work = pd.merge_asof(
            df_work,
            iwm_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        df_work = pd.merge_asof(
            df_work,
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )

    # ------------------------------------------------------------------ #
    # 2. Compute daily breadth signal: d_log(IWM/SPY)
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (df_work["spy"].values > 0) & (df_work["iwm"].values > 0),
            np.log(df_work["iwm"].values / df_work["spy"].values),
            np.nan,
        )
    d_breadth = np.diff(ratio, prepend=np.nan)   # first element NaN

    # ------------------------------------------------------------------ #
    # 3. Stock daily return
    # ------------------------------------------------------------------ #
    stock_ret = df_work["Close"].pct_change().values  # first element NaN

    # ------------------------------------------------------------------ #
    # 4. Rolling 120-day OLS beta
    # ------------------------------------------------------------------ #
    beta_arr = _rolling_beta(stock_ret, d_breadth, _BETA_WINDOW)

    # ------------------------------------------------------------------ #
    # 5. 20-day change in beta
    # ------------------------------------------------------------------ #
    chg_arr = np.full(len(beta_arr), np.nan)
    for t in range(_CHG_WINDOW, len(beta_arr)):
        prev = beta_arr[t - _CHG_WINDOW]
        if not (np.isnan(beta_arr[t]) or np.isnan(prev)):
            chg_arr[t] = beta_arr[t] - prev

    # ------------------------------------------------------------------ #
    # 6. Map back onto original df index (df_work is sorted, df may not be)
    # ------------------------------------------------------------------ #
    result = df_work[["Date"]].copy()
    result["ext4_breadth_sensitivity_beta"] = beta_arr
    result["ext4_breadth_sensitivity_chg"]  = chg_arr

    # Guard inf
    for col in ["ext4_breadth_sensitivity_beta", "ext4_breadth_sensitivity_chg"]:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Align back to original df (which may have a different sort order / index)
    df_orig_dates = pd.to_datetime(df["Date"])
    date_to_beta = result.set_index("Date")["ext4_breadth_sensitivity_beta"]
    date_to_chg  = result.set_index("Date")["ext4_breadth_sensitivity_chg"]

    df["ext4_breadth_sensitivity_beta"] = df_orig_dates.map(date_to_beta).values
    df["ext4_breadth_sensitivity_chg"]  = df_orig_dates.map(date_to_chg).values

    return df
