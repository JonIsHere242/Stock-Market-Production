"""
Coskewness with SPY over a 120-day trailing window.

Coskewness measures whether a stock tends to deliver poor returns precisely
when the market (SPY) makes large moves of either sign.  Negative coskewness
means the stock's losses cluster around high-variance market days -- a tail-
risk premium that should (per theory) earn a return premium.

E[(r_i - mu_i)(r_m - mu_m)^2] / (std_i * var_m)

A second column tracks the 20-day rolling z-score of that coskewness so the
model can see whether the relationship is strengthening or weakening.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helper: SPY index
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_factor_coskewness_spy_120",
    "description": (
        "120-day rolling coskewness of the ticker with SPY: "
        "E[(r_i-mu_i)(r_m-mu_m)^2] / (std_i * var_m).  "
        "Negative values signal that losses cluster on high-vol market days "
        "(coskewness risk premium).  Second column is the 20-day rolling "
        "z-score of the raw coskewness to capture regime changes."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_coskewness_spy_120_raw",
        "ff06282316b_factor_coskewness_spy_120_zscore",
    ],
    "tags": ["factor", "coskewness", "market", "tail-risk", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
_WINDOW = 120     # coskewness lookback in trading days
_Z_WIN  = 20      # z-score window for the regime signal
_EPS    = 1e-12   # guard against division by zero


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (gate requirement)
    raw_col    = "ff06282316b_factor_coskewness_spy_120_raw"
    zscore_col = "ff06282316b_factor_coskewness_spy_120_zscore"
    df[raw_col]    = np.nan
    df[zscore_col] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY Close and align to ticker dates (backward merge_asof)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order (df is guaranteed ascending by Date, but be safe)
    work = work.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Compute log-returns (ticker and SPY)
    # ------------------------------------------------------------------
    ri = np.log(work["Close"] / work["Close"].shift(1)).values        # shape (n,)
    rm = np.log(work["spy_close"] / work["spy_close"].shift(1)).values

    n = len(ri)

    # ------------------------------------------------------------------
    # 3. Rolling coskewness via numpy sliding view
    # ------------------------------------------------------------------
    from numpy.lib.stride_tricks import sliding_window_view  # numpy >= 1.20

    # We need at least _WINDOW valid return pairs.
    # First valid return is index 1, so first full window is index _WINDOW.
    start = _WINDOW  # first output index (0-based in the returns array)

    ri_wins = sliding_window_view(ri, _WINDOW)   # shape (n-W+1, W)
    rm_wins = sliding_window_view(rm, _WINDOW)

    # Compute means per window
    mu_i = np.nanmean(ri_wins, axis=1)   # (n-W+1,)
    mu_m = np.nanmean(rm_wins, axis=1)

    # Centred residuals
    di = ri_wins - mu_i[:, np.newaxis]   # (n-W+1, W)
    dm = rm_wins - mu_m[:, np.newaxis]

    # Coskewness numerator: mean[(r_i - mu_i)(r_m - mu_m)^2]
    numer = np.nanmean(di * dm ** 2, axis=1)

    # Denominator: std_i * var_m
    std_i = np.nanstd(ri_wins, axis=1, ddof=1)
    var_m = np.nanvar(rm_wins, axis=1, ddof=1)
    denom = std_i * var_m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coskew = np.where(np.abs(denom) < _EPS, np.nan, numer / denom)

    # Map back: sliding_window_view output index k corresponds to the window
    # ending at returns index k + W - 1, which maps to df row k + W - 1.
    # But ri[0] = NaN (first diff), so returns window ending at index W-1 in
    # ri corresponds to bars 0..W-1 in work.  Output row index = k + W - 1.
    out_indices = np.arange(len(coskew)) + _WINDOW - 1   # 0-based in work

    raw_vals = np.full(n, np.nan)
    raw_vals[out_indices] = coskew

    # ------------------------------------------------------------------
    # 4. 20-day rolling z-score of raw coskewness
    # ------------------------------------------------------------------
    raw_series = pd.Series(raw_vals)
    roll_mean  = raw_series.rolling(_Z_WIN, min_periods=_Z_WIN).mean()
    roll_std   = raw_series.rolling(_Z_WIN, min_periods=_Z_WIN).std(ddof=1)
    zscore_vals = np.where(
        roll_std.values < _EPS,
        np.nan,
        (raw_series.values - roll_mean.values) / roll_std.values,
    )

    # ------------------------------------------------------------------
    # 5. Write into df (work is sorted the same way as df which is ascending)
    # ------------------------------------------------------------------
    df[raw_col]    = raw_vals
    df[zscore_col] = zscore_vals

    return df
