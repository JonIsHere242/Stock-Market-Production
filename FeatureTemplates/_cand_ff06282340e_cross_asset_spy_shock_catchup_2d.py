"""
ff06282340e_cross_asset_spy_shock_catchup_2d

Per-ticker cross-asset feature: measures the average 2-day cumulative ticker
return following SPY shock days (|SPY 1-day return| > 1.5x trailing SPY return
std over a 120-day window), signed by the direction of the shock.

This captures whether a stock tends to lag SPY on shock days and catch up
(positive = catch-up momentum after shocks), or front-runs / fades the shock.
The 2-day cumulative return is used to model the delayed re-pricing dynamic.

A per-ticker proxy: no cross-sectional comparison needed; all data is
derived from OHLCV + SPY index (via _indexes helper).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load index helper (by file path — never via package import)
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282340e_cross_asset_spy_shock_catchup_2d",
    "description": (
        "Cross-asset SPY-shock catch-up: computes the trailing-120-bar average "
        "2-day cumulative ticker return following SPY shock days (|SPY 1d ret| "
        "> 1.5 * trailing-120 SPY std), signed by shock direction. Positive "
        "values indicate that the stock tends to catch up in the 2 days after "
        "a SPY shock; negative values indicate fade/reversal. Also produces "
        "a count of shock days in the window (density) and a recency-weighted "
        "variant. Per-ticker proxy; causal; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282340e_cross_asset_spy_shock_catchup_2d_avg",    # main level
        "ff06282340e_cross_asset_spy_shock_catchup_2d_cnt",    # shock-day count
        "ff06282340e_cross_asset_spy_shock_catchup_2d_wt",     # recency-weighted
    ],
    "tags": ["cross_asset", "spy", "shock", "catchup", "momentum", "2d"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120       # trailing window for SPY std and catch-up averaging
_SHOCK_MULT = 1.5   # threshold multiplier on trailing SPY std
_CATCHUP_DAYS = 2   # how many days after shock to measure cumulative return


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: single-ticker OHLCV, ascending by Date.
    Adds three columns defined in METADATA["produces"].
    """
    # Initialise all produced columns to NaN (satisfies every-code-path rule)
    col_avg = "ff06282340e_cross_asset_spy_shock_catchup_2d_avg"
    col_cnt = "ff06282340e_cross_asset_spy_shock_catchup_2d_cnt"
    col_wt  = "ff06282340e_cross_asset_spy_shock_catchup_2d_wt"

    df[col_avg] = np.nan
    df[col_cnt] = np.nan
    df[col_wt]  = np.nan

    n = len(df)
    if n < _WINDOW + _CATCHUP_DAYS + 2:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close, align to ticker dates
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spy_series = _indexes.index_close("SPY")
        except Exception:
            return df

    if spy_series is None or spy_series.empty:
        return df

    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_dates = pd.to_datetime(df["Date"].values)
    merged = pd.merge_asof(
        pd.DataFrame({"Date": ticker_dates}),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    spy_close = merged["spy_close"].values.astype(float)

    # ------------------------------------------------------------------
    # 2. SPY daily returns (shift-1 = prior close, causal)
    # ------------------------------------------------------------------
    spy_ret = np.empty(n, dtype=float)
    spy_ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        spy_ret[1:] = np.where(
            spy_close[:-1] != 0,
            (spy_close[1:] - spy_close[:-1]) / spy_close[:-1],
            np.nan,
        )

    # ------------------------------------------------------------------
    # 3. Ticker 2-day cumulative forward return (strictly from past bars)
    #    fwd2[t] = cum-ret from close[t] to close[t+2]
    #    To keep it causal, we compute at bar t using close[t+2]/close[t]-1
    #    BUT we must NEVER peek ahead at bar t — we do this in the loop below
    #    by treating fwd2[t] as observable only at bar t+2 (bar where catchup
    #    is complete). We assign the catch-up value to bar t+2 (the bar where
    #    the result is first knowable).
    # ------------------------------------------------------------------
    ticker_close = df["Close"].values.astype(float)

    # 2-day cumulative return anchored to bar i, knowable at bar i+2
    fwd2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(n - _CATCHUP_DAYS):
            if ticker_close[i] != 0 and not np.isnan(ticker_close[i]):
                fwd2[i + _CATCHUP_DAYS] = (
                    ticker_close[i + _CATCHUP_DAYS] / ticker_close[i] - 1.0
                )

    # ------------------------------------------------------------------
    # 4. Shock indicator (knowable at bar t from SPY return at bar t)
    #    shock_sign[t] = +1/-1 if |spy_ret[t]| > threshold, else 0
    #    The catch-up is knowable at bar t+2, so at bar t+2 we pair
    #    shock_sign[t] with fwd2[t+2].
    # ------------------------------------------------------------------
    # Trailing 120-bar std of SPY returns (excluding current bar → shift by 1)
    spy_ret_series = pd.Series(spy_ret)
    spy_std_trailing = (
        spy_ret_series.shift(1).rolling(_WINDOW, min_periods=30).std().values
    )

    shock_threshold = _SHOCK_MULT * spy_std_trailing
    abs_spy_ret = np.abs(spy_ret)

    shock_sign = np.where(
        (abs_spy_ret > shock_threshold) & (~np.isnan(shock_threshold)),
        np.sign(spy_ret),
        0.0,
    )

    # ------------------------------------------------------------------
    # 5. For each bar t, accumulate signed catch-up samples over past window
    #    A sample at bar s contributes if: shock at bar s-2 (i.e. shock_sign
    #    at bar s-2 != 0) and fwd2[s] is valid.
    #    At evaluation bar t, we look back _WINDOW bars.
    # ------------------------------------------------------------------
    avg_vals = np.full(n, np.nan)
    cnt_vals = np.full(n, np.nan)
    wt_vals  = np.full(n, np.nan)

    for t in range(_WINDOW + _CATCHUP_DAYS, n):
        # Look-back window: bars [t-WINDOW, t]
        start = t - _WINDOW
        signed_catchups = []
        weights = []
        for s in range(start + _CATCHUP_DAYS, t + 1):
            shock_bar = s - _CATCHUP_DAYS
            if shock_bar < 0:
                continue
            ss = shock_sign[shock_bar]
            if ss == 0.0:
                continue
            f2 = fwd2[s]
            if np.isnan(f2):
                continue
            signed_val = ss * f2
            signed_catchups.append(signed_val)
            # recency weight: more recent shock gets higher weight
            age = t - shock_bar
            weights.append(1.0 / max(age, 1))

        if len(signed_catchups) == 0:
            avg_vals[t] = 0.0
            cnt_vals[t] = 0.0
            wt_vals[t]  = 0.0
        else:
            arr = np.array(signed_catchups, dtype=float)
            w   = np.array(weights, dtype=float)
            avg_vals[t] = np.nanmean(arr)
            cnt_vals[t] = float(len(arr))
            w_sum = w.sum()
            wt_vals[t] = (arr * w).sum() / w_sum if w_sum > 0 else np.nan

    df[col_avg] = avg_vals
    df[col_cnt] = cnt_vals
    df[col_wt]  = wt_vals

    return df
