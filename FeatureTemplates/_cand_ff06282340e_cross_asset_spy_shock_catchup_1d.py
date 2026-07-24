"""
ff06282340e_cross_asset_spy_shock_catchup_1d

Capture the average signed catch-up / reaction of a stock on the day AFTER
a large SPY shock (|SPY return| > 1.5 * trailing SPY return std).

The "direction-adjusted next-day ticker return" on shock+1 days measures
whether the stock systematically leads/lags the market after large macro moves.
Positive = stock tends to catch up (follow) SPY shocks with a one-day delay.
Negative = stock tends to fade / mean-revert after SPY shocks.

This is a per-ticker proxy for the cross-sectional "shock catch-up beta at lag 1".
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path — no import-system coupling)
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
    "name": "ff06282340e_cross_asset_spy_shock_catchup_1d",
    "description": (
        "Per-ticker proxy for SPY-shock catch-up at lag 1. "
        "Identifies trailing-120 bars where |SPY daily return| > 1.5 × trailing "
        "SPY return std. Computes: (1) rolling mean of the ticker's next-day return "
        "on those shock days, signed by shock direction — positive means the stock "
        "systematically chases macro shocks with a one-bar delay (catch-up beta); "
        "(2) a recency-weighted variant (exponential decay, half-life 60 bars) to "
        "capture time-varying catch-up dynamics; (3) the fraction of the trailing "
        "window that actually contained SPY shocks (shock density). "
        "Inherently single-ticker; cross-sectional ranking done downstream."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282340e_catchup_mean",       # signed mean ticker ret on SPY shock+1 days
        "ff06282340e_catchup_ewm",        # exponentially-weighted variant (recency bias)
        "ff06282340e_shock_density",      # fraction of window that had SPY shocks
    ],
    "tags": ["cross_asset", "spy", "macro_shock", "lag1", "catch_up"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120          # rolling look-back (bars)
_SHOCK_MULT = 1.5     # number of trailing-std to define a large SPY move
_STD_MIN_PERIODS = 20 # minimum bars to estimate SPY vol
_HALFLIFE = 60        # EWM half-life in bars for recency-weighted variant


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add SPY-shock catch-up columns to the per-ticker DataFrame."""

    # Pre-initialise all produced columns to NaN so every code path covers them
    df["ff06282340e_catchup_mean"] = np.nan
    df["ff06282340e_catchup_ewm"] = np.nan
    df["ff06282340e_shock_density"] = np.nan

    if len(df) < _STD_MIN_PERIODS + 2:
        return df

    # -----------------------------------------------------------------------
    # 1. Fetch SPY close and align to ticker dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df  # helper unavailable — degrade gracefully

    if spy_close is None or spy_close.empty:
        return df

    # Build a small SPY frame and merge backward onto ticker dates
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_dates = df[["Date"]].copy()
    ticker_dates["Date"] = pd.to_datetime(ticker_dates["Date"])

    merged = pd.merge_asof(
        ticker_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.set_index(ticker_dates.sort_values("Date").index)
    merged = merged.sort_index()

    spy_c = merged["spy_close"].values.astype(float)

    # SPY daily returns (lag 0 return = close[t]/close[t-1] - 1)
    spy_ret = np.empty(len(spy_c))
    spy_ret[:] = np.nan
    spy_ret[1:] = spy_c[1:] / np.where(spy_c[:-1] == 0, np.nan, spy_c[:-1]) - 1.0

    # -----------------------------------------------------------------------
    # 2. Ticker daily returns (lag 0)
    # -----------------------------------------------------------------------
    close = df["Close"].values.astype(float)
    tkr_ret = np.empty(len(close))
    tkr_ret[:] = np.nan
    tkr_ret[1:] = close[1:] / np.where(close[:-1] == 0, np.nan, close[:-1]) - 1.0

    n = len(df)

    # -----------------------------------------------------------------------
    # 3. Rolling computation — iterate over anchored windows
    #    We use a stride=1 loop but vectorised inner ops; n<=~700 rows is fast.
    # -----------------------------------------------------------------------
    catchup_mean = np.full(n, np.nan)
    catchup_ewm = np.full(n, np.nan)
    shock_density = np.full(n, np.nan)

    # Pre-compute exponential weights for a window of size _WINDOW
    # weights[0] = oldest, weights[-1] = most recent
    alpha = 1.0 - np.exp(-np.log(2) / _HALFLIFE)
    raw_w = (1 - alpha) ** np.arange(_WINDOW - 1, -1, -1)  # shape (_WINDOW,)

    for t in range(_WINDOW, n):
        # Window indices: [t-_WINDOW .. t-1]  (all past, no lookahead)
        w_start = t - _WINDOW
        w_end = t  # exclusive

        spy_w = spy_ret[w_start:w_end]  # length _WINDOW
        tkr_w = tkr_ret[w_start:w_end]

        # SPY trailing std within this window (ignore first std-min guard)
        valid_spy = spy_w[~np.isnan(spy_w)]
        if len(valid_spy) < _STD_MIN_PERIODS:
            continue

        spy_std = valid_spy.std(ddof=1)
        if spy_std == 0 or np.isnan(spy_std):
            continue

        threshold = _SHOCK_MULT * spy_std

        # Identify shock days within the window: bar i is a shock if
        # |spy_ret[i]| > threshold. We need the ticker return on day i+1
        # (the catch-up bar). Since i is within [w_start, w_end-1] and
        # we need i+1 <= w_end (= t), we restrict shock bars to w_start..t-2
        # so the catch-up bar i+1 <= t-1 (still historical, no lookahead).
        shock_flags = np.zeros(_WINDOW - 1, dtype=bool)
        spy_slice = spy_w[:-1]   # bars [w_start .. t-2], length _WINDOW-1
        tkr_catchup = tkr_w[1:]  # bars [w_start+1 .. t-1], length _WINDOW-1

        abs_spy = np.abs(spy_slice)
        shock_flags = (abs_spy > threshold) & (~np.isnan(spy_slice)) & (~np.isnan(tkr_catchup))

        n_shocks = shock_flags.sum()
        shock_density[t] = n_shocks / (_WINDOW - 1)

        if n_shocks == 0:
            catchup_mean[t] = 0.0
            catchup_ewm[t] = 0.0
            continue

        # Signed catch-up returns: ticker ret on day after shock, signed by shock direction
        shock_signs = np.sign(spy_slice[shock_flags])
        signed_catchup = tkr_catchup[shock_flags] * shock_signs

        # Simple mean
        catchup_mean[t] = np.nanmean(signed_catchup)

        # Exponentially-weighted mean (more weight on recent shocks)
        # shock bar positions within the window (0=oldest, _WINDOW-2=most recent)
        positions = np.where(shock_flags)[0]  # positions in [0, _WINDOW-2]
        # Map to raw_w index: position p in window of size _WINDOW-1
        # raw_w has size _WINDOW; use positions directly for the EWM decay
        ew = (1 - alpha) ** (_WINDOW - 2 - positions)  # higher weight = more recent
        ew_sum = ew.sum()
        if ew_sum > 0:
            catchup_ewm[t] = np.sum(signed_catchup * ew) / ew_sum
        else:
            catchup_ewm[t] = catchup_mean[t]

    df["ff06282340e_catchup_mean"] = catchup_mean
    df["ff06282340e_catchup_ewm"] = catchup_ewm
    df["ff06282340e_shock_density"] = shock_density

    return df
