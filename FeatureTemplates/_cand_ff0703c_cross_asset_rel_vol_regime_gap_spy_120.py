"""
_cand_ff0703c_cross_asset_rel_vol_regime_gap_spy_120.py

VEIN: cross_asset | BATCH: 0703c | SPEC: ff0703c_cross_asset_rel_vol_regime_gap_spy_120

METHOD
------
Relative volatility-regime gap vs SPY.

For the ticker and for SPY independently:
  1. 20-bar realized vol = rolling std of 1-bar log returns (bar-window, causal).
  2. Percentile-rank of the CURRENT 20-bar vol reading within its own trailing
     120-bar window of vol readings (fraction of the last-120 vol values that are
     <= the current one, i.e. "how hot is vol right now vs its own recent regime").

Feature = ticker_percentile_rank - spy_percentile_rank.
  Positive  -> ticker is unusually volatile relative to where SPY sits in ITS OWN
               volatility regime (idiosyncratic vol expansion, not just a market-wide
               vol event).
  Negative  -> ticker is comparatively calm while SPY's own regime is elevated.

NaN for the first 120+20 bars (insufficient history) per the spec ("NaN if <120 bars").

IMPLEMENTATION NOTES
---------------------
- Fully causal: rolling std uses only past+current bars; the percentile-rank window
  is a trailing (not centered) window ending at the current bar, so truncating the
  series at any point in the past reproduces identical historical values (gate's
  causality test). No negative shifts, no full-series stats.
- SPY's vol/percentile-rank series is computed once from the shared `_indexes`
  index-close loader (independent of the per-ticker df) and merged onto df via a
  backward merge_asof on Date -- lookahead-safe (SPY's own current-day close is
  available intraday/EOD same as the ticker's, matching how other index-coupled
  blocks in this repo, e.g. beta/VIX features, join index data).
- If SPY index data is unavailable, the spy percentile-rank degrades to NaN and the
  gap feature is NaN (graceful degradation, no crash).
- rolling(...).apply(..., raw=True) is used for the percentile-rank step. Window=120
  on ~700-row per-ticker frames is well within the <100ms budget.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

SPEC_ID = "ff0703c_cross_asset_rel_vol_regime_gap_spy_120"

VOL_WIN = 20
RANK_WIN = 120

METADATA = {
    "name": "ff0703c_cross_asset_rel_vol_regime_gap_spy_120",
    "description": (
        "Percentile-rank of ticker's trailing-20-bar realized vol within its own "
        "trailing-120-bar vol history, minus the same percentile-rank computed for "
        "SPY. Positive => ticker is unusually volatile relative to where SPY sits "
        "within SPY's own volatility regime (idiosyncratic vol expansion vs a "
        "market-wide vol event). Causal, per-ticker; SPY leg computed once from the "
        "shared _indexes helper and joined via backward merge_asof on Date."
    ),
    "requires": ["Close"],
    "produces": [
        f"{SPEC_ID}_gap",
        f"{SPEC_ID}_gap_chg5",
    ],
    "tags": ["cross_asset", "volatility", "regime", "spy", "relative"],
    "version": "1.0",
    "author": (
        "feature-factory codegen (per-ticker proxy, faithful to spec: percentile-rank "
        "of trailing-20-bar realized vol within trailing-120-bar own-history, ticker "
        "minus SPY; SPY leg externally computed + merge_asof-joined since compute() "
        "is single-ticker)"
    ),
}


def _rolling_pct_rank(vol: pd.Series, window: int) -> pd.Series:
    """Causal percentile-rank of the current value within its trailing `window`."""
    def _rank(x: np.ndarray) -> float:
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        if valid.size == 0:
            return np.nan
        return float((valid <= last).sum()) / float(valid.size)

    return vol.rolling(window=window, min_periods=window).apply(_rank, raw=True)


def _spy_gap_series() -> pd.DataFrame:
    """Compute SPY's own vol percentile-rank series once, keyed by Date."""
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"),
                              "_spy_pct_rank": pd.Series(dtype="float64")})

    if spy_close is None or spy_close.empty:
        return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"),
                              "_spy_pct_rank": pd.Series(dtype="float64")})

    spy_close = spy_close.sort_index()
    spy_ret = np.log(spy_close / spy_close.shift(1).replace(0, np.nan))
    spy_vol = spy_ret.rolling(window=VOL_WIN, min_periods=VOL_WIN).std()
    spy_pct = _rolling_pct_rank(spy_vol, RANK_WIN)

    out = spy_pct.reset_index()
    out.columns = ["Date", "_spy_pct_rank"]
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    gap_col = f"{SPEC_ID}_gap"
    chg_col = f"{SPEC_ID}_gap_chg5"

    df[gap_col] = np.nan
    df[chg_col] = np.nan

    n = len(df)
    if n == 0 or "Close" not in df.columns:
        return df

    close = df["Close"].astype("float64")
    ret = np.log(close / close.shift(1).replace(0, np.nan))
    vol = ret.rolling(window=VOL_WIN, min_periods=VOL_WIN).std()
    tk_pct = _rolling_pct_rank(vol, RANK_WIN)

    spy_gap = _spy_gap_series()
    if spy_gap.empty or "Date" not in df.columns:
        return df

    tmp = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]),
        "_tk_pct_rank": tk_pct.to_numpy(),
    })
    tmp = tmp.sort_values("Date")

    merged = pd.merge_asof(
        tmp, spy_gap, on="Date", direction="backward",
    )

    gap_vals = merged["_tk_pct_rank"].to_numpy() - merged["_spy_pct_rank"].to_numpy()

    # merged is sorted by Date; restore original df order via the tmp sort index
    gap_series = pd.Series(gap_vals, index=tmp.index).sort_index()

    df[gap_col] = gap_series.to_numpy()
    df[chg_col] = df[gap_col] - df[gap_col].shift(5)

    return df
