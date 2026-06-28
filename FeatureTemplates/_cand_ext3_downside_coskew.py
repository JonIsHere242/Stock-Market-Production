"""
Downside-conditional coskewness with the SPY market index.

Coskewness of a stock with SPY, computed separately on market-down days
(the crash-coskew that is priced in the cross-section) vs all days, plus
their difference. Uses a 250-day rolling window.

Spec: ext3_downside_coskew  (Round-4 expansion; extends osap_coskewacx)
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional helper: market index
# ---------------------------------------------------------------------------
try:
    _s = _ilu.spec_from_file_location(
        "_indexes", _P(__file__).resolve().parent / "_indexes.py"
    )
    _indexes = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_indexes)
    _HAS_INDEXES = True
except Exception:
    _HAS_INDEXES = False

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_downside_coskew",
    "description": (
        "Downside-conditional coskewness with SPY over a 250-day rolling window. "
        "Produces: (1) ext3_downside_coskew_down — coskew restricted to market-down days "
        "(the crash-premium that is priced in expected returns); "
        "(2) ext3_downside_coskew_all — unconditional 250-day coskew with SPY; "
        "(3) ext3_downside_coskew_diff — downside minus all-days coskew "
        "(captures asymmetry of co-movement beyond symmetric coskew). "
        "Per-ticker proxy; cross-sectional ranking applied separately. "
        "Falls back to NaN if _indexes helper is unavailable or SPY data is missing."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_downside_coskew_down",
        "ext3_downside_coskew_all",
        "ext3_downside_coskew_diff",
    ],
    "tags": ["coskewness", "downside-risk", "market-beta", "crash-risk", "spy"],
    "version": "1.0.0",
    "author": "Round-4 expansion spec (osap_coskewacx); implementation by Claude",
}

# ---------------------------------------------------------------------------
# Helper: rolling coskewness via numpy sliding window
# ---------------------------------------------------------------------------
_WINDOW = 250
_MIN_OBS = 60  # minimum observations to emit a value


def _coskew(r_stock: np.ndarray, r_mkt: np.ndarray) -> float:
    """
    Coskewness: E[(r_s - mu_s)(r_m - mu_m)^2] /
                (std(r_s) * std(r_m)^2)
    Returns nan if insufficient variation.
    """
    n = len(r_stock)
    if n < 2:
        return np.nan
    mu_s = r_stock.mean()
    mu_m = r_mkt.mean()
    dev_s = r_stock - mu_s
    dev_m = r_mkt - mu_m
    num = (dev_s * dev_m ** 2).mean()
    std_s = r_stock.std(ddof=1)
    std_m = r_mkt.std(ddof=1)
    denom = std_s * std_m ** 2
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    val = num / denom
    return val if np.isfinite(val) else np.nan


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Initialise output columns to NaN
    df["ext3_downside_coskew_down"] = np.nan
    df["ext3_downside_coskew_all"] = np.nan
    df["ext3_downside_coskew_diff"] = np.nan

    if not _HAS_INDEXES or n < _MIN_OBS:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY daily close and compute market returns
    # ------------------------------------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build a small SPY-return frame aligned to df's dates
    spy_df = (
        spy_close.rename("spy_close")
        .reset_index()
        .rename(columns={"index": "Date", "Date": "Date"})
    )
    # Ensure Date column name is correct regardless of reset_index naming
    if "Date" not in spy_df.columns:
        spy_df.columns = ["Date", "spy_close"]

    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    spy_df["spy_ret"] = spy_df["spy_close"].pct_change()

    # Align with stock dates via merge_asof (backward = no lookahead)
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df[["Date", "spy_ret"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Stock daily returns
    work["stk_ret"] = work["Close"].pct_change()

    # Drop any row missing either return
    work_arr = work[["stk_ret", "spy_ret"]].values  # (n, 2)

    # ------------------------------------------------------------------
    # Rolling coskewness — iterate over each valid end-point
    # ------------------------------------------------------------------
    down_vals = np.full(n, np.nan)
    all_vals = np.full(n, np.nan)

    for end in range(_MIN_OBS - 1, n):
        start = max(0, end - _WINDOW + 1)
        block = work_arr[start : end + 1]

        # Drop rows where either series is NaN
        mask = np.isfinite(block[:, 0]) & np.isfinite(block[:, 1])
        r_s = block[mask, 0]
        r_m = block[mask, 1]

        if len(r_s) < _MIN_OBS:
            continue

        # All-days coskew
        all_vals[end] = _coskew(r_s, r_m)

        # Downside: market-down days only
        down_mask = r_m < 0
        if down_mask.sum() >= max(10, _MIN_OBS // 5):
            down_vals[end] = _coskew(r_s[down_mask], r_m[down_mask])

    # Re-align back to original df index order (work was sorted by Date)
    # work has the same length as df but may be reordered; we sorted by Date
    # so we need to map back using the original df index
    orig_order = work.index  # merge_asof preserves order of left (sorted)
    # Since df may not be perfectly sorted, we align by position of the sorted work
    # back to the original df positions using the original df index stored in work
    # Actually: we sorted work by Date for the merge; let's retrieve original positions
    # by re-merging the results back via Date.

    result_df = work[["Date"]].copy()
    result_df["_down"] = down_vals
    result_df["_all"] = all_vals

    # Merge back to original df (which keeps its original order)
    df2 = df.copy()
    df2["_orig_idx"] = np.arange(n)
    df2["Date"] = pd.to_datetime(df2["Date"])
    merged = pd.merge(
        df2[["Date", "_orig_idx"]],
        result_df,
        on="Date",
        how="left",
    ).sort_values("_orig_idx")

    df["ext3_downside_coskew_down"] = merged["_down"].values
    df["ext3_downside_coskew_all"] = merged["_all"].values
    diff = merged["_down"].values - merged["_all"].values
    diff = np.where(
        np.isfinite(merged["_down"].values) & np.isfinite(merged["_all"].values),
        diff,
        np.nan,
    )
    df["ext3_downside_coskew_diff"] = diff

    return df
