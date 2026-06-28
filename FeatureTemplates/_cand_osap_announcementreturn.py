"""
Per-ticker proxy for Earnings Announcement Return (Chan, Jegadeesh & Lakonishok 1996).

TRUE METHOD: Requires IBES earnings calendar. Sums excess-return (ret - mktrf + rf)
from day -1 to day +2 around each quarterly earnings announcement.

PROXY STRATEGY (no earnings calendar available):
  We detect likely earnings announcement windows via two signals that spike
  reliably at earnings: (a) abnormal volume (>= 2x 60-day median), and
  (b) large overnight gap (|open - prev_close| / prev_close >= 1%). Both
  conditions must fire on the same day to qualify as a candidate announcement.
  We enforce a minimum 55-day gap between detections to avoid double-counting
  the same quarter.

  Once a candidate announcement date is found, we compute the 3-day raw return
  (entry at prior close, exit at close on day +2 relative to the spike day)
  as the announcement return proxy. We then carry forward the most-recent
  announcement return as the feature value, and also compute a 4-quarter
  rolling mean (using the last 4 detections) to capture trend.

  A market-adjusted variant subtracts the concurrent SPY return over the same
  3-day window to approximate the excess-return in the original paper.

NOTE: Per-ticker proxy — no cross-sectional ranking, no IBES data.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Load _indexes helper (SPY for market-adjustment)                            #
# --------------------------------------------------------------------------- #
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
try:
    _spec_idx.loader.exec_module(_indexes)
    _INDEXES_OK = True
except Exception:
    _INDEXES_OK = False

# --------------------------------------------------------------------------- #
#  Metadata                                                                     #
# --------------------------------------------------------------------------- #
METADATA = {
    "name": "osap_announcementreturn",
    "description": (
        "Per-ticker proxy for earnings announcement return (Chan, Jegadeesh & "
        "Lakonishok 1996 / Chen-Zimmermann OpenSourceAP). True method uses IBES "
        "calendar; this proxy detects likely announcement days via abnormal volume "
        "(>=2x 60-day median) combined with large overnight gap (>=1%), then "
        "computes the 3-day raw and market-adjusted window return (day -1 to +2). "
        "Produces the most-recent announcement return and its 4-event rolling mean."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "osap_announcementreturn_last",   # most-recent 3-day announcement window return
        "osap_announcementreturn_mktadj", # market-adjusted version (subtract SPY 3-day)
        "osap_announcementreturn_mean4",  # rolling mean of last 4 detected announcement returns
    ],
    "tags": ["earnings", "event", "momentum", "price", "osap"],
    "version": "1.0",
    "author": "Chan, Jegadeesh & Lakonishok 1996 (OpenSourceAP / Chen-Zimmermann); proxy impl.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Initialise output columns with NaN
    df["osap_announcementreturn_last"] = np.nan
    df["osap_announcementreturn_mktadj"] = np.nan
    df["osap_announcementreturn_mean4"] = np.nan

    if n < 10:
        return df

    close = df["Close"].to_numpy(dtype=np.float64)
    open_ = df["Open"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  1. Detect candidate announcement days                               #
    # ------------------------------------------------------------------ #
    # Overnight gap: |Open_t - Close_{t-1}| / Close_{t-1}
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = np.where(
            (prev_close > 0) & np.isfinite(prev_close),
            np.abs(open_ - prev_close) / prev_close,
            np.nan,
        )

    # Abnormal volume: ratio vs 60-day rolling median (excluding current bar)
    WINDOW = 60
    vol_median = (
        pd.Series(volume)
        .shift(1)
        .rolling(WINDOW, min_periods=20)
        .median()
        .to_numpy(dtype=np.float64)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol_ratio = np.where(
            (vol_median > 0) & np.isfinite(vol_median),
            volume / vol_median,
            np.nan,
        )

    # Candidate = gap >= 1% AND volume ratio >= 2x
    GAP_THRESH = 0.01
    VOL_THRESH = 2.0
    candidate = (
        np.isfinite(gap) & (gap >= GAP_THRESH) &
        np.isfinite(vol_ratio) & (vol_ratio >= VOL_THRESH)
    )

    # Enforce minimum 55-day spacing between detections (one quarter ≈ 63 days)
    MIN_GAP = 55
    detection_days: list[int] = []
    last_det = -MIN_GAP - 1
    for i in range(n):
        if candidate[i] and (i - last_det) >= MIN_GAP:
            detection_days.append(i)
            last_det = i

    if not detection_days:
        return df

    # ------------------------------------------------------------------ #
    #  2. Compute 3-day window return for each detection                  #
    #     window: from prior day close (day -1) to close at day +2       #
    # ------------------------------------------------------------------ #
    # For safety (no lookahead into the future beyond what exists):
    # We record the RETURN but only write it at day +2 (when we have the exit).
    # Until day +2 is available the feature is NaN — this avoids any lookahead.

    # SPY series for market adjustment
    spy_close: pd.Series | None = None
    if _INDEXES_OK:
        try:
            spy_close = _indexes.index_close("SPY")
        except Exception:
            spy_close = None

    spy_arr: np.ndarray | None = None
    if spy_close is not None and len(spy_close) > 0:
        # Align SPY to df by Date via merge_asof
        df_dates = pd.DataFrame({"Date": df["Date"]})
        spy_df = spy_close.rename("spy_c").reset_index()
        spy_df.columns = ["Date", "spy_c"]
        merged = pd.merge_asof(
            df_dates.sort_values("Date").assign(_orig_idx=df_dates.index),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        ).set_index("_orig_idx").reindex(df_dates.index)
        spy_arr = merged["spy_c"].to_numpy(dtype=np.float64)

    ann_returns: list[tuple[int, float, float]] = []  # (write_idx, raw_ret, mkt_adj)

    for det in detection_days:
        # Entry: close at det-1 (the day before the announcement)
        entry_idx = det - 1
        # Exit: close at det+2
        exit_idx = det + 2

        if entry_idx < 0 or exit_idx >= n:
            continue

        entry_price = close[entry_idx]
        exit_price = close[exit_idx]

        if not (np.isfinite(entry_price) and entry_price > 0 and
                np.isfinite(exit_price) and exit_price > 0):
            continue

        raw_ret = (exit_price - entry_price) / entry_price

        # Market adjustment: SPY return over the same window
        mkt_ret = np.nan
        if spy_arr is not None:
            spy_entry = spy_arr[entry_idx]
            spy_exit = spy_arr[exit_idx]
            if (np.isfinite(spy_entry) and spy_entry > 0 and
                    np.isfinite(spy_exit) and spy_exit > 0):
                mkt_ret = (spy_exit - spy_entry) / spy_entry

        mkt_adj = raw_ret - mkt_ret if np.isfinite(mkt_ret) else raw_ret

        # Record: write at exit_idx (no lookahead — exit price is then known)
        ann_returns.append((exit_idx, raw_ret, mkt_adj))

    if not ann_returns:
        return df

    # ------------------------------------------------------------------ #
    #  3. Forward-fill features from each exit day onward                #
    # ------------------------------------------------------------------ #
    last_col = df["osap_announcementreturn_last"].to_numpy(dtype=np.float64)
    mktadj_col = df["osap_announcementreturn_mktadj"].to_numpy(dtype=np.float64)
    mean4_col = df["osap_announcementreturn_mean4"].to_numpy(dtype=np.float64)

    # Sort by write index (should already be sorted)
    ann_returns.sort(key=lambda x: x[0])

    # Build arrays of (write_idx, raw, mktadj)
    write_idxs = np.array([r[0] for r in ann_returns], dtype=np.int64)
    raws = np.array([r[1] for r in ann_returns], dtype=np.float64)
    mkts = np.array([r[2] for r in ann_returns], dtype=np.float64)

    # For each row, find the most recent completed detection
    # and compute rolling mean of last 4
    for i in range(n):
        mask = write_idxs <= i
        if not np.any(mask):
            continue
        valid_raws = raws[mask]
        valid_mkts = mkts[mask]

        last_col[i] = valid_raws[-1]
        mktadj_col[i] = valid_mkts[-1]

        # Rolling mean of last 4 events
        last4 = valid_raws[-4:]
        mean4_col[i] = np.nanmean(last4)

    df["osap_announcementreturn_last"] = last_col
    df["osap_announcementreturn_mktadj"] = mktadj_col
    df["osap_announcementreturn_mean4"] = mean4_col

    return df
