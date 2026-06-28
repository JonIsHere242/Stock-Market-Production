"""
SPEC ID : osap_volmkt
SOURCE  : OpenSourceAP (Chen-Zimmermann); Haugen and Baker 1996
SIGNAL  : Volume to market equity
          Average monthly dollar trading volume (vol * |price|) over the
          previous 12 months scaled by market value of equity.
          Predicted sign -1 (high ratio -> lower future returns).

PROXY NOTES
-----------
Market equity = shares_outstanding × Close (PIT via fundamentals).
When fundamentals are unavailable (~16% coverage gap) we fall back to a
pure-price proxy: we cannot know share count, so we use Close alone as a
size deflator (turning the ratio into turnover / price, which preserves the
cross-sectional ordering but is not the same magnitude).  The fallback column
is explicitly distinguished so downstream code can gate on it if desired.

Monthly dollar volume is approximated as:
  monthly_dolvol = sum of daily (Volume × Close) within the calendar month,
then averaged over up to 12 trailing complete months (minimum 6 months to
produce a value).

Additional produced column:
  osap_volmkt_slope  -- linear slope of monthly dollar-vol / mkt-eq over the
                        12 trailing months (momentum-in-the-ratio), capturing
                        whether crowding is accelerating.
  osap_volmkt_has_fund -- 1 if fundamentals were available (0 = proxy mode).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Optional: PIT fundamentals
# ---------------------------------------------------------------------------
try:
    _s2 = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _fundamentals = None
    _HAS_FUND = False

METADATA = {
    "name": "osap_volmkt",
    "description": (
        "Per-ticker proxy for Haugen & Baker (1996) 'volume to market equity' factor. "
        "Computes average monthly dollar trading volume (Volume*Close) over trailing "
        "12 months, scaled by market equity (PIT shares_outstanding*Close from SEC "
        "fundamentals, or Close-only proxy when fundamentals unavailable). "
        "Predicted sign -1 (overcrowded / lottery stocks underperform). "
        "Excludes days where Close < 5. Inherently cross-sectional in the original; "
        "implemented here as a pure per-ticker time-series ratio."
    ),
    "requires": ["Close", "Volume"],
    "produces": ["osap_volmkt", "osap_volmkt_slope", "osap_volmkt_has_fund"],
    "tags": ["volume", "liquidity", "size", "osap", "haugen_baker"],
    "version": "1.0",
    "author": "Haugen and Baker 1996 / OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 0. Initialise output columns with NaN
    # ------------------------------------------------------------------
    df = df.copy()
    df["osap_volmkt"] = np.nan
    df["osap_volmkt_slope"] = np.nan
    df["osap_volmkt_has_fund"] = np.nan

    if len(df) < 2:
        return df

    # ------------------------------------------------------------------
    # 1. Price filter: mask out rows where Close < 5 (per spec)
    #    We still compute on all rows but treat masked as NaN.
    # ------------------------------------------------------------------
    close = df["Close"].copy()
    volume = df["Volume"].copy()

    # Zero-guard
    close_safe = close.where(close > 0, np.nan)
    volume_safe = volume.where(volume >= 0, np.nan)

    # Price filter flag (< 5 -> dollar vol counts as NaN for that day)
    price_ok = close_safe >= 5.0

    # Daily dollar volume (NaN where price < 5 or close == 0)
    dolvol_daily = np.where(price_ok, volume_safe * close_safe, np.nan)
    dolvol_daily = pd.Series(dolvol_daily, index=df.index)

    # ------------------------------------------------------------------
    # 2. Attempt to load PIT market equity (shares_outstanding * close)
    # ------------------------------------------------------------------
    has_fund = False
    mktcap = None

    if _HAS_FUND and _fundamentals is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmp = _fundamentals.as_of(df, fields=["shares_outstanding"])
            so = tmp.get("fund_shares_outstanding", None)
            if so is not None:
                so_vals = pd.to_numeric(so, errors="coerce")
                # mktcap in $: shares * close (both forward-NaN if fund unavailable)
                mktcap_raw = so_vals * close_safe
                mktcap_raw = mktcap_raw.where(mktcap_raw > 0, np.nan)
                # Only consider valid where price >= 5 as well
                mktcap = mktcap_raw.where(price_ok, np.nan)
                has_fund = mktcap.notna().any()
        except Exception:
            mktcap = None
            has_fund = False

    # Fallback: use Close as size deflator (pure per-ticker proxy)
    if not has_fund or mktcap is None:
        mktcap = close_safe.where(price_ok, np.nan)
        has_fund_flag = 0.0
    else:
        has_fund_flag = 1.0

    mktcap = pd.Series(mktcap, index=df.index)

    # ------------------------------------------------------------------
    # 3. Aggregate to calendar-month buckets
    # ------------------------------------------------------------------
    dates = pd.to_datetime(df["Date"])
    month_key = dates.dt.to_period("M")

    tmp_df = pd.DataFrame(
        {
            "date": dates.values,
            "month": month_key.values,
            "dolvol": dolvol_daily.values,
            "mktcap": mktcap.values,
        },
        index=df.index,
    )

    # Monthly aggregates: sum of dolvol, last mktcap of month
    monthly_dolvol = tmp_df.groupby("month")["dolvol"].sum(min_count=1)
    monthly_mktcap = tmp_df.groupby("month")["mktcap"].last()

    # Ratio per month: avg monthly dolvol (itself) / end-of-month mktcap
    monthly_ratio = monthly_dolvol / monthly_mktcap.where(monthly_mktcap > 0, np.nan)

    # Convert to ordered series
    monthly_ratio = monthly_ratio.sort_index()
    ratio_values = monthly_ratio.values  # array aligned by sorted period index
    periods = monthly_ratio.index  # PeriodIndex

    # ------------------------------------------------------------------
    # 4. For each row in df compute trailing 12-month average ratio
    #    and slope over those months.
    # ------------------------------------------------------------------
    # Map each row to its period
    row_period = month_key.values  # array of Period objects

    # Build a dict: period -> integer position in monthly_ratio
    period_to_pos = {p: i for i, p in enumerate(periods)}

    n = len(df)
    result_level = np.full(n, np.nan)
    result_slope = np.full(n, np.nan)

    MIN_MONTHS = 6
    MAX_MONTHS = 12

    for i in range(n):
        p = row_period[i]
        pos = period_to_pos.get(p, None)
        if pos is None:
            continue
        # Use months BEFORE current (exclude current month -> avoid partial month lookahead)
        # pos-1 back to pos-12 (up to 12 complete prior months)
        end_idx = pos  # exclusive (don't include current month)
        start_idx = max(0, end_idx - MAX_MONTHS)
        window = ratio_values[start_idx:end_idx]
        valid = window[~np.isnan(window)]
        if len(valid) < MIN_MONTHS:
            continue
        result_level[i] = float(np.mean(valid))
        # Slope via linear regression on valid months only
        if len(valid) >= 2:
            x = np.arange(len(valid), dtype=float)
            xm = x - x.mean()
            ym = valid - valid.mean()
            denom = (xm * xm).sum()
            if denom > 0:
                result_slope[i] = float((xm * ym).sum() / denom)

    df["osap_volmkt"] = result_level
    df["osap_volmkt_slope"] = result_slope
    df["osap_volmkt_has_fund"] = has_fund_flag

    return df
