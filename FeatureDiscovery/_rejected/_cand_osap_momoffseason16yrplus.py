"""
Off-Season Return Seasonality (Years 16-20) — Heston & Sadka (2008)
Cross-sectional signal implemented as a per-ticker proxy.

Per the original paper, the predictor is the average stock return
in "off-season" months (calendar months != current month) measured
over the 16-20 year look-back window. The paper documents a reversal
(predicted sign = -1): stocks with high past off-season returns tend
to underperform going forward.

Per-ticker proxy: for each row's calendar month m, we look back
[16 yr, 20 yr] = [~192, ~240] months of daily Close data, compute
monthly returns within that window, then average only those months
whose calendar month != m. This faithfully captures the same seasonal
regularity at the individual ticker level.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_momoffseason16yrplus",
    "description": (
        "Per-ticker proxy for Heston & Sadka (2008) off-season return seasonality, "
        "years 16-20 look-back. For each date with calendar month m, computes the "
        "average monthly return over [16yr, 20yr] look-back using only months where "
        "calendar_month != m (off-season months). Predicted sign = -1 (reversal). "
        "Inherently cross-sectional in the original paper; this is a faithful "
        "per-ticker proxy capturing the same economic signal."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momoffseason16yrplus_main",   # average off-season monthly return, 16-20yr
        "osap_momoffseason16yrplus_n",      # number of off-season months used (quality flag)
    ],
    "tags": ["seasonality", "momentum", "reversal", "long_horizon", "heston_sadka"],
    "version": "1.0",
    "author": "Heston & Sadka (2008); Chen-Zimmermann OpenSourceAP; per-ticker proxy",
}

# Look-back bounds in approximate trading days
_YR_DAYS = 252
_LB_NEAR = 16 * _YR_DAYS   # ~4032 days
_LB_FAR  = 20 * _YR_DAYS   # ~5040 days


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute off-season return seasonality (years 16-20) per ticker."""
    n = len(df)
    out_main = np.full(n, np.nan, dtype=np.float64)
    out_n    = np.full(n, np.nan, dtype=np.float64)

    if n < _LB_NEAR + 1:
        df["osap_momoffseason16yrplus_main"] = out_main
        df["osap_momoffseason16yrplus_n"] = out_n
        return df

    # Work with a numpy Close array and date array for speed
    close = df["Close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["Date"]).to_numpy()  # numpy datetime64

    # Build a monthly return series from the full Close array
    # We resample to month-end to get monthly close prices, then compute returns
    # But we need per-row lookups referencing the daily index, so we work differently:
    # For each daily row i, we need off-season avg from rows [i - LB_FAR .. i - LB_NEAR].
    # To avoid O(n^2), we precompute a month-indexed return array and for each row do
    # a vectorised slice on the month array.

    # Build a DataFrame of month-end closes for interpolation
    dates_pd = pd.to_datetime(df["Date"])
    temp = pd.DataFrame({"Close": close}, index=dates_pd)

    # Month-end close: last close in each calendar month
    monthly = temp["Close"].resample("ME").last().dropna()
    if len(monthly) < 2:
        df["osap_momoffseason16yrplus_main"] = out_main
        df["osap_momoffseason16yrplus_n"] = out_n
        return df

    # Monthly returns (simple)
    m_ret = monthly.pct_change()          # Series indexed by month-end date
    m_dates = monthly.index               # DatetimeIndex of month-end dates
    m_months = m_dates.month.to_numpy(dtype=np.int8)   # calendar month 1-12
    m_ret_arr = m_ret.to_numpy(dtype=np.float64)        # aligned returns

    # Map each daily date to the last completed month-end index
    # For row i on date d, find the latest month-end <= d
    m_dates_np = m_dates.to_numpy()  # numpy datetime64

    # Process every row; skip rows where we don't have enough history
    dates_np = dates_pd.to_numpy()

    for i in range(n):
        d = dates_np[i]

        # Find position in monthly array: last month-end <= d
        pos = np.searchsorted(m_dates_np, d, side="right") - 1
        if pos < 1:
            continue

        # Window edges in calendar time
        # near boundary: d - 16 years
        # far  boundary: d - 20 years
        d_near = d - np.timedelta64(_LB_NEAR, "D")
        d_far  = d - np.timedelta64(_LB_FAR,  "D")

        # Find month indices that fall in [d_far, d_near]
        idx_far  = np.searchsorted(m_dates_np, d_far,  side="left")
        idx_near = np.searchsorted(m_dates_np, d_near, side="right") - 1

        if idx_near < idx_far or idx_near < 1:
            continue

        # Slice returns and months within that window (exclude first which is NaN from pct_change)
        window_ret    = m_ret_arr[idx_far:idx_near + 1]
        window_months = m_months[idx_far:idx_near + 1]

        # Current calendar month
        cur_month = dates_pd.iloc[i].month

        # Off-season mask: months != current month AND not NaN
        mask = (window_months != cur_month) & np.isfinite(window_ret)
        count = mask.sum()
        if count < 1:
            continue

        out_main[i] = window_ret[mask].mean()
        out_n[i] = float(count)

    df["osap_momoffseason16yrplus_main"] = out_main
    df["osap_momoffseason16yrplus_n"] = out_n
    return df
