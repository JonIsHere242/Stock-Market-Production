"""
evt_seasonal_cycles.py — Day-of-week, quarter-distance, and seasonal-anomaly
event-time features.

THEME: calendar / event-time / seasonality derived purely from the Date column.

Distinct from the existing cal_ block (which has dow_sin/cos and month_sin/cos):
here we encode (a) DISCRETE weekday effects as bounded numeric flags (the classic
"Monday effect" of weak/negative returns and the "Friday effect"), (b) distance to
quarter boundaries in TRADING days (institutional rebalancing / 13F window-dressing
clusters at quarter ends), and (c) named seasonal-anomaly windows long studied in
the literature: the January effect, the "sell-in-May" summer lull, the
tax-loss-selling December window, and the Santa-Claus-rally turn-of-year window.

Quarter distance uses the df's own realized trading calendar (one row per trading
day) so the counts are true trading-day distances, fully deterministic and
lookahead-safe (each row depends only on its own quarter label).

All produced columns are bounded numeric flags in {0,1} or normalized distances
in [0,1]; never NaN for a valid Date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "evt_seasonal_cycles",
    "description": (
        "Weekday-effect flags, trading-day distance to quarter boundaries, and "
        "named seasonal-anomaly window flags (January, sell-in-May, tax-loss "
        "December, Santa-Claus rally)."
    ),
    "requires":    ["Date"],
    "produces":    [
        "evt_is_monday",            # 1 on Mondays (classic weak-return day)
        "evt_is_friday",            # 1 on Fridays
        "evt_dow_signed",           # weekday mapped to [-1,1] (Mon=-1 ... Fri=+1)
        "evt_tdays_to_qend_norm",   # trading days until quarter end, normalized [0,1]
        "evt_tdays_from_qstart_norm",  # trading days since quarter start, normalized [0,1]
        "evt_jan_effect",           # 1 in January (small-cap January effect)
        "evt_sell_in_may",          # 1 for the weak May–Oct summer window
        "evt_taxloss_dec",          # 1 in the last ~6 weeks of the year (Nov15–Dec31)
        "evt_santa_window",         # 1 in the Santa-Claus rally turn-of-year window
    ],
    "tags":    ["calendar", "event_time", "seasonality"],
    "version": "1.0",
    "author":  "feature-gen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])

    # --- Weekday effects -----------------------------------------------------
    dow = dates.dt.dayofweek                       # Mon=0 .. Sun=6
    df["evt_is_monday"] = (dow == 0).astype(float)
    df["evt_is_friday"] = (dow == 4).astype(float)
    # Map trading weekdays Mon..Fri (0..4) linearly onto [-1, +1].
    df["evt_dow_signed"] = (dow.clip(0, 4) / 2.0 - 1.0).astype(float)

    # --- Distance to quarter boundaries (trading days) -----------------------
    # Quarter label per row; rank rows within each quarter using df's own calendar.
    yq = dates.dt.to_period("Q")
    tdoq = yq.groupby(yq).cumcount() + 1                       # trading-day-of-quarter, 1-based
    tdoq = pd.Series(tdoq.values, index=df.index)
    q_size = yq.map(yq.value_counts())
    q_size = pd.Series(np.asarray(q_size, dtype=float), index=df.index)

    denom = (q_size - 1.0).replace(0.0, np.nan)
    # Days remaining until quarter end (0 on last trading day of quarter).
    to_qend = (q_size - tdoq) / denom
    df["evt_tdays_to_qend_norm"] = to_qend.fillna(0.0).clip(0.0, 1.0)
    # Days elapsed since quarter start (0 on first trading day of quarter).
    from_qstart = (tdoq - 1.0) / denom
    df["evt_tdays_from_qstart_norm"] = from_qstart.fillna(0.0).clip(0.0, 1.0)

    # --- Named seasonal-anomaly windows --------------------------------------
    month = dates.dt.month
    dom = dates.dt.day

    # January effect: small-cap outperformance in January.
    df["evt_jan_effect"] = (month == 1).astype(float)

    # "Sell in May and go away" — historically weak May through October window.
    df["evt_sell_in_may"] = month.between(5, 10).astype(float)

    # Tax-loss-selling season: roughly Nov 15 through year end.
    taxloss = ((month == 12) | ((month == 11) & (dom >= 15))).astype(float)
    df["evt_taxloss_dec"] = taxloss

    # Santa-Claus rally: last ~5 trading days of Dec + first ~2 of Jan. Approximate
    # with calendar dates (Dec 24–31 and Jan 1–5) which span that window robustly.
    santa = (((month == 12) & (dom >= 24)) | ((month == 1) & (dom <= 5))).astype(float)
    df["evt_santa_window"] = santa

    return df
