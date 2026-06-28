"""
seasonality_calendar.py - Per-stock calendar return seasonality (Tier-2).

Heston & Sadka (2008, JFE) "Seasonality in the Cross-Section of Stock Returns": a
stock's historical return in a given calendar period (month-of-year, and more weakly
weekday) predicts its FUTURE return in that same period, and the effect is independent
of size, momentum, and -- crucially -- volatility. This is the most volatility-orthogonal
signal there is: it is a fixed-effect of the calendar slot on the *specific stock*, not
a function of dispersion at all.

We compute, using ONLY each stock's own PAST history (expanding, current row excluded):
  csa_seas_month     = mean own daily return in the current calendar month, de-meaned by
                       the stock's overall mean (isolates the seasonal component vs drift)
  csa_seas_month_ir  = that month's information ratio (mean / std) -- reliability of the
                       monthly seasonal
  csa_seas_dow       = mean own daily return on the current weekday, de-meaned

These are per-stock (each name has its own seasonal fingerprint), so they vary across the
cross-section even though the calendar slot is shared -- NOT a market-wide inert dummy.
Fully vectorized via grouped cumsum/cumcount (no leakage: current observation is removed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MINOBS = 20   # need this many prior same-slot observations before trusting the seasonal

METADATA = {
    "name":        "seasonality_calendar",
    "description": "Per-stock calendar seasonality from own history (expanding, leak-free): de-meaned mean daily return for the current calendar month and weekday, plus the month-effect information ratio, per Heston-Sadka 2008.",
    "requires":    ["Date", "Close"],
    "produces":    ["csa_seas_month", "csa_seas_month_ir", "csa_seas_dow"],
    "tags":        ["seasonality", "calendar", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Heston-Sadka 2008)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])
    ret = df["Close"].pct_change()

    tmp = pd.DataFrame({
        "ret":   ret.values,
        "ret2":  (ret * ret).values,
        "month": dates.dt.month.values,
        "dow":   dates.dt.dayofweek.values,
    })

    # Overall expanding mean EXCLUDING the current row (the stock's unconditional drift).
    prior_n = pd.Series(np.arange(len(tmp), dtype=float))          # rows strictly before current
    overall = (tmp["ret"].cumsum() - tmp["ret"]) / prior_n.replace(0, np.nan)

    def _grouped_excl(col, key):
        g = tmp.groupby(key)[col]
        return g.cumsum() - tmp[col], tmp.groupby(key).cumcount()   # (prior sum, prior count)

    # --- Month-of-year seasonal -------------------------------------------------
    sum_m, cnt_m = _grouped_excl("ret", "month")
    sum_m2, _ = _grouped_excl("ret2", "month")
    cnt_m_f = cnt_m.astype(float).replace(0, np.nan)
    mean_m = sum_m / cnt_m_f
    var_m = (sum_m2 / cnt_m_f) - mean_m ** 2
    std_m = np.sqrt(var_m.clip(lower=0))
    seas_month = (mean_m - overall).where(cnt_m >= _MINOBS)
    seas_month_ir = (mean_m / std_m.replace(0, np.nan)).where(cnt_m >= _MINOBS)

    # --- Weekday seasonal -------------------------------------------------------
    sum_d, cnt_d = _grouped_excl("ret", "dow")
    mean_d = sum_d / cnt_d.astype(float).replace(0, np.nan)
    seas_dow = (mean_d - overall).where(cnt_d >= _MINOBS)

    df["csa_seas_month"] = seas_month.clip(-0.1, 0.1).values
    df["csa_seas_month_ir"] = seas_month_ir.clip(-1, 1).values
    df["csa_seas_dow"] = seas_dow.clip(-0.1, 0.1).values

    return df
