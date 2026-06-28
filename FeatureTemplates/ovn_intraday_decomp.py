"""
ovn_intraday_decomp.py — Overnight vs intraday return decomposition (Tier-2).

Lou, Polk & Skouras (2019, JFE) "A tug of war: Overnight versus intraday
expected returns". The daily close-to-close return splits cleanly into:

    overnight return  = Open_t / Close_{t-1} - 1     (the gap)
    intraday return   = Close_t / Open_t   - 1        (the session)

Their central finding: these two components are persistently driven by DIFFERENT
clienteles and the OVERNIGHT component carries strong, persistent cross-sectional
return predictability (firms with high past overnight returns keep earning them),
while intraday returns tend to reverse. So a name's rolling overnight tilt is a
genuine Tier-2 sorter.

LEAKAGE NOTE: today's Open is known at prediction time (we trade after the open).
The overnight return Open_t/Close_{t-1}-1 uses ONLY today's Open and the PRIOR
close — both known. The intraday return Close_t/Open_t-1 uses today's Close,
which is NOT known same-day, so every intraday/spread feature here is built from
rolling windows that END at the PRIOR completed day (shift(1)) before averaging —
no same-day Close enters a feature value used to predict that same day.

Produces (all rolling, trailing):
  - ovn_ret_mean_21/63   : rolling mean overnight (gap) return.
  - ovn_intraday_mean_21/63 : rolling mean intraday return (prior-day-ended).
  - ovn_minus_intraday_21/63 : overnight-minus-intraday spread (the LPS tug-of-
        war signal; persistently positive names are the overnight-premium type).
  - ovn_ret_last        : the most recent realised overnight gap (today's Open
        vs yesterday's Close) — directly tradeable, fully known pre-decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [21, 63]
_MIN = {21: 12, 63: 35}

METADATA = {
    "name":        "ovn_intraday_decomp",
    "description": "Overnight (gap) vs intraday return decomposition per Lou-Polk-Skouras 2019: rolling means of each component, their spread (21d/63d), and the latest overnight gap.",
    "requires":    ["Open", "Close"],
    "produces":    (
        [f"ovn_ret_mean_{w}" for w in _WINDOWS]
        + [f"ovn_intraday_mean_{w}" for w in _WINDOWS]
        + [f"ovn_minus_intraday_{w}" for w in _WINDOWS]
        + ["ovn_ret_last"]
    ),
    "tags":        ["overnight", "return_decomposition", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Lou-Polk-Skouras 2019)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)

    # overnight gap: today's Open vs prior Close (both known pre-decision)
    overnight = (open_ / prev_close.replace(0.0, np.nan)) - 1.0
    overnight = overnight.replace([np.inf, -np.inf], np.nan).clip(-1.0, 2.0)

    # intraday: today's Close vs today's Open (uses same-day Close -> NOT known
    # at decision time; we lag by one full day before it can enter any feature)
    intraday = (close / open_.replace(0.0, np.nan)) - 1.0
    intraday = intraday.replace([np.inf, -np.inf], np.nan).clip(-1.0, 2.0)
    intraday_lag = intraday.shift(1)  # only completed-day intraday returns

    for w in _WINDOWS:
        mp = _MIN[w]
        # overnight rolling mean: overnight gap is known same-day, safe to use
        ovn_mean = overnight.rolling(w, min_periods=mp).mean()
        df[f"ovn_ret_mean_{w}"] = ovn_mean.values

        # intraday rolling mean: built only from prior completed days
        intr_mean = intraday_lag.rolling(w, min_periods=mp).mean()
        df[f"ovn_intraday_mean_{w}"] = intr_mean.values

        # tug-of-war spread (use the lagged-intraday mean to stay leak-free)
        df[f"ovn_minus_intraday_{w}"] = (ovn_mean - intr_mean).values

    # most recent realised overnight gap (directly tradeable)
    df["ovn_ret_last"] = overnight.values

    return df
