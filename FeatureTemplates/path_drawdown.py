"""
path_drawdown.py — drawdown / recovery dynamics & streaks (Tier-2).

How a stock sits relative to its own running peak — and how long it has been
under water — sorts resilient continuation names from broken ones. Names making
fresh highs (zero drawdown, time-under-water = 0) with positive up-streaks behave
very differently next day from names grinding through a long underwater stretch.
This is the maximum-drawdown / Calmar / underwater-curve framework (Magdon-Ismail
& Atiya 2004 on drawdown statistics) applied per ticker, all trailing-only and
fully vectorised (cummax + ffill index trick — NO python row loop).

Produces:
  path_drawdown_252        current drawdown from trailing 252d rolling peak (<=0)
  path_drawdown_63         current drawdown from trailing 63d peak (<=0)
  path_time_underwater     bars since the last 252d-window new high (time under
                           water), log1p-compressed; 0 == at a fresh high
  path_recovery_ratio      how far price has clawed back from the trough since
                           the last peak: (Close-trough)/(peak-trough), in [0,1]
  path_up_streak           length of the current consecutive up-close streak
                           (negative for a down streak), tanh-compressed
  path_max_dd_63           maximum drawdown experienced over the last 63d (<=0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


METADATA = {
    "name":        "path_drawdown",
    "description": "Drawdown/recovery dynamics: current 252d & 63d drawdown from running peak, time-under-water since last new high, recovery-from-trough ratio, signed up/down close streak, and rolling 63d max drawdown.",
    "requires":    ["Close"],
    "produces":    [
        "path_drawdown_252",
        "path_drawdown_63",
        "path_time_underwater",
        "path_recovery_ratio",
        "path_up_streak",
        "path_max_dd_63",
    ],
    "tags":        ["trend", "mean_reversion", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (drawdown / underwater dynamics)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].clip(lower=1e-8)
    n = len(close)

    # --- current drawdown from trailing rolling peaks (<= 0) ---
    peak_252 = close.rolling(252, min_periods=1).max()
    peak_63 = close.rolling(63, min_periods=1).max()
    df["path_drawdown_252"] = (close / peak_252 - 1.0).clip(-1.0, 0.0).values
    df["path_drawdown_63"] = (close / peak_63 - 1.0).clip(-1.0, 0.0).values

    # --- time under water: bars since last 252d-window new high ---
    # A bar is a new high when its close == trailing 252d rolling max.
    at_high_252 = close >= peak_252
    pos = pd.Series(np.arange(n, dtype=float), index=close.index)
    # index of the most recent new-high bar, carried forward
    last_high_idx = pos.where(at_high_252).ffill()
    tuw = (pos - last_high_idx).fillna(0.0)  # 0 at/ before first high
    df["path_time_underwater"] = np.log1p(tuw.clip(lower=0.0)).values

    # --- recovery ratio: fraction reclaimed from trough since last peak ---
    # peak carried forward = current running peak level (expanding-trailing).
    # We use the 252d peak level and the running trough since that peak.
    peak_level = peak_252
    # trough since the last new high: rolling min from the last-high bar to now.
    # Vectorised via cummin reset at each new high (group by the carried index).
    grp = last_high_idx.bfill()
    trough_since_peak = close.groupby(grp).cummin()
    span = (peak_level - trough_since_peak)
    rec = (close - trough_since_peak) / span.replace(0.0, np.nan)
    # When at a fresh high span==0 -> NaN -> fully recovered = 1.0
    df["path_recovery_ratio"] = rec.fillna(1.0).clip(0.0, 1.0).values

    # --- signed consecutive up/down close streak (vectorised) ---
    chg = np.sign(close.diff().fillna(0.0))  # +1 up, -1 down, 0 flat
    # streak resets whenever sign changes; group by cumulative break id.
    brk = (chg != chg.shift(1)).cumsum()
    run_len = chg.groupby(brk).cumcount() + 1
    signed_streak = run_len * chg
    df["path_up_streak"] = np.tanh(signed_streak.values / 5.0)

    # --- max drawdown experienced over the last 63d (<= 0) ---
    # rolling max drawdown = min over window of (close / running-peak-in-window - 1)
    roll_peak_63 = close.rolling(63, min_periods=20).max()
    dd_path = close / roll_peak_63 - 1.0
    df["path_max_dd_63"] = dd_path.rolling(63, min_periods=20).min().clip(-1.0, 0.0).values

    return df
