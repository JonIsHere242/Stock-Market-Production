"""
recurrence_interval_hazard.py - Recurrence-interval analysis of up-tail events (Tier-2).

Adapted from recurrence interval analysis in econophysics/seismology, e.g. Jiang, Zhou et al.
(arXiv:1610.08230) "Short-term prediction of extreme returns via recurrence intervals". Treat
up-tail returns (return above a trailing self-normalized threshold) as a point process and read
its conditional structure:

  - threshold Q_t = trailing-250d 90th percentile of returns (self-normalized -> the exceedance
    COUNT is volatility-invariant: a calm and a wild name both have ~10% exceedance days),
  - elapsed_t   = trading days since the last up-exceedance,
  - mu_tau      = mean realized recurrence interval (Goh-Barabasi renewal stats over events),
  - overdue     = elapsed / mu_tau   (how overdue an up-event is relative to this name's cadence),
  - burst       = (sigma_tau - mu_tau)/(sigma_tau + mu_tau)  (clustering of the event process),
  - rate        = exceedance frequency regime (deviation from the 0.10 baseline).

NOTE ON SIGN: the up-tail hazard's monotonicity in elapsed time FLIPS with clustering (for a
clustered/q>1 process an "overdue" name is LESS likely to fire, the opposite of a regular
process), so these are emitted as raw, sign-agnostic drivers for the model to combine/sign;
do not assume "overdue -> imminent". Fully vectorized, leak-free (events use only past data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 250          # threshold / renewal-stats window
_Q = 0.90         # up-tail quantile

METADATA = {
    "name":        "recurrence_interval_hazard",
    "description": "Recurrence-interval analysis of up-tail return events (self-normalized 90th-pct threshold): overdue ratio vs realized mean interval, Goh-Barabasi burstiness of the event process, and exceedance-rate regime. Sign-agnostic drivers (hazard monotonicity flips with clustering).",
    "requires":    ["Close"],
    "produces":    ["xdm_ri_overdue", "xdm_ri_burst", "xdm_ri_rate"],
    "tags":        ["tail", "econophysics", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (recurrence interval analysis, Jiang-Zhou)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    r = df["Close"].pct_change()

    q = r.rolling(_W, min_periods=_W // 2 + 25).quantile(_Q)
    exceed = (r > q).fillna(False).values
    idx = np.arange(n, dtype=float)

    # last event strictly BEFORE today (shift(1) keeps it leak-free and yields the closed interval)
    ev_strict = pd.Series(np.where(exceed, idx, np.nan)).shift(1).ffill().values
    elapsed = idx - ev_strict                              # days since last up-event

    interval_at_event = pd.Series(np.where(exceed, elapsed, np.nan))  # closed interval, only on event days
    mu_tau = interval_at_event.rolling(_W, min_periods=8).mean()
    sd_tau = interval_at_event.rolling(_W, min_periods=8).std()

    elapsed_s = pd.Series(elapsed)
    overdue = elapsed_s / mu_tau.replace(0, np.nan)
    burst = (sd_tau - mu_tau) / (sd_tau + mu_tau).replace(0, np.nan)
    rate = pd.Series(exceed, dtype=float).rolling(_W, min_periods=_W // 2 + 25).mean()

    df["xdm_ri_overdue"] = np.clip(overdue.values, 0, 10)
    df["xdm_ri_burst"] = np.clip(burst.values, -1, 1)
    df["xdm_ri_rate"] = np.clip(rate.values, 0, 0.5)

    return df
