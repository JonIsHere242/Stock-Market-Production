"""
Firm Age - Momentum (per-ticker proxy)
SPEC ID: osap_firmagemom
SOURCE: OpenSourceAP (Chen-Zimmermann); Zhang 2006

Economic signal: momentum (6-month past return) is stronger for younger firms.
Cross-sectional original: 6-month return restricted to the bottom quintile of
the cross-sectional firm age distribution; exclude if price < 5 or firm age < 12 months.

Per-ticker proxy: We approximate firm age using the elapsed calendar days since
the first observed trading date in the available price history for this ticker.
Because the pipeline sees one stock at a time, cross-sectional quintile ranking
is unavailable; instead we:
  1. Compute 6-month (126-bar) price return -- the momentum signal.
  2. Compute rolling relative age score: current age / max(age_so_far), a [0,1]
     proxy for the firm's position in its own life-cycle. Low values = "young"
     relative to its own history.
  3. Compute an interaction: 6-month return × (1 - relative_age), which up-weights
     the momentum signal when the firm is relatively young in its own history.
  4. Apply the price < 5 exclusion (set all outputs to NaN when Close < 5).
  5. Apply the 12-month age exclusion (set all outputs to NaN when the ticker has
     fewer than 252 trading bars of history).
These outputs are PER-TICKER only -- they cannot perfectly replicate the XS quintile
gate, so treat as a directional proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_firmagemom",
    "description": (
        "Firm Age - Momentum per-ticker proxy (Zhang 2006 / OpenSourceAP Chen-Zimmermann). "
        "6-month return as momentum signal, modulated by a per-ticker relative age proxy "
        "(days since first observed trade / rolling max of same). Interaction column "
        "up-weights momentum when the ticker is young relative to its own history. "
        "Price<5 and <252 bars of history are excluded (NaN). Original method requires "
        "cross-sectional firm-age quintile which is unavailable per-ticker; this is a "
        "faithful directional proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_firmagemom_ret6m",       # raw 6-month return (excludes price<5, age<12mo)
        "osap_firmagemom_relage",       # rolling relative age score in [0,1]
        "osap_firmagemom_interact",     # ret6m * (1 - relage): young-firm momentum proxy
    ],
    "tags": ["momentum", "firm_age", "cross_sectional_proxy", "price"],
    "version": "1.0",
    "author": "Zhang 2006; OpenSourceAP (Chen-Zimmermann); per-ticker proxy by Claude",
}

# Number of trading bars used as proxies
_BARS_6M = 126   # ~6 calendar months of trading days
_BARS_12M = 252  # ~12 calendar months -- minimum age gate


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ------------------------------------------------------------------ #
    # 1. 6-month momentum return: Close[t] / Close[t - 126] - 1
    # ------------------------------------------------------------------ #
    ret6m = np.full(n, np.nan)
    if n > _BARS_6M:
        denom = close[:-_BARS_6M]
        numer = close[_BARS_6M:]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom == 0.0, np.nan, numer / denom - 1.0)
        ret6m[_BARS_6M:] = r

    # ------------------------------------------------------------------ #
    # 2. Relative age proxy
    #    age_days[t] = calendar days from first bar to current bar
    #    rolling_max_age[t] = max(age_days[0..t])  (= age_days[t] since monotone)
    #    rel_age[t] = age_days[t] / rolling_max_age[-1]
    #    Since age_days is strictly non-decreasing and rolling_max == age_days,
    #    rel_age[t] = age_days[t] / age_days[-1], producing 0..1.
    # ------------------------------------------------------------------ #
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        first_date = dates.iloc[0]
        age_days = (dates - first_date).dt.days.to_numpy(dtype=np.float64)
    else:
        # Fallback: use bar index as age proxy
        age_days = np.arange(n, dtype=np.float64)

    max_age = age_days[-1] if n > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_age = np.where(max_age == 0.0, np.nan, age_days / max_age)

    # ------------------------------------------------------------------ #
    # 3. Interaction: ret6m × (1 - rel_age)
    #    Young (low rel_age) → weight near 1; old → weight near 0
    # ------------------------------------------------------------------ #
    with np.errstate(invalid="ignore"):
        interact = ret6m * (1.0 - rel_age)

    # ------------------------------------------------------------------ #
    # 4. Exclusion gates
    #    (a) price < 5  -> NaN for all outputs at that row
    #    (b) firm age < 12 months proxy (< 252 bars): mask first 252 bars
    # ------------------------------------------------------------------ #
    price_too_low = close < 5.0  # boolean mask

    # Age < 12 months: first _BARS_12M rows are excluded
    age_too_young = np.arange(n) < _BARS_12M

    mask = price_too_low | age_too_young  # True where we must set NaN

    ret6m[mask] = np.nan
    rel_age[mask] = np.nan
    interact[mask] = np.nan

    # ------------------------------------------------------------------ #
    # 5. Guard: replace inf
    # ------------------------------------------------------------------ #
    for arr in (ret6m, rel_age, interact):
        arr[~np.isfinite(arr)] = np.nan

    df["osap_firmagemom_ret6m"] = ret6m
    df["osap_firmagemom_relage"] = rel_age
    df["osap_firmagemom_interact"] = interact

    return df
