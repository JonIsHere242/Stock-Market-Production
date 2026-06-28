"""
fqg_filing_freshness.py  --  Filing-freshness & POST-EARNINGS-ANNOUNCEMENT-DRIFT (PEAD) signals.

Theme: the *timing* of fundamental information, not its level. After a firm files (10-Q / 10-K),
prices drift in the direction of the surprise for weeks (PEAD; Bernard-Thomas 1989). These features
flag how stale the current public fundamentals are and whether we are inside a fresh-filing drift
window, plus the SIGN of the most recent fundamental surprise.

DETECTING A FILING EVENT (PIT-safe)
-----------------------------------
_fundamentals.as_of() forward-fills the latest filed value onto every trading day. A NEW filing has
landed on the first day the merged fundamental values CHANGE. We track changes across several
robust fields (revenue/net_income/assets/shares) so a coincidental tie in one field doesn't hide a
filing. The very first non-NaN day is also treated as a filing event. Everything is derived purely
from already-public (filed_date<=Date) data, so it is fully lookahead-safe.

PRODUCED FEATURES
-----------------
  fqg_days_since_filing      : trading days since the last detected filing (freshness; capped 252).
  fqg_filing_recency_decay   : exp(-days/21) -> ~1 right after a filing, decaying over a month.
  fqg_post_filing_window_20  : 1.0 within 20 trading days of a new filing (PEAD drift window), else 0.
  fqg_filings_seen           : cumulative count of filings observed so far (information accumulation).
  fqg_revenue_surprise_sign  : sign of latest revenue_ttm change at the most recent filing
                               (+1 beat-direction / -1 miss-direction / 0 flat), held until next filing.
  fqg_earnings_accel_sign    : sign of (NI growth now - NI growth one filing ago) -> earnings
                               acceleration vs deceleration, held until next filing.

Windowing: history is ~2.5y so several filings exist per ticker; counts/decays populate early.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location(
    "_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "fqg_filing_freshness",
    "description": (
        "Filing-freshness & PEAD signals from filed-date SEC fundamentals: days since last filing, "
        "recency decay, post-filing drift-window flag, cumulative filings seen, and the sign of the "
        "latest revenue surprise and earnings acceleration."
    ),
    "requires":    ["Ticker", "Date"],
    "produces":    [
        "fqg_days_since_filing",
        "fqg_filing_recency_decay",
        "fqg_post_filing_window_20",
        "fqg_filings_seen",
        "fqg_revenue_surprise_sign",
        "fqg_earnings_accel_sign",
    ],
    "tags":        ["fundamentals", "pead", "freshness", "experimental"],
    "version":     "1.0",
    "author":      "feature-gen",
}

# Fields used to (a) detect filing-change events and (b) form the surprise signs.
_FIELDS = ["revenue_ttm", "net_income_ttm", "assets", "shares_outstanding"]

_DECAY_TAU = 21.0   # ~1 trading month
_PEAD_WIN  = 20     # trading days post-filing considered "fresh drift"
_CAP_DAYS  = 252    # cap days-since so a long-stale tail doesn't dominate


def _num(df: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(
        df.get(f"fund_{field}", pd.Series(np.nan, index=df.index)), errors="coerce"
    )


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = _fundamentals.as_of(df, fields=_FIELDS)

    rev    = _num(df, "revenue_ttm")
    ni     = _num(df, "net_income_ttm")
    assets = _num(df, "assets")
    shares = _num(df, "shares_outstanding")

    have_data = rev.notna() | ni.notna() | assets.notna() | shares.notna()

    # ---- Detect filing-change events (row index where merged values first differ) -----------
    # A new filing => at least one tracked field changes vs the previous row. fillna so a flip
    # from NaN->value (first filing) registers as a change.
    changed = (
        (rev.fillna(np.inf).diff() != 0)
        | (ni.fillna(np.inf).diff() != 0)
        | (assets.fillna(np.inf).diff() != 0)
        | (shares.fillna(np.inf).diff() != 0)
    )
    # Only count as an event once we actually have data on that row.
    event = (changed & have_data).to_numpy()

    idx = np.arange(n)
    last_event_idx = np.where(event, idx, np.nan)
    last_event_idx = pd.Series(last_event_idx).ffill().to_numpy()  # carry forward most recent event row

    days_since = idx - last_event_idx                      # NaN before the first event
    days_since = np.where(np.isnan(last_event_idx), np.nan, days_since)
    days_since = pd.Series(days_since, index=df.index)

    df["fqg_days_since_filing"]     = days_since.clip(upper=_CAP_DAYS)
    df["fqg_filing_recency_decay"]  = np.exp(-days_since / _DECAY_TAU)
    df["fqg_post_filing_window_20"] = (days_since < _PEAD_WIN).astype(float).where(days_since.notna())

    # Cumulative filings observed so far (information accumulation proxy).
    df["fqg_filings_seen"] = pd.Series(np.cumsum(event), index=df.index).astype(float).where(have_data.to_numpy())

    # ---- Surprise SIGNS, computed at each filing and held until the next filing -------------
    # The change at a filing day is the day-to-day diff there (the only day the value moves); on
    # non-event days we forward-fill the last event's sign so the signal is "held" until refreshed.
    event_s = pd.Series(event, index=df.index)
    rev_sign = np.sign(rev.diff()).where(event_s)
    df["fqg_revenue_surprise_sign"] = rev_sign.ffill()

    # Earnings acceleration: second difference of net income ACROSS consecutive filings. We build a
    # compact event-only series (one value per filing), take its 1st/2nd diff there (so the lag is
    # "previous filing", not "previous day"), then reindex the per-event 2nd-diff back onto the
    # filing rows and forward-fill across the held period.
    ni_event = ni.where(event_s).dropna()              # one value per filing event (by row index)
    ni_accel_compact = ni_event.diff().diff()          # change-in-change across filings
    ni_accel = ni_accel_compact.reindex(df.index)      # back onto full index (NaN off event rows)
    df["fqg_earnings_accel_sign"] = np.sign(ni_accel).ffill()

    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
