"""
_marketcap.py  --  Shared POINT-IN-TIME market-cap loader for feature blocks.

Auto-skipped by the framework (leading _ keeps it out of block discovery). It is the
market-cap analogue of _fundamentals.py / _indexes.py: a HELPER that size/liquidity blocks
import (by file path) to reach the per-ticker daily market-cap series that is NOT in the
per-ticker OHLCV frame.

WHY THIS EXISTS
---------------
The framework's compute(df) contract is per-ticker, OHLCV-only, and stateless. A stock's
market capitalization (price x shares-outstanding) lives outside that frame. Rather than let
blocks read the big shared parquet ad hoc, they call as_of_cap() here, which keeps the join
lookahead-safe and the file load cached once per process.

THE ONE RULE: POINT-IN-TIME ON `Date`
--------------------------------------
A market cap is PIT-knowable: the cap on day d (price_d x shares_d) is fully known at the
day-d close. as_of_cap() does a BACKWARD merge_asof on Date, so each trading row only ever
sees a cap stamped on a date <= that row's Date. This is lookahead-safe by construction.
(The cap panel ends earlier than the price panel; backward-asof simply carries the last
known cap forward -- stale but never a future peek.)

DATA SOURCE
-----------
`Data/MarketCaps/historical_market_caps.parquet` -- one wide table, columns
(Ticker, Date, MarketCap), ~2.68M rows, ~4065 tickers, daily. Loaded ONCE and split into a
per-ticker dict on first use; cached for the life of the process.

PUBLIC API
----------
    available()              -> list[str]                # tickers present in the panel
    load_caps(ticker)        -> pd.DataFrame             # ['Date','MarketCap'] PIT, fresh copy
    as_of_cap(df)            -> pd.DataFrame              # adds a 'mcap' column (NaN if no coverage)

as_of_cap returns the caller's df with one ADDED column 'mcap' (float). Existing columns are
never touched; row order is preserved. Rows before the ticker's first cap observation, or
tickers with no panel coverage, come back NaN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Data/MarketCaps lives at repo root; this file is repo_root/FeatureTemplates/_marketcap.py
CAP_PATH = Path(__file__).resolve().parent.parent / "Data" / "MarketCaps" / "historical_market_caps.parquet"

# Cache for THIS module instance.
_PANEL_BY_TICKER: dict[str, pd.DataFrame] | None = None

# Process-global cache key. Blocks import this helper via importlib exec_module (a FRESH
# module object each call, bypassing sys.modules) and the framework re-discovers blocks once
# per ticker -- so a module global alone re-runs the ~4s panel load on every block call
# (3 blocks x every ticker). We also stash the parsed panel on the singleton `sys` module so
# it loads exactly ONCE per process and is shared across every fresh _marketcap instance.
_CACHE_ATTR = "_MARKETCAP_PANEL_CACHE"


def _load_panel() -> dict[str, pd.DataFrame]:
    """Read the wide cap parquet once and split into a per-ticker {Date, MarketCap} dict."""
    global _PANEL_BY_TICKER
    if _PANEL_BY_TICKER is not None:
        return _PANEL_BY_TICKER

    # Shared across all fresh _marketcap instances in this process (see _CACHE_ATTR note).
    cached = getattr(sys, _CACHE_ATTR, None)
    if cached is not None:
        _PANEL_BY_TICKER = cached
        return cached

    if not CAP_PATH.exists():
        _PANEL_BY_TICKER = {}
        setattr(sys, _CACHE_ATTR, _PANEL_BY_TICKER)
        return _PANEL_BY_TICKER

    panel = pd.read_parquet(CAP_PATH, columns=["Ticker", "Date", "MarketCap"])
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel["MarketCap"] = pd.to_numeric(panel["MarketCap"], errors="coerce")
    # Drop unusable rows; non-positive caps are invalid (log undefined).
    panel = panel.dropna(subset=["Date", "MarketCap"])
    panel = panel[panel["MarketCap"] > 0]

    by_ticker: dict[str, pd.DataFrame] = {}
    for tic, grp in panel.groupby("Ticker", sort=False):
        g = grp[["Date", "MarketCap"]].sort_values("Date")
        # Collapse any accidental duplicate dates to the last observation of the day.
        g = g.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        by_ticker[str(tic).upper()] = g

    _PANEL_BY_TICKER = by_ticker
    setattr(sys, _CACHE_ATTR, by_ticker)
    return _PANEL_BY_TICKER


def available() -> list[str]:
    """Tickers that currently have a cap series in the panel."""
    return sorted(_load_panel().keys())


def load_caps(ticker: str) -> pd.DataFrame:
    """Fresh copy of one ticker's PIT cap series (empty frame if the ticker has no coverage)."""
    panel = _load_panel()
    g = panel.get(str(ticker).upper())
    return g.copy() if g is not None else pd.DataFrame(columns=["Date", "MarketCap"])


def as_of_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lookahead-safe join of daily market cap into a per-ticker OHLCV frame.

    For each row, attach the most recent MarketCap whose Date <= that row's Date
    (pd.merge_asof, direction="backward"). Ticker is read from df['Ticker']. Missing coverage
    or pre-first-observation rows -> NaN. Returns df with rows/order preserved plus one ADDED
    'mcap' column (existing columns untouched).
    """
    out = df  # add the column onto the caller's frame (block contract)
    ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns and len(df) else None

    caps = load_caps(ticker) if ticker else pd.DataFrame(columns=["Date", "MarketCap"])

    if caps.empty:
        out["mcap"] = np.nan
        return out

    left = out.copy()
    left["__row"] = range(len(left))
    left["__date"] = pd.to_datetime(left["Date"], errors="coerce")
    left = left.sort_values("__date")

    right = caps.rename(columns={"Date": "__cap_date"}).sort_values("__cap_date")
    merged = pd.merge_asof(
        left, right, left_on="__date", right_on="__cap_date", direction="backward"
    )
    merged = merged.sort_values("__row")

    out["mcap"] = merged["MarketCap"].to_numpy()
    return out
