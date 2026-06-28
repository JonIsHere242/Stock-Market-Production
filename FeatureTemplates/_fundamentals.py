"""
_fundamentals.py  --  Shared POINT-IN-TIME SEC-fundamentals loader for feature blocks.

Auto-skipped by the framework (leading _ keeps it out of block discovery). It is the SEC
analogue of _indexes.py: a HELPER that fundamentals-dependent blocks import to reach data
that is NOT in the per-ticker OHLCV frame.

WHY THIS EXISTS
---------------
The framework's compute(df) contract is per-ticker, OHLCV-only, and stateless. SEC fundamentals
(revenue, margins, P/E inputs, ...) live outside that frame. Rather than let blocks read files
ad hoc (which would also break the validator's leakage gate), they call as_of() here. The wide
per-ticker frames are pre-built by build_fundamentals_panel.py.

THE ONE RULE: POINT-IN-TIME ON `filed_date`
-------------------------------------------
Every fundamentals row is stamped with `filed_date` -- the day the number became public, NOT the
period it describes. as_of() does a BACKWARD merge_asof on `filed_date`, so each trading day only
ever sees filings already public by that day. This is what makes a fundamentals feature pass the
causality test in FeatureDiscovery/validate_feature.py: truncating future price bars can never
change a past as-of value. NEVER merge on the period-end date -- that is future leakage.

DATA SOURCE
-----------
`Data/Fundamentals/by_ticker/{TICKER}.parquet`, one row per filing event, sorted ascending by
`filed_date`, columns = canonical fields (revenue, net_income, ...), TTM rollups (*_ttm), and
fundamentals-only ratios (net_margin, roe, ...). Built by build_fundamentals_panel.py.

PUBLIC API (what feature blocks should use)
-------------------------------------------
    available()                 -> list[str]            # tickers with a by_ticker parquet
    canonical_fields()          -> list[str]            # column names a block may request
    load_fundamentals(ticker)   -> pd.DataFrame         # wide PIT frame (fresh copy), cached
    as_of(df, fields=None, prefix="fund_") -> pd.DataFrame
        Backward merge_asof of the named fundamental fields into a per-ticker OHLCV df, keyed on
        df['Date'] <= filed_date. Ticker is inferred from df['Ticker']. Adds prefixed columns
        (default 'fund_<field>'); rows before the first filing, or tickers with no SEC coverage,
        come back as NaN. Returns df with the same rows/order, plus the new columns.

All returns are fresh copies -- callers may use/mutate them freely without poisoning the cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Data/Fundamentals lives at repo root; this file is repo_root/FeatureTemplates/_fundamentals.py
FUND_DIR = Path(__file__).resolve().parent.parent / "Data" / "Fundamentals" / "by_ticker"

# Module-level cache (one load per process; ProcessPool workers each load once).
_FRAME_CACHE: dict[str, pd.DataFrame] = {}
_FIELDS_CACHE: list[str] | None = None

# Columns that are keys/metadata, not requestable feature fields.
_NON_FIELD = {"filed_date", "ticker", "cik"}


def _path(ticker: str) -> Path:
    return FUND_DIR / f"{ticker}.parquet"


def available() -> list[str]:
    """Tickers that currently have a by_ticker parquet on disk."""
    if not FUND_DIR.is_dir():
        return []
    return sorted(p.stem for p in FUND_DIR.glob("*.parquet"))


def _load_cached(ticker: str) -> pd.DataFrame | None:
    """Load and cache one ticker's wide PIT frame. Cached, not copied. None if absent."""
    key = ticker.upper()
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]

    path = _path(ticker)
    if not path.exists():
        _FRAME_CACHE[key] = None  # negative-cache so repeated misses are cheap
        return None

    f = pd.read_parquet(path)
    f["filed_date"] = pd.to_datetime(f["filed_date"], errors="coerce")
    f = f.dropna(subset=["filed_date"]).sort_values("filed_date").reset_index(drop=True)
    _FRAME_CACHE[key] = f
    return f


def load_fundamentals(ticker: str) -> pd.DataFrame:
    """Fresh copy of one ticker's wide PIT frame (empty frame if the ticker has no SEC data)."""
    f = _load_cached(ticker)
    return f.copy() if f is not None else pd.DataFrame()


def canonical_fields() -> list[str]:
    """The fundamental column names a block may request (union across a few sample tickers)."""
    global _FIELDS_CACHE
    if _FIELDS_CACHE is not None:
        return list(_FIELDS_CACHE)
    fields: list[str] = []
    seen: set[str] = set()
    for t in available()[:50]:
        f = _load_cached(t)
        if f is None:
            continue
        for c in f.columns:
            if c not in _NON_FIELD and c not in seen:
                seen.add(c)
                fields.append(c)
    _FIELDS_CACHE = sorted(fields)
    return list(_FIELDS_CACHE)


def as_of(df: pd.DataFrame, fields: list[str] | None = None, prefix: str = "fund_") -> pd.DataFrame:
    """
    Lookahead-safe join of SEC fundamentals into a per-ticker OHLCV frame.

    For each row, attach the most recent fundamental values whose `filed_date` <= that row's Date
    (pd.merge_asof, direction="backward"). Ticker is read from df['Ticker']. Missing coverage or
    pre-first-filing rows -> NaN. The returned frame preserves df's rows and order and only ADDS
    the requested `{prefix}{field}` columns (existing columns are never touched).
    """
    out = df  # we add columns onto the caller's frame, matching the block contract
    ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns and len(df) else None

    fund = _load_cached(ticker) if ticker else None
    req = list(fields) if fields else (
        [c for c in fund.columns if c not in _NON_FIELD] if fund is not None else [])

    # No data (or no fields): add NaN columns so downstream code/column contracts stay stable.
    if fund is None or fund.empty or not req:
        for f in (fields or []):
            out[f"{prefix}{f}"] = np.nan
        return out

    present = [c for c in req if c in fund.columns]
    left = out.copy()
    left["__row"] = range(len(left))
    left["__date"] = pd.to_datetime(left["Date"], errors="coerce")
    left = left.sort_values("__date")

    right = fund[["filed_date"] + present].sort_values("filed_date")
    merged = pd.merge_asof(left, right, left_on="__date", right_on="filed_date",
                           direction="backward")
    merged = merged.sort_values("__row")

    for c in present:
        # Coerce to numpy float64: some panel fields (e.g. total_debt) are stored as object/
        # nullable dtypes carrying pd.NA, which np.asarray(dtype=float) downstream cannot cast.
        # float64 (np.nan) is the framework's numeric contract and keeps blocks gate-safe.
        out[f"{prefix}{c}"] = pd.to_numeric(merged[c], errors="coerce").to_numpy(dtype="float64")
    # requested-but-absent fields still get a column (NaN) so the contract is honoured
    for c in req:
        if c not in present:
            out[f"{prefix}{c}"] = np.nan
    return out
