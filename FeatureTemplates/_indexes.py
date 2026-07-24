"""
_indexes.py  --  Shared market-index data loader for index-coupled feature blocks.

Auto-skipped by the framework (leading _ keeps it out of block discovery). It is a
HELPER that index-dependent blocks import, e.g. `vix_features.py`, `beta_metrics.py`.

WHY THIS EXISTS
---------------
The framework's compute(df) contract is per-ticker and stateless. A few legacy features
(the VIX suite and beta/corr/alpha vs the market indexes) need external index data that is
NOT in the per-ticker OHLCV frame. Rather than re-read parquet on every call, this module
loads each index once per process and caches it (ProcessPoolExecutor workers each load once).

DATA SOURCE
-----------
`Data/Indexes/{SYMBOL}.parquet`, one row per day, DatetimeIndex named "Date",
columns: Open, High, Low, Close, Adj Close, Volume.  Symbols: SPY, QQQ, IWM, DIA, VIX.

PUBLIC API (what feature blocks should use)
-------------------------------------------
    available()            -> list[str]                 # cached symbols actually on disk
    load_index(symbol)     -> pd.DataFrame              # DatetimeIndex 'Date', cached
    load_all()             -> dict[str, pd.DataFrame]   # {symbol: frame}
    index_close(symbol)    -> pd.Series                 # Close indexed by Date (DatetimeIndex)
    vix_daily_close()      -> pd.DataFrame[['Date','vix_close']]
        Daily-resampled (last + ffill) VIX close, a plain Date column, sorted ascending.
        Ready for: pd.merge_asof(df.sort_values('Date'), vix_daily_close(),
                                  on='Date', direction='backward')

All returns are fresh copies — callers may use/mutate them freely without poisoning the cache.
If a symbol's parquet is missing, load_index raises FileNotFoundError; index_close /
vix_daily_close return an empty/typed structure so a block can degrade gracefully.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Data/Indexes lives at repo root; this file is repo_root/FeatureTemplates/_indexes.py
# FF_INDEXES_DIR env override lets a deep-history rebuild point at Data/IndexesFull
# without touching the production default.
INDEXES_DIR = Path(os.environ.get(
    "FF_INDEXES_DIR",
    str(Path(__file__).resolve().parent.parent / "Data" / "Indexes"),
))

# Module-level caches (one load per process).
_FRAME_CACHE: dict[str, pd.DataFrame] = {}
_VIX_DAILY_CACHE: pd.DataFrame | None = None


def _path(symbol: str) -> Path:
    return INDEXES_DIR / f"{symbol.upper()}.parquet"


def available() -> list[str]:
    """Symbols whose parquet currently exists on disk (e.g. ['DIA','IWM','QQQ','SPY','VIX'])."""
    if not INDEXES_DIR.is_dir():
        return []
    return sorted(p.stem.upper() for p in INDEXES_DIR.glob("*.parquet"))


def _load_cached(symbol: str) -> pd.DataFrame:
    """Load and cache one index frame with a DatetimeIndex named 'Date'. Cached, not copied."""
    sym = symbol.upper()
    if sym in _FRAME_CACHE:
        return _FRAME_CACHE[sym]

    path = _path(sym)
    if not path.exists():
        raise FileNotFoundError(f"Index parquet not found: {path}")

    df = pd.read_parquet(path)

    # Normalize to a DatetimeIndex named 'Date' regardless of how it was stored.
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    _FRAME_CACHE[sym] = df
    return df


def load_index(symbol: str) -> pd.DataFrame:
    """Return a fresh copy of one index frame (DatetimeIndex named 'Date')."""
    return _load_cached(symbol).copy()


def load_all() -> dict[str, pd.DataFrame]:
    """Return {symbol: fresh-copy frame} for every symbol on disk."""
    return {sym: load_index(sym) for sym in available()}


def index_close(symbol: str) -> pd.Series:
    """Close price as a Series indexed by Date (DatetimeIndex). Empty Series if unavailable."""
    try:
        frame = _load_cached(symbol)
    except FileNotFoundError:
        return pd.Series(dtype="float64", name="Close")
    return frame["Close"].copy()


def vix_daily_close() -> pd.DataFrame:
    """
    Daily-resampled VIX close ready for a backward merge_asof on 'Date'.

    Columns: ['Date', 'vix_close'], sorted ascending by Date.
    Mirrors the monolith:  vix['Close'].resample('D').last().ffill().
    Returns an empty typed frame if VIX is unavailable.
    """
    global _VIX_DAILY_CACHE
    if _VIX_DAILY_CACHE is not None:
        return _VIX_DAILY_CACHE.copy()

    try:
        vix = _load_cached("VIX")
    except FileNotFoundError:
        return pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"),
                             "vix_close": pd.Series(dtype="float64")})

    daily = vix["Close"].resample("D").last().ffill()
    out = daily.reset_index()
    out.columns = ["Date", "vix_close"]
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)

    _VIX_DAILY_CACHE = out
    return out.copy()
