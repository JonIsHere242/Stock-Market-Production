"""Refresh the market-index lakes:  Data/Indexes  and  Data/IndexesFull.

WHY THIS FILE EXISTS
--------------------
Until now NOTHING in the production pipeline wrote these two directories. The only
writer in the repo was `update_all_indexes()` inside the RETIRED monolith
`experimental/3__AlphaSensitivity.py`, which was called from `process_data_files()`.
When stage 3 was replaced by `3__FeatureFramework.py` (which reads the lake through
`FeatureTemplates/_indexes.py` but never refreshes it) the index download silently
left the pipeline. `Data/Indexes` then froze at 2026-07-02 and every index-relative
panel column (~88 of them: alpha_*/beta_*/corr_* and the whole cross-asset /
beta_risk vein) went all-NaN for ~17 trading sessions with nobody noticing.

Root-cause fix = this fetcher (a real, runnable, nightly-able writer) plus the
fail-closed staleness gate in `lake_freshness.py`.

SOURCE
------
yfinance, matching what actually produced the bytes on disk: the existing
`Data/Indexes/*.parquet` carry a genuine dividend-back-adjusted `Adj Close`
(SPY: Adj Close != Close on 98.4% of rows) which the monolith's IBKR path could
never produce (it hard-set `Adj Close = Close`). `4__Predictor.spy_above_200ema_flag()`
already freshens SPY from yfinance, so this keeps the pipeline on one source.

SCHEMA (locked - `FeatureTemplates/_indexes.py` is the contract)
---------------------------------------------------------------
    index   : DatetimeIndex named 'Date', tz-naive, midnight-normalized
    columns : exactly ['Open','High','Low','Close','Adj Close','Volume']

APPEND-ONLY
-----------
Existing rows are NEVER rewritten: for any date already present the on-disk row
wins. This keeps backtests reproducible (a re-download would silently re-base
`Adj Close` and shift historical bars). Only genuinely new dates are appended.

SYMBOLS
-------
ORIGINAL (the 5 the pipeline has always had, load-bearing for existing columns):
    SPY QQQ IWM DIA VIX
EXTENDED (added 2026-07-28, PURELY ADDITIVE - no feature block reads them yet):
    IEF TLT SHY HYG LQD
    Rationale: `Data/FRED` already covers rates (DGS*, deep history) and the
    dollar (DTWEXBGS), so UUP/GLD were dropped as redundant. What FRED is WEAK on
    is credit - BAMLC0A0CM has only 794 rows from 2023-05 - so HYG/LQD add many
    more years of *tradeable* credit history, and IEF/TLT/SHY add tradeable
    duration proxies priced on the same calendar/adjustment basis as the equity panel.

    Adding files to Data/Indexes is safe: every production consumer either
    hard-codes the 5 symbols (`FeatureTemplates/orig_orig_market_beta.py`) or
    prefers SPY when present (`_paper_*` blocks). The one directory-enumerating
    consumer, `get_dynamic_signal_columns()` in `5__NightlyBackTester.py`, greps
    the *columns* of each index parquet for 'corr'/'alpha'/'beta' - OHLCV never
    matches - so it returns ([],[],[]) with 5 files and with 12. Verified, see
    `lake_freshness.py --verify-bt-columns`. `--originals-only` exists as the
    escape hatch if that ever stops being true.

USAGE
-----
    python fetchers/fetch_indexes.py                  # refresh both lakes, all symbols
    python fetchers/fetch_indexes.py --originals-only # only SPY/QQQ/IWM/DIA/VIX
    python fetchers/fetch_indexes.py --lake recent    # only Data/Indexes
    python fetchers/fetch_indexes.py --lake full      # only Data/IndexesFull
    python fetchers/fetch_indexes.py --dry-run        # report what would change
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import DATA_ROOT, log

RECENT_DIR = DATA_ROOT / "Indexes"
FULL_DIR = DATA_ROOT / "IndexesFull"

# Canonical column order/name set. Do not change - _indexes.py depends on it.
SCHEMA = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# lake symbol -> yahoo ticker
ORIGINAL_SYMBOLS: dict[str, str] = {
    "SPY": "SPY",     # S&P 500 ETF
    "QQQ": "QQQ",     # Nasdaq 100 ETF
    "IWM": "IWM",     # Russell 2000 ETF
    "DIA": "DIA",     # Dow Jones ETF
    "VIX": "^VIX",    # CBOE Volatility Index
}
EXTENDED_SYMBOLS: dict[str, str] = {
    "IEF": "IEF",     # 7-10y Treasury
    "TLT": "TLT",     # 20y+ Treasury
    "SHY": "SHY",     # 1-3y Treasury
    "HYG": "HYG",     # High-yield corporate credit
    "LQD": "LQD",     # Investment-grade corporate credit
}

# Seed window for a brand-new file in the SHALLOW lake: match what the existing
# 5 files span (2024-01-03 -> today) so the recent lake stays homogeneous.
RECENT_SEED_START = "2024-01-03"

ET = ZoneInfo("America/New_York")
# A daily bar is only final after the 16:00 ET close (+ a settle margin). Before
# that, drop today's row - the 2026-07-10 stale-book post-mortem and the
# trading_system.ps1 partial-bar guard both exist because in-progress bars
# contaminated downstream features.
CLOSE_SETTLED_ET = dt.time(16, 15)


def _last_final_session() -> pd.Timestamp:
    """Latest date whose daily bar can be considered final (ET-aware)."""
    now = dt.datetime.now(ET)
    d = now.date()
    if now.time() < CLOSE_SETTLED_ET:
        d = d - dt.timedelta(days=1)
    return pd.Timestamp(d)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame to the locked lake schema."""
    df = df.copy()
    # yfinance returns MultiIndex columns ('Price','Ticker') for list-style calls
    if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c) for c in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        # auto_adjust=True style payload: Close IS adjusted
        df["Adj Close"] = df["Close"]
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance payload missing columns {missing}; got {list(df.columns)}")
    df = df[SCHEMA]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0).astype("int64")
    return df


def _read_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()


def _download(ticker: str, start: str | None) -> pd.DataFrame:
    import yfinance as yf
    kw = dict(progress=False, auto_adjust=False, actions=False, threads=False)
    if start is None:
        raw = yf.download(ticker, period="max", **kw)
    else:
        raw = yf.download(ticker, start=start, **kw)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    return _normalize(raw)


def refresh_one(symbol: str, ticker: str, out_dir: Path, deep: bool,
                dry_run: bool = False) -> dict:
    """Append-only refresh of one symbol in one lake. Returns a report dict."""
    path = out_dir / f"{symbol}.parquet"
    existing = _read_existing(path)
    cutoff = _last_final_session()

    if existing is None or existing.empty:
        start = None if deep else RECENT_SEED_START
    else:
        # Re-request a small overlap so a mid-week gap heals; existing rows still win.
        start = str((existing.index.max() - pd.Timedelta(days=7)).date())
        if deep and existing.index.min() > pd.Timestamp("1990-01-02"):
            # Deep lake already holds max-available history for these tickers; only
            # the tail is ever missing. Keep the request cheap.
            pass

    fresh = _download(ticker, start)
    fresh = fresh[fresh.index <= cutoff]

    if existing is None or existing.empty:
        merged, added = fresh, len(fresh)
        col_name = None
    else:
        col_name = existing.columns.name
        new_rows = fresh[~fresh.index.isin(existing.index)]
        added = len(new_rows)
        if added == 0:
            merged = existing
        else:
            # Preserve the existing files' per-column dtypes (IndexesFull/VIX stores
            # Volume as int32); a gratuitous dtype flip is a pointless diff.
            new_rows = new_rows.astype(
                {c: existing[c].dtype for c in SCHEMA if c in existing.columns})
            merged = pd.concat([existing, new_rows]).sort_index()

    merged = merged[SCHEMA]
    merged.index.name = "Date"
    if col_name is not None:
        merged.columns.name = col_name

    rep = {
        "symbol": symbol, "lake": out_dir.name, "added": added,
        "rows": len(merged),
        "min": merged.index.min(), "max": merged.index.max(),
        "prev_max": None if existing is None or existing.empty else existing.index.max(),
    }
    if added and not dry_run:
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        tmp.replace(path)
    return rep


def fetch(lakes: tuple[str, ...] = ("recent", "full"), originals_only: bool = False,
          dry_run: bool = False) -> list[dict]:
    symbols = dict(ORIGINAL_SYMBOLS)
    if not originals_only:
        symbols.update(EXTENDED_SYMBOLS)

    targets = []
    if "recent" in lakes:
        targets.append((RECENT_DIR, False))
    if "full" in lakes:
        targets.append((FULL_DIR, True))

    reports: list[dict] = []
    for out_dir, deep in targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== {out_dir.name}  ({'max history' if deep else 'recent window'})"
            f"  cutoff={_last_final_session().date()}"
            f"{'  [DRY RUN]' if dry_run else ''}")
        for sym, tick in symbols.items():
            try:
                rep = refresh_one(sym, tick, out_dir, deep, dry_run=dry_run)
                reports.append(rep)
                log(f"  {sym:5s} +{rep['added']:<4d} rows -> {rep['rows']:>6,} total  "
                    f"{rep['min'].date()} .. {rep['max'].date()}"
                    + (f"  (was .. {rep['prev_max'].date()})" if rep["prev_max"] is not None else "  (new file)"))
            except Exception as e:
                log(f"  {sym:5s} FAIL: {type(e).__name__}: {e}")
                reports.append({"symbol": sym, "lake": out_dir.name, "error": str(e)})
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake", choices=["recent", "full", "both"], default="both")
    ap.add_argument("--originals-only", action="store_true",
                    help="restrict to SPY/QQQ/IWM/DIA/VIX (escape hatch if a consumer "
                         "ever starts enumerating the lake directory)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args()
    lakes = ("recent", "full") if a.lake == "both" else (a.lake,)
    fetch(lakes=lakes, originals_only=a.originals_only, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
