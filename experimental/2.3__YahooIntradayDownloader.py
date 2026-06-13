#!/usr/bin/env python3
"""
Yahoo Intraday Downloader  (2.3__YahooIntradayDownloader.py)
===========================================================
Free, fast alternative to 2__IntradayHistoricalDownloader.py (IBKR). IBKR's data
farm caps this account at ~1 ticker/min and concurrency makes it WORSE, so the full
universe is a ~64-hour grind there. Yahoo serves *consolidated* 60-minute bars going
back ~730 days for free, batch-downloadable, so the same universe is a ~hours job.

What it produces
----------------
Drop-in schema match for the IBKR downloader so the two are interchangeable:
    Data/IntradayData/{TICKER}_1hour.parquet
    columns: Date, Open, High, Low, Close, Volume, VWAP, BarCount, Ticker
(VWAP/BarCount are NaN — Yahoo doesn't provide them; kept for schema compatibility.)

Prices are AS-TRADED (auto_adjust=False) to match IBKR 'TRADES' bars, RTH only
(prepost=False). Timestamps are bar-START in US/Eastern, tz-stripped — same as the
IBKR file.

Hard limits (Yahoo, not us)
---------------------------
  * 60m  bars: ~730 calendar days of history   <- what we use
  * 1m   bars: only the last ~7 days
So this tool is hourly-only by design; for minute precision use Alpaca (free, IEX,
7yr) on the small traded-name subset instead.

Universes
---------
  --universe rf      every ticker in Data/RFpredictions/        (the full ~4.2k)
  --universe traded  symbols that actually traded (union of the trade_history files)
  --tickers AAPL ... explicit list (overrides --universe)

Resumable: skips any ticker whose output file already exists (use --force to redo).

Usage
-----
    python 2.3__YahooIntradayDownloader.py --limit 50          # benchmark sample
    python 2.3__YahooIntradayDownloader.py --universe traded   # ~800 traded names
    python 2.3__YahooIntradayDownloader.py --universe rf        # full universe
    python 2.3__YahooIntradayDownloader.py --tickers AAPL MSFT --force
"""

import os
import re
import sys
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import yfinance as yf

DATA_DIR    = os.path.join('Data', 'IntradayData')
RF_PRED_DIR = os.path.join('Data', 'RFpredictions')
TRADE_FILES = ['trade_history.parquet', os.path.join('Data', 'TradeHistory.parquet')]
os.makedirs(DATA_DIR, exist_ok=True)

# yfinance interval + period for each bar key (Yahoo caps 60m at ~730d)
BAR_MAP = {
    '1hour': {'interval': '60m', 'period': '730d', 'suffix': '1hour'},
}

OUT_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'BarCount', 'Ticker']


def out_path(ticker: str, suffix: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_{suffix}.parquet")


def to_yahoo(ticker: str) -> str:
    """Map our ticker symbology to Yahoo's (class shares: BRK.B -> BRK-B)."""
    return ticker.replace('.', '-').upper()


def universe_rf(limit=None):
    pat = re.compile(r"^(.*?)(?:_\d{8})?\.parquet$")
    out = set()
    for f in os.listdir(RF_PRED_DIR):
        if f.endswith('.parquet'):
            m = pat.match(f)
            if m and m.group(1):
                out.add(m.group(1).upper())
    out = sorted(out)
    return out[:limit] if limit else out


def universe_traded(limit=None):
    syms = set()
    for f in TRADE_FILES:
        if os.path.exists(f):
            syms |= set(pd.read_parquet(f, columns=['Symbol'])['Symbol'].astype(str).str.upper())
    out = sorted(syms)
    return out[:limit] if limit else out


def clean_one(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalize one ticker's yfinance frame to the IBKR-compatible schema."""
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    # Drop rows that are entirely NaN (Yahoo pads gaps)
    df = df.dropna(how='all')
    if df.empty:
        return None
    df = df.reset_index()
    # The datetime column is 'Datetime' (intraday) or 'Date'
    dtcol = 'Datetime' if 'Datetime' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
    df = df.rename(columns={dtcol: 'Date'})
    # tz -> US/Eastern, strip (match IBKR)
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    if df.empty:
        return None
    df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            return None
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if df.empty:
        return None
    df['VWAP'] = np.nan
    df['BarCount'] = np.nan
    df['Ticker'] = ticker
    df = (df[OUT_COLS]
          .drop_duplicates(subset=['Date'])
          .sort_values('Date')
          .reset_index(drop=True))
    return df if not df.empty else None


def download_batch(tickers, interval, period, retries=2):
    """Batch-download a list; return {ticker: cleaned_df}. Maps symbology for the query."""
    ymap = {to_yahoo(t): t for t in tickers}          # yahoo_symbol -> our_symbol
    ylist = list(ymap.keys())
    last_err = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                ylist,
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=False,
                prepost=False,
                threads=True,
                progress=False,
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    else:
        print(f"   batch failed after retries: {last_err}")
        return {}

    results = {}
    for ysym, our in ymap.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ysym in data.columns.get_level_values(0):
                    sub = data[ysym]
                else:
                    continue
            else:
                sub = data  # single-ticker frame
            cleaned = clean_one(sub, our)
            if cleaned is not None:
                results[our] = cleaned
        except Exception:
            continue
    return results


def main():
    ap = argparse.ArgumentParser(description='Free Yahoo hourly intraday downloader (IBKR-schema compatible)')
    ap.add_argument('--bar-size', default='1hour', choices=list(BAR_MAP.keys()))
    ap.add_argument('--universe', default='rf', choices=['rf', 'traded'])
    ap.add_argument('--tickers', nargs='+', help='explicit ticker list (overrides --universe)')
    ap.add_argument('--limit', type=int, help='cap number of tickers (benchmark)')
    ap.add_argument('--batch-size', type=int, default=50, help='tickers per Yahoo batch request')
    ap.add_argument('--sleep', type=float, default=1.0, help='seconds between batches (rate-limit politeness)')
    ap.add_argument('--force', action='store_true', help='re-download even if file exists')
    ap.add_argument('--period', default=None,
                    help="override Yahoo period (e.g. 1y, 6mo) — recovers recent IPOs that "
                         "have <730d of intraday history and error on the default 730d window")
    args = ap.parse_args()

    cfg = dict(BAR_MAP[args.bar_size])
    if args.period:
        cfg['period'] = args.period

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe == 'traded':
        tickers = universe_traded(args.limit)
    else:
        tickers = universe_rf(args.limit)

    if not args.force:
        todo = [t for t in tickers if not os.path.exists(out_path(t, cfg['suffix']))]
        skipped = len(tickers) - len(todo)
    else:
        todo, skipped = tickers, 0

    print("=" * 68)
    print("  YAHOO INTRADAY DOWNLOADER (free, hourly)")
    print("=" * 68)
    print(f"  Bar / interval : {args.bar_size}  ->  {cfg['interval']}  (period {cfg['period']})")
    print(f"  Universe       : {args.universe if not args.tickers else 'explicit'}")
    print(f"  Tickers        : {len(tickers)} total | {len(todo)} to fetch | {skipped} skipped (exist)")
    print(f"  Batch / sleep  : {args.batch_size} per request, {args.sleep}s between")
    print(f"  Output         : {DATA_DIR}/<TICKER>_{cfg['suffix']}.parquet")
    print("=" * 68)

    if not todo:
        print("  Nothing to do (all present). Use --force to redo.")
        return

    t0 = time.time()
    ok = fail = bars_total = 0
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    for bi in range(n_batches):
        batch = todo[bi * args.batch_size:(bi + 1) * args.batch_size]
        res = download_batch(batch, cfg['interval'], cfg['period'])
        for t in batch:
            df = res.get(t)
            if df is None:
                fail += 1
                continue
            try:
                df.to_parquet(out_path(t, cfg['suffix']), index=False, compression='snappy')
                ok += 1
                bars_total += len(df)
            except Exception:
                fail += 1

        done = ok + fail
        el = time.time() - t0
        rate = done / (el / 60.0) if el > 0 else 0
        eta_h = (len(todo) - done) / rate / 60.0 if rate > 0 else float('nan')
        # ascii-safe progress line
        line = (f"  batch {bi+1}/{n_batches} | done {done}/{len(todo)} "
                f"| ok={ok} fail={fail} | {rate:.0f} tic/min | ETA {eta_h:.2f}h | last={batch[-1]}")
        sys.stdout.buffer.write((line + "\n").encode('ascii', 'replace'))
        sys.stdout.flush()
        if bi < n_batches - 1:
            time.sleep(args.sleep)

    el = time.time() - t0
    print("=" * 68)
    print(f"  DONE: ok={ok} fail={fail} | {bars_total:,} bars | {el/60:.1f} min "
          f"| {ok/(el/60):.0f} tickers/min" if el > 0 else "")
    if ok:
        print(f"  Avg bars/ticker: {bars_total/ok:.0f}")
    print("=" * 68)


if __name__ == '__main__':
    main()
