#!/usr/bin/env python3
"""
Alpaca Minute Downloader  (2.4__AlpacaMinuteDownloader.py)
==========================================================
Multi-YEAR minute bars for the universe, free, via Alpaca's IEX historical feed.
This is the dataset the hourly Yahoo lake (2.3) could NOT give us: Yahoo caps 1m at
~7 days, Alpaca free serves 1-minute bars back to 2016. Minute granularity is what
lets an intraday backtest fire the 1.9% hard / 1.5% trailing stops on real bars
instead of next-day daily bars -> honest drawdowns (the daily sim hides -7.6% -> -20%).

Why Alpaca / what you get
-------------------------
  * FREE plan = IEX feed. IEX is ~2.5% of consolidated volume, so a minute with no
    IEX print has no bar (sparser than SIP) — fine for OHLC stop simulation, and the
    bars we DO get are real trades. Upgrade --feed sip later if you ever pay.
  * Real VWAP + trade-count per bar (Alpaca returns vw, n) — unlike Yahoo (NaN).
  * adjustment=raw  -> AS-TRADED prices, matching the IBKR/Yahoo lakes.

Schema (drop-in match for the hourly lake, 2.3):
    {root}/{TICKER}_{YEAR}_1min.parquet   (year-sharded)
    columns: Date, Open, High, Low, Close, Volume, VWAP, BarCount, Ticker
Default RTH only (09:30-16:00 ET); pass --all-sessions to keep pre/post too.

Keys
----
Put them in ./.alpaca_keys  (gitignored — see .alpaca_keys.example):
    ALPACA_API_KEY_ID=...
    ALPACA_API_SECRET_KEY=...
or export ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in the environment.

Robustness
----------
  * Resumable at (ticker, year) granularity — skips any shard already on disk, even
    empty ones (a 0-row shard is written when a name had no IEX data that year, so we
    never re-request it). A crash loses at most one (batch, year).
  * Rate-limited under the 200 req/min free cap, with 429/Retry-After backoff.
  * Year-chunked so memory stays bounded on a multi-hour, universe-scale pull.

Usage
-----
    python 2.4__AlpacaMinuteDownloader.py --limit 20                 # benchmark
    python 2.4__AlpacaMinuteDownloader.py --universe traded          # ~805 names first
    python 2.4__AlpacaMinuteDownloader.py --universe rf              # full universe
    python 2.4__AlpacaMinuteDownloader.py --tickers AAPL MSFT --start 2016-01-01
"""

import os
import re
import sys
import time
import json
import argparse
import warnings
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

RF_PRED_DIR = os.path.join('Data', 'RFpredictions')
TRADE_FILES = ['trade_history.parquet', os.path.join('Data', 'TradeHistory.parquet')]

DATA_URL   = 'https://data.alpaca.markets/v2/stocks/bars'
OUT_COLS   = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'BarCount', 'Ticker']
PAGE_LIMIT = 10000          # max bars per response (Alpaca hard cap)
MAX_PER_MIN = 190           # stay under the 200 req/min free-plan cap


# --------------------------------------------------------------------------- keys
def load_keys():
    kid = os.environ.get('ALPACA_API_KEY_ID')
    sec = os.environ.get('ALPACA_API_SECRET_KEY')
    if not (kid and sec) and os.path.exists('.alpaca_keys'):
        for line in open('.alpaca_keys', 'r', encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if k == 'ALPACA_API_KEY_ID':
                kid = kid or v
            elif k == 'ALPACA_API_SECRET_KEY':
                sec = sec or v
    if not (kid and sec):
        sys.exit("ERROR: no Alpaca keys. Create ./.alpaca_keys (see .alpaca_keys.example) "
                 "or export ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY.")
    return kid, sec


# ----------------------------------------------------------------------- universe
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


# ------------------------------------------------------------------- rate limiter
class RateLimiter:
    """Sliding 60s window; blocks to keep request rate under max_per_min."""
    def __init__(self, max_per_min):
        self.max = max_per_min
        self.calls = deque()

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= self.max:
            sleep_for = 60 - (now - self.calls[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.time()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
        self.calls.append(time.time())


# ---------------------------------------------------------------------- fetch
def fetch_window(symbols, start_iso, end_iso, timeframe, feed, headers, rl, session):
    """Page the multi-symbol bars endpoint for [start,end). -> {sym: [bar dicts]}."""
    out = {s: [] for s in symbols}
    params = {
        'symbols': ','.join(symbols),
        'timeframe': timeframe,
        'start': start_iso,
        'end': end_iso,
        'limit': PAGE_LIMIT,
        'adjustment': 'raw',
        'feed': feed,
        'sort': 'asc',
    }
    page_token = None
    while True:
        if page_token:
            params['page_token'] = page_token
        else:
            params.pop('page_token', None)
        rl.wait()
        for attempt in range(5):
            try:
                r = session.get(DATA_URL, headers=headers, params=params, timeout=60)
            except Exception:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                break
            if r.status_code == 429:
                ra = r.headers.get('Retry-After')
                time.sleep(float(ra) if ra else 5 * (attempt + 1))
                continue
            if r.status_code in (403, 401):
                raise RuntimeError(f"auth/subscription error {r.status_code}: {r.text[:200]}")
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"request failed repeatedly (last {r.status_code})")

        body = r.json()
        bars = body.get('bars') or {}
        for sym, lst in bars.items():
            if lst:
                out.setdefault(sym, []).extend(lst)
        page_token = body.get('next_page_token')
        if not page_token:
            break
    return out


def clean_bars(rows, ticker, rth=True):
    """Alpaca bar dicts -> IBKR/Yahoo-compatible frame. Bar timestamps are bar-START UTC."""
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.DataFrame(rows)
    ren = {'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low',
           'c': 'Close', 'v': 'Volume', 'vw': 'VWAP', 'n': 'BarCount'}
    df = df.rename(columns=ren)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            return pd.DataFrame(columns=OUT_COLS)
    if 'VWAP' not in df.columns:
        df['VWAP'] = np.nan
    if 'BarCount' not in df.columns:
        df['BarCount'] = np.nan
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    if rth:
        t = df['Date'].dt.time
        df = df[(t >= pd.Timestamp('09:30').time()) & (t < pd.Timestamp('16:00').time())]
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df['Ticker'] = ticker
    df = (df[OUT_COLS]
          .drop_duplicates(subset=['Date'])
          .sort_values('Date')
          .reset_index(drop=True))
    return df


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Free Alpaca multi-year minute downloader (IEX)')
    ap.add_argument('--universe', default='rf', choices=['rf', 'traded'])
    ap.add_argument('--tickers', nargs='+', help='explicit ticker list (overrides --universe)')
    ap.add_argument('--limit', type=int, help='cap number of tickers (benchmark)')
    ap.add_argument('--timeframe', default='1Min', help='1Min, 5Min, 15Min, 1Hour ...')
    ap.add_argument('--feed', default='iex', choices=['iex', 'sip'], help='iex=free, sip=paid')
    ap.add_argument('--start', default='2016-01-01', help='YYYY-MM-DD (Alpaca free goes back to 2016)')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (default: today)')
    ap.add_argument('--root', default=r'D:\MarketData\AlpacaMinute',
                    help='output root (default D: to spare C:)')
    ap.add_argument('--batch-size', type=int, default=100, help='symbols per request')
    ap.add_argument('--all-sessions', action='store_true', help='keep pre/post-market bars too')
    ap.add_argument('--force', action='store_true', help='re-download existing shards')
    args = ap.parse_args()

    kid, sec = load_keys()
    headers = {'APCA-API-KEY-ID': kid, 'APCA-API-SECRET-KEY': sec}
    rth = not args.all_sessions
    os.makedirs(args.root, exist_ok=True)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe == 'traded':
        tickers = universe_traded(args.limit)
    else:
        tickers = universe_rf(args.limit)

    end = args.end or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    y0 = int(args.start[:4])
    y1 = int(end[:4])
    years = list(range(y0, y1 + 1))
    suffix = '1min' if args.timeframe.lower() == '1min' else args.timeframe.lower()

    def shard_path(t, yr):
        return os.path.join(args.root, f"{t}_{yr}_{suffix}.parquet")

    print("=" * 70)
    print("  ALPACA MINUTE DOWNLOADER (free IEX, year-sharded, resumable)")
    print("=" * 70)
    print(f"  Universe   : {args.universe if not args.tickers else 'explicit'} | {len(tickers)} tickers")
    print(f"  Range      : {args.start} -> {end}  ({len(years)} year-shards/ticker)")
    print(f"  Bars/feed  : {args.timeframe} / {args.feed} | RTH={'no' if args.all_sessions else 'yes'} | raw")
    print(f"  Output     : {args.root}\\<TICKER>_<YEAR>_{suffix}.parquet")
    print(f"  Batch/cap  : {args.batch_size} sym/req, <= {MAX_PER_MIN} req/min")
    print("=" * 70)

    rl = RateLimiter(MAX_PER_MIN)
    session = requests.Session()
    t0 = time.time()
    shards_done = shards_skip = bars_total = req_est = 0
    total_shards = len(tickers) * len(years)

    for yr in years:
        ystart = f"{yr}-01-01T00:00:00Z"
        yend = f"{yr+1}-01-01T00:00:00Z" if yr < y1 else f"{end}T23:59:59Z"
        for bi in range(0, len(tickers), args.batch_size):
            batch = tickers[bi:bi + args.batch_size]
            todo = batch if args.force else [t for t in batch if not os.path.exists(shard_path(t, yr))]
            shards_skip += len(batch) - len(todo)
            if not todo:
                continue
            try:
                bars_by_sym = fetch_window(todo, ystart, yend, args.timeframe,
                                           args.feed, headers, rl, session)
            except RuntimeError as e:
                print(f"  !! {yr} batch @{bi}: {e}")
                if '403' in str(e) or '401' in str(e):
                    sys.exit("  Stopping: key rejected or feed not in your plan "
                             "(free plan must use --feed iex).")
                continue
            for t in todo:
                df = clean_bars(bars_by_sym.get(t, []), t, rth=rth)
                try:
                    df.to_parquet(shard_path(t, yr), index=False, compression='snappy')
                    shards_done += 1
                    bars_total += len(df)
                except Exception as ex:
                    print(f"  write fail {t} {yr}: {ex}")
            done = shards_done + shards_skip
            el = time.time() - t0
            rate = shards_done / (el / 60.0) if el > 0 else 0
            eta_h = (total_shards - done) / rate / 60.0 if rate > 0 else float('nan')
            line = (f"  yr {yr} batch@{bi:<5} | shards {done}/{total_shards} "
                    f"(new={shards_done} skip={shards_skip}) | {bars_total:,} bars "
                    f"| {rate:.0f} shard/min | ETA {eta_h:.1f}h | last={todo[-1]}")
            sys.stdout.buffer.write((line + "\n").encode('ascii', 'replace'))
            sys.stdout.flush()

    el = time.time() - t0
    print("=" * 70)
    print(f"  DONE: new={shards_done} skip={shards_skip} | {bars_total:,} bars "
          f"| {el/60:.1f} min")
    print("=" * 70)


if __name__ == '__main__':
    main()
