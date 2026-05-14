#!/usr/bin/env python3
"""
Historical Intraday Data Downloader
Downloads and maintains intraday OHLCV bar data via IBKR TWS (live, port 7496).
Supports 1min, 5min, 15min, 30min, and 1-hour bars.

Smart incremental updates: checks existing data and only downloads the missing range.
Multi-threaded: each thread holds its own IBKR connection and processes one ticker at
a time, chunking backwards through dates.  A global semaphore keeps total concurrent
historical-data requests within IBKR's pacing limits.

Usage:
    python 2__IntradayHistoricalDownloader.py --ticker AAPL
    python 2__IntradayHistoricalDownloader.py --ticker AAPL --bar-size 1min --lookback-days 7
    python 2__IntradayHistoricalDownloader.py --bar-size 5min --lookback-days 30
    python 2__IntradayHistoricalDownloader.py --bar-size 30min --lookback-days 60 --num-threads 8
    python 2__IntradayHistoricalDownloader.py --force  # re-download everything

Saved to: Data/IntradayData/{TICKER}_{bar_size}.parquet
"""

import os
import time
import argparse
import threading
import asyncio
import random
import warnings
import logging
import re
from collections import deque
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

from ib_insync import IB, Contract, util
import nest_asyncio
nest_asyncio.apply()

# Suppress ib_insync noise
logging.getLogger('ib_insync.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ib_insync.client').setLevel(logging.CRITICAL)

# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DATA_DIR    = os.path.join(script_dir, 'Data', 'IntradayData')
RF_PRED_DIR = os.path.join(script_dir, 'Data', 'RFpredictions')

os.makedirs(DATA_DIR, exist_ok=True)

# ── IBKR bar-size config ───────────────────────────────────────────────────────
# max_days_per_chunk: IBKR hard limit on how far back one request can reach
# max_lookback_days:  sensible cap for strategy testing
BAR_CONFIGS = {
    '1min':  {'bar_size': '1 min',  'max_days_per_chunk': 1,  'max_lookback_days': 7,   'desc': '1-minute bars'},
    '5min':  {'bar_size': '5 mins', 'max_days_per_chunk': 5,  'max_lookback_days': 90,  'desc': '5-minute bars'},
    '15min': {'bar_size': '15 mins','max_days_per_chunk': 10, 'max_lookback_days': 180, 'desc': '15-minute bars'},
    '30min': {'bar_size': '30 mins','max_days_per_chunk': 20, 'max_lookback_days': 365, 'desc': '30-minute bars'},
    '1hour': {'bar_size': '1 hour', 'max_days_per_chunk': 30, 'max_lookback_days': 730, 'desc': '1-hour bars'},
}

# ── Global pacing semaphore ────────────────────────────────────────────────────
# IBKR allows up to ~50 simultaneous historical-data requests across all connections.
# We stay conservative at 25 to leave headroom and avoid pacing violations.
IBKR_REQUEST_SEMAPHORE = asyncio.Semaphore(25)   # created fresh per event loop below
REQUEST_TIMEOUT   = 30   # seconds per request
INTER_CHUNK_DELAY = 0.35 # seconds between successive chunk requests on one thread


# ── Helpers ───────────────────────────────────────────────────────────────────

def parquet_path(ticker: str, bar_size_key: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_{bar_size_key}.parquet")


def load_existing(ticker: str, bar_size_key: str) -> pd.DataFrame | None:
    """Load existing parquet, or None if absent/corrupt."""
    fp = parquet_path(ticker, bar_size_key)
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_parquet(fp)
        df['Date'] = pd.to_datetime(df['Date'])
        return df if not df.empty else None
    except Exception:
        return None


def get_download_range(ticker: str, bar_size_key: str, lookback_days: int):
    """
    Return (start_dt, end_dt) for the data we still need to download.
    If we already have data, only fetch the missing tail (and a 3-day overlap
    to catch any late-arriving corrections).
    Returns (None, None) if data is already current.
    """
    cfg      = BAR_CONFIGS[bar_size_key]
    hard_cap = min(lookback_days, cfg['max_lookback_days'])
    end_dt   = datetime.now()
    full_start = end_dt - timedelta(days=hard_cap)

    existing = load_existing(ticker, bar_size_key)
    if existing is None:
        return full_start, end_dt

    last_date = existing['Date'].max()
    if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
        last_date = last_date.replace(tzinfo=None)

    days_stale = (end_dt - last_date).days
    if days_stale <= 1:
        return None, None   # already current

    # Re-download with a 3-day overlap to catch corrections
    incremental_start = last_date - timedelta(days=3)
    return max(full_start, incremental_start), end_dt


def merge_and_save(ticker: str, bar_size_key: str, new_df: pd.DataFrame) -> int:
    """Merge new_df with any existing file, deduplicate, sort, save. Returns row count."""
    existing = load_existing(ticker, bar_size_key)
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = (combined
                .drop_duplicates(subset=['Date'])
                .sort_values('Date')
                .reset_index(drop=True))

    fp = parquet_path(ticker, bar_size_key)
    combined.to_parquet(fp, index=False, compression='snappy')
    return len(combined)


def get_tickers_from_rf_predictions(limit: int | None = None) -> list[str]:
    """Pull unique ticker symbols from the RF predictions folder."""
    pattern = re.compile(r"^(.*?)(?:_\d{8})?\.parquet$")
    tickers = set()
    try:
        for fname in os.listdir(RF_PRED_DIR):
            if fname.endswith('.parquet'):
                m = pattern.match(fname)
                if m and m.group(1):
                    tickers.add(m.group(1))
    except Exception as e:
        print(f"Error reading RF predictions directory: {e}")
    result = sorted(tickers)
    return result[:limit] if limit else result


# ── Per-thread downloader ─────────────────────────────────────────────────────

class TickerDownloader:
    """
    One instance per worker thread.  Holds a single IBKR connection and
    processes tickers from the shared task queue.
    """

    def __init__(self, thread_id: int, host: str, port: int,
                 task_queue: deque, results_lock: threading.Lock,
                 counters: dict, bar_size_key: str, lookback_days: int,
                 stop_event: threading.Event, pbar, semaphore_holder: list):
        self.thread_id       = thread_id
        self.host            = host
        self.port            = port
        self.task_queue      = task_queue
        self.results_lock    = results_lock
        self.counters        = counters
        self.bar_size_key    = bar_size_key
        self.bar_cfg         = BAR_CONFIGS[bar_size_key]
        self.lookback_days   = lookback_days
        self.stop_event      = stop_event
        self.pbar            = pbar
        self.semaphore_holder = semaphore_holder  # mutable list so thread can set it

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(lambda l, ctx: None)
        # Create a per-loop semaphore and store it for this thread
        sem = asyncio.Semaphore(25)
        self.semaphore_holder.append(sem)
        try:
            loop.run_until_complete(self._run_async(sem))
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0.05))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _connect(self) -> IB | None:
        ib = IB()
        client_id = random.randint(1000, 9000) + self.thread_id
        ready = asyncio.Event()

        def on_error(reqId, code, msg, contract):
            if code in (2104, 2106):
                ready.set()

        try:
            await ib.connectAsync(self.host, self.port,
                                  clientId=client_id, timeout=REQUEST_TIMEOUT)
            ib.errorEvent += on_error
            if not ib.isConnected():
                return None
            # Wait for market-data-farm confirmation (or proceed after 5s)
            try:
                await asyncio.wait_for(ready.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return ib
        except Exception:
            return None

    async def _disconnect(self, ib: IB):
        try:
            if ib.isConnected():
                await ib.disconnectAsync()
                await asyncio.sleep(0.05)
        except Exception:
            pass
        try:
            if hasattr(ib, 'client') and hasattr(ib.client, 'conn') and ib.client.conn:
                ib.client.conn.disconnect()
                ib.client.conn = None
        except Exception:
            pass

    async def _run_async(self, sem: asyncio.Semaphore):
        ib = None
        for attempt in range(3):
            ib = await self._connect()
            if ib:
                break
            await asyncio.sleep(2 * (attempt + 1))

        if ib is None:
            return

        while not self.stop_event.is_set():
            task = None
            with self.results_lock:
                if self.task_queue:
                    task = self.task_queue.popleft()

            if task is None:
                await asyncio.sleep(0.1)
                continue

            ticker = task
            success, reason = await self._download_ticker(ib, ticker, sem)

            with self.results_lock:
                self.counters['processed'] += 1
                if success:
                    self.counters['success'] += 1
                else:
                    self.counters['failed'] += 1
                if self.pbar:
                    self.pbar.update(1)
                    sr = (self.counters['success'] / self.counters['processed'] * 100
                          if self.counters['processed'] else 0)
                    self.pbar.set_postfix(ok=self.counters['success'],
                                         fail=self.counters['failed'],
                                         pct=f"{sr:.0f}%",
                                         last=ticker)

        await self._disconnect(ib)

    async def _download_ticker(self, ib: IB, ticker: str, sem: asyncio.Semaphore):
        """Download all missing chunks for one ticker, merge and save."""
        if not ib.isConnected():
            return False, 'DISCONNECTED'

        start_dt, end_dt = get_download_range(ticker, self.bar_size_key, self.lookback_days)
        if start_dt is None:
            return True, 'UP_TO_DATE'

        contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')

        # Validate contract exists (one cheap call, no semaphore needed for details)
        try:
            details = await ib.reqContractDetailsAsync(contract)
            if not details:
                return False, 'NOT_FOUND'
        except Exception:
            return False, 'DETAILS_ERROR'

        all_bars = []
        chunk_end = end_dt
        max_chunk = self.bar_cfg['max_days_per_chunk']

        while chunk_end > start_dt and not self.stop_event.is_set():
            chunk_start = max(start_dt, chunk_end - timedelta(days=max_chunk))
            chunk_days  = max(1, (chunk_end - chunk_start).days)

            end_str = chunk_end.strftime('%Y%m%d %H:%M:%S')

            try:
                async with sem:
                    bars = await ib.reqHistoricalDataAsync(
                        contract       = contract,
                        endDateTime    = end_str,
                        durationStr    = f"{chunk_days} D",
                        barSizeSetting = self.bar_cfg['bar_size'],
                        whatToShow     = 'TRADES',
                        useRTH         = True,
                        formatDate     = 2,          # tz-aware timestamps
                        timeout        = REQUEST_TIMEOUT,
                    )

                if bars:
                    all_bars.extend(bars)

            except Exception as e:
                err = str(e).lower()
                if 'pacing violation' in err:
                    await asyncio.sleep(5)
                    continue                 # retry same chunk
                elif 'timeout' in err:
                    pass                    # skip chunk, move on
                elif 'connection' in err:
                    return False, 'CONNECTION_ERROR'
                # any other error: skip this chunk

            chunk_end = chunk_start
            if chunk_end > start_dt:
                await asyncio.sleep(INTER_CHUNK_DELAY)

        if not all_bars:
            return False, 'NO_DATA'

        df = util.df(all_bars)
        if df is None or df.empty:
            return False, 'EMPTY'

        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'average': 'VWAP', 'barCount': 'BarCount',
        }, inplace=True)

        df['Date']   = pd.to_datetime(df['Date'])
        # Strip timezone for consistent storage
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        df['Ticker'] = ticker

        try:
            rows = merge_and_save(ticker, self.bar_size_key, df)
            return True, f"SAVED_{rows}_rows"
        except Exception as e:
            return False, f"SAVE_ERROR_{e}"


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_download(tickers: list[str], bar_size_key: str, lookback_days: int,
                 host: str, port: int, num_threads: int):
    """Spin up worker threads and wait for completion."""
    task_queue    = deque(tickers)
    results_lock  = threading.Lock()
    stop_event    = threading.Event()
    counters      = {'processed': 0, 'success': 0, 'failed': 0}

    pbar = tqdm(total=len(tickers), desc="Downloading", unit="ticker",
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')

    threads = []
    for i in range(min(num_threads, len(tickers))):
        sem_holder = []
        dl = TickerDownloader(
            thread_id      = i,
            host           = host,
            port           = port,
            task_queue     = task_queue,
            results_lock   = results_lock,
            counters       = counters,
            bar_size_key   = bar_size_key,
            lookback_days  = lookback_days,
            stop_event     = stop_event,
            pbar           = pbar,
            semaphore_holder = sem_holder,
        )
        t = threading.Thread(target=dl.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.15)   # stagger connection attempts

    # Wait for all tickers to be processed
    while True:
        with results_lock:
            done = counters['processed'] >= len(tickers)
        if done:
            break
        # Also exit if queue is empty and all threads have finished
        all_dead = not any(t.is_alive() for t in threads)
        if all_dead:
            break
        time.sleep(0.5)

    stop_event.set()
    for t in threads:
        t.join(timeout=10)
    pbar.close()

    return counters['success'], counters['failed']


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download historical intraday data from IBKR TWS for strategy testing'
    )
    parser.add_argument('--ticker',        type=str,
                        help='Single ticker (overrides RF predictions list)')
    parser.add_argument('--tickers',       nargs='+',
                        help='Multiple tickers (e.g. --tickers AAPL MSFT TSLA)')
    parser.add_argument('--bar-size',      type=str, default='5min',
                        choices=list(BAR_CONFIGS.keys()),
                        help='Bar size (default: 5min)')
    parser.add_argument('--lookback-days', type=int, default=None,
                        help='Calendar days of history to download (default: bar-size max)')
    parser.add_argument('--num-threads',   type=int, default=8,
                        help='Parallel IBKR connections (default: 8). '
                             'Higher = faster but more pacing pressure.')
    parser.add_argument('--port',          type=int, default=7496,
                        help='TWS port — 7496 live (default), 7497 paper, 4001 Gateway live')
    parser.add_argument('--limit',         type=int,
                        help='Limit tickers from RF predictions (for quick tests)')
    parser.add_argument('--force',         action='store_true',
                        help='Re-download data even if file already exists')
    parser.add_argument('--show-data',     action='store_true',
                        help='Print a preview of each saved file after download')

    args = parser.parse_args()

    cfg          = BAR_CONFIGS[args.bar_size]
    lookback_days = args.lookback_days or cfg['max_lookback_days']
    lookback_days = min(lookback_days, cfg['max_lookback_days'])

    # ── Collect tickers ──────────────────────────────────────────────────────
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = get_tickers_from_rf_predictions(limit=args.limit)
        if not tickers:
            print("No tickers found in RF predictions folder. Use --ticker or --tickers.")
            return

    # ── Skip already-current files (unless --force) ──────────────────────────
    if args.force:
        to_process = tickers
        skipped    = []
    else:
        to_process, skipped = [], []
        for t in tickers:
            start, _ = get_download_range(t, args.bar_size, lookback_days)
            if start is None:
                skipped.append(t)
            else:
                to_process.append(t)

    # ── Banner ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  IBKR INTRADAY HISTORICAL DOWNLOADER")
    print("=" * 70)
    print(f"  Bar size     : {cfg['bar_size']}  ({cfg['desc']})")
    print(f"  Lookback     : {lookback_days} calendar days")
    print(f"  Port         : {args.port}  ({'live TWS' if args.port == 7496 else 'paper/gateway'})")
    print(f"  Threads      : {args.num_threads}")
    print(f"  Tickers total: {len(tickers)}  |  to download: {len(to_process)}  |  skipped (current): {len(skipped)}")
    print(f"  Output dir   : {DATA_DIR}")
    print("=" * 70)

    if lookback_days > cfg['max_lookback_days']:
        print(f"\n  NOTE: {args.bar_size} bars are only available for "
              f"~{cfg['max_lookback_days']} days on IBKR. Capping lookback.")

    if not to_process:
        print("\n  All tickers are already up-to-date.  Use --force to re-download.")
        return

    print(f"\n  Starting download of {len(to_process)} tickers...\n")
    t0 = time.time()

    success, failed = run_download(
        tickers      = to_process,
        bar_size_key = args.bar_size,
        lookback_days= lookback_days,
        host         = '127.0.0.1',
        port         = args.port,
        num_threads  = args.num_threads,
    )

    elapsed = time.time() - t0
    rate    = (success / len(to_process) * 100) if to_process else 0

    print()
    print("=" * 70)
    print("  DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"  Successful : {success}")
    print(f"  Failed     : {failed}")
    print(f"  Success %%  : {rate:.1f}%%")
    print(f"  Time taken : {elapsed/60:.1f} min")
    if len(to_process) > 1:
        print(f"  Avg/ticker : {elapsed/len(to_process):.1f} s")
    print("=" * 70)

    # ── Optional data preview ─────────────────────────────────────────────────
    if args.show_data or (args.ticker and success > 0):
        ticker = (args.ticker or to_process[0]).upper()
        fp = parquet_path(ticker, args.bar_size)
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            print(f"\n  DATA PREVIEW — {ticker} ({args.bar_size})")
            print(f"  {len(df)} bars  |  {df['Date'].min()}  →  {df['Date'].max()}")
            print()
            print(df.head(5).to_string(index=False))
            print("  ...")
            print(df.tail(5).to_string(index=False))
            print()


if __name__ == '__main__':
    main()
