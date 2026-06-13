#!/usr/bin/env python3
"""
Trade-History Intraday Downloader
=================================
Pulls the highest *normal* resolution (1-minute RTH bars) from IBKR TWS for EXACTLY
the days each symbol in trade_history.parquet was actually held, into a dedicated
directory so it never collides with the rest of the data lake.

Why this exists
---------------
trade_history.parquet was produced by 5__NightlyBackTester.py, which fills every
trade at the DAILY OPEN of daily bars. The live broker (9_SuperFastBroker.py) instead
enters at 10:00 ET and exits at the close. This downloader fetches the minute data the
companion simulator (8__IntradayFillSim.py) needs to reprice every trade at realistic
intraday fills and check whether the backtest's headline numbers survive.

It reuses the connection / threading / pacing machinery of
2__IntradayHistoricalDownloader.py, but instead of one contiguous lookback per ticker,
each ticker carries an EXPLICIT set of target trading days (the union of every
EntryDate..ExitDate hold window). Requests run day-by-day (durationStr="1 D") which is
the proven-safe pattern for 1-min bars (cf. auxiliary/0__EDA_IntradayDownloader.py,
which already pulled 1-min data months old).

Coverage tracking
-----------------
A sidecar `_coverage.json` records which (ticker, day) pairs have been *attempted*, so
re-runs skip days already fetched AND days that legitimately returned nothing (market
holidays) — without that, holidays inside a hold window would force a full re-pull
every run. Use --force to clear coverage for the selected tickers and re-download.

Usage
-----
    python 2.2__TradeHistoryIntradayDownloader.py                 # all symbols, port 7496
    python 2.2__TradeHistoryIntradayDownloader.py --num-threads 12
    python 2.2__TradeHistoryIntradayDownloader.py --tickers AAPL MSFT --force
    python 2.2__TradeHistoryIntradayDownloader.py --port 7497     # paper

Prerequisite: TWS (or IB Gateway) running and logged in on the chosen port.
Saved to: Data/IntradayTradeSim/{TICKER}_1min.parquet
"""

import os
import json
import time
import argparse
import threading
import asyncio
import random
import warnings
import logging
from collections import deque
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

from ib_insync import IB, Contract, util
import nest_asyncio
nest_asyncio.apply()

logging.getLogger('ib_insync.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ib_insync.client').setLevel(logging.CRITICAL)

# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DEFAULT_OUT_DIR       = os.path.join(script_dir, 'Data', 'IntradayTradeSim')
DEFAULT_TRADE_HISTORY = os.path.join(script_dir, 'trade_history.parquet')
COVERAGE_FILENAME     = '_coverage.json'
BAR_SIZE_KEY          = '1min'
BAR_SIZE_SETTING      = '1 min'

# ── Pacing ───────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 30    # seconds per request
INTER_DAY_DELAY   = 0.30  # seconds between successive day requests on one thread
SEM_LIMIT         = 25    # max concurrent historical-data requests (per loop)


# ── Target-day computation ───────────────────────────────────────────────────────

def compute_targets(trade_history_path: str) -> dict[str, list]:
    """
    Return {ticker: sorted list of datetime.date} — the union of every
    EntryDate..ExitDate hold window (business days only) for each symbol.
    """
    df = pd.read_parquet(trade_history_path)
    df['EntryDate'] = pd.to_datetime(df['EntryDate'])
    df['ExitDate']  = pd.to_datetime(df['ExitDate'])

    targets: dict[str, set] = {}
    for sym, entry, exit_ in zip(df['Symbol'], df['EntryDate'], df['ExitDate']):
        days = pd.bdate_range(entry.normalize(), exit_.normalize())
        targets.setdefault(sym, set()).update(d.date() for d in days)

    return {sym: sorted(days) for sym, days in targets.items()}


# ── Coverage sidecar ─────────────────────────────────────────────────────────────

def coverage_path(out_dir: str) -> str:
    return os.path.join(out_dir, COVERAGE_FILENAME)


def load_coverage(out_dir: str) -> dict[str, set]:
    fp = coverage_path(out_dir)
    if not os.path.exists(fp):
        return {}
    try:
        with open(fp, 'r') as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}
    except Exception:
        return {}


def save_coverage(out_dir: str, coverage: dict[str, set]):
    fp = coverage_path(out_dir)
    tmp = fp + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({k: sorted(v) for k, v in coverage.items()}, f)
        os.replace(tmp, fp)
    except Exception:
        pass


def parquet_path(out_dir: str, ticker: str) -> str:
    return os.path.join(out_dir, f"{ticker}_{BAR_SIZE_KEY}.parquet")


def load_existing(out_dir: str, ticker: str) -> pd.DataFrame | None:
    fp = parquet_path(out_dir, ticker)
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_parquet(fp)
        df['Date'] = pd.to_datetime(df['Date'])
        return df if not df.empty else None
    except Exception:
        return None


def file_covered_days(out_dir: str, ticker: str) -> set:
    """Distinct calendar dates that already have >=1 bar in the saved file."""
    df = load_existing(out_dir, ticker)
    if df is None:
        return set()
    return set(df['Date'].dt.date.map(lambda d: d.isoformat()))


def missing_days(out_dir: str, ticker: str, needed: list, coverage: dict[str, set]) -> list:
    """Needed days not yet attempted (per coverage json) and not already in the file."""
    attempted = coverage.get(ticker, set()) | file_covered_days(out_dir, ticker)
    return [d for d in needed if d.isoformat() not in attempted]


def merge_and_save(out_dir: str, ticker: str, new_df: pd.DataFrame) -> int:
    existing = load_existing(out_dir, ticker)
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = (combined
                .drop_duplicates(subset=['Date'])
                .sort_values('Date')
                .reset_index(drop=True))

    combined.to_parquet(parquet_path(out_dir, ticker), index=False, compression='snappy')
    return len(combined)


# ── Per-thread downloader ─────────────────────────────────────────────────────────

class TickerDownloader:
    """One instance per worker thread; holds a single IBKR connection."""

    def __init__(self, thread_id, host, port, task_queue, results_lock,
                 counters, out_dir, coverage, stop_event, pbar):
        self.thread_id    = thread_id
        self.host         = host
        self.port         = port
        self.task_queue   = task_queue
        self.results_lock = results_lock
        self.counters     = counters
        self.out_dir      = out_dir
        self.coverage     = coverage
        self.stop_event   = stop_event
        self.pbar         = pbar

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(lambda l, ctx: None)
        sem = asyncio.Semaphore(SEM_LIMIT)
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

            ticker, days = task
            success, reason = await self._download_ticker(ib, ticker, days, sem)

            with self.results_lock:
                # Mark days covered ONLY when the ticker actually produced data. A ticker that
                # timed out / returned nothing stays UNMARKED, so a plain re-run retries it
                # automatically (no --force needed). Holidays inside a successful ticker come
                # back as a clean empty and get marked, so they are not re-pulled.
                if success:
                    self.coverage.setdefault(ticker, set()).update(d.isoformat() for d in days)
                    save_coverage(self.out_dir, self.coverage)

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
                                          pct=f"{sr:.0f}%", last=ticker)

        await self._disconnect(ib)

    async def _download_ticker(self, ib: IB, ticker: str, days: list, sem):
        if not ib.isConnected():
            return False, 'DISCONNECTED'
        if not days:
            return True, 'UP_TO_DATE'

        contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')
        try:
            details = await ib.reqContractDetailsAsync(contract)
            if not details:
                return False, 'NOT_FOUND'
        except Exception:
            return False, 'DETAILS_ERROR'

        all_bars = []
        for day in days:
            if self.stop_event.is_set():
                break
            # endDateTime at end of the target day; "1 D" RTH returns that session
            end_str = datetime(day.year, day.month, day.day, 23, 59, 59).strftime('%Y%m%d %H:%M:%S')
            try:
                async with sem:
                    bars = await ib.reqHistoricalDataAsync(
                        contract       = contract,
                        endDateTime    = end_str,
                        durationStr    = "1 D",
                        barSizeSetting = BAR_SIZE_SETTING,
                        whatToShow     = 'TRADES',
                        useRTH         = True,
                        formatDate     = 2,
                        timeout        = REQUEST_TIMEOUT,
                    )
                if bars:
                    all_bars.extend(bars)
            except Exception as e:
                err = str(e).lower()
                if 'pacing violation' in err:
                    await asyncio.sleep(5)
                    # one retry for this day
                    try:
                        async with sem:
                            bars = await ib.reqHistoricalDataAsync(
                                contract=contract, endDateTime=end_str, durationStr="1 D",
                                barSizeSetting=BAR_SIZE_SETTING, whatToShow='TRADES',
                                useRTH=True, formatDate=2, timeout=REQUEST_TIMEOUT)
                        if bars:
                            all_bars.extend(bars)
                    except Exception:
                        pass
                elif 'connection' in err:
                    return False, 'CONNECTION_ERROR'
                # timeout / other: skip this day
            await asyncio.sleep(INTER_DAY_DELAY)

        if not all_bars:
            return False, 'NO_DATA'

        df = util.df(all_bars)
        if df is None or df.empty:
            return False, 'EMPTY'

        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume', 'average': 'VWAP', 'barCount': 'BarCount',
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        df['Ticker'] = ticker

        try:
            rows = merge_and_save(self.out_dir, ticker, df)
            return True, f"SAVED_{rows}_rows"
        except Exception as e:
            return False, f"SAVE_ERROR_{e}"


# ── Orchestrator ──────────────────────────────────────────────────────────────────

def run_download(tasks, out_dir, coverage, host, port, num_threads):
    task_queue   = deque(tasks)
    results_lock = threading.Lock()
    stop_event   = threading.Event()
    counters     = {'processed': 0, 'success': 0, 'failed': 0}

    pbar = tqdm(total=len(tasks), desc="Downloading", unit="ticker", ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')

    threads = []
    for i in range(min(num_threads, len(tasks))):
        dl = TickerDownloader(
            thread_id=i, host=host, port=port, task_queue=task_queue,
            results_lock=results_lock, counters=counters, out_dir=out_dir,
            coverage=coverage, stop_event=stop_event, pbar=pbar)
        t = threading.Thread(target=dl.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.15)

    while True:
        with results_lock:
            done = counters['processed'] >= len(tasks)
        if done:
            break
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.5)

    stop_event.set()
    for t in threads:
        t.join(timeout=10)
    pbar.close()
    return counters['success'], counters['failed']


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download 1-min RTH intraday data for the days in trade_history.parquet')
    parser.add_argument('--trade-history', type=str, default=DEFAULT_TRADE_HISTORY)
    parser.add_argument('--out-dir',       type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--tickers',       nargs='+', help='Restrict to these symbols')
    parser.add_argument('--num-threads',   type=int, default=8)
    parser.add_argument('--port',          type=int, default=7496,
                        help='TWS port — 7496 live (default), 7497 paper, 4001 Gateway live')
    parser.add_argument('--force',         action='store_true',
                        help='Clear coverage for selected tickers and re-download')
    parser.add_argument('--fill-holes',    action='store_true',
                        help='Refetch only days MISSING from each file (ignores coverage); '
                             'recovers interior timeout-holes without a full re-pull')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    targets = compute_targets(args.trade_history)
    if args.tickers:
        want = {t.upper() for t in args.tickers}
        targets = {k: v for k, v in targets.items() if k in want}
        if not targets:
            print("None of the requested tickers appear in the trade history.")
            return

    coverage = load_coverage(args.out_dir)
    if args.force:
        for sym in targets:
            coverage.pop(sym, None)
        save_coverage(args.out_dir, coverage)

    # Build the work list: (ticker, [missing days])
    tasks, skipped = [], 0
    total_needed_days = 0
    total_missing_days = 0
    for sym, needed in sorted(targets.items()):
        total_needed_days += len(needed)
        if args.fill_holes:
            # Days literally absent from the saved file (interior timeout-holes, never-fetched).
            have = file_covered_days(args.out_dir, sym)
            miss = [d for d in needed if d.isoformat() not in have]
        elif args.force:
            miss = needed
        else:
            miss = missing_days(args.out_dir, sym, needed, coverage)
        total_missing_days += len(miss)
        if miss:
            tasks.append((sym, miss))
        else:
            skipped += 1

    print()
    print("=" * 72)
    print("  TRADE-HISTORY INTRADAY DOWNLOADER  (1-min RTH, TRADES)")
    print("=" * 72)
    print(f"  Trade history : {args.trade_history}")
    print(f"  Output dir    : {args.out_dir}")
    print(f"  Port          : {args.port}  ({'live TWS' if args.port == 7496 else 'paper/gateway'})")
    print(f"  Threads       : {args.num_threads}")
    print(f"  Symbols total : {len(targets)}  |  to download: {len(tasks)}  |  already covered: {skipped}")
    print(f"  Symbol-days   : {total_needed_days} needed  |  {total_missing_days} still to fetch")
    print("=" * 72)

    if not tasks:
        print("\n  All symbols already covered. Use --force to re-download.")
        return

    print(f"\n  Starting download for {len(tasks)} symbols "
          f"({total_missing_days} symbol-days)...\n")
    t0 = time.time()
    success, failed = run_download(tasks, args.out_dir, coverage,
                                   host='127.0.0.1', port=args.port,
                                   num_threads=args.num_threads)
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print("  DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"  Symbols ok   : {success}")
    print(f"  Symbols fail : {failed}  (NO_DATA / NOT_FOUND / errors — see coverage.json)")
    print(f"  Time taken   : {elapsed/60:.1f} min")
    print("=" * 72)
    print("\n  Next: python 8__IntradayFillSim.py\n")


if __name__ == '__main__':
    main()
