#!/usr/bin/env python3
"""
EDA Intraday Data Downloader
Downloads recent 1-minute bar data for specific tickers via IBKR TWS/Gateway.

Usage:
    python 0__EDA_IntradayDownloader.py                    # defaults: 4 portfolio stocks, 5 days
    python 0__EDA_IntradayDownloader.py --tickers TNDM AAPL --days 7
    python 0__EDA_IntradayDownloader.py --port 7497        # paper trading port

Data is saved to Data/MinuteData/{TICKER}_{start}_{end}.parquet
"""

import os
import asyncio
import time
import random
import argparse
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ib_insync import IB, Contract, util
import nest_asyncio
nest_asyncio.apply()

# ── Config ────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

MINUTE_DATA_DIR = os.path.join(script_dir, 'Data', 'MinuteData')
os.makedirs(MINUTE_DATA_DIR, exist_ok=True)

DEFAULT_TICKERS  = ['TNDM', 'CMBT', 'ASC', 'ABVX']
DEFAULT_PORT     = 7496   # TWS live (7497 = paper, 4001 = Gateway live)
DEFAULT_DAYS     = 5      # IBKR 1-min bars max lookback ~7 trading days
REQUEST_DELAY    = 0.6    # seconds between IBKR requests (avoid pacing violations)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_recent_trading_days(n: int) -> list:
    """Return the last N weekdays (no holiday check, IBKR returns empty on holidays)."""
    days = []
    d = datetime.now().date()
    # If before market close today, don't include today (data may be incomplete)
    # Include today only if it's a weekday and we explicitly want it
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:   # Mon–Fri
            days.append(d)
    return list(reversed(days))   # oldest → newest


def save_minute_data(ticker: str, dfs: list, trading_days: list) -> str | None:
    """Merge & save list of daily DataFrames to a single parquet file."""
    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined = (combined
                .drop_duplicates(subset=['Date'])
                .sort_values('Date')
                .reset_index(drop=True))

    start_s = trading_days[0].strftime('%Y%m%d')
    end_s   = trading_days[-1].strftime('%Y%m%d')
    save_path = os.path.join(MINUTE_DATA_DIR, f"{ticker}_{start_s}_{end_s}.parquet")
    combined.to_parquet(save_path, index=False, compression='snappy')
    return save_path


# ── Download logic ────────────────────────────────────────────────────────────
async def fetch_one_day(ib: IB, ticker: str, date) -> pd.DataFrame | None:
    """Download 1-min TRADES bars for a single date from IBKR."""
    contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')

    # End at market close on that day (23:59:59 lets IBKR pick the right session)
    end_str = datetime.combine(date, datetime.min.time()).strftime('%Y%m%d') + ' 23:59:59'

    try:
        # Validate contract exists
        details = await ib.reqContractDetailsAsync(contract)
        if not details:
            print(f"    [{ticker}] Not found in IBKR — skipping")
            return None

        bars = await ib.reqHistoricalDataAsync(
            contract        = contract,
            endDateTime     = end_str,
            durationStr     = '1 D',
            barSizeSetting  = '1 min',
            whatToShow      = 'TRADES',
            useRTH          = True,    # regular trading hours only
            formatDate      = 2,       # UTC epoch → ib_insync converts to tz-aware
        )

        if not bars:
            print(f"    [{ticker}] No bars on {date} (holiday or no data)")
            return None

        df = util.df(bars)
        df.rename(columns={
            'date':     'Date',
            'open':     'Open',
            'high':     'High',
            'low':      'Low',
            'close':    'Close',
            'volume':   'Volume',
            'average':  'VWAP',
            'barCount': 'BarCount',
        }, inplace=True)
        df['Ticker'] = ticker
        df['Date']   = pd.to_datetime(df['Date'])
        return df

    except Exception as e:
        print(f"    [{ticker}] Error on {date}: {e}")
        return None


async def download_all(tickers: list, days: int, port: int):
    ib = IB()
    client_id = random.randint(1000, 9000)

    print(f"\nConnecting to IBKR on port {port} (clientId={client_id})...")
    try:
        await ib.connectAsync('127.0.0.1', port, clientId=client_id, timeout=20)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure TWS or IB Gateway is running and API connections are enabled.")
        return

    if not ib.isConnected():
        print("Could not connect to IBKR. Exiting.")
        return

    print(f"Connected ✓")

    trading_days = get_recent_trading_days(days)
    print(f"Target dates ({days} trading days): {[str(d) for d in trading_days]}\n")

    results_summary = []

    for ticker in tickers:
        print(f"── {ticker} ─────────────────────────────────")
        daily_dfs = []

        for date in trading_days:
            df = await fetch_one_day(ib, ticker, date)
            if df is not None and not df.empty:
                daily_dfs.append(df)
                print(f"    {date}: {len(df):4d} bars  "
                      f"({df['Date'].min().strftime('%H:%M')} – {df['Date'].max().strftime('%H:%M')})")
            await asyncio.sleep(REQUEST_DELAY)

        if daily_dfs:
            save_path = save_minute_data(ticker, daily_dfs, trading_days)
            total_bars = sum(len(d) for d in daily_dfs)
            print(f"  → Saved {total_bars} total bars to {os.path.basename(save_path)}")
            results_summary.append((ticker, total_bars, 'OK'))
        else:
            print(f"  → No data saved for {ticker}")
            results_summary.append((ticker, 0, 'NO DATA'))

    ib.disconnect()

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for ticker, bars, status in results_summary:
        print(f"  {ticker:6s}  {bars:5d} bars  [{status}]")
    print("=" * 50)
    print(f"Files saved to: {MINUTE_DATA_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Download 1-min intraday data from IBKR for EDA')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        help=f'Tickers to download (default: {DEFAULT_TICKERS})')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                        help=f'Number of recent trading days (default: {DEFAULT_DAYS}, max ~7 for 1-min)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'IBKR TWS/Gateway port (default: {DEFAULT_PORT})')
    args = parser.parse_args()

    print("=" * 50)
    print("EDA INTRADAY DOWNLOADER")
    print("=" * 50)
    print(f"Tickers : {args.tickers}")
    print(f"Days    : {args.days}")
    print(f"Port    : {args.port}")
    print("=" * 50)

    asyncio.run(download_all(args.tickers, args.days, args.port))


if __name__ == '__main__':
    main()
