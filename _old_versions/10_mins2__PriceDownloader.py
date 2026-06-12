#!/usr/bin/env python

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import asyncio
from tqdm import tqdm
import traceback
import glob
import re
import json
import random
import warnings
import sys
from collections import deque
from functools import lru_cache
from Util import get_logger

warnings.filterwarnings('ignore')

from ib_insync import IB, Contract, util
import nest_asyncio
nest_asyncio.apply()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

BASE_DIRECTORY = script_dir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'PriceData')
RF_PREDICTIONS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'RFpredictions')
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'logging')
TICKERS_CIK_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'TickerCikData')
PROGRESS_FILE = os.path.join(LOG_DIRECTORY, 'download_progress.json')

DAILY_BAR_SIZE = '1 day'
NUM_THREADS = 32  # Keep 50 requests in the air at all times
REQUEST_TIMEOUT = 10

os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(RF_PREDICTIONS_DIRECTORY, exist_ok=True)

# Global state for threading approach
class DataFetcher:
    """Multi-threaded data fetcher using deque-based reading system"""
    def __init__(self, host='127.0.0.1', port=7497, num_threads=NUM_THREADS, data_directory=None):
        self.host = host
        self.port = port
        self.num_threads = num_threads
        self.data_directory = data_directory or DATA_DIRECTORY

        # Task queue: items are (ticker_index, ticker, duration_str, end_date_time)
        self.task_queue = deque()

        # Results: dict mapping ticker_index -> (ticker, df, fail_reason)
        self.results = {}
        self.results_lock = threading.Lock()

        # Progress tracking
        self.success_count = 0
        self.fail_count = 0
        self.processed_count = 0
        self.total_tasks = 0
        self.processed_tickers = []  # Track processed tickers for incremental saving

        # Thread management
        self.threads = []
        self.stop_event = threading.Event()

        # Statistics
        self.requests_count = 0
        self.connections_created = 0
        self.connections_failed = 0

        # Rate limiting
        self.last_request_times = deque(maxlen=50)  # Track last 50 request times
        self.rate_limit_lock = threading.Lock()

        print(f"DataFetcher initialized with {num_threads} threads")
        print(f"Host: {host}, Port: {port}")
        print("=" * 80)

    async def create_connection_async(self, client_id):
        """Create a single IB connection for a worker thread (async version)"""
        ib = IB()

        def handle_error(reqId, errorCode, errorString, contract):
            """Custom error handler to suppress contract not found errors"""
            # Silently ignore these common/expected errors
            if errorCode in [200, 2104, 2106, 2158, 2119, 2105]:
                # 200: No security definition (ticker not found)
                # 2104, 2106, 2158: Market data farm connection messages
                # 2119: Market data farm is connected
                # 2105: HMDS data farm connection is broken (will reconnect)
                pass
            else:
                # Only log unexpected errors
                logger.error(f"Error {errorCode}, reqId {reqId}: {errorString}")

        try:
            await ib.connectAsync(self.host, self.port, clientId=client_id, timeout=REQUEST_TIMEOUT)
            ib.errorEvent += handle_error

            if ib.isConnected():
                self.connections_created += 1
                return ib
            else:
                raise Exception("Connection failed to establish")

        except Exception as e:
            self.connections_failed += 1
            # Only log non-timeout errors
            if "timeout" not in str(e).lower():
                logger.error(f"Failed to create connection (client {client_id}): {str(e)}")
            return None

    def worker_thread(self, thread_id, pbar=None):
        """Worker thread that processes tasks from the deque"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Suppress Windows ProactorEventLoop connection errors
        loop.set_exception_handler(lambda loop, context: None)

        try:
            loop.run_until_complete(self._worker_thread_async(thread_id, pbar))
        finally:
            # Give pending tasks a moment to complete
            try:
                loop.run_until_complete(asyncio.sleep(0.1))
            except:
                pass
            # Close the loop
            try:
                loop.close()
            except:
                pass


                
                
                
    async def _worker_thread_async(self, thread_id, pbar=None):
        """Async worker thread implementation"""
        client_id = random.randint(1000, 9000) + thread_id
        ib = None
        retry_count = 0
        max_connection_retries = 3

        # Try to establish connection
        while retry_count < max_connection_retries and not self.stop_event.is_set():
            ib = await self.create_connection_async(client_id)
            if ib is not None:
                break
            retry_count += 1
            await asyncio.sleep(2 * retry_count)

        if ib is None:
            logger.error(f"Thread {thread_id}: Failed to establish connection after {max_connection_retries} retries")
            return

        # Process tasks
        consecutive_errors = 0
        while not self.stop_event.is_set():
            try:
                # Try to get a task from the deque
                if len(self.task_queue) == 0:
                    await asyncio.sleep(0.1)
                    continue

                with self.results_lock:
                    if len(self.task_queue) == 0:
                        continue
                    task = self.task_queue.popleft()

                ticker_index, ticker, duration_str, end_date_time = task

                # Apply rate limiting
                await self._apply_rate_limit()

                # Download data
                df, fail_reason = await self._download_data_async(ib, ticker, duration_str, end_date_time)

                # Check if connection is broken and needs reconnection
                if fail_reason == "CONNECTION_ERROR" and consecutive_errors > 2:
                    logger.warning(f"Thread {thread_id}: Reconnecting due to connection errors...")
                    try:
                        if ib and ib.isConnected():
                            await ib.disconnectAsync()
                        await asyncio.sleep(1)
                        ib = await self.create_connection_async(client_id)
                        if ib is None:
                            logger.error(f"Thread {thread_id}: Failed to reconnect")
                            return
                        consecutive_errors = 0
                    except Exception as e:
                        logger.error(f"Thread {thread_id}: Error reconnecting: {str(e)}")

                # Save data immediately if successful
                if df is not None:
                    yahoo_df = format_yahoo_style_data(df)
                    if yahoo_df is not None:
                        success, row_count = update_ticker_data(ticker, self.data_directory, yahoo_df)
                        if success:
                            with self.results_lock:
                                self.processed_tickers.append(ticker)
                                self.success_count += 1
                                self.processed_count += 1
                                consecutive_errors = 0
                        else:
                            with self.results_lock:
                                self.fail_count += 1
                                self.processed_count += 1
                                consecutive_errors += 1
                    else:
                        with self.results_lock:
                            self.fail_count += 1
                            self.processed_count += 1
                            consecutive_errors += 1
                elif fail_reason in ["UP_TO_DATE", "SECURITY_NOT_FOUND"]:
                    with self.results_lock:
                        self.processed_tickers.append(ticker)
                        self.success_count += 1
                        self.processed_count += 1
                        consecutive_errors = 0
                else:
                    with self.results_lock:
                        self.fail_count += 1
                        self.processed_count += 1
                        consecutive_errors += 1

                # Store result for tracking
                with self.results_lock:
                    self.results[ticker_index] = (ticker, df, fail_reason)

                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        success_rate = (self.success_count / self.processed_count * 100) if self.processed_count > 0 else 0
                        pbar.set_postfix(failed=self.fail_count, success_rate=f"{success_rate:.1f}%", last=ticker)

                self.requests_count += 1

            except Exception as e:
                logger.error(f"Thread {thread_id}: Error processing task: {str(e)}")
                with self.results_lock:
                    if 'ticker_index' in locals():
                        self.results[ticker_index] = (ticker, None, f"ERROR: {str(e)}")
                        self.fail_count += 1
                        self.processed_count += 1
                        if pbar:
                            pbar.update(1)
                consecutive_errors += 1
    
        # Cleanup - properly disconnect without causing destructor errors
        if ib:
            try:
                if ib.isConnected():
                    await ib.disconnectAsync()
                    await asyncio.sleep(0.1)
            except Exception:
                pass
            
            # Safely mark connection as disconnected to prevent destructor errors
            try:
                if hasattr(ib, 'client') and hasattr(ib.client, 'conn') and ib.client.conn is not None:
                    # Close the connection properly
                    try:
                        ib.client.conn.disconnect()
                    except:
                        pass
                    # Now it's safe to set to None
                    ib.client.conn = None
            except:
                pass
            
            # Clear the reference to help garbage collection
            del ib

    async def _apply_rate_limit(self):
        """Apply rate limiting to prevent overwhelming IBKR servers"""
        with self.rate_limit_lock:
            current_time = time.time()

            # If we have 50 requests in the last second, wait
            if len(self.last_request_times) >= 50:
                oldest_request = self.last_request_times[0]
                time_since_oldest = current_time - oldest_request

                if time_since_oldest < 1.0:
                    # Wait until we can make another request
                    wait_time = 1.0 - time_since_oldest
                    await asyncio.sleep(wait_time)

            self.last_request_times.append(time.time())

    async def _download_data_async(self, ib, ticker, duration_str, end_date_time):
        """Download historical data for a single ticker (async version)"""
        if not ib or not ib.isConnected():
            return None, "CONNECTION_FAILED"

        fallback_durations = ['2 Y', '1 Y', '6 M'] if duration_str in ['5 Y', '2 Y'] else [duration_str]

        for attempt_duration in fallback_durations:
            try:
                contract = get_contract(ticker)

                # Request contract details
                contract_details = await ib.reqContractDetailsAsync(contract)
                if not contract_details:
                    return None, "SECURITY_NOT_FOUND"

                # Request historical data
                bars = await ib.reqHistoricalDataAsync(
                    contract=contract,
                    endDateTime=end_date_time,
                    durationStr=attempt_duration,
                    barSizeSetting=DAILY_BAR_SIZE,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )

                if not bars:
                    logger.warning(f"No data returned for {ticker} with duration {attempt_duration}")
                    continue

                df = util.df(bars)

                if df.empty:
                    logger.warning(f"Empty DataFrame for {ticker} with duration {attempt_duration}")
                    continue

                # Rename columns
                df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'average': 'Average',
                    'barCount': 'BarCount'
                }, inplace=True)

                df['Date'] = pd.to_datetime(df['Date'])
                df['Ticker'] = ticker

                return df, None

            except Exception as e:
                error_message = str(e)

                if "pacing violation" in error_message.lower():
                    await asyncio.sleep(1)  # Brief pause for pacing violations
                    continue
                elif "security definition" in error_message.lower():
                    return None, "SECURITY_NOT_FOUND"
                elif "connection reset" in error_message.lower() or "timeout" in error_message.lower():
                    return None, "CONNECTION_ERROR"
                else:
                    logger.warning(f"Error downloading {ticker} with duration {attempt_duration}: {error_message}")
                    continue

        return None, "ALL_ATTEMPTS_FAILED"

    def start_workers(self, pbar=None):
        """Start all worker threads"""
        self.stop_event.clear()

        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker_thread, args=(i, pbar), daemon=True)
            thread.start()
            self.threads.append(thread)
            time.sleep(0.15)  # Small delay to stagger connection attempts 50ms was found online but 150 works better

    def add_task(self, ticker_index, ticker, duration_str, end_date_time):
        """Add a task to the deque"""
        with self.results_lock:
            self.task_queue.append((ticker_index, ticker, duration_str, end_date_time))
            self.total_tasks += 1

    def wait_for_completion(self):
        """Wait for all tasks to be processed"""
        while True:
            with self.results_lock:
                if self.processed_count >= self.total_tasks:
                    break
            time.sleep(0.5)

        # Stop all threads
        self.stop_event.set()

        # Give threads time to cleanup connections properly
        time.sleep(0.5)

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=10)

    def get_result(self, ticker_index):
        """Get result for a specific ticker index"""
        with self.results_lock:
            return self.results.get(ticker_index)

    def get_stats(self):
        """Get statistics about the fetcher"""
        return {
            "total_requests": self.requests_count,
            "connections_created": self.connections_created,
            "connections_failed": self.connections_failed,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "processed_count": self.processed_count,
            "total_tasks": self.total_tasks
        }

data_fetcher = None

@lru_cache(maxsize=1000)
def get_contract(ticker, security_type='STK', exchange='SMART', currency='USD'):
    return Contract(symbol=ticker, secType=security_type, exchange=exchange, currency=currency)

def get_last_date(file_path):
    try:
        if not os.path.exists(file_path):
            return None

        df = pd.read_parquet(file_path)
        if df.empty or 'Date' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['Date'])
        return df['Date'].max()

    except Exception as e:
        logger.error(f"Error reading last date from {file_path}: {str(e)}")
        return None

def calculate_missing_duration(last_date, target_end_date=None):
    """Simplified approach: always request overlapping data to ensure we get recent updates"""
    if last_date is None:
        return "2 Y"  # For new tickers, get 2 years

    end_date = target_end_date or datetime.now()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    days_missing = (end_date - last_date).days

    if days_missing <= 0:
        return None  # Data is current
    elif days_missing <= 30:
        return "60 D"  # Request 60 days to ensure we get the missing data + overlap
    elif days_missing <= 90:
        return "120 D"  # Request 120 days for good overlap
    else:
        return "6 M"  # For longer gaps, request 6 months

def merge_data(existing_df, new_df, ticker):
    if existing_df is None or existing_df.empty:
        if new_df is not None and not new_df.empty:
            new_df['Ticker'] = ticker
        return new_df

    if new_df is None or new_df.empty:
        return existing_df

    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df['Ticker'] = ticker

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    return combined_df

def update_ticker_data(ticker, data_directory, new_data_df=None):
    """Simplified update function - just merge and deduplicate"""
    file_path = os.path.join(data_directory, f"{ticker}.parquet")

    try:
        existing_df = None
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])

        # Merge data (this function already handles deduplication and sorting)
        updated_df = merge_data(existing_df, new_data_df, ticker)

        if updated_df is not None and not updated_df.empty:
            # Save the merged data
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            updated_df.to_parquet(file_path, index=False, compression='snappy')

            return True, len(updated_df)

        return False, 0

    except Exception as e:
        logger.error(f"Error updating {ticker}: {str(e)}")
        return False, 0

def format_yahoo_style_data(df):
    try:
        if df is None or df.empty:
            return None

        yahoo_df = df.copy()
        yahoo_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']

        if all(col in yahoo_df.columns for col in yahoo_columns):
            yahoo_df = yahoo_df[yahoo_columns]
        else:
            missing_cols = [col for col in yahoo_columns if col not in yahoo_df.columns]
            logger.warning(f"Missing columns in data: {missing_cols}")
            return None

        yahoo_df.sort_values('Date', inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        yahoo_df[numeric_cols] = yahoo_df[numeric_cols].round(4)

        return yahoo_df

    except Exception as e:
        logger.error(f"Error formatting data to Yahoo style: {str(e)}")
        return None

def save_progress(processed_tickers, total_tickers, remaining_tickers=None):
    try:
        progress = {
            'processed_tickers': processed_tickers,
            'total_tickers': total_tickers,
            'remaining_tickers': remaining_tickers if remaining_tickers else [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        temp_file = PROGRESS_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(progress, f)

        if os.path.exists(PROGRESS_FILE):
            os.replace(temp_file, PROGRESS_FILE)
        else:
            os.rename(temp_file, PROGRESS_FILE)
    except Exception as e:
        logger.error(f"Error saving progress: {str(e)}")

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return [], []

    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        print(f"Loaded progress: {len(progress['processed_tickers'])}/{progress.get('total_tickers', 0)} tickers processed")
        return progress['processed_tickers'], progress.get('remaining_tickers', [])
    except Exception as e:
        print(f"Could not load progress file: {str(e)}")
        logger.error(f"Could not load progress file: {str(e)}")
        return [], []

def find_latest_ticker_cik_file(directory):
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Latest TickerCIKs file found: {latest_file}")
    return latest_file

def get_existing_tickers(data_dir):
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []

    try:
        for file in os.listdir(data_dir):
            match = pattern.match(file)
            if match and match.group(1):
                ticker = match.group(1)
                if ticker and isinstance(ticker, str) and len(ticker) > 0:
                    tickers.append(ticker)
    except Exception as e:
        print(f"Error getting existing tickers: {str(e)}")

    tickers = [t for t in tickers if t]
    print(f"Found {len(tickers)} existing tickers in {data_dir} for RefreshMode.")
    return tickers

def process_all_tickers(tickers, host='127.0.0.1', port=7497, use_rth=True, resume=False, num_threads=NUM_THREADS):
    """Process all tickers using the multi-threaded deque approach"""
    global data_fetcher

    # Note: IB Gateway/TWS automatically cleans up disconnected client IDs
    # No manual cleanup needed - just use unique random client IDs

    processed_tickers = []
    remaining_tickers = tickers.copy()

    if resume:
        processed_tickers, loaded_remaining = load_progress()
        if loaded_remaining:
            remaining_tickers = loaded_remaining
        else:
            remaining_tickers = [t for t in tickers if t not in processed_tickers]

        print(f"Resuming download: {len(processed_tickers)} already processed, {len(remaining_tickers)} remaining")

    total_count = len(tickers)

    print(f"Processing {len(remaining_tickers)} tickers with {num_threads} threads")

    os.system('cls' if os.name == 'nt' else 'clear')

    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - MULTI-THREADED DEQUE VERSION")
    print("="*80)
    print(f"Tickers: {len(remaining_tickers)} pending, {len(processed_tickers)} completed")
    print(f"Threads: {num_threads} (keeping 50 requests in the air)")
    print("="*80)

    # Create the data fetcher with data directory for incremental saving
    data_fetcher = DataFetcher(host=host, port=port, num_threads=num_threads, data_directory=DATA_DIRECTORY)

    # Create progress bar
    pbar = tqdm(total=len(remaining_tickers), desc="Download Progress",
                unit="ticker", ncols=100, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    # Start worker threads
    data_fetcher.start_workers(pbar=pbar)

    # Start a background thread to periodically save progress
    progress_saver_stop = threading.Event()

    def save_progress_periodically():
        """Save progress every 30 seconds"""
        while not progress_saver_stop.is_set():
            time.sleep(30)
            with data_fetcher.results_lock:
                current_processed = processed_tickers + data_fetcher.processed_tickers
            save_progress(current_processed, total_count, [])

    progress_thread = threading.Thread(target=save_progress_periodically, daemon=True)
    progress_thread.start()

    # Add all tasks to the deque
    for idx, ticker in enumerate(remaining_tickers):
        if not ticker or not isinstance(ticker, str):
            continue

        # Calculate duration for this ticker
        file_path = os.path.join(DATA_DIRECTORY, f"{ticker}.parquet")
        last_date = get_last_date(file_path)
        duration_str = calculate_missing_duration(last_date)

        if duration_str is None:
            # Data is up to date, skip
            with data_fetcher.results_lock:
                data_fetcher.results[idx] = (ticker, None, "UP_TO_DATE")
                data_fetcher.success_count += 1
                data_fetcher.processed_count += 1
                data_fetcher.total_tasks += 1
            pbar.update(1)
            continue

        end_date_time = datetime.now().strftime('%Y%m%d-23:59:59')
        data_fetcher.add_task(idx, ticker, duration_str, end_date_time)

    # Wait for all tasks to complete
    data_fetcher.wait_for_completion()

    pbar.close()

    # Stop the progress saver thread
    progress_saver_stop.set()
    progress_thread.join(timeout=5)

    # Get final counts (data was saved incrementally in worker threads)
    print("\nFinalizing results...")
    with data_fetcher.results_lock:
        success_count = data_fetcher.success_count
        fail_count = data_fetcher.fail_count
        processed_tickers.extend(data_fetcher.processed_tickers)

    # Save final progress
    save_progress(processed_tickers, total_count, [])

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Total processed: {success_count + fail_count} tickers")
    success_rate = (success_count/(success_count+fail_count)*100) if (success_count+fail_count) > 0 else 0
    print(f"Successful: {success_count} ({success_rate:.1f}%)")
    print(f"Failed: {fail_count}")

    stats = data_fetcher.get_stats()
    print("\nDataFetcher stats:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Connections created: {stats['connections_created']}")
    print(f"Connections failed: {stats['connections_failed']}")
    print("="*80)

    return success_count, fail_count, total_count

def process_single_ticker(ticker, host='127.0.0.1', port=7497, use_rth=True):
    """Process a single ticker"""
    global data_fetcher

    print(f"Processing single ticker: {ticker}")

    # Create the data fetcher with just 1 thread
    data_fetcher = DataFetcher(host=host, port=port, num_threads=1)

    # Create progress bar
    pbar = tqdm(total=1, desc="Download Progress", unit="ticker")

    # Start worker thread
    data_fetcher.start_workers(pbar=pbar)

    # Calculate duration
    file_path = os.path.join(DATA_DIRECTORY, f"{ticker}.parquet")
    last_date = get_last_date(file_path)
    duration_str = calculate_missing_duration(last_date)

    if duration_str is None:
        print(f"{ticker}: Data is up to date")
        pbar.close()
        return True

    end_date_time = datetime.now().strftime('%Y%m%d-23:59:59')
    data_fetcher.add_task(0, ticker, duration_str, end_date_time)

    # Wait for completion
    data_fetcher.wait_for_completion()

    pbar.close()

    # Process result
    result = data_fetcher.get_result(0)
    if result:
        ticker_name, df, fail_reason = result

        if df is not None:
            yahoo_df = format_yahoo_style_data(df)
            if yahoo_df is not None:
                success, row_count = update_ticker_data(ticker_name, DATA_DIRECTORY, yahoo_df)

                # Display dataframe information
                if success:
                    ticker_file = os.path.join(DATA_DIRECTORY, f"{ticker}.parquet")
                    if os.path.exists(ticker_file):
                        try:
                            df_read = pd.read_parquet(ticker_file)
                            df_read['Date'] = pd.to_datetime(df_read['Date'])
                            df_sorted = df_read.sort_values('Date')

                            print(f"\n{'='*60}")
                            print(f"DATA SUMMARY FOR {ticker}")
                            print(f"{'='*60}")
                            print(f"Total records: {len(df_sorted)}")
                            print(f"Date range: {df_sorted['Date'].min().strftime('%Y-%m-%d')} to {df_sorted['Date'].max().strftime('%Y-%m-%d')}")
                            print(f"File location: {ticker_file}")

                            print(f"\nFirst 5 records:")
                            print("-" * 60)
                            display_df = df_sorted.head()
                            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                            print(display_df.to_string(index=False))

                            print(f"\nLast 5 records:")
                            print("-" * 60)
                            display_df_tail = df_sorted.tail()
                            display_df_tail['Date'] = display_df_tail['Date'].dt.strftime('%Y-%m-%d')
                            print(display_df_tail.to_string(index=False))
                            print(f"{'='*60}")

                        except Exception as e:
                            print(f"Error reading ticker data file: {str(e)}")
                            logger.error(f"Error reading ticker data file for {ticker}: {str(e)}")

                return success
        elif fail_reason == "UP_TO_DATE":
            print(f"{ticker}: Data is already up to date")
            return True
        elif fail_reason == "SECURITY_NOT_FOUND":
            print(f"{ticker}: Security not found in IBKR")
            return False

    return False

def main(logger):
    parser = argparse.ArgumentParser(description='Download stock data from IBKR with multi-threaded deque approach')

    parser.add_argument('--port', type=int, default=None, help='TWS/Gateway port (7497 for paper, 7496 for real)')
    parser.add_argument('--ibgateway', action='store_true', help='Use IB Gateway ports (4001 live / 4002 paper) instead of TWS (7496 live / 7497 paper)')
    parser.add_argument('--ticker', type=str, help='Single ticker symbol to process')
    parser.add_argument('--RefreshMode', action='store_true', help='Refresh existing data by appending the latest missing data')
    parser.add_argument('--ColdStart', action='store_true', help='Initial download of all tickers from the CIK file')
    parser.add_argument('--Resume', action='store_true', help='Resume from the last saved progress point')
    parser.add_argument('--num-threads', type=int, default=NUM_THREADS, help=f'Number of worker threads (default: {NUM_THREADS})')

    args = parser.parse_args()

    # Determine port based on --ibgateway flag
    if args.port is not None:
        port = args.port
    elif args.ibgateway:
        port = 4001  # IB Gateway default for live (4002 for paper)
    else:
        port = 7496  # TWS default for live (7497 for paper)

    os.system('cls' if os.name == 'nt' else 'clear')

    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - MULTI-THREADED DEQUE VERSION")
    print("="*80)
    print(f"Connection: {'IB Gateway' if args.ibgateway else 'TWS'}")
    print(f"Port: {port}")
    print(f"Threads: {args.num_threads}")
    print("="*80)

    tickers = []

    if args.ticker:
        tickers = [args.ticker]
        print(f"Processing single ticker: {args.ticker}")
    elif args.RefreshMode and args.ColdStart:
        print("Cannot use both --RefreshMode and --ColdStart simultaneously. Exiting.")
        exit(1)
    elif args.RefreshMode:
        print("Running in Refresh Mode: Refreshing data for tickers from RF_PREDICTIONS_DIRECTORY.")
        tickers = get_existing_tickers(RF_PREDICTIONS_DIRECTORY)
        if not tickers:
            print("No tickers found in RF_PREDICTIONS_DIRECTORY for RefreshMode. Exiting.")
            exit(1)
    elif args.ColdStart or args.Resume:
        print(f"Running in {'Resume' if args.Resume else 'ColdStart'} Mode: Downloading data for all tickers from the CIK file.")
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            print("No TickerCIKs file found. Exiting.")
            exit(1)
        tickers_df = pd.read_parquet(ticker_cik_file)
        if 'ticker' not in tickers_df.columns:
            print("The TickerCIKs file does not contain a 'ticker' column. Exiting.")
            exit(1)
        tickers = tickers_df['ticker'].dropna().unique().tolist()
    else:
        print("No mode selected. Please use --ticker, --RefreshMode, --ColdStart, or --Resume.")
        parser.print_help()
        exit(1)

    print(f"Found {len(tickers)} tickers to process")

    try:
        start_time = time.time()

        if len(tickers) == 1:
            success = process_single_ticker(
                ticker=tickers[0],
                host='127.0.0.1',
                port=port,
                use_rth=True
            )

            success_count = 1 if success else 0
            fail_count = 0 if success else 1
            total_count = 1
        else:
            # Process multiple tickers
            success_count, fail_count, total_count = process_all_tickers(
                tickers=tickers,
                host='127.0.0.1',
                port=port,
                use_rth=True,
                resume=args.Resume,
                num_threads=args.num_threads
            )

        elapsed_time = time.time() - start_time

        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Process completed in {elapsed_time/60:.2f} minutes")
        print(f"Total tickers: {total_count}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed/Skipped: {fail_count}")
        print(f"Success rate: {(success_count/total_count)*100:.2f}%")

        if success_count > 0:
            print(f"Data saved to {DATA_DIRECTORY}")
        else:
            print("No tickers were successfully processed")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Cleaning up...")
        if data_fetcher is not None:
            data_fetcher.stop_event.set()
        print("Cleanup complete. Exiting.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    logger = get_logger(script_name="2__PriceDownloader")
    main(logger)
