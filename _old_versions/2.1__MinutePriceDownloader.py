#!/usr/bin/env python

import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import warnings
from tqdm import tqdm
import glob
import re
import threading
import random
from collections import deque

warnings.filterwarnings('ignore')

from ib_insync import IB, Contract, util
import nest_asyncio
nest_asyncio.apply()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

BASE_DIRECTORY = script_dir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'IntradayData')
RF_PREDICTIONS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'RFpredictions')

REQUEST_TIMEOUT = 120
MIN_REQUEST_DELAY = 0.2

BAR_CONFIGS = {
    '5min': {
        'bar_size': '5 mins',
        'max_duration_days': 3,
        'description': '5-minute bars'
    },
    '15min': {
        'bar_size': '15 mins',
        'max_duration_days': 10,
        'description': '15-minute bars'
    },
    '30min': {
        'bar_size': '30 mins',
        'max_duration_days': 20,
        'description': '30-minute bars'
    },
    '1hour': {
        'bar_size': '1 hour',
        'max_duration_days': 30,
        'description': '1-hour bars'
    }
}

os.makedirs(DATA_DIRECTORY, exist_ok=True)


def check_ticker_data_status(ticker, bar_size_key, lookback_years=2.0):
    """Check if ticker already has up-to-date data"""
    filename = f"{ticker}_{bar_size_key}.parquet"
    filepath = os.path.join(DATA_DIRECTORY, filename)
    
    if not os.path.exists(filepath):
        return False, "MISSING"
    
    try:
        df = pd.read_parquet(filepath)
        
        if df.empty:
            return False, "EMPTY"
        
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        first_date = df['Date'].min()
        
        days_old = (datetime.now(last_date.tz if last_date.tz else None) - last_date).days
        
        if days_old > 3:
            return False, f"OUTDATED_{days_old}d"
        
        expected_start = datetime.now() - timedelta(days=int(lookback_years * 365))
        data_span_days = (last_date - first_date).days
        expected_span_days = int(lookback_years * 365)
        
        if data_span_days < (expected_span_days * 0.9):
            return False, f"INCOMPLETE_{data_span_days}d"
        
        return True, "CURRENT"
    
    except Exception as e:
        return False, f"ERROR_{str(e)[:20]}"


def get_tickers_from_rf_predictions(directory, limit=None):
    """Extract unique ticker symbols from RF predictions folder"""
    pattern = re.compile(r"^(.*?)(?:_\d{8})?\.parquet$")
    tickers = set()
    
    try:
        for file in os.listdir(directory):
            if file.endswith('.parquet'):
                match = pattern.match(file)
                if match and match.group(1):
                    ticker = match.group(1)
                    if ticker and isinstance(ticker, str) and len(ticker) > 0:
                        tickers.add(ticker)
    except Exception as e:
        print(f"Error reading RF predictions directory: {str(e)}")
        return []
    
    tickers_list = sorted(list(tickers))
    
    if limit and limit > 0:
        tickers_list = tickers_list[:limit]
    
    return tickers_list


class ProgressDisplay:
    """Thread-safe progress display for active downloads"""
    
    def __init__(self, total_tickers):
        self.active_downloads = {}
        self.lock = threading.Lock()
        self.total_tickers = total_tickers
        self.completed = 0
        self.failed = 0
        self.display_thread = None
        self.stop_display = threading.Event()
        self.last_update = ""
        self.queue_size = 0
        
    def update_download(self, thread_id, ticker, current_date, start_date, end_date):
        """Update progress for a specific thread's download"""
        with self.lock:
            self.active_downloads[thread_id] = {
                'ticker': ticker,
                'current_date': current_date,
                'start_date': start_date,
                'end_date': end_date
            }
    
    def clear_download(self, thread_id):
        """Clear download info when thread completes a ticker"""
        with self.lock:
            if thread_id in self.active_downloads:
                del self.active_downloads[thread_id]
    
    def increment_completed(self):
        """Increment completed counter"""
        with self.lock:
            self.completed += 1
    
    def increment_failed(self):
        """Increment failed counter"""
        with self.lock:
            self.failed += 1
    
    def update_queue_size(self, size):
        """Update queue size"""
        with self.lock:
            self.queue_size = size
    
    def start(self):
        """Start the display thread"""
        self.stop_display.clear()
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def stop(self):
        """Stop the display thread"""
        self.stop_display.set()
        if self.display_thread:
            self.display_thread.join(timeout=2)
    
    def _display_loop(self):
        """Display loop that updates console"""
        while not self.stop_display.is_set():
            self._update_display()
            time.sleep(0.5)
    
    def _update_display(self):
        """Update the console display"""
        with self.lock:
            lines = []
            lines.append("=" * 100)
            lines.append(f"Progress: {self.completed}/{self.total_tickers} completed | {self.failed} failed | {len(self.active_downloads)} active | Queue: {self.queue_size}")
            lines.append("-" * 100)
            
            if self.active_downloads:
                lines.append("Active Downloads:")
                for thread_id, info in sorted(self.active_downloads.items()):
                    ticker = info['ticker']
                    current_date = info['current_date']
                    start_date = info['start_date']
                    end_date = info['end_date']
                    
                    total_days = (end_date - start_date).days
                    elapsed_days = (end_date - current_date).days
                    
                    if total_days > 0:
                        progress_pct = (elapsed_days / total_days) * 100
                    else:
                        progress_pct = 0
                    
                    current_str = current_date.strftime('%Y-%m-%d')
                    start_str = start_date.strftime('%Y-%m-%d')
                    
                    lines.append(f"  Thread {thread_id:2d}: {ticker:8s} | {current_str} <- {start_str} | {progress_pct:5.1f}%")
            else:
                if self.queue_size > 0:
                    lines.append(f"Waiting for rate limit clearance... ({self.queue_size} tasks queued)")
                else:
                    lines.append("No tasks remaining...")
            
            lines.append("=" * 100)
            
            output = "\n".join(lines)
            
            if output != self.last_update:
                print("\033[2J\033[H", end="")
                print(output)
                self.last_update = output


class IntradayDataFetcher:
    """Multi-threaded intraday data fetcher"""
    
    def __init__(self, host='127.0.0.1', port=7496, num_threads=15, bar_size='30min', lookback_years=2.0):
        self.host = host
        self.port = port
        self.num_threads = num_threads
        self.bar_config = BAR_CONFIGS.get(bar_size, BAR_CONFIGS['30min'])
        self.bar_size_key = bar_size
        self.lookback_years = lookback_years
        
        self.task_queue = deque()
        self.results = {}
        self.results_lock = threading.Lock()
        
        self.success_count = 0
        self.fail_count = 0
        self.processed_count = 0
        self.total_tasks = 0
        
        self.threads = []
        self.stop_event = threading.Event()
        
        self.progress_display = None
        
        print(f"IntradayDataFetcher initialized")
        print(f"Bar size: {self.bar_config['bar_size']} ({self.bar_config['max_duration_days']} day chunks)")
        print(f"Threads: {num_threads}")
        print(f"Lookback: {lookback_years} years")
        print(f"Note: IBKR will naturally throttle requests if needed")
        print("=" * 100)
    
    def worker_thread(self, thread_id):
        """Worker thread processing tickers"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(lambda loop, context: None)
        
        try:
            loop.run_until_complete(self._worker_thread_async(thread_id))
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0.1))
            except:
                pass
            try:
                loop.close()
            except:
                pass
    
    async def _worker_thread_async(self, thread_id):
        """Async worker implementation"""
        client_id = random.randint(1000, 9000) + thread_id
        ib = None
        retry_count = 0
        max_connection_retries = 3
        
        while retry_count < max_connection_retries and not self.stop_event.is_set():
            ib = await self._create_connection(client_id)
            if ib is not None:
                break
            retry_count += 1
            await asyncio.sleep(2 * retry_count)
        
        if ib is None:
            return
        
        while not self.stop_event.is_set():
            ticker_idx = None
            ticker = None
            
            try:
                task = None
                with self.results_lock:
                    if len(self.task_queue) > 0:
                        task = self.task_queue.popleft()
                        if self.progress_display:
                            self.progress_display.update_queue_size(len(self.task_queue))
                
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                ticker_idx, ticker = task
                
                df, fail_reason = await self._download_ticker_data(ib, ticker, thread_id)
                
                if self.progress_display:
                    self.progress_display.clear_download(thread_id)
                
                success = False
                if df is not None and not df.empty:
                    success = self._save_ticker_data(ticker, df)
                
                with self.results_lock:
                    self.results[ticker_idx] = (ticker, success, fail_reason, len(df) if df is not None else 0)
                    self.processed_count += 1
                    
                    if success or fail_reason in ["UP_TO_DATE", "SECURITY_NOT_FOUND"]:
                        self.success_count += 1
                        if self.progress_display:
                            self.progress_display.increment_completed()
                    else:
                        self.fail_count += 1
                        if self.progress_display:
                            self.progress_display.increment_failed()
            
            except Exception as e:
                if self.progress_display:
                    self.progress_display.clear_download(thread_id)
                
                with self.results_lock:
                    if ticker_idx is not None and ticker is not None:
                        self.results[ticker_idx] = (ticker, False, f"ERROR: {str(e)}", 0)
                        self.fail_count += 1
                        self.processed_count += 1
                        if self.progress_display:
                            self.progress_display.increment_failed()
        
        if ib:
            try:
                if ib.isConnected():
                    await ib.disconnectAsync()
                    await asyncio.sleep(0.1)
            except:
                pass
            finally:
                try:
                    if hasattr(ib, 'client') and hasattr(ib.client, 'conn') and ib.client.conn:
                        ib.client.conn.disconnect()
                        ib.client.conn = None
                except:
                    pass
    
    async def _create_connection(self, client_id):
        """Create IB connection"""
        ib = IB()
        
        def handle_error(reqId, errorCode, errorString, contract):
            if errorCode not in [200, 2104, 2106, 2158, 2119]:
                pass
        
        try:
            await ib.connectAsync(self.host, self.port, clientId=client_id, timeout=REQUEST_TIMEOUT)
            ib.errorEvent += handle_error
            
            if ib.isConnected():
                return ib
            else:
                return None
        
        except Exception:
            return None
    
    async def _download_ticker_data(self, ib, ticker, thread_id):
        """Download all data for a single ticker"""
        if not ib or not ib.isConnected():
            return None, "CONNECTION_FAILED"
        
        try:
            contract = Contract(symbol=ticker, secType='STK', exchange='SMART', currency='USD')
            
            contract_details = await ib.reqContractDetailsAsync(contract)
            if not contract_details:
                return None, "SECURITY_NOT_FOUND"
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(self.lookback_years * 365))
            
            all_bars = []
            current_end = end_date
            max_days_per_chunk = self.bar_config['max_duration_days']
            
            while current_end > start_date and not self.stop_event.is_set():
                current_start = max(start_date, current_end - timedelta(days=max_days_per_chunk))
                chunk_days = (current_end - current_start).days
                
                if chunk_days < 1:
                    break
                
                if self.progress_display:
                    self.progress_display.update_download(thread_id, ticker, current_start, start_date, end_date)
                
                duration_str = f"{chunk_days} D"
                
                try:
                    bars = await ib.reqHistoricalDataAsync(
                        contract=contract,
                        endDateTime=current_end.strftime('%Y%m%d %H:%M:%S US/Eastern'),
                        durationStr=duration_str,
                        barSizeSetting=self.bar_config['bar_size'],
                        whatToShow='TRADES',
                        useRTH=False,
                        formatDate=1,
                        timeout=REQUEST_TIMEOUT
                    )
                    
                    if bars:
                        all_bars.extend(bars)
                    
                    current_end = current_start
                    
                    if current_end > start_date:
                        await asyncio.sleep(MIN_REQUEST_DELAY)
                
                except Exception as chunk_error:
                    error_msg = str(chunk_error)
                    if "pacing violation" in error_msg.lower():
                        await asyncio.sleep(2)
                        continue
                    elif "timeout" in error_msg.lower():
                        current_end = current_start
                        continue
                    else:
                        break
            
            if not all_bars:
                return None, "NO_DATA"
            
            df = util.df(all_bars)
            
            if df.empty:
                return None, "EMPTY_DATA"
            
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
            df = df.drop_duplicates(subset=['Date'], keep='last')
            df = df.sort_values('Date').reset_index(drop=True)
            
            return df, None
        
        except Exception as e:
            error_message = str(e)
            
            if "security definition" in error_message.lower():
                return None, "SECURITY_NOT_FOUND"
            else:
                return None, f"ERROR: {error_message}"
    
    def _save_ticker_data(self, ticker, df):
        """Save ticker data to parquet file"""
        try:
            filename = f"{ticker}_{self.bar_size_key}.parquet"
            filepath = os.path.join(DATA_DIRECTORY, filename)
            
            if df is None or df.empty:
                return False
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df.to_parquet(filepath, index=False, compression='snappy')
            
            return True
        
        except Exception:
            return False
    
    def start_workers(self):
        """Start worker threads"""
        self.stop_event.clear()
        
        for i in range(self.num_threads):
            thread = threading.Thread(target=self.worker_thread, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)
            time.sleep(0.1)
    
    def add_task(self, ticker_idx, ticker):
        """Add task to queue"""
        with self.results_lock:
            self.task_queue.append((ticker_idx, ticker))
            self.total_tasks += 1
            if self.progress_display:
                self.progress_display.update_queue_size(len(self.task_queue))
    
    def wait_for_completion(self):
        """Wait for all tasks to complete"""
        while True:
            with self.results_lock:
                if self.processed_count >= self.total_tasks:
                    break
            time.sleep(1)
        
        self.stop_event.set()
        time.sleep(1)
        
        for thread in self.threads:
            thread.join(timeout=15)
    
    def get_result(self, ticker_idx):
        """Get result for task"""
        with self.results_lock:
            return self.results.get(ticker_idx)


def main():
    parser = argparse.ArgumentParser(description='Download intraday data for tickers from RF predictions')
    
    parser.add_argument('--ticker', type=str,
                       help='Single ticker symbol (overrides RF predictions folder)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of tickers to process (for testing)')
    parser.add_argument('--port', type=int, default=7496,
                       help='TWS/Gateway port (7497 for paper, 7496 for real)')
    parser.add_argument('--bar-size', type=str, default='30min',
                       choices=['5min', '15min', '30min', '1hour'],
                       help='Bar size (default: 30min)')
    parser.add_argument('--lookback-years', type=float, default=2.0,
                       help='Years of data to download (default: 2.0)')
    parser.add_argument('--num-threads', type=int, default=15,
                       help='Number of worker threads (default: 15)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files are up-to-date')
    
    args = parser.parse_args()
    
    print("\nIMPORTANT: Make sure TWS or IB Gateway is running and connected!")
    print(f"Using {args.num_threads} parallel connections for faster downloads.\n")
    
    try:
        if args.ticker:
            tickers = [args.ticker]
            print(f"Processing single ticker: {args.ticker}\n")
        else:
            tickers = get_tickers_from_rf_predictions(RF_PREDICTIONS_DIRECTORY, limit=args.limit)
            
            if not tickers:
                print("No tickers found in RF predictions folder. Exiting.")
                return
            
            if args.limit:
                print(f"Limited to first {args.limit} tickers\n")
        
        bar_config = BAR_CONFIGS[args.bar_size]
        
        print("=" * 100)
        print("IBKR INTRADAY DATA DOWNLOADER - MULTI-THREADED")
        print("=" * 100)
        print(f"Source: RF Predictions folder")
        print(f"Total tickers found: {len(tickers)}")
        print(f"Bar size: {bar_config['bar_size']} ({bar_config['max_duration_days']} day chunks)")
        print(f"Lookback: {args.lookback_years} years")
        print(f"Threads: {args.num_threads} parallel connections")
        print(f"Output: {DATA_DIRECTORY}")
        print("=" * 100)
        
        print("\nChecking existing data files...")
        tickers_to_process = []
        tickers_skipped = []
        status_counts = {}
        
        if args.force:
            print("Force mode enabled - will re-download all tickers")
            tickers_to_process = tickers
            status_counts['FORCE_REDOWNLOAD'] = len(tickers)
        else:
            for ticker in tickers:
                is_current, status = check_ticker_data_status(ticker, args.bar_size, args.lookback_years)
                
                if is_current:
                    tickers_skipped.append((ticker, status))
                else:
                    tickers_to_process.append(ticker)
                
                status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nData Status Summary:")
        print(f"  Already up-to-date: {len(tickers_skipped)}")
        print(f"  Need processing: {len(tickers_to_process)}")
        
        if status_counts:
            print(f"\nStatus breakdown:")
            for status, count in sorted(status_counts.items()):
                print(f"  {status}: {count}")
        
        if tickers_skipped and len(tickers_skipped) <= 10:
            print(f"\nSkipped tickers: {', '.join([t[0] for t in tickers_skipped])}")
        
        if not tickers_to_process:
            print("\nAll tickers are up-to-date. Nothing to download.")
            print("Use --force to re-download existing data.")
            return
        
        print(f"\nQueueing {len(tickers_to_process)} tickers for download...")
        print("=" * 100)
        
        fetcher = IntradayDataFetcher(
            host='127.0.0.1',
            port=args.port,
            num_threads=args.num_threads,
            bar_size=args.bar_size,
            lookback_years=args.lookback_years
        )
        
        progress_display = ProgressDisplay(len(tickers_to_process))
        fetcher.progress_display = progress_display
        
        print("\nAdding tasks to queue...")
        task_count = 0
        for idx, ticker in enumerate(tickers_to_process):
            if not ticker or not isinstance(ticker, str):
                continue
            fetcher.add_task(idx, ticker)
            task_count += 1
        
        print(f"Added {task_count} tasks to queue")
        print(f"Starting {args.num_threads} worker threads...")
        
        fetcher.start_workers()
        progress_display.start()
        
        start_time = time.time()
        fetcher.wait_for_completion()
        elapsed_time = time.time() - start_time
        
        progress_display.stop()
        
        print("\n" + "=" * 100)
        print("DOWNLOAD COMPLETE")
        print("=" * 100)
        print(f"Total tickers checked: {len(tickers)}")
        print(f"Already up-to-date: {len(tickers_skipped)}")
        print(f"Processed: {len(tickers_to_process)}")
        success_rate = (fetcher.success_count / len(tickers_to_process) * 100) if len(tickers_to_process) > 0 else 0
        print(f"Successful downloads: {fetcher.success_count} ({success_rate:.1f}%)")
        print(f"Failed downloads: {fetcher.fail_count}")
        print(f"Time taken: {elapsed_time/60:.2f} minutes")
        if len(tickers_to_process) > 0:
            print(f"Average time per ticker: {elapsed_time/len(tickers_to_process):.1f} seconds")
        print(f"Data saved to: {DATA_DIRECTORY}")
        print("=" * 100)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if 'progress_display' in locals():
            progress_display.stop()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'progress_display' in locals():
            progress_display.stop()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()