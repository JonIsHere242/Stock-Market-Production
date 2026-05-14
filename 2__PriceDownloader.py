#!/usr/bin/env python

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
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
DEFAULT_BATCH_SIZE = 8
MAX_WORKERS_PER_BATCH = 32
REQUEST_DELAY = 0.1
BATCH_PAUSE = 0.5
CONNECTION_POOL_SIZE = 8
REQUEST_TIMEOUT = 10

os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(RF_PREDICTIONS_DIRECTORY, exist_ok=True) 

class ConnectionPool:
    def __init__(self, host='127.0.0.1', port=7497, max_connections=CONNECTION_POOL_SIZE):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.connection_queue = deque()
        self.active_connections = {}
        self.connection_count = 0
        self.lock = asyncio.Lock()
        self.next_client_id = random.randint(1000, 8000)
        
        self.requests_count = 0
        self.connections_created = 0
        self.connections_failed = 0
        self.requests_per_connection = {}
        self.connection_health = {}
        self.consecutive_failures = 0
        self.last_reconnect_time = time.time()
        
        print(f"Connection pool initialized with max {max_connections} connections")
        print(f"Host: {host}, Port: {port}")
        print("=" * 80)
        
    async def get_connection(self):
        async with self.lock:
            await self.reconnect_strategy()
            
            while self.connection_queue:
                tag, ib = self.connection_queue.popleft()
                
                if ib is not None and ib.isConnected():
                    if tag not in self.connection_health:
                        self.connection_health[tag] = {'failures': 0, 'successes': 0}
                    self.connection_health[tag]['successes'] += 1
                    return tag, ib
                else:
                    try:
                        if ib is not None:
                            await ib.disconnectAsync()
                    except:
                        pass
                    
                    if tag in self.connection_health:
                        self.connection_health[tag]['failures'] += 1
                    
                    if tag in self.active_connections:
                        del self.active_connections[tag]
                    if tag in self.requests_per_connection:
                        del self.requests_per_connection[tag]
            
            return await self._create_new_connection()
    
    async def _create_new_connection(self):
        client_id = self.next_client_id
        self.next_client_id = (self.next_client_id + 1) % 9000 + 1000
        tag = f"conn_{self.connection_count}"
        self.connection_count += 1
        
        ib = IB()
        
        def handle_error(reqId, errorCode, errorString, contract):
            """Custom error handler to suppress contract not found errors"""
            if errorCode == 200 and "No security definition" in errorString:
                # Log contract not found errors instead of printing
                logger.info(f"Contract not found (reqId {reqId}): {errorString}")
            else:
                # Log other errors with appropriate level
                logger.error(f"Error {errorCode}, reqId {reqId}: {errorString}")
        
        try:
            await ib.connectAsync(self.host, self.port, clientId=client_id, timeout=REQUEST_TIMEOUT)
            
            # Connect the custom error handler
            ib.errorEvent += handle_error
            
            if not ib.isConnected():
                raise Exception("Connection failed to establish")
                
            self.connections_created += 1
            self.active_connections[tag] = ib
            self.requests_per_connection[tag] = 0
            self.connection_health[tag] = {'failures': 0, 'successes': 1}
            self.consecutive_failures = 0
            
            return tag, ib
            
        except Exception as e:
            self.connections_failed += 1
            self.consecutive_failures += 1
            error_msg = str(e)
            logger.error(f"Failed to create connection {tag}: {error_msg}")
            tqdm.write(f"✗ Failed to create connection {tag}: {error_msg}")
            
            try:
                if ib:
                    await ib.disconnectAsync()
            except:
                pass
                
            if "already in use" in error_msg.lower():
                await asyncio.sleep(2.0)
            elif "timeout" in error_msg.lower():
                await asyncio.sleep(5.0)
            else:
                await asyncio.sleep(1.0)
                
            if self.connections_failed > 200:
                tqdm.write("⚠️ Too many connection failures. Check your TWS/Gateway settings or restart it.")
                raise Exception("Too many connection failures")
                
            return await self._create_new_connection()
    
    async def release_connection(self, tag, ib, force_disconnect=False):
        if ib is None:
            return
            
        async with self.lock:
            if not tag or not isinstance(tag, str):
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
                
            if not ib.isConnected() or force_disconnect:
                if tag in self.active_connections:
                    del self.active_connections[tag]
                if tag in self.requests_per_connection:
                    del self.requests_per_connection[tag]
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
            
            if len(self.connection_queue) >= self.max_connections:
                if tag in self.active_connections:
                    del self.active_connections[tag]
                if tag in self.requests_per_connection:
                    del self.requests_per_connection[tag]
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
            
            self.connection_queue.append((tag, ib))
    
    async def record_request(self, tag):
        async with self.lock:
            self.requests_count += 1
            if tag in self.requests_per_connection:
                self.requests_per_connection[tag] += 1
    
    async def close_all(self):
        async with self.lock:
            while self.connection_queue:
                tag, ib = self.connection_queue.popleft()
                try:
                    await ib.disconnectAsync()
                except:
                    pass
            
            for tag, ib in list(self.active_connections.items()):
                try:
                    await ib.disconnectAsync()
                except:
                    pass
            
            self.active_connections = {}
            self.requests_per_connection = {}
            
    def get_stats(self):
        return {
            "total_requests": self.requests_count,
            "connections_created": self.connections_created,
            "connections_failed": self.connections_failed,
            "active_connections": len(self.active_connections),
            "queued_connections": len(self.connection_queue),
            "requests_per_connection": self.requests_per_connection
        }

    async def reconnect_strategy(self):
        current_time = time.time()
        
        if self.consecutive_failures >= 3:
            cooldown = min(30, self.consecutive_failures * 2)
            
            if current_time - self.last_reconnect_time < cooldown:
                logger.warning(f"Connection cooldown: waiting {cooldown}s before new connection attempts")
                await asyncio.sleep(cooldown / 2)
                return True
            else:
                self.last_reconnect_time = current_time
        
        return False

connection_pool = None

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
            #print(f"{ticker}: Found {len(existing_df)} existing records")
        
        #if new_data_df is not None and not new_data_df.empty:
            #print(f"{ticker}: Downloaded {len(new_data_df)} new records")
        
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





async def download_historical_data(connection_info, ticker, duration_str, use_rth=True, end_date_time=''):
    tag, ib = connection_info
    fail_reason = None
    
    fallback_durations = ['2 Y', '1 Y', '6 M'] if duration_str == '5 Y' or duration_str == '2 Y' else [duration_str]
    
    for attempt_duration in fallback_durations:
        try:
            contract = get_contract(ticker)
            await connection_pool.record_request(tag)
            
            contract_details = await ib.reqContractDetailsAsync(contract)
            if not contract_details:
                return None, "No contract details found"
            
            await asyncio.sleep(0.1)
            
            bars = await ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_date_time,
                durationStr=attempt_duration,
                barSizeSetting=DAILY_BAR_SIZE,
                whatToShow='TRADES',
                useRTH=use_rth,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No data returned for {ticker} with duration {attempt_duration}")
                continue
            
            df = util.df(bars)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker} with duration {attempt_duration}")
                continue
            
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
            logger.warning(f"Error downloading {ticker} with duration {attempt_duration}: {error_message}")
            
            if "pacing violation" in error_message.lower():
                fail_reason = "PACING"
                continue
            elif "security definition" in error_message.lower():
                return None, "SECURITY_NOT_FOUND"
            elif "connection reset" in error_message.lower() or "timeout" in error_message.lower():
                fail_reason = "CONNECTION"
                continue
            elif "session is connected from a different IP address" in error_message:
                fail_reason = "SESSION"
                return None, fail_reason
            else:
                fail_reason = "OTHER"
                continue
    
    return None, fail_reason or "ALL_ATTEMPTS_FAILED"




async def download_incremental_data(connection_info, ticker, data_directory, use_rth=True, target_end_date=None):
    """Simplified incremental download with overlap strategy"""
    file_path = os.path.join(data_directory, f"{ticker}.parquet")
    
    last_date = get_last_date(file_path)
    duration_str = calculate_missing_duration(last_date, target_end_date)
    
    if duration_str is None:
        #print(f"{ticker}: Data is up to date")
        return None, "UP_TO_DATE"
    
    #if last_date is not None:
        #days_missing = (datetime.now().date() - last_date.date()).days

        #print(f"{ticker}: {days_missing} days missing, requesting {duration_str} with overlap")
    #else:
        #print(f"{ticker}: New ticker, requesting {duration_str} of historical data")
    
    # Always use current time as end date for incremental updates
    end_date_time = datetime.now().strftime('%Y%m%d-23:59:59')
    
    return await download_historical_data(
        connection_info=connection_info,
        ticker=ticker,
        duration_str=duration_str,
        use_rth=use_rth,
        end_date_time=end_date_time
    )









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

def save_progress(processed_tickers, total_tickers, remaining_tickers=None, pbar=None):
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




async def process_ticker_incremental(ticker, data_directory, use_rth=True, max_retries=3):
    """Simplified ticker processing"""
    global connection_pool
    
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return False
    
    retry_count = 0
    success = False
    connection_established = False
    
    while retry_count <= max_retries and not success:
        try:
            connection_info = await connection_pool.get_connection()
            tag, ib = connection_info
            
            if not ib or not ib.isConnected():
                logger.warning(f"Got invalid connection for {ticker}, retrying...")
                retry_count += 1
                await asyncio.sleep(1.0)
                continue
            
            connection_established = True
            
            df, fail_reason = await download_incremental_data(
                connection_info=connection_info,
                ticker=ticker,
                data_directory=data_directory,
                use_rth=use_rth
            )
            
            if fail_reason == "SECURITY_NOT_FOUND":
                logger.info(f"{ticker}: Security not found in IBKR")
                await connection_pool.release_connection(tag, ib)
                return False
            
            if fail_reason == "UP_TO_DATE":
                success = True
            elif df is not None:
                yahoo_df = format_yahoo_style_data(df)
                if yahoo_df is not None:
                    success, row_count = update_ticker_data(ticker, data_directory, yahoo_df)
                    if not success:
                        logger.warning(f"{ticker}: Failed to save data")
            else:
                logger.info(f"{ticker}: No data retrieved, reason: {fail_reason}")
                success = False
            
            await connection_pool.release_connection(tag, ib)
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            connection_established = False
            if 'tag' in locals() and 'ib' in locals():
                try:
                    await connection_pool.release_connection(tag, ib, force_disconnect=True)
                except:
                    pass
        
        if not success:
            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(min(retry_count * 1.5, 4))
    
    if not connection_established:
        logger.warning(f"{ticker}: Failed to establish connection to IBKR")
        return False
        
    return success






async def process_batch(tickers, data_directory, use_rth=True, pbar=None, semaphore=None):
    success_count = 0
    fail_count = 0
    
    async def process_with_semaphore(ticker):
        try:
            async with semaphore:
                success = await process_ticker_incremental(
                    ticker=ticker,
                    data_directory=data_directory,
                    use_rth=use_rth
                )
                
                if pbar:
                    pbar.update(1)
                    postfix_str = str(pbar.postfix) if hasattr(pbar, 'postfix') and pbar.postfix else ""
                    current_fails = 0
                    if postfix_str and "failed=" in postfix_str:
                        try:
                            current_fails = int(postfix_str.split('failed=')[1].split(',')[0])
                        except (IndexError, ValueError):
                            current_fails = 0
                    
                    if not success:
                        current_fails += 1
                    success_rate = (pbar.n - current_fails) / pbar.n * 100 if pbar.n > 0 else 0
                    pbar.set_postfix(failed=current_fails, success_rate=f"{success_rate:.1f}%", last=ticker)
                
                return success
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            if pbar:
                pbar.update(1)
            return False
    
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_WORKERS_PER_BATCH)
    
    tasks = []
    for ticker in tickers:
        if ticker:
            tasks.append(process_with_semaphore(ticker))
    
    if not tasks:
        return 0, 0
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            fail_count += 1
            logger.error(f"Task failed with exception: {result}")
        elif result is True:
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

async def process_all_tickers(tickers, host='127.0.0.1', port=7497, batch_size=DEFAULT_BATCH_SIZE, 
                             use_rth=True, resume=False, max_workers=MAX_WORKERS_PER_BATCH):
    global connection_pool
    
    connection_pool = ConnectionPool(host=host, port=port, max_connections=CONNECTION_POOL_SIZE)
    
    processed_tickers = []
    remaining_tickers = tickers.copy()
    
    if resume:
        processed_tickers, loaded_remaining = load_progress()
        if loaded_remaining:
            remaining_tickers = loaded_remaining
        else:
            remaining_tickers = [t for t in tickers if t not in processed_tickers]
        
        print(f"Resuming download: {len(processed_tickers)} already processed, {len(remaining_tickers)} remaining")
    
    success_count = len(processed_tickers)
    fail_count = 0
    total_count = len(tickers)
    
    print(f"Processing {len(remaining_tickers)} tickers in batches of {batch_size}")
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - INCREMENTAL UPDATE VERSION")
    print("="*80)
    print(f"Tickers: {len(remaining_tickers)} pending, {len(processed_tickers)} completed")
    print(f"Batch size: {batch_size}, Workers: {max_workers}")
    print(f"Connection pool size: {CONNECTION_POOL_SIZE}")
    print("="*80)
    
    pbar = tqdm(total=len(remaining_tickers), desc="Download Progress", 
                unit="ticker", ncols=100, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    semaphore = asyncio.Semaphore(max_workers)
    
    i = 0
    while i < len(remaining_tickers):
        try:
            batch = remaining_tickers[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size
            
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            
            batch_success, batch_fail = await process_batch(
                tickers=batch,
                data_directory=DATA_DIRECTORY,
                use_rth=use_rth,
                pbar=pbar,
                semaphore=semaphore
            )
            
            success_count += batch_success
            fail_count += batch_fail
            
            batch_processed = [t for idx, t in enumerate(batch) if idx < batch_success]
            processed_tickers.extend(batch_processed)
            remaining_after_batch = remaining_tickers[i+batch_size:]
            
            save_progress(processed_tickers, total_count, remaining_after_batch)
            
            i += batch_size
            
            if i < len(remaining_tickers):
                await asyncio.sleep(BATCH_PAUSE)
                
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {str(e)}")
            logger.error(traceback.format_exc())
            save_progress(processed_tickers, total_count, remaining_tickers[i:])
            semaphore = asyncio.Semaphore(max_workers)
            await connection_pool.close_all()
            connection_pool = ConnectionPool(host=host, port=port, max_connections=CONNECTION_POOL_SIZE)
            await asyncio.sleep(5)
            continue
    
    pbar.close()
    await connection_pool.close_all()
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Total processed: {success_count + fail_count} tickers")
    success_rate = (success_count/(success_count+fail_count)*100) if (success_count+fail_count) > 0 else 0
    print(f"Successful: {success_count} ({success_rate:.1f}%)")
    print(f"Failed: {fail_count}")
    
    stats = connection_pool.get_stats()
    print("\nConnection pool stats:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Connections created: {stats['connections_created']}")
    print(f"Connections failed: {stats['connections_failed']}")
    print("="*80)
    
    return success_count, fail_count, total_count



def main(logger):
    parser = argparse.ArgumentParser(description='Download stock data from IBKR with incremental updates')
    
    parser.add_argument('--port', type=int, default=7497, help='TWS/Gateway port (7497 for paper, 7496 for real)')
    parser.add_argument('--ticker', type=str, help='Single ticker symbol to process')
    parser.add_argument('--RefreshMode', action='store_true', help='Refresh existing data by appending the latest missing data')
    parser.add_argument('--ColdStart', action='store_true', help='Initial download of all tickers from the CIK file')
    parser.add_argument('--Resume', action='store_true', help='Resume from the last saved progress point')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Number of tickers to process in each batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS_PER_BATCH, help=f'Maximum number of concurrent workers per batch (default: {MAX_WORKERS_PER_BATCH})')
    parser.add_argument('--pool-size', type=int, default=CONNECTION_POOL_SIZE, help=f'Size of the connection pool (default: {CONNECTION_POOL_SIZE})')
    parser.add_argument('--request-delay', type=float, default=REQUEST_DELAY, help=f'Delay between requests in seconds (default: {REQUEST_DELAY})')
    parser.add_argument('--batch-pause', type=float, default=BATCH_PAUSE, help=f'Pause between batches in seconds (default: {BATCH_PAUSE})')
    
    args = parser.parse_args()
    
    globals()['REQUEST_DELAY'] = args.request_delay
    globals()['BATCH_PAUSE'] = args.batch_pause
    globals()['CONNECTION_POOL_SIZE'] = args.pool_size
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - INCREMENTAL UPDATE VERSION")
    print("="*80)
    print(f"Port: {args.port}")
    print(f"Batch size: {args.batch_size}, Workers: {args.max_workers}")
    print(f"Connection pool size: {args.pool_size}")
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
            print(f"Processing single ticker {tickers[0]}...")
            
            global connection_pool
            connection_pool = ConnectionPool(host='127.0.0.1', port=args.port, max_connections=CONNECTION_POOL_SIZE)
            
            try:
                success = asyncio.run(process_ticker_incremental(
                    ticker=tickers[0],
                    data_directory=DATA_DIRECTORY,
                    use_rth=True
                ))
                
                asyncio.run(connection_pool.close_all())
                
                # Display dataframe information for single ticker
                if success:
                    ticker_file = os.path.join(DATA_DIRECTORY, f"{tickers[0]}.parquet")
                    if os.path.exists(ticker_file):
                        try:
                            df = pd.read_parquet(ticker_file)
                            df['Date'] = pd.to_datetime(df['Date'])
                            df_sorted = df.sort_values('Date')
                            
                            print(f"\n{'='*60}")
                            print(f"DATA SUMMARY FOR {tickers[0]}")
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
                            logger.error(f"Error reading ticker data file for {tickers[0]}: {str(e)}")
                    else:
                        print(f"Data file not found for {tickers[0]} at {ticker_file}")
                else:
                    print(f"Failed to process ticker {tickers[0]}")
                
                success_count = 1 if success else 0
                fail_count = 0 if success else 1
                total_count = 1
                
            except Exception as e:
                print(f"Error processing ticker {tickers[0]}: {str(e)}")
                traceback.print_exc()
                success_count = 0
                fail_count = 1
                total_count = 1
                
                try:
                    asyncio.run(connection_pool.close_all())
                except:
                    pass
        else:
            # Process multiple tickers
            success_count, fail_count, total_count = asyncio.run(process_all_tickers(
                tickers=tickers,
                host='127.0.0.1',
                port=args.port,
                batch_size=args.batch_size,
                use_rth=True,
                resume=args.Resume,
                max_workers=args.max_workers
            ))
        
        
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
        if 'connection_pool' in globals() and connection_pool is not None:
            asyncio.run(connection_pool.close_all())
        print("Cleanup complete. Exiting.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    
    finally:
        if 'connection_pool' in globals() and connection_pool is not None:
            try:
                asyncio.run(connection_pool.close_all())
                print("All connections closed.")
            except:
                print("Error while closing connections.")


if __name__ == "__main__":
    logger = get_logger(script_name="2__BulkPriceDownloader")
    main(logger)