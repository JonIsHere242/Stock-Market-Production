#!/usr/bin/env python 
import os
import time
import logging
import argparse
import random
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import pyarrow.parquet as pq
import multiprocessing
from numba import njit
import traceback
from collections import Counter
import pandas_market_calendars as mcal
import math
import glob
import warnings
import yfinance as yf
import concurrent.futures
from collections import defaultdict
from scipy.stats import spearmanr
import scipy.stats as stats
from Util import *

from Util import (
    STRATEGY_PARAMS_TUPLE as STRATEGY_PARAMS,  # Note we're using the tuple version for backtrader
)

# Silence the verbose dprint output — file logging still works, console is quiet
def dprint(*_):  # noqa: silence verbose debug output, file logging still active
    pass

class IBKRAdaptiveCommission(bt.CommInfoBase):

    """
    Interactive Brokers Adaptive Commission for Backtrader
    Fixed version with correct method signatures and no Unicode characters
    """

    params = (
        ('commission_per_share', 0.0035),  # $0.0035 per share  
        ('min_per_order', 0.35),           # $0.35 minimum per order
        ('max_per_order_pct', 0.01),       # 1.0% cap of trade value
        ('exchange_fees', 0.0002),         # $0.0002 per share for SEC/TAF/etc
        ('partial_fill_min', 0.35),        # $0.35 minimum for partial fills
        
        # Standard commission info params
        ('stocklike', True),               # Stock-like instrument
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Use FIXED commission type
        ('percabs', False),                # Commission is absolute, not percentage
    )

    def __init__(self):
        super(IBKRAdaptiveCommission, self).__init__()
        # Debug tracking
        self.total_commission_charged = 0.0
        self.trade_count = 0
        
        # Set commission to 0 since we'll calculate it ourselves
        self.p.commission = 0.0
    
    def calculate_commission(self, size, price):
        """Calculate commission according to IBKR tiered pricing structure"""
        abs_size = abs(size)
        per_share_comm = abs_size * self.p.commission_per_share
        exchange_fee = abs_size * self.p.exchange_fees
        order_value = abs_size * price
        value_cap = order_value * self.p.max_per_order_pct
        
        # Apply minimum commission
        base_commission = max(per_share_comm, self.p.min_per_order)
        
        # Ensure commission doesn't exceed the percentage cap
        commission = min(base_commission, value_cap) + exchange_fee
        
        return commission
    
    def _getcommission(self, size, price, pseudoexec):
        """
        Main commission calculation method that Backtrader calls
        """
        commission = self.calculate_commission(size, price)
        
        if not pseudoexec:
            # Only track real executions, not simulated ones
            self.total_commission_charged += commission
            self.trade_count += 1
            
            logging.info(f"Commission charged: ${commission:.4f} for {abs(size)} shares at ${price:.2f}")
        
        return commission
    
    def get_credit_interest(self, data, pos, dt0):
        """
        FIXED: Correct signature for Backtrader credit interest calculation
        Backtrader calls this with (data, pos, dt0) not (size, price, days, dt0, dt1)
        """
        return 0.0  # No credit interest in this model
    
    def getsize(self, price, cash):
        """Calculate position size that can be bought with available cash"""
        if price <= 0:
            return 0
        return int(cash / price)
    
    def getoperationcost(self, size, price):
        """Total cost of operation including commission"""
        return abs(size) * price + self.calculate_commission(size, price)
    
    def getvaluesize(self, size, price):
        """Return value of the operation (without commission)"""
        return abs(size) * price
    
    def getvalue(self, position, price):
        """Returns the value of a position given a price"""
        return position.size * price
    
    def get_margin(self, price):
        """Return margin needed for single item"""
        return price  # Full price for cash stocks
    
    def profitandloss(self, size, price, newprice):
        """Calculate P&L for a position"""
        return size * (newprice - price)
    
    def cashadjust(self, size, price, newprice):
        """Calculate cash adjustment for position"""
        return self.profitandloss(size, price, newprice)
    
    def get_commission_stats(self):
        """Return commission statistics for analysis"""
        avg_commission = self.total_commission_charged / max(self.trade_count, 1)
        return {
            'total_commission': self.total_commission_charged,
            'total_trades': self.trade_count,
            'avg_commission_per_trade': avg_commission,
        }
    
    def reset_stats(self):
        """Reset commission statistics"""
        self.total_commission_charged = 0.0
        self.trade_count = 0


class SimpleIBKRCommission(bt.CommInfoBase):
    """
    Simplified IBKR commission model for testing
    """
    
    params = (
        ('commission_per_share', 0.0035),
        ('min_commission', 0.35),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('percabs', False),
    )
    
    def __init__(self):
        super(SimpleIBKRCommission, self).__init__()
        self.total_commission = 0.0
        self.trade_count = 0
    
    def _getcommission(self, size, price, pseudoexec):
        """Simple commission calculation"""
        abs_size = abs(size)
        commission = max(abs_size * self.p.commission_per_share, self.p.min_commission)
        
        if not pseudoexec:
            self.total_commission += commission
            self.trade_count += 1
            logging.info(f"Simple commission: ${commission:.4f} for {abs_size} shares")
        
        return commission
    
    def get_credit_interest(self, data, pos, dt0):
        """Fixed signature for credit interest"""
        return 0.0
    
    def get_stats(self):
        return {
            'total_commission': self.total_commission,
            'trade_count': self.trade_count,
            'avg_commission': self.total_commission / max(self.trade_count, 1)
        }






class IBKRSlippageModel:
    """
    Separate slippage model that can be used with the commission model
    or standalone. This addresses the integration issues.
    """
    
    def __init__(self, 
                 slip_base=0.0001,
                 slip_size_factor=0.15,
                 slip_vol_factor=0.1,
                 slip_atr_factor=1.5,
                 min_dollar_volume=1000000,
                 min_avg_volume=10000):
        
        self.slip_base = slip_base
        self.slip_size_factor = slip_size_factor
        self.slip_vol_factor = slip_vol_factor
        self.slip_atr_factor = slip_atr_factor
        self.min_dollar_volume = min_dollar_volume
        self.min_avg_volume = min_avg_volume
        
        self.volume_cache = {}
        self.atr_cache = {}
        self.total_slippage = 0.0
    
    def calculate_slippage(self, data, size, price):
        """
        Calculate slippage based on order size, market conditions and volatility
        Returns slippage amount (to be added to buy price or subtracted from sell price)
        """
        try:
            # Get volume data
            current_volume, avg_volume = self._get_volume_data(data)
            
            # Get volatility measure (ATR / price)
            atr = self._get_atr_estimate(data)
            volatility = atr / price if price > 0 else 0.02
            
            # Order size relative to average volume (capped at 30%)
            volume_ratio = min(abs(size) / avg_volume, 0.3)
            
            # Calculate base slippage percentage
            slippage_pct = (
                self.slip_base +
                (volume_ratio * self.slip_size_factor) +
                (1 / max(avg_volume, 1) * self.slip_vol_factor) +
                (volatility * self.slip_atr_factor)
            )
            
            # Dollar volume filter - increase slippage for low liquidity
            dollar_volume = price * current_volume
            if dollar_volume < self.min_dollar_volume:
                slippage_pct *= 1.5
            
            # Volume filter - increase slippage for low volume
            if current_volume < self.min_avg_volume:
                slippage_pct *= 1.5
            
            # Cap slippage at reasonable bounds (0.05% to 1.5%)
            slippage_pct = max(0.0005, min(0.015, slippage_pct))
            
            # Calculate actual slippage amount
            slippage_amount = price * slippage_pct
            
            # Slippage direction depends on order direction
            # Buying: price goes up, Selling: price goes down
            final_slippage = slippage_amount if size > 0 else -slippage_amount
            
            self.total_slippage += abs(final_slippage)
            
            return final_slippage
            
        except Exception as e:
            logging.warning(f"Error calculating slippage: {e}")
            # Return minimal slippage as fallback
            return price * 0.0001 * (1 if size > 0 else -1)
    
    def _get_volume_data(self, data, lookback=20):
        """Get volume data safely"""
        try:
            current_volume = data.volume[0] if len(data.volume) > 0 else 1000
            
            # Use cached average volume if available
            data_id = id(data)
            if data_id in self.volume_cache:
                return current_volume, self.volume_cache[data_id]
            
            # Calculate average volume
            volumes = []
            for i in range(min(lookback, len(data.volume))):
                try:
                    vol = data.volume.get(ago=i, size=1)
                    if vol and len(vol) > 0:
                        volumes.append(vol[0])
                except:
                    pass
            
            avg_volume = sum(volumes) / len(volumes) if volumes else current_volume
            self.volume_cache[data_id] = max(avg_volume, 1)
            
            return current_volume, self.volume_cache[data_id]
            
        except Exception as e:
            logging.warning(f"Error getting volume data for slippage: {e}")
            return 1000, 1000
    
    def _get_atr_estimate(self, data, period=14):
        """Get ATR estimate safely"""
        try:
            data_id = id(data)
            if data_id in self.atr_cache:
                return self.atr_cache[data_id]
            
            # Calculate simple ATR
            ranges = []
            for i in range(min(period, len(data.high))):
                try:
                    high = data.high.get(ago=i, size=1)
                    low = data.low.get(ago=i, size=1)
                    if high and low and len(high) > 0 and len(low) > 0:
                        ranges.append(high[0] - low[0])
                except:
                    pass
            
            atr = sum(ranges) / len(ranges) if ranges else data.close[0] * 0.02
            self.atr_cache[data_id] = max(atr, 0.01)
            
            return self.atr_cache[data_id]
            
        except Exception as e:
            logging.warning(f"Error calculating ATR for slippage: {e}")
            return data.close[0] * 0.02 if hasattr(data, 'close') else 0.01



def get_last_trading_date():
    """Get the last trading date from NYSE calendar."""
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    
    if schedule.empty:
        raise Exception("No trading days found in the past 10 days.")
    
    if today in schedule.index.date:
        today_market_open = schedule.loc[schedule.index.date == today, 'market_open'].iloc[0]
        
        if today_market_open.tzinfo is None:
            today_market_open = today_market_open.replace(tzinfo=timezone.utc)
        
        now_utc = datetime.now(timezone.utc)
        
        if now_utc < today_market_open:
            schedule = schedule[schedule.index.date < today]
    
    if not schedule.empty:
        last_trading_date = schedule.index[-1].date()
        return last_trading_date
    else:
        raise Exception("No trading days found in the past 10 days after excluding today.")

def get_previous_trading_day(current_date, days_back=1):
    """Get the nth previous trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    end_date = current_date.date()
    start_date = end_date - timedelta(days=days_back * 2)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    valid_days = schedule[schedule.index.date <= end_date]
    if len(valid_days) < days_back:
        raise ValueError(f"Not enough trading days found before {end_date}")
    
    return valid_days.index[-days_back].date()

def get_next_trading_day(current_date):
    """Get the next trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    start_date = current_date.date() + timedelta(days=1)
    end_date = start_date + timedelta(days=10)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    if schedule.empty:
        raise ValueError(f"No trading days found after {start_date}")
    
    return schedule.index[0].date()

# Optimized data loading
@njit
def calculate_up_prob_variance(up_probs):
    """Calculate variance of UpProbability using numba for speed."""
    if len(up_probs) < 10:  # Need minimum sample size
        return 0.0
    return np.var(up_probs)

def filter_stocks_by_signal_quality(directory, min_variance=0.01, min_up_prob=0.5, variance_weight=1.0):

    all_files = glob.glob(os.path.join(directory, '*.parquet'))
    quality_stocks = []
    
    logging.info(f"Evaluating {len(all_files)} stocks for signal quality...")
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.starmap(
                evaluate_stock_quality, 
                [(f, min_variance, min_up_prob, variance_weight) for f in all_files]
            ),
            total=len(all_files),
            desc="Filtering stocks"
        ))
        
    stock_quality_pairs = [(r[0], r[1]) for r in results if r is not None]
    
    stock_quality_pairs.sort(key=lambda x: x[1], reverse=True)
    
    quality_stocks = [pair[0] for pair in stock_quality_pairs]
    
    if len(stock_quality_pairs) > 0:
        top_5 = stock_quality_pairs[:5]
        bottom_5 = stock_quality_pairs[-5:] if len(stock_quality_pairs) >= 5 else stock_quality_pairs
        
        logging.info("Top 5 quality stocks:")
        for file_path, score in top_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
            
        logging.info("Bottom 5 quality stocks:")
        for file_path, score in bottom_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
    
    logging.info(f"Found {len(quality_stocks)} stocks meeting quality criteria")
    return quality_stocks

def evaluate_stock_quality(file_path, min_variance, min_up_prob, variance_weight=1.0):
    try:
        table = pq.read_table(file_path, columns=['UpProbability'])
        df = table.to_pandas()
        
        if len(df) < 60:  # Require at least 60 days of data
            return None
        
        up_probs = df['UpProbability'].values
        
        variance = calculate_up_prob_variance(up_probs)
        
        max_up_prob = np.max(up_probs)
        
        if variance < min_variance or max_up_prob < min_up_prob:
            return None
            
        norm_variance = min(variance / 0.10, 1.0)  # Cap at 1.0
        norm_max_prob = (max_up_prob - 0.5) / 0.5  # Normalize to 0-1 range
        
        quality_score = (norm_variance * variance_weight + norm_max_prob) / (1 + variance_weight)
        
        return (file_path, quality_score)
            
    except Exception as e:
        logging.error(f"Error evaluating {file_path}: {str(e)}")
    
    return None




def load_data(file_path, last_trading_date):
    """Load a single data file with basic validation, without alignment."""
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        yesterday = last_trading_date
        start_date = yesterday - timedelta(days=400)
        
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= yesterday)]
        
        if len(df) < 252:  # Need at least 1 year of data
            logging.info(f"Skipping {file_path} due to insufficient data: {len(df)} days")
            return None
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'UpProbability', 'UpPrediction', 'VIX_Close']
        
        if all(col in df.columns for col in required_columns):
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(4).astype(np.float32)
            
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            return (stock_name, df)
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            logging.warning(f"Skipping {file_path} due to missing columns: {missing_cols}")

        ##in the UpProbability add an epsilon value to avoid zero values if it is under 0.01
        if 'UpProbability' in df.columns:
            df['UpProbability'] = df['UpProbability'].apply(lambda x: x if x > 0.01 else 0.01)




    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        traceback.print_exc()

    return None



def parallel_load_data(file_paths, last_trading_date, align_start_date=False, retention_pct=95, min_days=270):
    """
    Load data files in parallel and align them by start date.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to load
    last_trading_date : datetime.date
        The last trading date to consider
    align_start_date : bool, default=True
        Whether to align all datasets to have a common start date
    retention_pct : float, default=95
        Target percentage of stocks to retain after alignment (0-100)
    min_days : int, default=252
        Minimum number of trading days required after alignment
        
    Returns:
    --------
    list of tuples: (stock_name, DataFrame) with aligned data
    """
    # Step 1: Load all data in parallel
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.starmap(load_data, [(fp, last_trading_date) for fp in file_paths]), 
            total=len(file_paths), 
            desc="Loading Files"
        ))
    
    loaded_data = [result for result in results if result is not None]
    
    if not align_start_date or not loaded_data:
        return loaded_data
    
    # Step 2: Analyze start date distribution
    logging.info(f"Loaded {len(loaded_data)} valid datasets. Analyzing for alignment...")
    
    # Get list of all available dates across all datasets
    all_dates = set()
    date_presence = {}  # For each date, how many datasets have it
    
    for _, df in loaded_data:
        dates = set(df['Date'].dt.date)
        all_dates.update(dates)
        for date in dates:
            date_presence[date] = date_presence.get(date, 0) + 1
    
    all_dates = sorted(all_dates)
    
    # Find dates that appear in at least retention_pct% of datasets
    min_datasets = (retention_pct / 100) * len(loaded_data)
    common_dates = [date for date, count in date_presence.items() if count >= min_datasets]
    common_dates.sort()
    
    if not common_dates:
        logging.warning(f"No dates appear in {retention_pct}% of datasets. Using most common date.")
        # Fall back to finding the most common date
        best_date = max(date_presence.items(), key=lambda x: x[1])[0]
        common_dates = [best_date]
    
    # Step 3: Find the earliest date that still gives us enough data points
    best_start_date = common_dates[0]  # Start with earliest common date
    target_len = None
    
    # Try different start dates to see which gives us the most data while keeping desired stocks
    for start_date in common_dates:
        # Calculate how many datasets would have at least min_days after this start date
        valid_datasets = []
        lengths = []
        
        for name, df in loaded_data:
            filtered_df = df[df['Date'].dt.date >= start_date]
            if len(filtered_df) >= min_days:
                valid_datasets.append((name, filtered_df))
                lengths.append(len(filtered_df))
        
        if len(valid_datasets) >= min_datasets:
            # We found a good start date, now find a common length
            if lengths:
                # Sort lengths and find the one that keeps ~95% of stocks
                lengths.sort()
                target_idx = int(len(lengths) * (retention_pct / 100))
                target_len = lengths[target_idx] if target_idx < len(lengths) else lengths[-1]
                
                # If this length is enough, use this start date
                if target_len >= min_days:
                    best_start_date = start_date
                    break
    
    if target_len is None or target_len < min_days:
        logging.warning(f"Could not find a common length of at least {min_days} days. Using {min_days}.")
        target_len = min_days
    
    # Step 4: Align all datasets to the best start date and common length
    aligned_data = []
    for name, df in loaded_data:
        # Filter to start on or after best_start_date
        aligned_df = df[df['Date'].dt.date >= best_start_date]
        
        # Check if we have enough data after applying the start date filter
        if len(aligned_df) >= target_len:
            # Trim to the common length
            aligned_df = aligned_df.iloc[:target_len]
            aligned_data.append((name, aligned_df))
        else:
            logging.info(f"Skipping {name}: Has only {len(aligned_df)} days after alignment (need {target_len})")
    
    # Step 5: Verify alignment
    start_dates = {df['Date'].dt.date.min() for _, df in aligned_data}
    lengths = {len(df) for _, df in aligned_data}
    
    if len(start_dates) == 1 and len(lengths) == 1:
        start_date = next(iter(start_dates))
        length = next(iter(lengths))
        logging.info(f"Perfect alignment achieved: {len(aligned_data)} stocks with {length} trading days")
        logging.info(f"Start date: {start_date}, End dates may vary slightly")
    else:
        logging.warning(f"Imperfect alignment: {len(start_dates)} different start dates, {len(lengths)} different lengths")
        logging.warning(f"Start dates: {start_dates}")
        logging.warning(f"Lengths: {lengths}")
    
    # Calculate what percentage of original stocks were kept
    retention_actual = (len(aligned_data) / len(loaded_data)) * 100
    logging.info(f"Retained {retention_actual:.1f}% of stocks after alignment ({len(aligned_data)} of {len(loaded_data)})")
    
    return aligned_data






def read_trading_data():
    """RETIRED: the per-ticker state ledger is no longer persisted (it used to
    pollute _Buy_Signals.parquet, which is now exclusively the broker's narrowed
    book). Returns an empty ledger-schema frame so legacy callers keep working
    without touching any file. write_trading_data() is a no-op; per-ticker state
    does not carry across runs."""
    return pd.DataFrame(columns=[
        'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
        'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
    ])




def write_trading_data(df):
    dtype_schema = {
        'Symbol': 'string',
        'LastBuySignalPrice': 'float64',
        'IsCurrentlyBought': 'bool',
        'ConsecutiveLosses': 'int64',
        'UpProbability': 'float64',
        'LastSellPrice': 'float64',
        'PositionSize': 'float64'
    }
    
    df = df.copy()
    
    date_columns = ['LastBuySignalDate', 'LastTradedDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col, dtype in dtype_schema.items():
        if col not in df.columns:
            if dtype == 'float64':
                df[col] = pd.Series(dtype='float64')
            elif dtype == 'int64':
                df[col] = pd.Series(dtype='int64')
            elif dtype == 'bool':
                df[col] = pd.Series(dtype='bool')
            elif dtype == 'string':
                df[col] = pd.Series(dtype='string')
        else:
            if dtype == 'float64':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif dtype == 'int64':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            elif dtype == 'bool':
                if df[col].dtype != 'bool':
                    df[col] = df[col].astype('bool')
            elif dtype == 'string':
                if df[col].dtype != 'string':
                    df[col] = df[col].astype('string')
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df[col] = df[col].astype('object').where(df[col].notnull(), None)
    # _Buy_Signals.parquet retired — Z_signals.parquet is the canonical signals file

def update_buy_signal(symbol, date, price, up_probability):
    try:
        price = round(float(price), 4)
        up_probability = round(float(up_probability), 4)
        
        df = read_trading_data()
        
        new_data = pd.DataFrame([{
            'Symbol': str(symbol),
            'LastBuySignalDate': pd.Timestamp(date),
            'LastBuySignalPrice': price,
            'IsCurrentlyBought': False,
            'ConsecutiveLosses': 0,
            'LastTradedDate': pd.NaT,
            'UpProbability': up_probability,
            'LastSellPrice': float('nan'),
            'PositionSize': float('nan')
        }])
        
        # Set proper dtypes for the new DataFrame
        new_data = new_data.astype({
            'Symbol': 'string',
            'LastBuySignalPrice': 'float64',
            'IsCurrentlyBought': 'bool',
            'ConsecutiveLosses': 'int64',
            'UpProbability': 'float64',
            'LastSellPrice': 'float64',
            'PositionSize': 'float64'
        })
        
        new_data['LastBuySignalDate'] = pd.to_datetime(new_data['LastBuySignalDate'])
        new_data['LastTradedDate'] = pd.to_datetime(new_data['LastTradedDate'])
        
        df = df[df['Symbol'] != symbol]
        
        for col in new_data.columns:
            if col in df.columns and col not in ['LastBuySignalDate', 'LastTradedDate']:
                df[col] = df[col].astype(new_data[col].dtype)
        
        df = pd.concat([df, new_data], ignore_index=True)
        
        write_trading_data(df)
        
        logging.info(f"Updated buy signal for {symbol} at price {price}")
        
    except Exception as e:
        logging.error(f"Error in update_buy_signal for {symbol}: {str(e)}")
        raise

def mark_position_as_bought(symbol, position_size):
    df = read_trading_data()
    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
    df.loc[df['Symbol'] == symbol, 'PositionSize'] = position_size
    write_trading_data(df)

def update_trade_result(symbol, is_loss, exit_price=None, exit_date=None):
    df = read_trading_data()

    if symbol in df['Symbol'].values:
        if is_loss:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] += 1
        else:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] = 0
        
        df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(exit_date or datetime.now().date())
        if exit_price is not None:
            df.loc[df['Symbol'] == symbol, 'LastSellPrice'] = exit_price
        df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
        write_trading_data(df)




class AdaptiveSlippageCommissionScheme(bt.CommInfoBase):
    params = (
        ('commission', 0.0),  # Set commission separately
        ('min_slippage', 0.0005),  # 5 basis points minimum
        ('max_slippage', 0.015),   # 150 basis points maximum
        ('atr_period', 14),        # Same as your strategy
    )
    
    def __init__(self):
        super(AdaptiveSlippageCommissionScheme, self).__init__()
        self.atr_cache = {}
    
    def _get_credit_interest(self, size, price, days, dt):
        return 0.0  # No interest in this model
    
    def _get_commission(self, size, price, pseudoexec):
        return self.p.commission * abs(size) * price
    
    def get_slippage(self, order, data, dt):
        """Calculate dynamic slippage based on order size and market conditions"""
        if order.exectype != bt.Order.Market:
            return 0.0  # Only apply to market orders as per your docs
        
        # Get required metrics
        order_size = abs(order.size)
        current_volume = data.volume[0]
        avg_volume = sum(data.volume.get(size=20, fallback=True)) / 20
        price = data.close[0]
        dollar_volume = price * current_volume
        
        # Get or calculate ATR for volatility estimate
        if data._name not in self.atr_cache:
            atr = bt.indicators.ATR(data, period=self.p.atr_period)
            self.atr_cache[data._name] = atr[-1]
        volatility = self.atr_cache[data._name]
        
        # Apply core slippage formula
        volume_ratio = min(order_size / max(avg_volume, 1), 0.3)
        liquidity_factor = min(1_000_000 / max(dollar_volume, 1), 0.02)
        
        base_slippage = self.p.min_slippage + (volume_ratio * 0.015) + (liquidity_factor * 0.01)
        volatility_multiplier = 1.0 + (volatility / price) * 5.0 if price > 0 else 1.0
        
        slippage_pct = base_slippage * volatility_multiplier
        final_slippage = max(self.p.min_slippage, min(self.p.max_slippage, slippage_pct))
        
        # Direction matters - buys get worse prices (higher), sells get worse prices (lower)
        slippage_value = price * final_slippage * (1 if order.isbuy() else -1)
        return slippage_value



def get_dynamic_signal_columns(index_folder="Data/Indexes"):
    """Scan the folder for parquet files and collect corr/alpha/beta column names."""
    corr_cols, alpha_cols, beta_cols = set(), set(), set()

    # Find all parquet files in the folder
    parquet_files = glob.glob(os.path.join(index_folder, "*.parquet"))

    for file in parquet_files:
        try:
            df = pd.read_parquet(file, engine="pyarrow")
            cols = df.columns.str.lower()  # lower for consistent comparison

            corr_cols.update([col for col in df.columns if "corr" in col.lower()])
            alpha_cols.update([col for col in df.columns if "alpha" in col.lower()])
            beta_cols.update([col for col in df.columns if "beta" in col.lower()])
        except Exception as e:
            print(f" Skipping {file}: {e}")

    return sorted(corr_cols), sorted(alpha_cols), sorted(beta_cols)

class EnhancedPandasData(bt.feeds.PandasData):
    """Enhanced PandasData class that includes ML signals and technical indicators."""

    # Base lines always present
    base_lines = ('UpProbability', 'VIX_Close', 'atr')

    # Dynamically add corr/alpha/beta lines from parquet data
    corr_cols, alpha_cols, beta_cols = get_dynamic_signal_columns()

    lines = base_lines + tuple(corr_cols) + tuple(alpha_cols) + tuple(beta_cols)

    # Base params for PandasData fields
    base_params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('UpProbability', 'UpProbability'),
        ('VIX_Close', 'VIX_Close'),
        ('atr', None),
    )

    # Combine base + dynamic columns into params
    params = base_params + tuple((col, col) for col in corr_cols + alpha_cols + beta_cols)









class FixedCommissionScheme(bt.CommInfoBase):
    params = (
        ('commission', 3.0),  # Fixed commission per trade
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission  # Return fixed commission

class Rule201Monitor:
    def __init__(self, threshold=-9.99, cooldown_days=1):
        self.threshold = threshold
        self.cooldown_days = cooldown_days
        self.violations = set()
        self.trigger_dates = {}
        self.triggerCount = 0
    
    def check_rule_201(self, symbol, prev_close, current_price, current_date):
        if prev_close <= 0:
            return False
            
        daily_return = (current_price / prev_close - 1) * 100
        
        if daily_return <= self.threshold:
            self.violations.add(symbol)
            self.trigger_dates[symbol] = current_date
            self.triggerCount += 1
            return True
        return False
    
    def clear_expired_restrictions(self, current_date):
        expired = []
        
        for symbol, trigger_date in self.trigger_dates.items():
            days_since = (current_date - trigger_date).days
            if days_since > self.cooldown_days:
                expired.append(symbol)
                logging.info(f"Rule 201 cooldown expired for {symbol}")
        
        for symbol in expired:
            self.violations.discard(symbol)
            del self.trigger_dates[symbol]
    
    def is_restricted(self, symbol):
        """Check if a symbol is currently restricted under Rule 201."""
        return symbol in self.violations



class PositionSizer:
    def __init__(self, risk_per_trade=0.6, max_position_pct=0.06, min_position_pct=0.025, reserve_pct=0.20):
        self.risk_per_trade = risk_per_trade  # Risk per trade in percent of account
        self.max_position_pct = max_position_pct  # Maximum position size as % of account
        self.min_position_pct = min_position_pct  # Minimum position size as % of account
        self.reserve_pct = reserve_pct  # Cash reserve percentage
    
    def calculate_position_size(self, account_value, cash, price, atr, max_positions):
        # Target 80% of account invested (20% cash buffer)
        workable_capital = account_value * (1.0 - self.reserve_pct)
        
        # Ensure adequate cash buffer before taking positions
        if cash < account_value * self.reserve_pct:
            return 0
            
        # Calculate position size based on risk management
        risk_amount = account_value * (self.risk_per_trade / 100.0)
        risk_per_share = 2.0 * atr
        
        if risk_per_share <= 0.01:
            risk_per_share = 0.01
            
        shares_by_risk = risk_amount / risk_per_share
        
        # Position limits based on account percentages
        min_position = (account_value * self.min_position_pct) / price
        max_position = (account_value * self.max_position_pct) / price
        
        # Use risk-based sizing within percentage bounds
        position_size = np.clip(shares_by_risk, min_position, max_position)
        
        # Ensure portfolio diversification across max_positions
        max_per_position = workable_capital / (price * max_positions)
        position_size = min(position_size, max_per_position)
        position_size = int(position_size)
        
        # Final liquidity check
        required_capital = position_size * price
        if required_capital > cash * 0.90:  # Keep 10% cash buffer within available cash
            position_size = int(cash * 0.90 / price)
            
        return max(0, position_size)


class PositionSize222r:
    def __init__(self, risk_per_trade=1.0, max_position_pct=0.20, min_position_pct=0.05, reserve_pct=0.10):
        self.risk_per_trade = risk_per_trade  # Risk per trade in percent of account
        self.max_position_pct = max_position_pct  # Maximum position size as % of account
        self.min_position_pct = min_position_pct  # Minimum position size as % of account
        self.reserve_pct = reserve_pct  # Cash reserve percentage
    
    def calculate_position_size(self, account_value, cash, price, atr, max_positions):
        workable_capital = account_value * (1.0 - self.reserve_pct) # Check if we have enough free cash
        
        if cash < account_value * self.reserve_pct: # Don't take a position if cash is too low
            return 0
            
        
        risk_amount = account_value * (self.risk_per_trade / 100.0) # Calculate position size based on risk and ATR
        risk_per_share = 2.0 * atr # Use 2x ATR as the risk per share
        
        if risk_per_share <= 0.01:  # Avoid division by zero or very small numbers
            risk_per_share = 0.01
            
        shares_by_risk = risk_amount / risk_per_share
        
        
        min_position = (account_value * self.min_position_pct) / price # Calculate position limits based on account percentages
        max_position = (account_value * self.max_position_pct) / price
        
        # Position size is the middle ground between risk-based and percentage-based
        position_size = np.clip(shares_by_risk, min_position, max_position)
        
        # Also consider max positions to avoid over-concentration
        position_size = min(position_size, workable_capital / (price * max_positions))
        position_size = int(position_size)
        
        # Final check: Ensure we have enough cash
        if position_size * price > cash * 0.95:  # Leave some buffer
            position_size = int(cash * 0.95 / price)
            
        return max(0, position_size)





class TradeRecorder:
    def __init__(self, filename='Data/TradeHistory.parquet'):
        self.filename = filename
        self.trades = []
        
    def record_trade(self, trade_data):
        """Record a trade with detailed metadata."""
        self.trades.append(trade_data)
        
    def save_trades(self):
        """Save all recorded trades to a parquet file."""
        if not self.trades:
            logging.info("No trades to save")
            return
            
        df = pd.DataFrame(self.trades)
        
        numeric_cols = [
            'EntryPrice', 'ExitPrice', 'Quantity', 'PnL', 'PnLPct',
            'Commission', 'Slippage', 'ATR', 'UpProbability'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        date_cols = ['EntryDate', 'ExitDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df.to_parquet(self.filename, index=False)
        logging.info(f"Saved {len(self.trades)} trades to {self.filename}")














class StockSniperStrategy(bt.Strategy):
    # Use the parameters from STRATEGY_PARAMS_TUPLE
    params = STRATEGY_PARAMS
    
    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=self.p.atr_period)
            # FAST: removed up_prob_ma3/ma5/roc — computed on every feed every bar but
            # never referenced anywhere. ATR-14 still sets the warmup, so results are
            # identical. (lossless ~indicator-cost reduction)
            self.inds[d]['up_prob'] = d.UpProbability



        # Rest of your initialization code remains the same
        self.order_list = []  # Track pending orders
        self.signal_metadata = {}
        self.bracket_orders = {}
        self.entry_prices = {}  # Track entry prices for positions
        self.position_dates = {}  # Track entry dates for positions
        self.stop_loss_orders = {}  # Track stop loss orders
        self.take_profit_orders = {}  # Track take profit orders
        self.trailing_stops = {}  # Track trailing stop levels
        self.asset_groups = {}  # Track asset groups for correlation
        self.group_allocations = {}  # Track group allocations
        
        self.trade_history = []  # Detailed trade history for analysis
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0
        self.longest_win_streak = 0
        self.longest_loss_streak = 0
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.recent_outcomes = []  # Store last 10 trade outcomes (1=win, 0=breakeven, -1=loss)
        self.trade_pct_returns = []  # For rolling Sharpe calculation
        
        self.trade_recorder = TradeRecorder('Data/TradeHistory.parquet')
        
        self.open_positions = 0
        
        self.correlation_df = pd.read_parquet('Correlations.parquet')
        logging.info(f"Loaded correlation dataframe with columns: {list(self.correlation_df.columns)}")

        if 'Ticker' in self.correlation_df.columns:
            self.correlation_df_by_ticker = self.correlation_df.copy()
            self.correlation_df.set_index('Ticker', inplace=True)
            logging.info("Set 'Ticker' column as index in correlation dataframe")
        else:
            verbose = False
            if verbose:
                logging.warning("'Ticker' column not found in correlation dataframe. Available columns: " 
                               f"{list(self.correlation_df.columns)}")

        self.total_groups = self.correlation_df['Cluster'].nunique()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}
        
        # Use the parameters from Util for PositionSizer
        self.position_sizer = PositionSizer(
            risk_per_trade=self.p.risk_per_trade_pct,
            max_position_pct=self.p.max_position_pct,
            min_position_pct=self.p.min_position_pct,
            reserve_pct=self.p.reserve_percent
        )
        
        # Use the parameters from Util for Rule201Monitor
        self.rule_201_monitor = Rule201Monitor(
            threshold=self.p.rule_201_threshold,
            cooldown_days=self.p.rule_201_cooldown
        )
        
        self.last_trading_date = get_last_trading_date() 
        self.second_last_trading_date = get_previous_trading_day(self.last_trading_date)
        self.trading_lockup_start = get_previous_trading_day(self.last_trading_date, self.p.lockup_days)
        
        self.positions_cleared_for_lockup = False
        self.trading_locked = False
        self.last_logged_date = None
        
        self.monthly_performance = {}  # {YYYY-MM: percent_return}
        self.yearly_performance = {}   # {YYYY: percent_return}
        self.last_month_equity = None
        self.last_year_equity = None
        self.month_high_equity = None
        self.month_low_equity = None
        self.current_month = None
        self.current_year = None
        
        self.day_count = 0
        self.total_bars = 252  # Expected trading days in backtest
        self.progress_bar = tqdm(
            total=self.total_bars,
            desc="Strategy Progress",
            unit="day",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )
        

    def prenext(self):
        """Run Rule 201 checks before each new trading day AKA down by more than 10% and will prevent child parent orders with trailing stops."""
        current_date = self.datetime.date()
        
        self.rule_201_monitor.clear_expired_restrictions(current_date)
        
        for d in self.datas:
            if len(d) > 1:  # Need at least 2 data points
                symbol = d._name
                prev_close = d.close[-1]
                current_price = d.open[0]
                
                self.rule_201_monitor.check_rule_201(
                    symbol, prev_close, current_price, current_date
                )
    








    def next(self):
        """Main strategy logic executed on each bar."""
        self.day_count += 1
        self.progress_bar.update(1)

        current_date = self.datetime.date()
        current_month = current_date.strftime('%Y-%m')
        current_year = current_date.strftime('%Y')

        if any(len(data) == 0 for data in self.datas):
            return

        if self.last_logged_date != current_date:
            logging.info(f"Processing date: {current_date}")
            self.last_logged_date = current_date

        # Continue with normal position management
        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        if self.open_positions < self.p.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates or current_date == self.last_trading_date:
                # This will now handle selecting the best stock if needed
                #print(f"Buy candidates: {[(d._name, size, correlation) for d, size, correlation in buy_candidates]}")
                self.process_buy_candidates(buy_candidates, current_date, verbose=False)

        current_equity = self.broker.getvalue()
        self.update_performance_tracking(current_equity, current_month, current_year)












    def update_performance_tracking(self, current_equity, current_month, current_year):
        # Monthly performance tracking
        if current_month != self.current_month:
            if self.current_month is not None and self.last_month_equity is not None:
                monthly_return = (current_equity / self.last_month_equity - 1) * 100
                self.monthly_performance[self.current_month] = monthly_return
                logging.info(f"Month {self.current_month} performance: {monthly_return:.2f}%")
                logging.info(f"Month high: {self.month_high_equity:.2f}, low: {self.month_low_equity:.2f}")

            self.current_month = current_month
            self.last_month_equity = current_equity
            self.month_high_equity = current_equity
            self.month_low_equity = current_equity
            logging.info(f"Starting new month: {current_month}")
        else:
            if current_equity > self.month_high_equity:
                self.month_high_equity = current_equity
            if current_equity < self.month_low_equity:
                self.month_low_equity = current_equity

        # Yearly performance tracking
        if current_year != self.current_year:
            if self.current_year is not None and self.last_year_equity is not None:
                yearly_return = (current_equity / self.last_year_equity - 1) * 100
                self.yearly_performance[self.current_year] = yearly_return
                logging.info(f"Year {self.current_year} performance: {yearly_return:.2f}%")
            self.current_year = current_year
            self.last_year_equity = current_equity
            logging.info(f"Starting new year: {current_year}")





    def get_buy_candidates(self, current_date):

        buy_candidates = []
        
        for d in self.datas:
            if self.getposition(d).size > 0:
                continue
                
            if self.can_buy(d, current_date):
                size = self.calculate_position_size(d)
                
                if size > 0:
                    correlation = self.get_mean_correlation(
                        d._name, 
                        [data._name for data in self.datas if self.getposition(data).size > 0]
                    )
                    buy_candidates.append((d, size, correlation))

        return buy_candidates
    
    def calculate_position_size(self, data):
        return self.position_sizer.calculate_position_size(
            account_value=self.broker.getvalue(),
            cash=self.broker.getcash(),
            price=data.close[0],
            atr=self.inds[data]['atr'][0],
            max_positions=self.p.max_positions
        )
    








    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    ##=================================================[relative thresholds]=================================================##
    







    def can_buy__NO__CONFIG(self, data, current_date):
        symbol = data._name

        # HARD FILTER: No trading in first 30 days
        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < 30:
            return False

        # Restrict based on UpProbability raw bounds
        if data.UpProbability[0] < 0.45:
            return False
        if data.UpProbability[0] > 0.70:
            return False

        # Basic gates
        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        # Current values
        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False

        # Hard rejects
        if current_close is None or current_close < 3.00:
            return False
        if current_volume is None or current_volume < 1_000_000:
            return False
        if current_prob is None:
            return False

        # Liquidity check
        dollar_volume = current_close * current_volume
        if dollar_volume < 2_000_000:  # $2M minimum
            return False

        # Build historical UpProbability series
        historic_probs = []
        i = 1
        while len(historic_probs) < 45 and i < 200:
            try:
                prob_val = data.UpProbability[i]
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        # Limited data sanity check
        if len(historic_probs) < 30:
            # Reject symbols with recent drops > 10% when data is limited
            try:
                for lookback in range(1, min(11, 100)):  # Check last 10 days
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]

                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < -0.05:  # Reject on >10% single-day drop
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break  # No more historical data
            except Exception:
                return False

            # Need minimum viable data even with sanity check
            if len(historic_probs) < 5:
                return False

            # For limited data, use simplified thresholds
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, 85)  # Lower threshold for limited data
                p97_5 = np.percentile(historic_probs, 97.5)
            else:
                return False
        else:
            # Standard percentile thresholds for sufficient data
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, 90)
                p97_5 = np.percentile(historic_probs, 95)
            else:
                return False

        # --- Extra Failure Filters ---

        # 1. 52-week high filter
        try:
            closes_252 = data.close.get(size=252)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                lowest_52w = min(closes_252)
                ##if its within 10% of high or low reject 
                if highest_52w and lowest_52w and highest_52w > 0:
                    if (current_close / highest_52w) > 0.90:  # Within 10% of 52-week high
                        return False
        except Exception:
            pass

        # 2. Momentum filter (5d return)
        try:
            prev_close_5 = data.close[-5]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > 0.15:  # > +15% in last 5 days
                    return False
                
                if ret_5d < -0.075:  # > -7.5% in last 5 days
                    return False
        except Exception:
            pass

        # 3. Volume spike filter reject if 3.5x 20d avg volume
        try:
            vols = data.volume.get(size=20)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > 3.5:
                    return False
        except Exception:
            pass

        # 4. Volatility filter (20d) - make more restrictive
        try:
            closes = data.close.get(size=20)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > 0.04:  # Reduce from 0.06 to 0.04 (4% daily volatility cutoff)
                        return False
        except Exception:
            pass


        # --- Core Buy Condition ---
        if current_prob >= p96 and current_prob < p97_5:
            return True

        return False






    def can_buy(self, data, current_date):
        """Optimized 2026-05-15 via can_buy_optimizer.py.

        Restored 2026-05-17 after the EDA-driven retune
        (can_buy_may_17th_eda_attempt_FAILED) regressed badly in backtest:
        73.7% -> 23.5% annualised, Sharpe 1.85 -> 0.56.

        Original docstring follows.

        Tuned by 2 x 800-trial Optuna runs over the continuous filter space
        (top_k=3/hold=2 seed=0, top_k=5/hold=1 seed=42). Held-out window:
        2025-05-01 onwards. Only threshold changes where BOTH runs agreed on
        direction AND the cross-seed midpoint moved measurably are shipped;
        params with conflicting directions were left at legacy values.

        Lite-backtest holdout deltas vs legacy can_buy_may15th:
          Sharpe   +1.38 -> +1.76 (k=3/h=2)   |   +1.16 -> +1.19 (k=5/h=1)
          MaxDD    19.4% -> 11.6%             |   17.8% -> 14.2%
          AnnRet   +90%  -> +100%             |   +332% -> +358%

        Single-line rollback: swap can_buy <-> can_buy_may15th in
        get_buy_candidates (line ~1279).
        """

        # Strategy Timing (unchanged)
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        # Probability Bounds (unchanged -- runs disagreed on direction)
        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        # Financial Thresholds
        MIN_CLOSE_PRICE = 2.00       # was 1.50 -- both runs preferred ~2.0
        MAX_CLOSE_PRICE = 1650.00    # was 1000.00 -- both runs preferred ~1650
        MIN_VOLUME_SHARES = 10_000
        MIN_DOLLAR_VOLUME = 1_000_000  # unchanged -- runs disagreed

        # Risk Management - Drop Protection (unchanged)
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10

        # Risk Management - 52-Week Position (unchanged -- runs disagreed)
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85
        WEEK_52_LOOKBACK_DAYS = 252

        # Risk Management - Momentum
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15     # unchanged -- runs disagreed
        MAX_MOMENTUM_LOSS = -0.18    # was -0.075 -- both runs preferred deeper allowance

        # Risk Management - Volume & Volatility
        VOLUME_SPIKE_MULTIPLIER = 3.5      # unchanged -- runs disagreed
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.10    # was 0.04 -- both runs preferred ~0.08-0.11
        VOLATILITY_LOOKBACK_DAYS = 20

        # Percentile Thresholds (sufficient-data path)
        # Single biggest finding: both runs converged on ~0.65 low bound. The legacy
        # 0.90 only fires when current prob is in the stock's top 10%; 0.65 fires
        # when it's in the top 35% -- captures the bulk of the model's per-stock
        # alpha that the legacy cutoff was excluding.
        SUFFICIENT_DATA_P_LOW = 65.0       # was 90.0
        SUFFICIENT_DATA_P_HIGH = 98.0      # was 95.0 -- stop rejecting the top tail

        # For limited data (unchanged -- not in optimizer; keep strict on low-data)
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99

        # ============ END CONFIGURATION ============

        symbol = data._name

        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < MIN_DAYS_BEFORE_TRADING:
            return False

        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False

        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
            return False

        try:
            if len(data.UpProbability) < 6:
                return False
            recent_probs = [data.UpProbability[i] for i in range(1, 6) if data.UpProbability[i] is not None]
            if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                return False
            if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                return False
        except Exception:
            pass

        if current_close is None or current_close < MIN_CLOSE_PRICE:
            return False
        if current_close is None or current_close > MAX_CLOSE_PRICE:
            return False
        if current_volume is None or current_volume < MIN_VOLUME_SHARES:
            return False
        if current_prob is None:
            return False

        dollar_volume = current_close * current_volume
        if dollar_volume < MIN_DOLLAR_VOLUME:
            return False

        historic_probs = []
        i = 1
        while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
            try:
                prob_val = data.UpProbability[i]
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
            try:
                for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]
                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < MAX_SINGLE_DAY_DROP:
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break
            except Exception:
                return False

            if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                return False

            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
            else:
                return False
        else:
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
            else:
                return False

        # 1. 52-week high filter
        try:
            closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                lowest_52w = min(closes_252)
                if highest_52w and lowest_52w and highest_52w > 0:
                    if (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                        return False
        except Exception:
            pass

        # 2. Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > MAX_MOMENTUM_GAIN:
                    return False
                if ret_5d < MAX_MOMENTUM_LOSS:
                    return False
        except Exception:
            pass

        # 3. Volume spike filter
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                    return False
        except Exception:
            pass

        # 4. Volatility filter
        try:
            closes = data.close.get(size=VOLATILITY_LOOKBACK_DAYS)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > MAX_VOLATILITY_THRESHOLD:
                        return False
        except Exception:
            pass

        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 20   # was 30 -- both runs preferred lower (14, 25); midpoint

        # 5. RSI filter
        try:
            closes_for_rsi = data.close.get(size=RSI_PERIOD + 1)
            if closes_for_rsi is not None and len(closes_for_rsi) >= RSI_PERIOD + 1:
                deltas = np.diff(closes_for_rsi)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-RSI_PERIOD:])
                avg_loss = np.mean(losses[-RSI_PERIOD:])
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

        # --- Core Buy Condition ---
        if current_prob >= p96 and current_prob < p97_5:
            return True

        return False



    def can_buy_may_17th_eda_attempt_FAILED(self, data, current_date):
        """ARCHIVED -- DO NOT USE.  Tuned 2026-05-17 from eda_prob_distribution.py
        findings.  Backtest result was a regression vs can_buy_may_17th:

          can_buy_may_17th : 73.7% ann ret  Sharpe 1.85  win 53.25%  (n=323)
          THIS VERSION    : 23.5% ann ret  Sharpe 0.56  win 45.48%  (n=299)

        Why it failed: the EDA looked at which percentile bands of the
        EXISTING gate's 323 trades had the best per-trade quality and saw
        [p65, p90) doing well on n=16.  But tightening P_HIGH 98 -> 90 does
        NOT retain those 16 trades -- it RESHAPES the trade distribution.
        The new gate fired 299 different trades, and they were worse.

        Lesson: when tuning filters from EDA on backtest outcomes, validate
        with a full backtest BEFORE deploying.  The selection bias is real.
        Original docstring follows.

        ----
        Tuned 2026-05-17 from eda_prob_distribution.py findings.

        Run that script against Data/TradeHistory.parquet + Data/RFpredictions/ and
        you'll see three things this version acts on:

          1. The current [p65, p98) per-stock percentile gate fires 167 trades
             in the EDA sample, but [p95, p98) alone is 119 of them with
             mean +0.25% / win 0.43.  Tightening P_HIGH to 90 keeps only the
             zone where the EDA shows mean +1.26% / sharpe 7.55.  Trade count
             drops ~10x but per-trade quality jumps.
          2. SUFFICIENT_DATA_P_LOW=65 was a no-op: no trades fired below p80
             anyway, since the other filters (price, volume, momentum, RSI)
             gate that out.  Dropping it to 0 has zero effect and is cleaner.
          3. UP_PROB_MIN/MAX_BOUND in [0.2, 0.8] passes 100% of universe rows
             because predict_to_rf maps everything into [0.30, 0.70].  Dead
             code.  Replaced with a real UpProb floor of 0.40: trades in
             [0.30, 0.40) won 42% with mean +0.14% (n=90) in the EDA --
             barely above breakeven, can_buy fired them only because the
             percentile gate said yes.

        Single-line rollback: swap can_buy <-> can_buy_may_17th in
        get_buy_candidates.

        Caveat: the [p65, p90) decision is supported by n=16 in the EDA.
        Direction is consistent with the wider bands ([p85, p90) n=12 mean
        +1.17%, [p90, p95) n=32 mean +0.31%) so I trust the shape, but the
        exact p90 cut-off is a judgment call that should be validated by a
        full backtest before live deployment.
        """

        # Strategy Timing
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        # Absolute UpProbability floor (NEW).  predict_to_rf squashes universe
        # into [0.30, 0.70]; the [0.30, 0.40) band is the low-confidence tail
        # that produced 90 trades with mean +0.14% in the EDA.
        UP_PROB_FIRE_FLOOR = 0.40

        # Financial Thresholds (unchanged from May 17th tune)
        MIN_CLOSE_PRICE = 2.00
        MAX_CLOSE_PRICE = 1650.00
        MIN_VOLUME_SHARES = 10_000
        MIN_DOLLAR_VOLUME = 1_000_000

        # Risk Management - Drop Protection
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10

        # Risk Management - 52-Week Position
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85
        WEEK_52_LOOKBACK_DAYS = 252

        # Risk Management - Momentum
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.18

        # Risk Management - Volume & Volatility
        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.10
        VOLATILITY_LOOKBACK_DAYS = 20

        # Percentile Thresholds -- THE CHANGE
        # P_LOW kept as a safety net (still effectively a no-op given other
        # filters, but cheap and self-documenting).  P_HIGH = 90 is the EDA-
        # backed choice that captures the high-edge zone and excludes the
        # large, low-quality [p95, p98) cluster.
        SUFFICIENT_DATA_P_LOW = 65.0
        SUFFICIENT_DATA_P_HIGH = 90.0   # was 98.0 -- main change

        # Limited-data path (kept strict because sample is small)
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99

        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 20

        # ============ END CONFIGURATION ============

        symbol = data._name

        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < MIN_DAYS_BEFORE_TRADING:
            return False

        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False

        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        # NEW: absolute fire floor in place of the dead [0.2, 0.8] bound.
        if current_prob is None or current_prob < UP_PROB_FIRE_FLOOR:
            return False

        if current_close is None or current_close < MIN_CLOSE_PRICE:
            return False
        if current_close > MAX_CLOSE_PRICE:
            return False
        if current_volume is None or current_volume < MIN_VOLUME_SHARES:
            return False

        dollar_volume = current_close * current_volume
        if dollar_volume < MIN_DOLLAR_VOLUME:
            return False

        historic_probs = []
        i = 1
        while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
            try:
                prob_val = data.UpProbability[i]
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
            try:
                for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]
                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < MAX_SINGLE_DAY_DROP:
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break
            except Exception:
                return False

            if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                return False

            if len(historic_probs) > 1:
                p_low = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                p_high = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
            else:
                return False
        else:
            if len(historic_probs) > 1:
                p_low = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                p_high = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
            else:
                return False

        # 1. 52-week high filter
        try:
            closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                if highest_52w and highest_52w > 0:
                    if (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                        return False
        except Exception:
            pass

        # 2. Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > MAX_MOMENTUM_GAIN:
                    return False
                if ret_5d < MAX_MOMENTUM_LOSS:
                    return False
        except Exception:
            pass

        # 3. Volume spike filter
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                    return False
        except Exception:
            pass

        # 4. Volatility filter
        try:
            closes = data.close.get(size=VOLATILITY_LOOKBACK_DAYS)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > MAX_VOLATILITY_THRESHOLD:
                        return False
        except Exception:
            pass

        # 5. RSI filter
        try:
            closes_for_rsi = data.close.get(size=RSI_PERIOD + 1)
            if closes_for_rsi is not None and len(closes_for_rsi) >= RSI_PERIOD + 1:
                deltas = np.diff(closes_for_rsi)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-RSI_PERIOD:])
                avg_loss = np.mean(losses[-RSI_PERIOD:])
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

        # --- Core Buy Condition: per-stock percentile in [p65, p90) ---
        if current_prob >= p_low and current_prob < p_high:
            return True

        return False



    def can_buy_may15th(self, data, current_date):
        """Pre-optimization snapshot (2026-05-15). Kept for one-line rollback:
        swap can_buy <-> can_buy_may15th in get_buy_candidates if the optimized
        version misbehaves. This is the configuration that ran before the
        can_buy_optimizer.py Optuna sweep on 2026-05-15.
        """

        # Strategy Timing
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        # Probability Bounds
        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        # Financial Thresholds  
        MIN_CLOSE_PRICE = 1.50
        MAX_CLOSE_PRICE = 1000.00
        MIN_VOLUME_SHARES = 10_000 ## at least 10k shares traded 
        MIN_DOLLAR_VOLUME = 1_000_000 ## at least a mill traded so that i will be 1/1000 of the volume

        # Risk Management - Drop Protection
        MAX_SINGLE_DAY_DROP = -0.15  # -5%
        RECENT_DROP_LOOKBACK_DAYS = 10

        # Risk Management - 52-Week Position
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85  # Within 90% of 52-week high
        WEEK_52_LOOKBACK_DAYS = 252

        # Risk Management - Momentum
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15   # +15%
        MAX_MOMENTUM_LOSS = -0.075 # -7.5%

        # Risk Management - Volume & Volatility

        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.04  # 4% daily volatility
        VOLATILITY_LOOKBACK_DAYS = 20

        # Percentile Thresholds
        # For sufficient data
        SUFFICIENT_DATA_P_LOW = 90.0   # p96 equivalent 
        SUFFICIENT_DATA_P_HIGH = 95.0   # p97_5 equivalent

        # For limited data  
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99

        # ============ END CONFIGURATION ============

        symbol = data._name

        # HARD FILTER: No trading in first X days
        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < MIN_DAYS_BEFORE_TRADING:
            return False

        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False
        
        # Basic gates #if it dropped by more than 10% yesterday the trail limit orders will not work as in a child parent order group
        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        # Restrict based on UpProbability raw bounds
        # 3. Basic probability bounds HARD REJECT
        if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
            return False

        try:
            #make sure up prob is within the range that shows the best results and not some weird values like 0.1 or 0.9
            if len(data.UpProbability) < 6:
                return False
            recent_probs = [data.UpProbability[i] for i in range(1, 6) if data.UpProbability[i] is not None]
        
            if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                return False
            if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                return False
        except Exception:
            pass

        # Hard rejects
        if current_close is None or current_close < MIN_CLOSE_PRICE:
            return False
        if current_close is None or current_close > MAX_CLOSE_PRICE:
            return False
        if current_volume is None or current_volume < MIN_VOLUME_SHARES:
            return False
        if current_prob is None:
            return False

        # Liquidity check
        dollar_volume = current_close * current_volume
        if dollar_volume < MIN_DOLLAR_VOLUME:
            return False

        # Build historical UpProbability series
        historic_probs = []
        i = 1
        while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
            try:
                prob_val = data.UpProbability[i]
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        # Limited data sanity check
        if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
            # Reject symbols with recent drops when data is limited
            try:
                for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]

                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < MAX_SINGLE_DAY_DROP:
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break
            except Exception:
                return False

            # Need minimum viable data even with sanity check
            if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                return False

            # For limited data, use simplified thresholds
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
            else:
                return False
        else:
            # Standard percentile thresholds for sufficient data
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
            else:
                return False

        # --- Extra Failure Filters ---

        # 1. 52-week high filter
        try:
            closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                lowest_52w = min(closes_252)
                if highest_52w and lowest_52w and highest_52w > 0:
                    if (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                        return False
        except Exception:
            pass

        # 2. Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > MAX_MOMENTUM_GAIN:
                    return False
                if ret_5d < MAX_MOMENTUM_LOSS:
                    return False
        except Exception:
            pass

        # 3. Volume spike filter
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                    return False
        except Exception:
            pass

        # 4. Volatility filter
        try:
            closes = data.close.get(size=VOLATILITY_LOOKBACK_DAYS)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > MAX_VOLATILITY_THRESHOLD:
                        return False
        except Exception:
            pass



        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 30

        # 5. RSI filter
        try:
            closes_for_rsi = data.close.get(size=RSI_PERIOD + 1)
            if closes_for_rsi is not None and len(closes_for_rsi) >= RSI_PERIOD + 1:
                # Calculate price changes
                deltas = np.diff(closes_for_rsi)

                # Separate gains and losses
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                # Calculate average gain and loss
                avg_gain = np.mean(gains[-RSI_PERIOD:])
                avg_loss = np.mean(losses[-RSI_PERIOD:])

                # Calculate RSI
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

 

        # --- Core Buy Condition ---
        if current_prob >= p96 and current_prob < p97_5:
            return True

        return False















    def can_buy222(self, data, current_date):
        # Data-driven update 2026-05-15. Conservative tweaks vs can_buy_15May2026:
        #   MAX_VOLATILITY_THRESHOLD: 0.04 -> 0.05   (EDA: CAGR +16pts, DD improves)
        #   MIN_RSI_THRESHOLD:        30   -> 20    (EDA: small consistent win)
        # Companion change outside this function: sort_buy_candidates reverse=False -> True.

        # Strategy Timing
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        # Probability Bounds
        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        # Financial Thresholds
        MIN_CLOSE_PRICE = 1.50
        MAX_CLOSE_PRICE = 1000.00
        MIN_VOLUME_SHARES = 10_000
        MIN_DOLLAR_VOLUME = 1_000_000

        # Risk Management - Drop Protection
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10

        # Risk Management - 52-Week Position
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85
        WEEK_52_LOOKBACK_DAYS = 252

        # Risk Management - Momentum
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.075

        # Risk Management - Volume & Volatility
        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.05  # was 0.04 - EDA shows 5% cap raises CAGR and lowers DD
        VOLATILITY_LOOKBACK_DAYS = 20

        # Percentile Thresholds (sufficient data)
        SUFFICIENT_DATA_P_LOW = 90.0
        SUFFICIENT_DATA_P_HIGH = 95.0

        # Percentile Thresholds (limited data)
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99

        # RSI
        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 20  # was 30 - EDA shows softening adds trades without quality loss

        # ============ END CONFIGURATION ============

        symbol = data._name

        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date

        days_since_start = (current_date - self.strategy_start_date).days
        if days_since_start < MIN_DAYS_BEFORE_TRADING:
            return False

        try:
            current_close = data.close[0]
            current_prob = data.UpProbability[0]
            current_volume = data.volume[0]
        except (IndexError, AttributeError, TypeError):
            return False

        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
            return False

        try:
            if len(data.UpProbability) < 6:
                return False
            recent_probs = [data.UpProbability[i] for i in range(1, 6) if data.UpProbability[i] is not None]

            if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                return False
            if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                return False
        except Exception:
            pass

        if current_close is None or current_close < MIN_CLOSE_PRICE:
            return False
        if current_close is None or current_close > MAX_CLOSE_PRICE:
            return False
        if current_volume is None or current_volume < MIN_VOLUME_SHARES:
            return False
        if current_prob is None:
            return False

        dollar_volume = current_close * current_volume
        if dollar_volume < MIN_DOLLAR_VOLUME:
            return False

        historic_probs = []
        i = 1
        while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
            try:
                prob_val = data.UpProbability[i]
                if prob_val is not None:
                    historic_probs.append(float(prob_val))
                i += 1
            except (IndexError, AttributeError, TypeError):
                break

        if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
            try:
                for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                    try:
                        prev_close = data.close[-lookback]
                        next_close = data.close[-(lookback-1)] if lookback > 1 else data.close[0]

                        if prev_close is not None and next_close is not None and prev_close > 0:
                            daily_return = (next_close / prev_close) - 1
                            if daily_return < MAX_SINGLE_DAY_DROP:
                                return False
                    except (IndexError, AttributeError, TypeError):
                        break
            except Exception:
                return False

            if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                return False

            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
            else:
                return False
        else:
            if len(historic_probs) > 1:
                p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
            else:
                return False

        # 52-week high filter
        try:
            closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                lowest_52w = min(closes_252)
                if highest_52w and lowest_52w and highest_52w > 0:
                    if (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                        return False
        except Exception:
            pass

        # Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if ret_5d > MAX_MOMENTUM_GAIN:
                    return False
                if ret_5d < MAX_MOMENTUM_LOSS:
                    return False
        except Exception:
            pass

        # Volume spike filter
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and len(vols) > 0:
                avg_vol_20 = np.mean(vols)
                if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                    return False
        except Exception:
            pass

        # Volatility filter (cap raised from 0.04 to 0.05)
        try:
            closes = data.close.get(size=VOLATILITY_LOOKBACK_DAYS)
            if closes is not None and len(closes) > 1:
                returns_20d = np.diff(np.log(closes))
                if len(returns_20d) > 0:
                    vol_20d = np.std(returns_20d)
                    if vol_20d > MAX_VOLATILITY_THRESHOLD:
                        return False
        except Exception:
            pass

        # RSI filter (hard-reject threshold lowered from 30 to 20)
        try:
            closes_for_rsi = data.close.get(size=RSI_PERIOD + 1)
            if closes_for_rsi is not None and len(closes_for_rsi) >= RSI_PERIOD + 1:
                deltas = np.diff(closes_for_rsi)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-RSI_PERIOD:])
                avg_loss = np.mean(losses[-RSI_PERIOD:])

                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

        # --- Core Buy Condition ---
        if current_prob >= p96 and current_prob < p97_5:
            return True

        return False




    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    
    # Optional: Add a method to track blacklisted stocks
    def update_blacklist(self):
        """
        Maintain a blacklist of stocks with recent major events
        This can be called periodically to update the blacklist
        """
        if not hasattr(self, 'stock_blacklist'):
            self.stock_blacklist = {}
        
        current_date = self.datetime.date()
        
        # Clean up old blacklist entries (older than 30 days)
        for symbol in list(self.stock_blacklist.keys()):
            if (current_date - self.stock_blacklist[symbol]['date']).days > 110:
                del self.stock_blacklist[symbol]
    
    
    def force_best_signal_for_current_day(self, data=None):
        """Find and save the best possible stock for the current day using IDENTICAL logic to can_buy function."""
        logging.info("Finding best signals using identical can_buy criteria...")
    
        # Use the exact same configuration constants as can_buy
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5
    
        UP_PROB_MIN_BOUND = 0.35
        UP_PROB_MAX_BOUND = 0.60
    
        MIN_CLOSE_PRICE = 1.50
        MAX_CLOSE_PRICE = 1000.00
        MIN_VOLUME_SHARES = 10_000
        MIN_DOLLAR_VOLUME = 500_000
    
        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10
    
        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.85
        WEEK_52_LOOKBACK_DAYS = 252
    
        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.075
    
        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.04
        VOLATILITY_LOOKBACK_DAYS = 20
    
        SUFFICIENT_DATA_P_LOW = 90.0
        SUFFICIENT_DATA_P_HIGH = 95.0
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 99
    
        # Strategy start date check (same as can_buy)
        current_date = self.datetime.date()
        if not hasattr(self, 'strategy_start_date'):
            self.strategy_start_date = current_date
    
        days_since_start = (current_date - self.strategy_start_date).days
    
        # Create a list to store candidates that pass ALL can_buy criteria
        valid_candidates = []
    
        for d in self.datas:
            if self.getposition(d).size > 0:
                continue  # Skip stocks we already have positions in
            
            symbol = d._name
            if self.rule_201_monitor.is_restricted(symbol):
                continue  # Skip stocks restricted by Rule 201
            
            # Apply EXACT same filters as can_buy function
            try:
                # 1. Strategy timing filter
                if days_since_start < MIN_DAYS_BEFORE_TRADING:
                    continue
                
                # 2. Current values extraction
                try:
                    current_close = d.close[0]
                    current_prob = d.UpProbability[0]
                    current_volume = d.volume[0]
                except (IndexError, AttributeError, TypeError):
                    continue
                
                # 3. Basic probability bounds
                if current_prob < UP_PROB_MIN_BOUND or current_prob > UP_PROB_MAX_BOUND:
                    continue

                try:
                    #make sure up prob is within the range that shows the best results and not some weird values like 0.1 or 0.9
                    if len(d.UpProbability) < 6:
                        continue
                    recent_probs = [d.UpProbability[i] for i in range(1, 6) if d.UpProbability[i] is not None]

                    if recent_probs and min(recent_probs) < UP_PROB_MIN_BOUND:
                        continue
                    if recent_probs and max(recent_probs) > UP_PROB_MAX_BOUND:
                        continue
                except Exception:
                    pass
                
                # 4. Hard rejects (price, volume, prob)
                if current_close is None or current_close < MIN_CLOSE_PRICE:
                    continue
                if current_close > MAX_CLOSE_PRICE:
                    continue
                if current_volume is None or current_volume < MIN_VOLUME_SHARES:
                    continue
                if current_prob is None:
                    continue
                
                # 5. Liquidity check
                dollar_volume = current_close * current_volume
                if dollar_volume < MIN_DOLLAR_VOLUME:
                    continue
                
                # 6. Build historical UpProbability series (identical to can_buy)
                historic_probs = []
                i = 1
                while len(historic_probs) < TARGET_HISTORIC_PROB_COUNT and i < MAX_HISTORIC_LOOKBACK:
                    try:
                        prob_val = d.UpProbability[i]
                        if prob_val is not None:
                            historic_probs.append(float(prob_val))
                        i += 1
                    except (IndexError, AttributeError, TypeError):
                        break
                    
                # 7. Limited data sanity check (identical to can_buy)
                if len(historic_probs) < MIN_HISTORIC_PROB_THRESHOLD:
                    # Check for recent drops when data is limited
                    drop_detected = False
                    try:
                        for lookback in range(1, min(RECENT_DROP_LOOKBACK_DAYS + 1, 100)):
                            try:
                                prev_close = d.close[-lookback]
                                next_close = d.close[-(lookback-1)] if lookback > 1 else d.close[0]
    
                                if prev_close is not None and next_close is not None and prev_close > 0:
                                    daily_return = (next_close / prev_close) - 1
                                    if daily_return < MAX_SINGLE_DAY_DROP:
                                        drop_detected = True
                                        break
                            except (IndexError, AttributeError, TypeError):
                                break
                    except Exception:
                        drop_detected = True
    
                    if drop_detected:
                        continue
                    
                    # Need minimum viable data
                    if len(historic_probs) < MIN_VIABLE_DATA_POINTS:
                        continue
                    
                    # Use limited data thresholds
                    if len(historic_probs) > 1:
                        p96 = np.percentile(historic_probs, LIMITED_DATA_P_LOW)
                        p97_5 = np.percentile(historic_probs, LIMITED_DATA_P_HIGH)
                    else:
                        continue
                else:
                    # Use standard thresholds for sufficient data
                    if len(historic_probs) > 1:
                        p96 = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW)
                        p97_5 = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH)
                    else:
                        continue
                    
                # 8. 52-week high filter
                try:
                    closes_252 = d.close.get(size=WEEK_52_LOOKBACK_DAYS)
                    if closes_252 and len(closes_252) > 0:
                        highest_52w = max(closes_252)
                        lowest_52w = min(closes_252)
                        if highest_52w and lowest_52w and highest_52w > 0:
                            if (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                                continue
                except Exception:
                    pass
                
                # 9. Momentum filter
                try:
                    prev_close_5 = d.close[-MOMENTUM_LOOKBACK_DAYS]
                    if prev_close_5 and prev_close_5 > 0:
                        ret_5d = (current_close / prev_close_5) - 1
                        if ret_5d > MAX_MOMENTUM_GAIN or ret_5d < MAX_MOMENTUM_LOSS:
                            continue
                except Exception:
                    pass
                
                # 10. Volume spike filter
                try:
                    vols = d.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
                    if vols is not None and len(vols) > 0:
                        avg_vol_20 = np.mean(vols)
                        if avg_vol_20 and current_volume / avg_vol_20 > VOLUME_SPIKE_MULTIPLIER:
                            continue
                except Exception:
                    pass
                
                # 11. Volatility filter
                try:
                    closes = d.close.get(size=VOLATILITY_LOOKBACK_DAYS)
                    if closes is not None and len(closes) > 1:
                        returns_20d = np.diff(np.log(closes))
                        if len(returns_20d) > 0:
                            vol_20d = np.std(returns_20d)
                            if vol_20d > MAX_VOLATILITY_THRESHOLD:
                                continue
                except Exception:
                    pass

                # 12. CORE BUY CONDITION (identical to can_buy)
                if current_prob >= p96 and current_prob < p97_5:
                    # This stock passes ALL can_buy criteria
                    size = self.calculate_position_size(d)
                    if size > 0:
                        # Calculate the same quality score as before for ranking
                        quality_score = current_prob * 100  # Simple ranking by probability
                        valid_candidates.append((d, size, 0, quality_score, current_prob, p96, p97_5))
    
            except Exception as e:
                logging.warning(f"Error evaluating candidate {symbol} with can_buy logic: {str(e)}")
    
        # Sort by quality score (highest probability first, just like can_buy preference)
        valid_candidates.sort(key=lambda x: x[3], reverse=True)
    
        if valid_candidates:
            # Log the results
            logging.info(f"Found {len(valid_candidates)} stocks that pass identical can_buy criteria:")
            for i, (d, size, _, quality_score, current_prob, p96, p97_5) in enumerate(valid_candidates[:5]):
                logging.info(f"  {i+1}. {d._name}: UpProb={current_prob:.4f} (threshold: {p96:.4f}-{p97_5:.4f})")
    
            # Return only the data/size/correlation tuples for compatibility
            best_candidates = [(candidate[0], candidate[1], candidate[2]) for candidate in valid_candidates]

            logging.info(f"Selected {len(best_candidates)} signals using identical can_buy criteria")
        else:
            logging.warning("No stocks passed the identical can_buy criteria filters")
            # If no stocks pass the strict criteria, you could choose to:
            # 1. Return empty list (no signals)
            # 2. Use a fallback method
            # For now, return empty to maintain consistency
            pass























    
    
    def process_buy_candidates(self, buy_candidates, current_date, verbose=False):
        """Process buy candidates and execute trades."""
        if verbose:
            dprint("Starting process_buy_candidates", "INFO")
            dprint(f"Current date: {current_date}", "INFO")
            dprint(f"Number of buy candidates: {len(buy_candidates)}", "INFO")
            dprint(f"Last trading date: {self.last_trading_date}", "INFO")
        
        self.force_best_signal_for_current_day()
        if verbose:
            dprint("Completed force_best_signal_for_current_day", "INFO")
    
        buy_candidates = self.sort_buy_candidates(buy_candidates)
        if verbose:
            dprint(f"Sorted buy candidates: {len(buy_candidates)}", "INFO")
    
        # Save the most-recent-day candidate POOL to Data/0__signals.parquet (canonical signals file)
        # FAST: gate to the LAST trading date only. This export block scrapes finviz +
        # runs FinBERT sentiment (save_guaranteed_signals_to_parquet) — on every historical
        # bar that was ~64% of backtest runtime (1225s of SSL reads). It's post-decision
        # (live-signal export), so running it only on the final bar is LOSSLESS for backtest
        # metrics and matches the experimental backtester's fix.
        if buy_candidates and current_date == self.last_trading_date:
            if verbose:
                dprint("Converting candidates to new signal format", "INFO")
            signals = []
            
            # Get real-world current date for live trading
            real_current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(real_current_date)
            if verbose:
                dprint(f"Real current date: {real_current_date}", "INFO")
                dprint(f"Next trading day for signals: {next_trading_day}", "INFO")
            
            # Write a candidate POOL (not just max_positions) so the morning
            # FilterRubric narrowing has a surplus to trim down to the final book.
            # The broker only ever trades the narrowed _Buy_Signals.parquet, never
            # this raw pool.
            SIGNAL_POOL_SIZE = 12
            for d, size, correlation in buy_candidates[:SIGNAL_POOL_SIZE]:
                price = d.close[0]
                atr = self.inds[d]['atr'][0] if d in self.inds and 'atr' in self.inds[d] else price * 0.02
                
                signal = {
                    'Symbol': d._name,
                    'Price': price,
                    'UpProbability': d.UpProbability[0],
                    'ATR': atr,
                    'Quality': d.UpProbability[0],  # Use probability as quality measure
                    'DollarVolume': d.volume[0] * price,
                    'ThresholdValue': d.UpProbability[0],  # The threshold that triggered this signal
                }
                signals.append(signal)
                if verbose:
                    dprint(f"Created signal for {d._name} at ${price:.2f} with UpProb {d.UpProbability[0]:.3f}", "DETAIL")
            
            # Save to new consolidated format
            try:
                success = save_guaranteed_signals_to_parquet(signals, next_trading_day)
                if success:
                    if verbose:
                        dprint(f"Successfully saved {len(signals)} signals to consolidated format", "SUCCESS")
                else:
                    if verbose:
                        dprint("Failed to save signals to consolidated format", "ERROR")
            except Exception as e:
                if verbose:
                    dprint(f"Error saving to consolidated format: {str(e)}", "ERROR")
        else:
            if verbose:
                dprint("No buy candidates to save to consolidated format", "WARN")
    
        # Execute trades as normal
        for d, size, _ in buy_candidates:
            if self.open_positions < self.p.max_positions:
                if self.check_group_allocation(d):
                    self.execute_buy(d, size, current_date)
                    if verbose:
                        dprint(f"Executed buy order for {d._name}", "SUCCESS")
                else:
                    if verbose:
                        dprint(f"Skipped {d._name} due to group allocation limits", "INFO")
            else:
                if verbose:
                    dprint("Max positions reached, stopping buy executions", "INFO")
                break
            





    def sort_buy_candidates(self, buy_candidates):
        # 2026-05-15: flipped to descending after EDA showed the asc sort discarded
        # the model's highest-conviction qualifying candidates. Sharpe 0.80 -> 3.66.
        sorted_candidates = sorted(buy_candidates, key=lambda x: x[0].UpProbability[0], reverse=True)
        Verbose = False

        if Verbose:
            if sorted_candidates:
                logging.info("Top buy candidates based on UpProbability:")
                for i, (d, size, corr) in enumerate(sorted_candidates[:min(5, len(sorted_candidates))]):
                    logging.info(f"  {i+1}. {d._name}: UpProb={d.UpProbability[0]:.4f}")

        return sorted_candidates
    





    



    def get_mean_correlation(self, candidate_ticker, current_positions):
        try:
            if not current_positions:
                return 0

            if candidate_ticker not in self.correlation_df.index:

                ##logging.warning(f"Ticker {candidate_ticker} not found in correlation data")
                return 0

            correlations = []

            candidate_row = self.correlation_df.loc[candidate_ticker]

            for pos in current_positions:
                if pos not in self.correlation_df.index:
                    continue

                position_cluster = self.correlation_df.loc[pos, 'Cluster']

                cluster_column = f"correlation_{position_cluster}"
                if cluster_column in self.correlation_df.columns:
                    corr_value = candidate_row[cluster_column]
                    correlations.append(abs(corr_value))  # Use absolute correlation

            return np.mean(correlations) if correlations else 0

        except Exception as e:
            logging.error(f"Error calculating correlations: {str(e)}")
            return 0
    




    def check_group_allocation(self, data):
        """Check if adding a position would exceed group allocation limits.
        Also reject stocks that are in the outlier group (-1).
        """
        symbol = data._name

        # Find which group the stock belongs to
        group = None
        try:
            if hasattr(self.correlation_df, 'index') and hasattr(self.correlation_df.index, 'contains'):
                if symbol in self.correlation_df.index:
                    group = int(self.correlation_df.loc[symbol, 'Cluster'])
            elif 'Ticker' in self.correlation_df.columns:
                ticker_row = self.correlation_df[self.correlation_df['Ticker'] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
            else:
                first_col = self.correlation_df.columns[0]
                ticker_row = self.correlation_df[self.correlation_df[first_col] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
        except Exception as e:
            logging.warning(f"Error finding group for {symbol}: {str(e)}")

        # Reject stocks in the outlier group (-1)
        if group == -1:
            logging.info(f"Rejecting {symbol} because it's in the outlier group (-1)")
            return False

        if group is None:
            logging.info(f"No group found for {symbol}, allowing trade")
            return True  # If no group data, allow the trade

        # Check if adding this stock would exceed the group allocation limit
        current_allocation = self.group_allocations.get(group, 0)
        return current_allocation < self.p.max_group_allocation




    def execute_buy(self, data, size, current_date):
        """Replace the old execute_buy with bracket order version"""
        return self.execute_buy_with_bracket(data, size, current_date)
    



    def execute_buy_with_bracket(self, data, size, current_date):
        """Alternative: Execute buy using buy_bracket method with trailing stop"""
        symbol = data._name
        current_price = data.close[0]

        # Calculate target price and trailing stop parameters
        take_profit_percent = 20
        trailing_stop_percent = 3.0

        target_price = current_price * (1 + take_profit_percent / 100.0)

        logging.info(f"BUY BRACKET {symbol}: Entry=${current_price:.2f}, Size={size}, "
                    f"TrailStop={trailing_stop_percent}%, Target=${target_price:.2f}")

        # METHOD 2: Single bracket order method with trailing stop
        bracket_orders = self.buy_bracket(
            data=data,
            size=size,
            price=current_price,  # Main order entry price
            exectype=bt.Order.Market,  # Main order type

            # Trailing stop loss configuration
            stopexec=bt.Order.StopTrail,  # Use trailing stop instead of regular stop
            trailpercent=trailing_stop_percent / 100.0,  # 2% trailing stop

            # Take profit configuration
            limitprice=target_price,
            limitexec=bt.Order.Limit
        )

        if bracket_orders:
            main_order, stop_order, target_order = bracket_orders

            # Store order info for tracking
            self.entry_prices[data] = current_price
            self.position_dates[data] = current_date
            self.open_positions += 1

            # Track the bracket orders
            self.bracket_orders[data] = {
                'main': main_order,
                'stop': stop_order, 
                'target': target_order,
                'entry_price': current_price,
                'stop_type': 'trailing',
                'trail_percent': trailing_stop_percent,
                'target_price': target_price
            }

            self.update_group_data(data)
            update_buy_signal(symbol, current_date, current_price, data.UpProbability[0])

            return True

        return False




    def update_group_data(self, data):
        try:
            symbol = data._name
            
            if symbol in self.correlation_df.index:
                group = int(self.correlation_df.loc[symbol, 'Cluster'])
                self.asset_groups[symbol] = group
            
            self.update_group_allocations()
            
        except Exception as e:
            logging.error(f"Error updating group data: {str(e)}")
    



    def update_group_allocations(self):
        """Update the allocation percentages for each cluster group based on current positions."""
        # Calculate total portfolio value
        total_value = self.broker.getvalue()

        # Initialize all possible group IDs including:
        # - Regular groups (could start at 0 or 1 depending on clustering algorithm)
        # - Special outlier group (-1)
        # - Any possible group values in asset_groups

        # First reset allocations for numbered groups
        self.group_allocations = {}

        # Initialize all potential group IDs from 0 to total_groups
        for i in range(0, self.total_groups + 1):
            self.group_allocations[i] = 0.0

        # Also add the outlier group (-1)
        self.group_allocations[-1] = 0.0

        # Collect any additional group IDs that might exist in asset_groups
        for symbol, group in self.asset_groups.items():
            if group not in self.group_allocations:
                self.group_allocations[group] = 0.0

        # If portfolio is empty, set equal allocations and return
        if total_value == 0:
            group_count = len([g for g in self.group_allocations.keys() if g >= 0])  # Don't count outlier group
            if group_count > 0:
                equal_alloc = 1.0 / group_count
                for group in self.group_allocations.keys():
                    if group >= 0:  # Only allocate to real groups, not outliers
                        self.group_allocations[group] = equal_alloc
            return

        # Update allocations based on current positions
        for data in self.datas:
            position = self.getposition(data)
            if position.size > 0:
                symbol = data._name

                # Default to outlier group if not found
                group = self.asset_groups.get(symbol, -1)

                # Make sure the group is in our allocations dict (defensive programming)
                if group not in self.group_allocations:
                    logging.warning(f"Found unexpected group ID {group} for {symbol}, adding to allocations")
                    self.group_allocations[group] = 0.0

                # Calculate position value
                position_value = position.size * data.close[0]

                # Update allocation - now safe since we've handled all possible groups
                self.group_allocations[group] += position_value / total_value













    def evaluate_sell_conditions(self, data, current_date):
        """Enhanced: Timeout + 5-day Momentum-based exit"""
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

        entry_date = self.position_dates.get(data, current_date)
        days_held = (current_date - entry_date).days

        try:
            current_prob = data.UpProbability[0]
            momentum = current_prob - data.UpProbability[-1]

            # Early exit conditions
            if momentum <= -0.05:
                logging.info(f"MOMENTUM SELL {symbol}: 5-day momentum {momentum:.3f} (Current: {current_prob:.3f})")
                return self.exit_position(data)

            if days_held >= self.p.position_timeout:
                logging.info(f"TIMEOUT SELL {symbol}: Held for {days_held} days")
                return self.exit_position(data)

            # Probability drop check (only after 3+ days)
            if days_held >= 3:
                recent_probs = [float(data.UpProbability[-i]) for i in range(1, min(11, len(data))) 
                               if data.UpProbability[-i] is not None]

                if recent_probs and max(recent_probs) > 0.55 and current_prob < 0.48:
                    logging.info(f"PROB DROP SELL {symbol}: Max recent {max(recent_probs):.3f}, Current {current_prob:.3f}")
                    return self.exit_position(data)

        except (IndexError, AttributeError, TypeError) as e:
            logging.warning(f"No probability data for {symbol} on {current_date}: {e}")







    def evaluate_sell_conditions__UpProbDropCondition(self, data, current_date):
        """Enhanced: Timeout + Momentum + Day 1 UpProbability Drop Signal"""
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

        entry_date = self.position_dates.get(data, current_date)
        days_held = (current_date - entry_date).days

        try:
            current_prob = data.UpProbability[0]

            # NEW: Day 1 UpProbability Drop Analysis
            if days_held == 1:
                try:
                    # Get yesterday's UpProbability (entry day)
                    entry_prob = data.UpProbability[-1]

                    if entry_prob is not None and current_prob is not None:
                        # Calculate the drop from entry to current
                        prob_drop = entry_prob - current_prob
                        drop_percentage = prob_drop / entry_prob if entry_prob > 0 else 0

                        # If there's LESS than 3% drop, this might be a losing trade - consider exit
                        if drop_percentage < 0.03:  # Less than 3% drop
                            logging.info(f"DAY 1 WEAK DROP SELL {symbol}: Only {drop_percentage:.1%} drop "
                                       f"(Entry: {entry_prob:.3f} -> Current: {current_prob:.3f}). "
                                       f"Pattern suggests potential loser.")
                            return self.exit_position(data)
                        else:
                            # Good drop pattern - log but continue holding
                            logging.info(f"DAY 1 GOOD DROP {symbol}: {drop_percentage:.1%} drop "
                                       f"(Entry: {entry_prob:.3f} -> Current: {current_prob:.3f}). "
                                       f"Pattern suggests potential winner - holding.")

                except (IndexError, AttributeError, TypeError) as e:
                    logging.warning(f"Could not analyze day 1 drop for {symbol}: {e}")

            # Existing momentum check (5-day)
            if len(data.UpProbability) > 5:
                momentum = current_prob - data.UpProbability[-5]
                if momentum <= -0.05:
                    logging.info(f"MOMENTUM SELL {symbol}: 5-day momentum {momentum:.3f} (Current: {current_prob:.3f})")
                    return self.exit_position(data)

            # Existing timeout check
            if days_held >= self.p.position_timeout:
                logging.info(f"TIMEOUT SELL {symbol}: Held for {days_held} days")
                return self.exit_position(data)

            # Existing probability drop check (only after 3+ days)
            if days_held >= 3:
                recent_probs = [float(data.UpProbability[-i]) for i in range(1, min(11, len(data))) 
                               if data.UpProbability[-i] is not None]

                if recent_probs and max(recent_probs) > 0.55 and current_prob < 0.48:
                    logging.info(f"PROB DROP SELL {symbol}: Max recent {max(recent_probs):.3f}, Current {current_prob:.3f}")
                    return self.exit_position(data)

        except (IndexError, AttributeError, TypeError) as e:
            logging.warning(f"No probability data for {symbol} on {current_date}: {e}")










    def exit_position(self, data):
        """Close position and clean up bracket tracking."""
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

        logging.info(f"EXIT POSITION {symbol}: Closing position of size {position.size}")
        self.close(data=data)

        # Clean up bracket tracking
        if data in self.bracket_orders:
            try:
                bracket = self.bracket_orders[data]
                if bracket['stop'] and bracket['stop'].alive():
                    self.cancel(bracket['stop'])
                if bracket['target'] and bracket['target'].alive():
                    self.cancel(bracket['target'])
            except Exception as e:
                logging.warning(f"Error canceling bracket orders for {symbol}: {e}")
            del self.bracket_orders[data]


















    def save_best_buy_signals___NONLOGGING(self, buy_candidates):
        """Save the best buy signals to a parquet file for the live trader."""
        # Use REAL-WORLD current date, not backtest date
        current_real_date = datetime.now().date()  # Real world date
        backtest_date = self.datetime.date()       # Backtest data date

        # Add data freshness check using real world date
        try:
            market_last_date = get_last_trading_date()
            if current_real_date <= market_last_date:
                data_age = (market_last_date - backtest_date).days
                if data_age < 5:
                    logging.info(f"Data processing: Backtest date: {backtest_date}, Real date: {current_real_date}, Market: {market_last_date}")
        except Exception as e:
            logging.error(f"Error checking data freshness: {str(e)}")

        # Calculate next trading day from REAL-WORLD date
        next_trading_day = get_next_trading_day(current_real_date)

        # Create a fresh DataFrame for new signals
        signal_data = []
        for d, size, correlation in buy_candidates[:self.p.max_positions]:
            price = round(d.close[0], 3)

            signal_data.append({
                'Symbol': str(d._name),
                'LastBuySignalDate': pd.Timestamp(next_trading_day),
                'LastBuySignalPrice': float(price),
                'IsCurrentlyBought': False,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.NaT,
                'UpProbability': float(d.UpProbability[0]),
                'LastSellPrice': float('nan'),
                'PositionSize': float('nan')
            })

            logging.info(f"Prepared buy signal for {d._name} at {price} for {next_trading_day}")

        if signal_data:
            # Create completely new DataFrame with just these signals
            new_signals_df = pd.DataFrame(signal_data)

            # Option 1: Complete overwrite - replace ALL signals
            # write_trading_data(new_signals_df)

            # Option 2: Maintain currently bought positions but replace all signals
            df = read_trading_data()

            # Keep only currently bought positions
            currently_bought = df[df['IsCurrentlyBought'] == True]

            # Create final DataFrame with both bought positions and new signals
            final_df = pd.concat([currently_bought, new_signals_df], ignore_index=True)

            # Remove duplicates in case a symbol is both bought and in new signals
            final_df = final_df.drop_duplicates(subset=['Symbol'], keep='first')

            write_trading_data(final_df)

            #logging.info(f"Successfully wrote {len(signal_data)} new buy signals and synchronized with live trader")




    def save_best_buy_signals(self, buy_candidates):
        """Save the best buy signals to a parquet file for the live trader."""
        dprint("Starting save_best_buy_signals function", "INFO")

        # Use REAL-WORLD current date, not backtest date
        current_real_date = datetime.now().date()
        dprint(f"Real world current date: {current_real_date}", "INFO")

        backtest_date = self.datetime.date()
        dprint(f"Backtest data date: {backtest_date}", "INFO")

        # Add data freshness check using real world date
        try:
            market_last_date = get_last_trading_date()
            dprint(f"Market last trading date: {market_last_date}", "INFO")

            if current_real_date <= market_last_date:
                data_age = (market_last_date - backtest_date).days
                dprint(f"Data age: {data_age} days", "INFO")

                if data_age < 5:
                    dprint(f"Data processing: Backtest date: {backtest_date}, Real date: {current_real_date}, Market: {market_last_date}", "INFO")
            else:
                dprint(f"Current date {current_real_date} is after market last date {market_last_date}", "WARN")

        except Exception as e:
            dprint(f"Error checking data freshness: {str(e)}", "ERROR")

        # Calculate next trading day from REAL-WORLD date
        dprint(f"Calculating next trading day from real date: {current_real_date}", "INFO")

        try:
            next_trading_day = get_next_trading_day(current_real_date)
            dprint(f"Calculated next trading day: {next_trading_day}", "SUCCESS")
        except Exception as e:
            dprint(f"Error calculating next trading day: {str(e)}", "ERROR")
            # Fallback to a simple calculation
            next_trading_day = current_real_date + timedelta(days=1)
            dprint(f"Using fallback next trading day: {next_trading_day}", "WARN")

        dprint(f"Number of buy candidates: {len(buy_candidates)}", "INFO")
        dprint(f"Max positions from params: {self.p.max_positions}", "INFO")

        # Create a fresh DataFrame for new signals
        signal_data = []
        candidates_to_process = buy_candidates[:self.p.max_positions]
        dprint(f"Processing {len(candidates_to_process)} candidates", "INFO")

        for i, (d, size, correlation) in enumerate(candidates_to_process):
            dprint(f"Processing candidate {i+1}: {d._name}", "DETAIL")

            price = round(d.close[0], 3)
            dprint(f"Stock {d._name} price: {price}", "DETAIL")
            dprint(f"Stock {d._name} UpProbability: {d.UpProbability[0]}", "DETAIL")

            signal_record = {
                'Symbol': str(d._name),
                'LastBuySignalDate': pd.Timestamp(next_trading_day),
                'LastBuySignalPrice': float(price),
                'IsCurrentlyBought': False,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.NaT,
                'UpProbability': float(d.UpProbability[0]),
                'LastSellPrice': float('nan'),
                'PositionSize': float('nan')
            }

            dprint(f"Created signal record for {d._name} with date: {signal_record['LastBuySignalDate']}", "DETAIL")
            signal_data.append(signal_record)

            dprint(f"Prepared buy signal for {d._name} at {price} for {next_trading_day}", "INFO")

        dprint(f"Total signals prepared: {len(signal_data)}", "INFO")

        if signal_data:
            dprint("Creating new signals DataFrame", "INFO")

            # Create completely new DataFrame with just these signals
            new_signals_df = pd.DataFrame(signal_data)
            dprint(f"New signals DataFrame shape: {new_signals_df.shape}", "DETAIL")
            dprint(f"Sample of new signals dates: {new_signals_df['LastBuySignalDate'].unique()}", "INFO")

            dprint("Reading existing trading data", "INFO")
            df = read_trading_data()
            dprint(f"Existing trading data shape: {df.shape}", "DETAIL")

            # Keep only currently bought positions
            currently_bought = df[df['IsCurrentlyBought'] == True]
            dprint(f"Currently bought positions: {len(currently_bought)}", "INFO")

            # Create final DataFrame with both bought positions and new signals
            dprint("Concatenating bought positions with new signals", "INFO")
            final_df = pd.concat([currently_bought, new_signals_df], ignore_index=True)
            dprint(f"Final DataFrame shape before dedup: {final_df.shape}", "DETAIL")

            # Remove duplicates in case a symbol is both bought and in new signals
            final_df = final_df.drop_duplicates(subset=['Symbol'], keep='first')
            dprint(f"Final DataFrame shape after dedup: {final_df.shape}", "DETAIL")

            # Debug: Check what dates are actually in the final DataFrame
            if 'LastBuySignalDate' in final_df.columns:
                unique_dates = final_df['LastBuySignalDate'].unique()
                dprint(f"Final DataFrame signal dates: {unique_dates}", "INFO")

            dprint("Writing trading data to file", "INFO")
            try:
                write_trading_data(final_df)
                #dprint("Successfully wrote trading data", "SUCCESS")
            except Exception as e:
                dprint(f"Error writing trading data: {str(e)}", "ERROR")
                raise

            #dprint(f"Successfully wrote {len(signal_data)} new buy signals and synchronized with live trader", "SUCCESS")

            # Verification step - read back the file to confirm what was written
            dprint("Verifying written signals by reading back from file", "INFO")
            try:
                verification_df = read_trading_data()
                if not verification_df.empty and 'LastBuySignalDate' in verification_df.columns:
                    written_dates = verification_df['LastBuySignalDate'].unique()
                    dprint(f"Verification: Actual dates in file: {written_dates}", "INFO")

                    pending_signals = verification_df[verification_df['IsCurrentlyBought'] == False]
                    dprint(f"Verification: Found {len(pending_signals)} pending signals", "INFO")

                    for _, row in pending_signals.iterrows():
                        signal_date = row['LastBuySignalDate']
                        dprint(f"Verification: {row['Symbol']} has signal date {signal_date}", "DETAIL")
                else:
                    dprint("Verification: No signals found in written file", "WARN")

            except Exception as e:
                dprint(f"Error during verification: {str(e)}", "ERROR")
        else:
            dprint("No signal data to save", "WARN")




    def notify_order(self, order):

        if order.status in [order.Completed, order.Partial]:
            self.handle_order_execution(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.handle_order_failure(order)
    

        ##logging of oopen / close close and the comissions/ slippage 



    def handle_order_execution(self, order):

        if order.isbuy():
            self.handle_buy_execution(order)
        elif order.issell():
            self.handle_sell_execution(order)
    
    def handle_buy_execution(self, order):

        data = order.data
        symbol = data._name

        mark_position_as_bought(symbol, order.executed.size)
        logging.info(f"BUY EXECUTED for {symbol}: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}")

        # Correct entry price to actual fill (market orders fill at next-bar open,
        # not the close stored at signal time in execute_buy_with_bracket).
        self.entry_prices[data] = order.executed.price
    



    def handle_sell_execution(self, order):
        """Handle the completion of a sell order."""
        data = order.data
        symbol = data._name
        
        if self.getposition(data).size == 0:
            self.open_positions -= 1
            
            entry_price = self.entry_prices.get(data)
            exit_price = order.executed.price
            entry_date = self.position_dates.get(data)
            exit_date = self.datetime.date()
            
            if entry_price is not None:
                profit_pct = ((exit_price / entry_price) - 1) * 100
                profit_abs = (exit_price - entry_price) * abs(order.executed.size)
                days_held = (exit_date - entry_date).days if entry_date else 0

                _color = "\033[32m" if profit_pct >= 0 else "\033[31m"
                _sign  = "+" if profit_pct >= 0 else ""

                self.trade_pct_returns.append(profit_pct)
                _total_ret = (self.broker.getvalue() / self.broker.startingcash - 1) * 100
                _n = len(self.trade_pct_returns)
                if _n >= 2:
                    import numpy as _np
                    _arr = _np.array(self.trade_pct_returns)
                    _sharpe = (_arr.mean() / _arr.std(ddof=1)) * (_n ** 0.5) if _arr.std(ddof=1) > 0 else 0.0
                else:
                    _sharpe = 0.0
                _ret_sign = "+" if _total_ret >= 0 else ""
                print(f"{_color}{symbol:<6}  {_sign}{profit_pct:.2f}%\033[0m  |  Total: {_ret_sign}{_total_ret:.2f}%  Sharpe: {_sharpe:.2f}  (n={_n})")
                
                is_win = profit_abs > 0
                is_loss = profit_abs < 0
                
                if is_win:
                    self.winning_trades += 1
                    self.total_win_pnl += profit_abs
                    self.current_win_streak += 1
                    self.current_loss_streak = 0
                    self.recent_outcomes.append(1)
                    if self.current_win_streak > self.longest_win_streak:
                        self.longest_win_streak = self.current_win_streak
                elif is_loss:
                    self.losing_trades += 1
                    self.total_loss_pnl += profit_abs  # This will be negative
                    self.current_loss_streak += 1
                    self.current_win_streak = 0
                    self.recent_outcomes.append(-1)
                    if self.current_loss_streak > self.longest_loss_streak:
                        self.longest_loss_streak = self.current_loss_streak
                else:
                    self.breakeven_trades += 1
                    self.recent_outcomes.append(0)
                
                if len(self.recent_outcomes) > 10:
                    self.recent_outcomes = self.recent_outcomes[-10:]
                
                # Record in our internal trade history
                trade_data = {
                    'Symbol': symbol,
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Quantity': abs(order.executed.size),
                    'PnL': profit_abs,
                    'PnLPct': profit_pct,
                    'DaysHeld': days_held,
                    'Commission': order.executed.comm,
                    'TradeType': 'Long',
                    'ExitReason': self.determine_exit_reason(data),
                    'ATR': self.inds[data]['atr'][0],
                    'UpProbability': data.UpProbability[0],
                    'AccountValue': self.broker.getvalue(),
                }
                
                # Add to internal history
                self.trade_history.append(trade_data)
                
                # Record using TradeRecorder for Data/TradeHistory.parquet
                self.trade_recorder.record_trade(trade_data)
                
                # ALSO DIRECTLY ADD TO COMPLETED TRADES FILE
                # Convert to format expected by Util.add_completed_trade
                from Util import add_completed_trade
                completed_trade_data = {
                    'Symbol': symbol,
                    'EntryDate': pd.Timestamp(entry_date),
                    'ExitDate': pd.Timestamp(exit_date),
                    'EntryPrice': float(entry_price),
                    'ExitPrice': float(exit_price),
                    'PositionSize': float(abs(order.executed.size)),
                    'PnL': float(profit_abs),
                    'PnLPct': float(profit_pct),
                    'DaysHeld': int(days_held),
                    'Commission': float(order.executed.comm),
                    'Slippage': 0.0,    # Default or calculate if available
                    'TradeType': 'Long',
                    'ExitReason': self.determine_exit_reason(data),
                    'ATR': float(self.inds[data]['atr'][0]),
                    'UpProbability': float(data.UpProbability[0]),
                    'AccountValue': float(self.broker.getvalue()),
                    'Source': 'Backtest'
                }
                
                add_completed_trade(completed_trade_data)
                
                is_loss_for_tracking = profit_abs < 0
                update_trade_result(symbol, is_loss_for_tracking, exit_price, exit_date)
            
            if data in self.entry_prices:
                del self.entry_prices[data]
            if data in self.position_dates:
                del self.position_dates[data]
            if data in self.trailing_stops:
                del self.trailing_stops[data]
            if symbol in self.asset_groups:
                del self.asset_groups[symbol]
            
            self.update_group_allocations()
            
            profit_pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0
            logging.info(f"SELL EXECUTED for {symbol}: Price={exit_price:.2f}, "
                       f"Profit={profit_pct:.2f}%, Value={order.executed.value:.2f}")
        



    def determine_exit_reason(self, data):
        """Determine the reason for exiting a position."""
        current_price = data.close[0]
        entry_price = self.entry_prices.get(data, current_price)
        entry_date = self.position_dates.get(data, self.datetime.date())
        trailing_stop = self.trailing_stops.get(data)
        
        # Calculate metrics
        days_held = (self.datetime.date() - entry_date).days
        profit_pct = (current_price / entry_price - 1) * 100
        take_profit_level = entry_price * (1 + self.p.take_profit_percent / 100.0)
        
        # Check conditions
        if trailing_stop and current_price <= trailing_stop:
            if profit_pct >= 0:
                return "Trailing Stop (In Profit)"
            else:
                return "Stop Loss"
        elif current_price >= take_profit_level:
            return "Take Profit"
        elif days_held >= self.p.position_timeout:
            return "Max Hold Time"
        elif days_held > 5 and profit_pct < (days_held * self.p.min_daily_return):
            return "Poor Performance"
        else:
            return "Manual Exit"
    


    
    def handle_order_failure(self, order):

        if order in self.order_list:
            self.order_list.remove(order)
            
        reason = "Unknown"
        if order.status == order.Canceled:
            reason = "Canceled"
        elif order.status == order.Margin:
            reason = "Insufficient Margin"
        elif order.status == order.Rejected:
            reason = "Rejected"
        elif order.status == order.Expired:
            reason = "Expired"
            
        #logging.warning(f"Order failed for {order.data._name}: {reason}")
    
    def stop(self):
        self.progress_bar.close()
        self.trade_recorder.save_trades()
        

##===========================================================[Control]=========================================================##













##===========================================================[Control]=========================================================##













##===========================================================[Control]=========================================================##












##===========================================================[Control]=========================================================##



def save_guaranteed_signals_to_parquet(signals, next_trading_day=None):
    """
    Save signals with market cap and sentiment analysis integrated.
    Generates all columns in one pass while creating the file.
    """
    from finvizfinance.quote import finvizfinance
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    logger = logging.getLogger(__name__)
    
    if not signals:
        logger.error("CRITICAL: No signals to save! Check your data pipeline.")
        return False
    
    # Load sentiment model once for all signals
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    
    def get_market_cap_data(ticker):
        """Get market cap classification"""
        try:
            stock = finvizfinance(ticker)
            fundamentals = stock.ticker_fundament()
            cap_str = fundamentals.get('Market Cap', '-')
            
            if not cap_str or cap_str == '-':
                return 'Unknown', None
            
            cap_str = cap_str.upper().strip()
            multiplier = 1
            
            if cap_str.endswith('B'):
                multiplier = 1000
                cap_str = cap_str[:-1]
            elif cap_str.endswith('M'):
                multiplier = 1
                cap_str = cap_str[:-1]
            elif cap_str.endswith('K'):
                multiplier = 0.001
                cap_str = cap_str[:-1]
            
            cap_millions = float(cap_str) * multiplier
            
            if cap_millions >= 200000:
                return 'Mega', cap_millions
            elif cap_millions >= 10000:
                return 'Large', cap_millions
            elif cap_millions >= 2000:
                return 'Mid', cap_millions
            elif cap_millions >= 300:
                return 'Small', cap_millions
            elif cap_millions >= 50:
                return 'Micro', cap_millions
            else:
                return 'Nano', cap_millions
        except:
            return 'Unknown', None
    
    def get_sentiment_score(ticker):
        """Get sentiment score from news"""
        try:
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()
            
            if news_df is None or news_df.empty:
                return None
            
            headlines = ' '.join(news_df.head(10)['Title'].tolist())
            inputs = tokenizer(headlines, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].cpu().numpy()
            positive = probs[0]
            negative = probs[1]
            neutral = probs[2]
            
            score = (positive + (neutral * 0.5)) / (positive + negative + neutral)
            return float(score)
        except:
            return None
    
    try:
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")
        
        STOP_LOSS_PERCENT = 2.0  
        TAKE_PROFIT_PERCENT = 10.0
        
        signal_data = []
        for signal in signals:
            symbol = str(signal['Symbol']).upper()
            up_prob = float(signal['UpProbability']) if signal['UpProbability'] is not None else 0.0
            price = float(signal['Price']) if signal['Price'] is not None else 0.0
            
            atr = signal.get('ATR', price * 0.02)
            if isinstance(atr, str) or atr == 0:
                atr = price * 0.02
            atr = float(atr)
            
            stop_price = price * (1 - STOP_LOSS_PERCENT / 100.0)
            target_price = price * (1 + TAKE_PROFIT_PERCENT / 100.0)
            
            cap_bucket, cap_millions = get_market_cap_data(symbol)
            sentiment_score = get_sentiment_score(symbol)
            
            signal_record = {
                'Symbol': symbol,
                'Status': 'Pending',
                'TargetDate': pd.Timestamp(next_trading_day),
                
                'CurrentPrice': price,
                'SignalPrice': price,
                'EntryPrice': price,
                'StopPrice': round(stop_price, 4),
                'TargetPrice': round(target_price, 4),
                
                'UpProbability': up_prob,
                'ATR': atr,
                'SignalStrength': up_prob,
                
                'SignalDate': pd.Timestamp(next_trading_day),
                'CreatedDate': pd.Timestamp(datetime.now()),
                'LastUpdated': pd.Timestamp(datetime.now()),
                'LastUpdate': pd.Timestamp(datetime.now()),
                
                'PositionSize': 0,
                'PnL': 0.0,
                'PnLPct': 0.0,
                'ExitPrice': np.nan,
                'EntryDate': pd.NaT,
                'ExitDate': pd.NaT,
                'ExitReason': '',
                'ConsecutiveLosses': 0,
                
                'CapBucket': cap_bucket,
                'CapMillions': cap_millions,
                'Sentiment': sentiment_score
            }
            
            signal_data.append(signal_record)
            
            threshold_used = signal.get('ThresholdValue', signal.get('Threshold', 'Unknown'))
            quality_score = signal.get('Quality', 'Unknown')
            
            try:
                threshold_str = f"{float(threshold_used):.3f}" if threshold_used != 'Unknown' else 'Unknown'
                quality_str = f"{float(quality_score):.1f}" if quality_score != 'Unknown' else 'Unknown'
            except (ValueError, TypeError):
                threshold_str = str(threshold_used)
                quality_str = str(quality_score)
            
            sentiment_str = f"{sentiment_score:.3f}" if sentiment_score is not None else "N/A"
            
            logger.info(f"SIGNAL CREATED: {symbol} | Price: ${price:.2f} | "
                       f"Stop: ${stop_price:.2f} | Target: ${target_price:.2f} | "
                       f"UpProb: {up_prob:.3f} | Cap: {cap_bucket} | Sentiment: {sentiment_str}")
        
        new_signals_df = pd.DataFrame(signal_data)
        
        datetime_columns = ['TargetDate', 'EntryDate', 'ExitDate', 'CreatedDate', 
                           'LastUpdated', 'LastUpdate', 'SignalDate']
        
        for col in datetime_columns:
            if col in new_signals_df.columns:
                new_signals_df[col] = pd.to_datetime(new_signals_df[col])
        
        signals_file_path = 'Data/0__signals.parquet'
        _sig_dir = os.path.dirname(signals_file_path)
        if _sig_dir:
            os.makedirs(_sig_dir, exist_ok=True)

        # Mechanical pre-filter: bake FilterRubric Step-1 (non-web) hard-exclusion
        # audit columns into the dataframe BEFORE the single final write, so the
        # morning analyst sees price/cap/RSI/weekly-vol verdicts without hand-running
        # the rubric. Annotate only — no rows dropped. (see Util.annotate_signals_*)
        if annotate_signals_mechanical_filter is not None:
            try:
                new_signals_df = annotate_signals_mechanical_filter(new_signals_df)
                n_ex = int(new_signals_df["MechExclude"].sum())
                logger.info(f"Mechanical pre-filter: annotated {len(new_signals_df)} signals, "
                            f"{n_ex} flagged MechExclude (rows kept, just flagged)")
                for _, r in new_signals_df[new_signals_df["MechExclude"]].iterrows():
                    logger.info(f"  FLAG {r['Symbol']}: {r['MechReasons']}")
            except Exception as e:
                logger.error(f"Mechanical pre-filter annotation failed (signals still written): {e}")

        new_signals_df.to_parquet(signals_file_path, index=False)

        logger.info(f"SUCCESS: Wrote {len(signal_data)} signals to {signals_file_path}")
        logger.info(f"Signals ready for live trading on {next_trading_day}")
        
        try:
            verification_df = pd.read_parquet(signals_file_path)
            pending_signals = verification_df[verification_df['Status'] == 'Pending']
            logger.info(f"VERIFICATION: File contains {len(pending_signals)} pending signals")
            
            for _, row in pending_signals.iterrows():
                sentiment_disp = f"{row['Sentiment']:.3f}" if pd.notna(row['Sentiment']) else "N/A"
                logger.info(f"  {row['Symbol']}: Cap={row['CapBucket']}, Sentiment={sentiment_disp}, "
                           f"Target=${row['TargetPrice']:.2f}, Stop=${row['StopPrice']:.2f}")
            
        except Exception as e:
            logger.error(f"ERROR: Could not verify saved file: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving signals: {e}")
        logger.error(traceback.format_exc())
        return False







def save_best_signals_to_parquet(signals, next_trading_day=None):
    """
    Save the generated best signals to the buy signals parquet file.
    
    Args:
        signals: List of dictionaries containing signal information
        next_trading_day: Optional next trading day, if None will calculate it
    """
    
    logger = logging.getLogger(__name__)
    
    if not signals:
        logger.warning("No signals to save")
        return
    
    try:
        # Get next trading day if not provided
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")
        
        # Read existing signal file
        try:
            df = read_trading_data()
            logger.info(f"Read existing trading data with {len(df)} records")
        except:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame(columns=[
                'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
                'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
            ])
            logger.info("Created new trading data DataFrame")
        
        # Create new signal data
        signal_data = []
        for signal in signals:
            signal_data.append({
                'Symbol': signal['Symbol'],
                'LastBuySignalDate': pd.Timestamp(next_trading_day),
                'LastBuySignalPrice': float(signal['Price']),
                'IsCurrentlyBought': False,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.NaT,
                'UpProbability': float(signal['UpProbability']),
                'LastSellPrice': float('nan'),
                'PositionSize': float('nan')
            })
        
        # Create new signals DataFrame
        new_signals_df = pd.DataFrame(signal_data)
        
        # Keep only currently bought positions
        currently_bought = df[df['IsCurrentlyBought'] == True]
        
        # Create final DataFrame with both bought positions and new signals
        final_df = pd.concat([currently_bought, new_signals_df], ignore_index=True)
        
        # Remove duplicates in case a symbol is both bought and in new signals
        final_df = final_df.drop_duplicates(subset=['Symbol'], keep='first')
        
        # Write to file
        write_trading_data(final_df)
        
        
        #logger.info(f"Successfully wrote {len(signal_data)} new buy signals")
        
    except Exception as e:
        logger.error(f"Error saving best signals to parquet: {e}")
        logger.error(traceback.format_exc())











def filter_stocks_by_signal_quality(data_dir, min_variance=0.1, min_up_prob=0.50):
    
    logger = logging.getLogger(__name__)
    filtered_files = []
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    logger.info(f"Found {len(all_files)} stock prediction files")
    
    def meets_criteria(file_path):
        try:
            df = pd.read_parquet(file_path, columns=['up_prob'])
            
            max_up_prob = df['up_prob'].max()
            if max_up_prob < min_up_prob:
                return (False, f"max_up_prob {max_up_prob:.2f} < {min_up_prob:.2f}")
                
            variance = df['up_prob'].var()
            if variance < min_variance:
                return (False, f"variance {variance:.4f} < {min_variance:.4f}")
                
            return (True, f"Passed: max_up_prob={max_up_prob:.2f}, var={variance:.4f}")
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(meets_criteria, all_files),
            total=len(all_files),
            desc="Filtering stocks by signal quality"
        ))
    
    filtered_files = []
    rejected_counts = {"max_up_prob": 0, "variance": 0, "error": 0}
    
    for file_path, (meets, reason) in zip(all_files, results):
        file_name = os.path.basename(file_path)
        if meets:
            filtered_files.append(file_path)
            logger.debug(f"Accepted {file_name}: {reason}")
        else:
            if "max_up_prob" in reason:
                rejected_counts["max_up_prob"] += 1
            elif "variance" in reason:
                rejected_counts["variance"] += 1
            else:
                rejected_counts["error"] += 1
            logger.debug(f"Rejected {file_name}: {reason}")
    
    logger.info(f"Filtered to {len(filtered_files)} stocks with quality signals")
    logger.info(f"Rejected: {rejected_counts['max_up_prob']} for low up_prob, " 
                f"{rejected_counts['variance']} for low variance, "
                f"{rejected_counts['error']} due to errors")
    
    return filtered_files




# ------------------------------------------------------------------------------
# Main function and setup routines
# ------------------------------------------------------------------------------




def main():
    """Modified main function that ensures you get signals"""
    logger = get_logger(script_name="5__NightlyBackTester")

    # Mute all console (StreamHandler) output — file handlers keep full detail
    for _handler in logging.root.handlers + logger.handlers:
        if isinstance(_handler, logging.StreamHandler) and not isinstance(_handler, logging.FileHandler):
            _handler.setLevel(logging.WARNING)
    logging.root.setLevel(logging.WARNING)

    start_time = time.time()
    
    try:
        args = arg_parser()

        _t_load = time.time()
        cerebro, data_feeds = setup_backtest_environment(args, logger)
        print(f"[PHASE] load+setup: {time.time()-_t_load:.1f}s", flush=True)

        if not data_feeds:
            logger.error("No data feeds available. Exiting.")
            return None

        # Run backtest (optionally profiled via BT_PROFILE=1)
        _t_run = time.time()
        if os.environ.get('BT_PROFILE') == '1':
            import cProfile
            _pr = cProfile.Profile(); _pr.enable()
            strategies = cerebro.run()
            _pr.disable(); _pr.dump_stats('Data/_speedup/cerebro_profile.out')
        else:
            strategies = cerebro.run()
        print(f"[PHASE] cerebro.run (sim+analyzers): {time.time()-_t_run:.1f}s", flush=True)
        if not strategies:
            logger.error("No strategies were executed.")
            return None

        # Process results
        _t_ext = time.time()
        first_strategy = strategies[0]
        results = extract_backtest_results(first_strategy, cerebro, logger)
        print(f"[PHASE] extract_results: {time.time()-_t_ext:.1f}s", flush=True)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        # Display results
        print_detailed_results(results, execution_time)
        
        # Optional plotting
        try_plot_results(cerebro, logger)
        
        # Return summary
        return create_results_summary(results)
    
    except Exception as e:
        logger.error(f"Critical error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nA critical error occurred: {str(e)}")
        return None






def prepare_data_feed(name_df_tuple):
    """Prepares a single data feed (can be run in parallel)"""
    name, df = name_df_tuple
    data_feed = EnhancedPandasData(dataname=df)
    return name, data_feed



def setup_backtest_environment(args, logger):
    """Set up the backtest environment with WORKING commission model."""
    
    cerebro = bt.Cerebro(maxcpus=None)
    cerebro.broker.set_cash(10000)  # Initial cash

    # FIXED: Use the corrected commission model without emojis
    commission_added = False
    
    try:
        # Try the full IBKR commission model first
        ibkr_commission = IBKRAdaptiveCommission(
            commission_per_share=0.0035,     
            min_per_order=0.35,              
            max_per_order_pct=0.005,         
            exchange_fees=0.0002,            
        )
        
        cerebro.broker.addcommissioninfo(ibkr_commission)
        cerebro._commission_model = ibkr_commission  # Store for later analysis
        logger.info("SUCCESS: Added IBKR commission model")
        commission_added = True
        
    except Exception as e:
        logger.warning(f"IBKR model failed: {e}")
        
        # Fallback to simple commission
        try:
            simple_commission = SimpleIBKRCommission()
            cerebro.broker.addcommissioninfo(simple_commission)
            cerebro._commission_model = simple_commission
            logger.info("SUCCESS: Added simple commission model")
            commission_added = True
            
        except Exception as e2:
            logger.warning(f"Simple model failed: {e2}")
    
    # Final fallback if both failed
    if not commission_added:
        # Use built-in Backtrader commission
        cerebro.broker.setcommission(commission=0.0035, mult=1.0, margin=None)
        logger.info("FALLBACK: Using built-in commission: $0.0035 per share")

    
    cerebro.broker.set_coo(False)
    cerebro.broker.set_coc(False)
    
    # Get data files (read-only override via --data_dir; default = live signals)
    data_dir = getattr(args, 'data_dir', 'Data/RFpredictions')
    file_paths = select_data_files(args, data_dir, logger)
    
    if not file_paths:
        return cerebro, []
    
    # Process data files
    aligned_data = process_data_files(args, file_paths, logger)
    
    if not aligned_data:
        return cerebro, []
    
    # Prepare data feeds in parallel
    logger.info(f"Preparing {len(aligned_data)} data feeds in parallel...")
    start_time = time.time()
    
    # Use process pool for true parallelism (32 cores or however many are available)
    num_cores = min(32, multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_cores) as pool:
        prepared_feeds = list(tqdm(
            pool.imap(prepare_data_feed, aligned_data),
            total=len(aligned_data),
            desc="Preparing Data Feeds"
        ))
    
    prep_time = time.time() - start_time
    logger.info(f"Data feed preparation completed in {prep_time:.2f} seconds using {num_cores} cores")
    
    # Add prepared data feeds to cerebro (this part still needs to be sequential)
    add_start_time = time.time()
    for name, data_feed in tqdm(prepared_feeds, desc="Adding Data Feeds to Cerebro"):
        cerebro.adddata(data_feed, name=name)
    
    add_time = time.time() - add_start_time
    logger.info(f"Data feed addition completed in {add_time:.2f} seconds")
    
    # Add analyzers
    add_analyzers(cerebro, logger)
    
    # Add strategy with parameters
    strategy_params = {}
    
    if hasattr(args, 'up_prob') and args.up_prob is not None:
        strategy_params['up_prob_threshold'] = args.up_prob
        strategy_params['up_prob_min_trigger'] = args.up_prob + 0.02
    
    cerebro.addstrategy(StockSniperStrategy, **strategy_params)
    
    return cerebro, aligned_data





####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####








def select_data_files(args, data_dir, logger):
    """Select data files based on sampling or filtering criteria."""
    
    # First, check if the directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure you have generated prediction data first")
        return []
    
    if args.sample > 0:
        all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
        num_files = len(all_files)
        
        # Check if there are any files
        if num_files == 0:
            logger.error(f"No .parquet files found in directory: {data_dir}")
            logger.info("Available files in directory:")
            try:
                all_files_any = os.listdir(data_dir)
                if all_files_any:
                    for file in all_files_any[:10]:  # Show first 10 files
                        logger.info(f"  - {file}")
                    if len(all_files_any) > 10:
                        logger.info(f"  ... and {len(all_files_any) - 10} more files")
                else:
                    logger.info("  Directory is empty")
            except Exception as e:
                logger.error(f"  Error listing directory contents: {e}")
            
            logger.info("\nPossible solutions:")
            logger.info("1. Run your data generation/prediction script first")
            logger.info("2. Check if prediction files are in a different directory")
            logger.info("3. Verify the data pipeline is working correctly")
            return []
        
        # Calculate number to select with proper bounds checking
        sample_pct = min(100.0, max(0.1, args.sample))  # Clamp between 0.1% and 100%
        num_to_select = max(1, int(round(num_files * sample_pct / 100)))
        
        # Ensure we don't try to sample more files than exist
        num_to_select = min(num_to_select, num_files)
        
        file_paths = random.sample(all_files, num_to_select)
        logger.info(f"Selected {len(file_paths)} random files ({sample_pct}% of {num_files})")
        
    else:
        # When sample is 0, use filtering instead
        logger.info("Sample percentage is 0, using quality filtering instead")
        file_paths = filter_stocks_by_signal_quality(
            data_dir, 
            min_variance=args.filter,
            min_up_prob=args.up_prob
        )
        
    if not file_paths:
        logger.error("No stock files found or passed filtering. Exiting.")
        return []
        
    logger.info(f"Processing {len(file_paths)} stock files")
    return file_paths



def process_data_files(args, file_paths, logger):
    """Load and process data files, ensuring proper alignment."""
    last_trading_date = get_last_trading_date()
    logger.info(f"Last trading date: {last_trading_date}")
    
    # Configure alignment parameters (you could add these to args if you want them configurable)
    align_start_date = False  # Set to False to disable alignment
    retention_pct = 90       # Target to keep 95% of stocks
    min_days = 501           # Minimum trading days required
    
    # Load data with alignment
    aligned_data = parallel_load_data(
        file_paths, 
        last_trading_date, 
        align_start_date=align_start_date,
        retention_pct=retention_pct,
        min_days=min_days
    )
    
    if not aligned_data:
        logger.error("No data remains after processing. Exiting.")
        return []
    
    logger.info(f"Final dataset: {len(aligned_data)} stocks with {len(aligned_data[0][1])} trading days")
    return aligned_data




####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####
####========================================[Alignment fix testing ]========================================####



def add_analyzers(cerebro, logger):
    """Add analyzers to the Cerebro instance."""
    analyzers_to_add = [
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeStats"}),
        (bt.analyzers.DrawDown, {"_name": "DrawDown"}),
        (bt.analyzers.SharpeRatio, {"_name": "SharpeRatio", "riskfreerate": 0.05}),
        (bt.analyzers.SQN, {"_name": "SQN"}),
        (bt.analyzers.Returns, {"_name": "Returns"}),
        (bt.analyzers.VWR, {"_name": "VWR"}),
        (bt.analyzers.TimeReturn, {"_name": "TimeReturn"}),
        (bt.analyzers.PeriodStats, {"_name": "PeriodStats"}),
        (bt.analyzers.Transactions, {"_name": "Transactions"}),
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeAnalyzer"}),
        (bt.analyzers.PositionsValue, {"_name": "PositionsValue"}),
        (bt.analyzers.TimeDrawDown, {"_name": "TimeDrawDown"}),
        # FAST: removed PyFolio — registered but its result is NEVER extracted, and it's
        # one of the heaviest analyzers (tracks per-position value across all feeds every
        # bar + builds full return/transaction frames). Lossless drop.
    ]
    
    for analyzer_class, kwargs in analyzers_to_add:
        try:
            cerebro.addanalyzer(analyzer_class, **kwargs)
            logger.debug(f"Added analyzer: {kwargs.get('_name', 'unnamed')}")
        except Exception as e:
            logger.error(f"Failed to add analyzer {kwargs.get('_name', 'unnamed')}: {str(e)}")


# ------------------------------------------------------------------------------
# Results extraction and processing routines
# ------------------------------------------------------------------------------






def extract_backtest_results(strategy, cerebro, logger):
    """Extract detailed results from the backtest."""
    results = initialize_results_dict(cerebro)
    
    try:
        # Set day count
        try:
            day_count = strategy.day_count
        except AttributeError:
            logger.warning("Could not get day_count from strategy, using 252 as fallback")
            day_count = 252
            
        results['day_count'] = day_count
        
        # Calculate total and annualized returns
        results['total_return'] = (results['final_value'] / results['initial_value'] - 1) * 100
        try:
            results['annualized_return'] = ((results['final_value'] / results['initial_value']) ** (252 / day_count) - 1) * 100
        except Exception as e:
            logger.warning(f"Failed to calculate annualized return: {str(e)}")
        
        # Extract data from analyzers
        analyzer_data = get_analyzer_data(strategy, logger)
        
        # Get strategy-specific data
        if hasattr(strategy, 'monthly_performance'):
            results['monthly_performance'] = strategy.monthly_performance
        
        if hasattr(strategy, 'yearly_performance'):
            results['yearly_performance'] = strategy.yearly_performance
        
        # Process analyzer data into results
        process_trade_statistics(results, analyzer_data, logger)
        process_drawdown_statistics(results, analyzer_data, logger)
        process_sharpe_ratio(results, analyzer_data, logger)
        process_trade_analyzer_data(results, analyzer_data, logger)
        process_daily_returns_data(results, analyzer_data, logger)
        calculate_risk_of_ruin(results, logger)
        determine_sqn_description(results)
    
        
        return results


    except Exception as e:
        logger.error(f"Error extracting metrics from analyzers: {str(e)}")
    
    return results


def initialize_results_dict(cerebro):
    """Initialize the results dictionary with default values."""
    return {
        # Core metrics
        'initial_value': cerebro.broker.startingcash,
        'final_value': cerebro.broker.getvalue(),
        'total_return': 0,
        'annualized_return': 0,
        'daily_return': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0,
        'sqn_value': 0,
        'sqn_description': "Unknown",
        'vwr': 0,
        'gain_to_pain_ratio': 0,
        'omega_ratio': 0,
        'information_ratio': 0,
        
        # Risk metrics
        'max_dd': 0,
        'max_dd_duration': 0,
        'avg_dd': 0,
        'avg_dd_duration': 0,
        'ulcer_index': 0,
        'recovery_factor': 0,
        'common_sense_ratio': 0,
        'risk_of_ruin': 1.0,
        'daily_volatility': 0,
        'annualized_volatility': 0,
        'var_95': 0,
        'cvar_95': 0,
        
        # Trade statistics
        'total_closed': 0,
        'won_total': 0,
        'lost_total': 0,
        'won_pnl_total': 0,
        'lost_pnl_total': 0,
        'won_avg': 0,
        'lost_avg': 0,
        'won_max': 0,
        'lost_max': 0,
        'net_total': 0,
        'profit_factor': 0,
        'percent_profitable': 0,
        'risk_reward_ratio': 0,
        'expectancy': 0,
        'kelly_percentage': 0,
        'avg_win_pct': 0,
        'avg_loss_pct': 0,
        'largest_win_pct': 0,
        'largest_loss_pct': 0,
        'avg_profit_per_trade': 0,
        'net_profit_drawdown_ratio': 0,
        
        # Trade management
        'avg_trade_len': 0,
        'longest_trade': 0,
        'shortest_trade': 0,
        'time_in_market_pct': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'current_streak': None,
        'win_loss_count_ratio': 0,
        
        # Advanced metrics
        'positive_days_pct': 0.0,
        'max_pos_streak': 0,
        'max_neg_streak': 0,
        'mfe_avg': 0,
        'mae_avg': 0,
        'mfe_max': 0,
        'mae_max': 0,
        'profit_per_day': 0,
        
        # Strategy specific
        'monthly_performance': {},
        'yearly_performance': {}
    }


def get_analyzer_data(strategy, logger):
    """Safely extract data from all analyzers."""
    analyzer_data = {}
    
    # Define the list of analyzers to extract: (key, analyzer_name)
    analyzers = [
        ('trade_stats', 'TradeStats'),
        ('drawdown', 'DrawDown'),
        ('sharpe_ratio', 'SharpeRatio'),
        ('sqn', 'SQN'),
        ('returns', 'Returns'),
        ('vwr', 'VWR'),
        ('time_return', 'TimeReturn'),
        ('period_stats', 'PeriodStats'),
        ('transactions', 'Transactions'),
        ('trade_analyzer', 'TradeAnalyzer'),
        ('positions_value', 'PositionsValue'),
        ('time_drawdown', 'TimeDrawDown')
    ]
    
    # Extract data from each analyzer with error handling
    for key, name in analyzers:
        try:
            analyzer = getattr(strategy.analyzers, name)
            analyzer_data[key] = analyzer.get_analysis()
            
            # Fix for SQN
            if key == 'sqn':
                sqn_value = analyzer_data[key].get('sqn', None)
                if sqn_value is not None:
                    strategy.sqn_value = sqn_value
                    #logger.info(f"SQN value: {sqn_value}")
                else:
                    # Try to calculate SQN manually if analyzer doesn't provide it
                    if 'trade_stats' in analyzer_data and strategy.winning_trades + strategy.losing_trades > 0:
                        trade_results = []
                        won_total = analyzer_data['trade_stats'].get('won', {}).get('total', 0)
                        lost_total = analyzer_data['trade_stats'].get('lost', {}).get('total', 0)
                        won_pnl = analyzer_data['trade_stats'].get('won', {}).get('pnl', {}).get('total', 0)
                        lost_pnl = analyzer_data['trade_stats'].get('lost', {}).get('pnl', {}).get('total', 0)
                        
                        total_trades = won_total + lost_total
                        if total_trades > 0:
                            avg_win = won_pnl / won_total if won_total > 0 else 0
                            avg_loss = lost_pnl / lost_total if lost_total > 0 else 0
                            
                            # Approximate trade results for SQN calculation
                            trade_results = [avg_win] * won_total + [avg_loss] * lost_total
                            
                            if trade_results:
                                mean_r = sum(trade_results) / len(trade_results)
                                std_dev = (sum((r - mean_r) ** 2 for r in trade_results) / len(trade_results)) ** 0.5
                                
                                if std_dev > 0:
                                    strategy.sqn_value = (mean_r / std_dev) * (len(trade_results) ** 0.5)
                                    logger.info(f"Manually calculated SQN value: {strategy.sqn_value}")
            elif key == 'returns':
                strategy.daily_return = analyzer_data[key].get('rtot', 0) / strategy.day_count
            elif key == 'vwr':
                strategy.vwr = analyzer_data[key].get('vwr', 0)
            elif key == 'period_stats':
                strategy.time_in_market_pct = analyzer_data[key].get('inmarket', 0) * 100
                
        except Exception as e:
            logger.warning(f"Failed to get {name} analysis: {str(e)}")
            analyzer_data[key] = {}
    
    return analyzer_data

def process_trade_statistics(results, analyzer_data, logger):
    """Process trade statistics from the analyzer data."""
    try:
        trade_stats = analyzer_data['trade_stats']
        
        results['total_closed'] = trade_stats.get('total', {}).get('closed', 0)
        results['won_total'] = trade_stats.get('won', {}).get('total', 0)
        results['lost_total'] = trade_stats.get('lost', {}).get('total', 0)
        
        results['won_pnl_total'] = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
        results['lost_pnl_total'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('total', 0))
        
        results['won_avg'] = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
        results['lost_avg'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('average', 0))
        
        results['won_max'] = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
        results['lost_max'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('max', 0))
        
        results['net_total'] = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)
        
        # Calculate derived metrics
        if results['lost_pnl_total'] > 0:
            results['profit_factor'] = results['won_pnl_total'] / results['lost_pnl_total']
        else:
            results['profit_factor'] = float('inf')
            
        if results['total_closed'] > 0:
            results['percent_profitable'] = (results['won_total'] / results['total_closed'] * 100)
        
        # Calculate average trade size and commission impact
        if 'avg_trade_len' in results:
            avg_trade_price = results['initial_value'] / 50  # Rough approximation of average position size
            avg_trade_size = avg_trade_price * results['avg_trade_len'] / 252  # Size based on duration
        else:
            avg_trade_price = results['initial_value'] / 50
            avg_trade_size = avg_trade_price
            
        # Calculate commission impact as percentage
        commission_per_trade = 3.0  # Fixed commission from your strategy
        commission_impact_pct = (commission_per_trade / avg_trade_price) * 100 if avg_trade_price > 0 else 0
        
        # Store commission metrics
        results['avg_trade_price'] = avg_trade_price
        results['commission_impact_pct'] = commission_impact_pct
        results['breakeven_threshold_pct'] = commission_impact_pct * 2  # Entry and exit commissions
        
        # Estimate gross win rate (before commissions)
        if results['total_closed'] > 0:
            # Estimate number of marginally profitable trades that become losers due to commission
            marginal_trades = sum(1 for trade in trade_stats.get('trades', []) 
                               if isinstance(trade, dict) and 
                               0 < trade.get('pnl', 0) < commission_per_trade * 2)
            
            # If we can't get individual trades, make an estimation based on average profits
            if marginal_trades == 0 and results['won_avg'] > 0:
                # Estimate percentage of winning trades that would be losers if commissions were higher
                margin_pct = min(1.0, (commission_per_trade * 2) / results['won_avg'])
                marginal_trades = int(results['won_total'] * margin_pct * 0.2)  # Assume 20% are near the threshold
            
            # Calculate gross win rate (adding back marginal trades)
            results['gross_win_rate'] = ((results['won_total'] + marginal_trades) / results['total_closed']) * 100
        else:
            results['gross_win_rate'] = 0.0
            
        if results['lost_avg'] > 0:
            results['risk_reward_ratio'] = abs(results['won_avg'] / results['lost_avg'])
        else:
            results['risk_reward_ratio'] = float('inf')
        






        p_win = results['percent_profitable'] / 100
        results['expectancy'] = (p_win * results['won_avg']) + ((1 - p_win) * -results['lost_avg'])
        
        if results['risk_reward_ratio'] > 0:
            results['kelly_percentage'] = ((p_win) - ((1 - p_win) / results['risk_reward_ratio'])) * 100
        


        ## its like EV but kelly assumes that the % profit is unrelated to how often you win 
        ## your biggest winners will be more rare than your average winners 


        if results['total_closed'] > 0:
            results['avg_profit_per_trade'] = results['net_total'] / results['total_closed']
        
        # Percentage metrics
        results['avg_win_pct'] = results['won_avg'] / results['initial_value'] * 100
        results['avg_loss_pct'] = results['lost_avg'] / results['initial_value'] * 100
        results['largest_win_pct'] = results['won_max'] / results['initial_value'] * 100
        results['largest_loss_pct'] = results['lost_max'] / results['initial_value'] * 100
        
        if results['lost_total'] > 0:
            results['win_loss_count_ratio'] = results['won_total'] / results['lost_total']
        else:
            results['win_loss_count_ratio'] = float('inf')
        
        if results['day_count'] > 0:
            results['profit_per_day'] = results['net_total'] / results['day_count']
            
    except Exception as e:
        logger.warning(f"Error processing trade statistics: {str(e)}")




def process_drawdown_statistics(results, analyzer_data, logger):
    """Process drawdown statistics from the analyzer data."""
    try:
        drawdown = analyzer_data['drawdown']
        
        results['max_dd'] = drawdown.get('max', {}).get('drawdown', 0)
        results['max_dd_duration'] = drawdown.get('max', {}).get('len', 0)
        results['avg_dd'] = drawdown.get('average', {}).get('drawdown', 0)
        results['avg_dd_duration'] = drawdown.get('average', {}).get('len', 0)
        
        # Calculate derived metrics
        if results['max_dd'] > 0:
            results['calmar_ratio'] = results['annualized_return'] / results['max_dd']
        else:
            results['calmar_ratio'] = float('inf')
        
        if results['max_dd'] > 0:
            results['recovery_factor'] = results['total_return'] / results['max_dd']
        else:
            results['recovery_factor'] = float('inf')
        
        if results['total_return'] > 0 and results['max_dd'] > 0:
            results['common_sense_ratio'] = results['total_return'] / results['max_dd_duration']
        
        if results['max_dd'] > 0:
            results['net_profit_drawdown_ratio'] = results['net_total'] / (results['max_dd'] * results['initial_value'] / 100)
        else:
            results['net_profit_drawdown_ratio'] = float('inf')
    except Exception as e:
        logger.warning(f"Error processing drawdown statistics: {str(e)}")


def process_sharpe_ratio(results, analyzer_data, logger):
    """Process Sharpe ratio from the analyzer data."""
    try:
        results['sharpe_ratio'] = analyzer_data['sharpe_ratio'].get('sharperatio', 0)
    except Exception as e:
        logger.warning(f"Error processing Sharpe ratio: {str(e)}")


def process_trade_analyzer_data(results, analyzer_data, logger):
    """Process trade analyzer data for streaks and trade lengths."""
    try:
        trade_analyzer = analyzer_data['trade_analyzer']
        streak_data = trade_analyzer.get('streak', {})
        won_streak = streak_data.get('won', {})
        lost_streak = streak_data.get('lost', {})
        
        results['max_consecutive_wins'] = won_streak.get('longest', 0)
        results['max_consecutive_losses'] = lost_streak.get('longest', 0)
        
        if 'current' in streak_data:
            if streak_data['current'] > 0:
                results['current_streak'] = f"{streak_data['current']} wins"
            elif streak_data['current'] < 0:
                results['current_streak'] = f"{abs(streak_data['current'])} losses"
        
        trade_len = trade_analyzer.get('len', {})
        results['avg_trade_len'] = trade_len.get('average', 0)
        results['longest_trade'] = trade_len.get('max', 0)
        results['shortest_trade'] = trade_len.get('min', 0)
        
        mfe_stats = trade_analyzer.get('mfe', {})
        mae_stats = trade_analyzer.get('mae', {})
        
        results['mfe_avg'] = mfe_stats.get('average', 0)
        results['mfe_max'] = mfe_stats.get('max', 0)
        results['mae_avg'] = mae_stats.get('average', 0)
        results['mae_max'] = mae_stats.get('max', 0)
    except Exception as e:
        logger.warning(f"Error processing trade analyzer data: {str(e)}")



# Omega ratio - CORRECT calculation



def process_daily_returns_data(results, analyzer_data, logger):
    """Process daily returns data for volatility and related metrics."""
    try:
        daily_returns = []
        for date, ret in analyzer_data['time_return'].items():
            if isinstance(ret, (int, float)):
                daily_returns.append(ret)
                
        if not daily_returns:
            return
            
        # Sortino ratio calculation
        # downside_returns are fractional; convert to % to match annualized_return units
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) * 100 if downside_returns else 0
        if downside_deviation > 0:
            results['sortino_ratio'] = (results['annualized_return'] - 5) / downside_deviation
        else:
            results['sortino_ratio'] = float('inf')
        
        # Gain to pain ratio
        sum_of_positive_returns = sum(max(0, r) for r in daily_returns)
        sum_of_negative_returns = abs(sum(min(0, r) for r in daily_returns))
        if sum_of_negative_returns > 0:
            results['gain_to_pain_ratio'] = sum_of_positive_returns / sum_of_negative_returns
        else:
            results['gain_to_pain_ratio'] = float('inf')
        
        # Ulcer index calculation
        # daily_returns are fractional (0.01 = 1%) — use geometric compounding
        equity_curve = list(results['initial_value'] * np.cumprod(1 + np.array(daily_returns)))
        drawdowns = []
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
                drawdowns.append(0)
            else:
                dd_pct = (peak - value) / peak * 100
                drawdowns.append(dd_pct)
        results['ulcer_index'] = np.sqrt(np.mean(np.array(drawdowns) ** 2))

        # Volatility metrics
        results['daily_volatility'] = np.std(daily_returns) * 100
        results['annualized_volatility'] = results['daily_volatility'] * np.sqrt(252)

        # Override backtrader's SharpeRatio (bt.analyzers.SharpeRatio inflates the
        # Sharpe when the strategy spends many days in cash — near-zero portfolio
        # changes make std artificially tiny). Recompute directly from the daily
        # TimeReturn series, which already includes 0-return idle days.
        if np.std(daily_returns, ddof=1) > 0:
            _daily_rf = (1.05 ** (1.0 / 252)) - 1
            _excess   = np.array(daily_returns) - _daily_rf
            results['sharpe_ratio'] = float(
                np.mean(_excess) / np.std(_excess, ddof=1) * np.sqrt(252)
            )

        # VaR and CVaR
        if len(daily_returns) > 5:
            results['var_95'] = np.percentile(daily_returns, 5) * 100
            cvar_values = [r for r in daily_returns if r < results['var_95'] / 100]
            if cvar_values and results['var_95'] < 0:
                results['cvar_95'] = np.mean(cvar_values) * 100

            # Tail Ratio: P95 daily return / |P5 daily return| — > 1.0 means fat right tail
            p95 = np.percentile(daily_returns, 95)
            p5  = np.percentile(daily_returns, 5)
            if p5 < 0:
                results['tail_ratio'] = p95 / abs(p5)


        def calculate_omega_ratio_inline(returns, threshold=0.0):
            """Calculate true Omega ratio inline."""
            if not returns or len(returns) == 0:
                return 0.0

            returns_array = np.array(returns)
            probability = 1.0 / len(returns_array)

            # Probability-weighted gains above threshold
            weighted_gains = np.sum(np.maximum(0, returns_array - threshold) * probability)

            # Probability-weighted losses below threshold  
            weighted_losses = np.sum(np.maximum(0, threshold - returns_array) * probability)

            return weighted_gains / weighted_losses if weighted_losses > 0 else float('inf')


        
        # Calculate Omega ratio with 0% threshold
        results['omega_ratio'] = calculate_omega_ratio_inline(daily_returns, threshold=0.0)
        
        # Optional: Also calculate with risk-free rate threshold
        daily_risk_free = 0.05 / 252  # 5% annual risk-free rate converted to daily
        results['omega_ratio_rf'] = calculate_omega_ratio_inline(daily_returns, threshold=daily_risk_free)



        # Streak and positive days analysis
        if len(daily_returns) > 20:
            results['positive_days_pct'] = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
            
            pos_streak = 0
            max_pos_streak = 0
            neg_streak = 0
            max_neg_streak = 0
            
            for r in daily_returns:
                if r > 0:
                    pos_streak += 1
                    neg_streak = 0
                    max_pos_streak = max(pos_streak, max_pos_streak)
                else:
                    neg_streak += 1
                    pos_streak = 0
                    max_neg_streak = max(neg_streak, max_neg_streak)
                    
            results['max_pos_streak'] = max_pos_streak
            results['max_neg_streak'] = max_neg_streak

            # Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)
            # P(true annualized SR > 1.0), corrected for skewness and kurtosis
            dr_arr  = np.array(daily_returns)
            sr_hat  = np.mean(dr_arr) / np.std(dr_arr, ddof=1)   # daily SR
            sr_star = 1.0 / np.sqrt(252)                          # daily equiv of annual 1.0
            gamma3  = stats.skew(dr_arr)
            gamma4  = stats.kurtosis(dr_arr, fisher=False)        # raw kurtosis (normal = 3)
            variance_sr = 1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat ** 2
            if variance_sr > 0:
                psr_stat = (sr_hat - sr_star) * np.sqrt(len(daily_returns) - 1) / np.sqrt(variance_sr)
                results['psr'] = float(stats.norm.cdf(psr_stat))

        # Serenity Ratio = (Annualized Excess Return) / Ulcer Index
        # Rewards fast recovery; penalises lingering drawdowns unlike Calmar's single-event max-DD
        if results.get('ulcer_index', 0) > 0:
            results['serenity_ratio'] = (results.get('annualized_return', 0) - 5.0) / results['ulcer_index']

    except Exception as e:
        logger.warning(f"Error calculating advanced metrics from daily returns: {str(e)}")




##evaluates to gain to pain ratio because the gain/loss metrics are the same after the prob is not being calculated and set to a defult
def calculate_omega_ratio(daily_returns, threshold=0.0002380):  # Changed to 10% annual (0.10/252)
    """
    Calculate Omega ratio with meaningful default threshold.
    
    Default threshold = 0.000397 daily ≈ 10% annual risk-free rate
    Alternative thresholds:
    - 8% annual:  threshold=0.000317 (0.08/252)
    - 12% annual: threshold=0.000476 (0.12/252)
    
    This ensures Omega ratio != Gain-to-Pain ratio
    
    Parameters:
    -----------
    daily_returns : list or array
        Daily returns as decimals (e.g., 0.01 for 1%)
    threshold : float
        Target return threshold (default: 0.000397 ≈ 10% annual)
        
    Returns:
    --------
    float : Omega ratio value
    """
    if not daily_returns or len(daily_returns) == 0:
        return 0.0
    
    returns = np.array(daily_returns)
    n = len(returns)
    
    # Each return has equal probability (1/n)
    probability = 1.0 / n
    
    # Calculate probability-weighted gains above threshold
    gains_above_threshold = np.maximum(0, returns - threshold)
    weighted_gains = np.sum(gains_above_threshold * probability)
    
    # Calculate probability-weighted losses below threshold  
    losses_below_threshold = np.maximum(0, threshold - returns)
    weighted_losses = np.sum(losses_below_threshold * probability)
    
    # Calculate Omega ratio
    if weighted_losses > 0:
        omega_ratio = weighted_gains / weighted_losses
    else:
        # If no losses below threshold, ratio is infinite
        omega_ratio = float('inf')
    
    return omega_ratio





def calculate_omega_ratio_risk_free(daily_returns, risk_free_rate=0.05):
 
    # Convert annual risk-free rate to daily
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    
    return calculate_omega_ratio(daily_returns, threshold=daily_rf_rate)












def process_daily_returns_data_corrected(results, analyzer_data, logger):
    """Process daily returns data for volatility and related metrics with CORRECTED Omega ratio."""
    try:
        daily_returns = []
        for date, ret in analyzer_data['time_return'].items():
            if isinstance(ret, (int, float)):
                daily_returns.append(ret)
                
        if not daily_returns:
            return
            
        # Convert to numpy array for calculations
        returns_array = np.array(daily_returns)
        
        # Sortino ratio calculation
        # downside_returns are fractional; convert to % to match annualized_return units
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) * 100 if downside_returns else 0
        if downside_deviation > 0:
            results['sortino_ratio'] = (results['annualized_return'] - 5) / downside_deviation
        else:
            results['sortino_ratio'] = float('inf')
        
        # Gain to pain ratio (keep this as is - it's different from Omega now)
        sum_of_positive_returns = sum(max(0, r) for r in daily_returns)
        sum_of_negative_returns = abs(sum(min(0, r) for r in daily_returns))
        if sum_of_negative_returns > 0:
            results['gain_to_pain_ratio'] = sum_of_positive_returns / sum_of_negative_returns
        else:
            results['gain_to_pain_ratio'] = float('inf')
        
        # CORRECTED Omega ratio calculation - uses meaningful threshold (not zero)
        results['omega_ratio'] = calculate_omega_ratio(daily_returns)  # Uses 5% annual threshold by default
        
        # Omega ratio with risk-free rate threshold (additional metric)
        results['omega_ratio_rf'] = calculate_omega_ratio_risk_free(daily_returns, risk_free_rate=0.05)
        
        # Ulcer index calculation
        # daily_returns are fractional (0.01 = 1%) — use geometric compounding
        equity_curve = list(results['initial_value'] * np.cumprod(1 + np.array(daily_returns)))
        drawdowns = []
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
                drawdowns.append(0)
            else:
                dd_pct = (peak - value) / peak * 100
                drawdowns.append(dd_pct)
        results['ulcer_index'] = np.sqrt(np.mean(np.array(drawdowns) ** 2))

        # Volatility metrics
        results['daily_volatility'] = np.std(daily_returns) * 100
        results['annualized_volatility'] = results['daily_volatility'] * np.sqrt(252)

        # Override backtrader's SharpeRatio (bt.analyzers.SharpeRatio inflates the
        # Sharpe when the strategy spends many days in cash — near-zero portfolio
        # changes make std artificially tiny). Recompute directly from the daily
        # TimeReturn series, which already includes 0-return idle days.
        if np.std(daily_returns, ddof=1) > 0:
            _daily_rf = (1.05 ** (1.0 / 252)) - 1
            _excess   = np.array(daily_returns) - _daily_rf
            results['sharpe_ratio'] = float(
                np.mean(_excess) / np.std(_excess, ddof=1) * np.sqrt(252)
            )

        # VaR and CVaR
        if len(daily_returns) > 5:
            results['var_95'] = np.percentile(daily_returns, 5) * 100
            cvar_values = [r for r in daily_returns if r < results['var_95'] / 100]
            if cvar_values and results['var_95'] < 0:
                results['cvar_95'] = np.mean(cvar_values) * 100
        
        # Streak and positive days analysis
        if len(daily_returns) > 20:
            results['positive_days_pct'] = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
            
            pos_streak = 0
            max_pos_streak = 0
            neg_streak = 0
            max_neg_streak = 0
            
            for r in daily_returns:
                if r > 0:
                    pos_streak += 1
                    neg_streak = 0
                    max_pos_streak = max(pos_streak, max_pos_streak)
                else:
                    neg_streak += 1
                    pos_streak = 0
                    max_neg_streak = max(neg_streak, max_neg_streak)
                    
            results['max_pos_streak'] = max_pos_streak
            results['max_neg_streak'] = max_neg_streak
            
    except Exception as e:
        logger.warning(f"Error calculating advanced metrics from daily returns: {str(e)}")



def verify_omega_fix():
    """Verify that Omega ratio is now different from Gain-to-Pain ratio."""
    
    # Sample returns
    test_returns = [0.02, -0.01, 0.03, -0.015, 0.025, -0.005, 0.01]
    
    # Gain-to-Pain calculation
    positive_returns = [r for r in test_returns if r > 0]
    negative_returns = [r for r in test_returns if r < 0]
    
    sum_positive = sum(positive_returns)
    sum_negative_abs = sum(abs(r) for r in negative_returns)
    
    gain_to_pain = sum_positive / sum_negative_abs
    
    # Omega ratio calculations
    omega_risk_free = calculate_omega_ratio(test_returns)  # Uses meaningful threshold
    omega_rf = calculate_omega_ratio_risk_free(test_returns)
    
    print("VERIFICATION RESULTS:")
    print(f"Gain-to-Pain Ratio: {gain_to_pain:.4f}")
    print(f"Omega Ratio (5% threshold): {omega_risk_free:.4f}")
    print(f"Omega Ratio (Risk-Free): {omega_rf:.4f}")
    print(f"Are they different? {abs(gain_to_pain - omega_risk_free) > 0.001}")
    
    return gain_to_pain, omega_risk_free, omega_rf






def calculate_risk_of_ruin(results, logger):
    """Calculate risk of ruin based on win rate and risk/reward ratio."""
    try:
        if results['percent_profitable'] > 0 and results['risk_reward_ratio'] > 0:
            win_rate_decimal = results['percent_profitable'] / 100
            edge = win_rate_decimal - (1 - win_rate_decimal) / results['risk_reward_ratio']
            if edge > 0:
                results['risk_of_ruin'] = ((1 - edge) / (1 + edge)) ** 20
            else:
                results['risk_of_ruin'] = 1.0
    except Exception as e:
        logger.warning(f"Error calculating risk of ruin: {str(e)}")


def determine_sqn_description(results):
    """Determine the SQN description based on the SQN value."""
    sqn_descriptions = {
        (float('-inf'), 0): "Negative",
        (0, 1.6): "Poor",
        (1.6, 2.0): "Below Average",
        (2.0, 2.5): "Average",
        (2.5, 3.0): "Good",
        (3.0, 5.0): "Excellent",
        (5.0, 7.0): "Superb",
        (7.0, float('inf')): "Holy Grail Potential"
    }
    
    for (low, high), desc in sqn_descriptions.items():
        if low <= results['sqn_value'] < high:
            results['sqn_description'] = desc
            break





# ------------------------------------------------------------------------------
# Printing routines for results
# ------------------------------------------------------------------------------

def print_detailed_results(results, execution_time):
    """Print detailed results to the console with colorized output."""
    print("\n" + "=" * 80)
    print(" Stock Sniper Strategy Backtest Results ".center(80))
    print("=" * 80)

    # Core Performance Metrics
    print("\nCore Performance Metrics:")
    print(colorize_output(results['total_return'], "Total Return %:", 50, 10))
    print(colorize_output(results['annualized_return'], "Annualized Return %:", 25, 10))
    print(colorize_output(results['final_value'], "Final Portfolio Value:", results['initial_value'] * 1.5, results['initial_value'] * 1.1))
    print(colorize_output(results['initial_value'], "Initial Portfolio Value:", results['initial_value'], results['initial_value']))
    print(colorize_output(results['sharpe_ratio'], "Sharpe Ratio:", 1.5, 0.75))
    print(colorize_output(results['sortino_ratio'], "Sortino Ratio:", 2.0, 1.0))
    print(colorize_output(results['calmar_ratio'], "Calmar Ratio:", 2.0, 0.5))
    print(colorize_output(results['gain_to_pain_ratio'], "Gain to Pain Ratio:", 1.5, 1.0))
    print(colorize_output(results['omega_ratio'], "Omega Ratio:", 1.5, 1.0))
    print(colorize_output(results['omega_ratio_rf'], "Omega Ratio (Risk-Free):", 1.5, 1.0))
    #verify_omega_fix()
    
    # SQN Metrics (Original and Enhanced)
    #print(colorize_output(results['sqn_value'], "SQN:", 3.0, 1.6))
    #print_sqn_quality(results)

    ##SQN CURRENTLY BROKERN - FIX LATER also the std on the postive returns is making this low when it should be high


    if not results['vwr'] == 0.0 or results['vwr'] == None:
        print(colorize_output(results['vwr'], "Variability-Weighted Return:", 5, 0.5))
    
    # Add enhanced SQN if available
    if 'enhanced_modified_sqn' in results:
        print(colorize_output(results['enhanced_modified_sqn'], "Modified SQN (% normalized):", 3.0, 1.6))
        

    # Compute capture ratios before printing so they appear in Risk Metrics
    _compute_capture_ratios(results)

    # Risk metrics (includes capture ratios)
    print_risk_metrics(results)

    # Trade statistics
    print_trade_statistics(results)

    # Statistical quality metrics (PSR, Serenity, Tail Ratio)
    print_signal_quality_metrics(results)

    # Trade management metrics
    print_trade_management_metrics(results)

    # Advanced trade quality metrics
    print_advanced_trade_metrics(results)
    
    # Enhanced strategy consistency metrics (if available)
    #print_enhanced_consistency_metrics(results)
    
    # Position sizing recommendations (if available)
    #print_position_sizing_recommendations(results)
    
    # Monthly and yearly performance
    print_period_performance(results)
    
    # System interpretation (if enhanced metrics available)
    
    # Execution time and trade data notice
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("Trade data saved to Data/TradeHistory.parquet for further analysis")



def print_sqn_quality(results):
    """Print colorized SQN quality description."""
    sqn_value = results['sqn_value']
    sqn_description = results['sqn_description']
    
    if sqn_value is not None and not math.isnan(sqn_value):
        sqn_color_map = {
            "Holy Grail Potential": 0.0,  # Best
            "Superb": 0.1,
            "Excellent": 0.2,
            "Good": 0.3,
            "Average": 0.5,
            "Below Average": 0.7,
            "Poor": 0.85,
            "Negative": 1.0  # Worst
        }
        normalized_value = sqn_color_map.get(sqn_description, 0.5)  # Default to Average

        if sqn_value >= 15:  # Unicorn level for SQN
            color_code = "\033[38;2;100;149;237m"  # Cornflower blue
            sqn_description = "Unicorn"
        else:
            colors = [
                (0, 235, 0),    # Bright Green
                (0, 180, 0),    # Normal Green
                (220, 220, 0),  # Yellow
                (220, 140, 0),  # Orange
                (220, 0, 0),    # Red
                (240, 0, 0)     # Bright Red
            ]
            index = min(int(normalized_value * (len(colors) - 1)), len(colors) - 2)
            t = (normalized_value * (len(colors) - 1)) - index
            r = int(colors[index][0] * (1 - t) + colors[index+1][0] * t)
            g = int(colors[index][1] * (1 - t) + colors[index+1][1] * t)
            b = int(colors[index][2] * (1 - t) + colors[index+1][2] * t)
            color_code = f"\033[38;2;{r};{g};{b}m"
        print(f"{'SQN Quality:':<30}{color_code}{sqn_description:<10}\033[0m")
    else:
        print(f"{'SQN Quality:':<30}\033[38;2;150;150;150mN/A        \033[0m")


def _compute_capture_ratios(results):
    """Download S&P 500 monthly data and store up/down capture ratios in results."""
    strat_monthly = results.get('monthly_performance', {})
    if not strat_monthly:
        return
    try:
        months   = sorted(strat_monthly.keys())
        sy, sm   = map(int, months[0].split('-'))
        ey, em   = map(int, months[-1].split('-'))
        start_dt = datetime(sy, sm, 1) - timedelta(days=5)
        end_dt   = datetime(ey, em, 28) + timedelta(days=10)

        sp500 = yf.download("^GSPC",
                            start=start_dt.strftime('%Y-%m-%d'),
                            end=end_dt.strftime('%Y-%m-%d'),
                            interval="1d", auto_adjust=True, progress=False)
        if len(sp500) == 0:
            return

        daily_ret = sp500['Close'].pct_change().dropna()
        market_monthly = {}
        for (yr, mo), grp in daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]):
            market_monthly[f"{yr}-{mo:02d}"] = ((1 + grp).cumprod().iloc[-1] - 1) * 100

        common = set(strat_monthly.keys()) & set(market_monthly.keys())
        if len(common) < 3:
            return

        up_s, up_m, dn_s, dn_m = [], [], [], []
        for month in common:
            sp = market_monthly[month]
            st = strat_monthly[month]
            if sp > 0:
                up_s.append(st); up_m.append(sp)
            elif sp < 0:
                dn_s.append(st); dn_m.append(sp)

        def _gm(vals):
            return (np.prod([1 + v / 100 for v in vals]) ** (1 / len(vals)) - 1) * 100

        if len(up_m) >= 2:
            gm_u = _gm(up_m)
            if gm_u != 0:
                results['up_capture'] = (_gm(up_s) / gm_u) * 100
        if len(dn_m) >= 2:
            gm_d = _gm(dn_m)
            if gm_d != 0:
                results['down_capture'] = (_gm(dn_s) / gm_d) * 100
    except Exception:
        pass  # silently skip on network failure


def print_risk_metrics(results):
    """Print risk metrics with dynamic, context-aware thresholds."""
    print("\nRisk Metrics:")
    
    # Get key metrics for dynamic calculations
    sortino = results.get('sortino_ratio', 0)
    calmar = results.get('calmar_ratio', 0)
    annual_return = results.get('annualized_return', 0)
    avg_loss_pct = results.get('avg_loss_pct', 0)

    # ===== Dynamic Threshold Logic =====
    # Annualized Volatility Thresholds
    if sortino > 2 and calmar > 5:  # Exceptional risk-adjusted returns
        vol_good = 30.0  # Green if <30%
        vol_bad = 50.0   # Red if >50%
    elif annual_return > 150:  # Ultra-high return strategy
        vol_good = 40.0
        vol_bad = 60.0
    else:  # Standard thresholds
        vol_good = 20.0
        vol_bad = 40.0

    # Daily Volatility (derived from annualized thresholds)
    daily_vol_multiplier = 1/15.8  # ≈ sqrt(252 trading days)
    daily_good = vol_good * daily_vol_multiplier
    daily_bad = vol_bad * daily_vol_multiplier

    # ===== Updated Print Statements =====
    print(colorize_output(results['max_dd'], "Max Drawdown %:", 10, 25, lower_is_better=True))
    print(colorize_output(results['max_dd_duration'], "Max Drawdown Duration (days):", 
                         results['max_consecutive_wins'] * 10, 
                         results['max_consecutive_wins'] * 15, 
                         lower_is_better=True))
    print(colorize_output(results['ulcer_index'], "Ulcer Index:", 1, 3, 
                         lower_is_better=True, unicorn_multiplier=10000.0))
    print(colorize_output(results['recovery_factor'], "Recovery Factor:", 3.0, 1.0))
    print(colorize_output(results['common_sense_ratio'], "Common Sense Ratio:", 0.5, 0.2))
    print(colorize_output(results['risk_of_ruin'], "Risk of Ruin:", 0.001, 0.05, 
                         lower_is_better=True, unicorn_multiplier=10000.0))
    
    # Updated Volatility Lines with Dynamic Thresholds
    print(colorize_output(results['daily_volatility'], "Daily Volatility %:", 
                         daily_good, daily_bad, lower_is_better=True))
    
    print(colorize_output(results['annualized_volatility'], "Annualized Volatility %:", 
                         vol_good, vol_bad, lower_is_better=True))
    
    # VaR/CVaR thresholds scaled to strategy performance
    var_cvar_multiplier = 2 if annual_return > 100 else 1  # Aggressive vs conservative
    print(colorize_output(results['var_95'], "Daily VaR (95%):", 
                         avg_loss_pct * 1.2 * var_cvar_multiplier, 
                         avg_loss_pct * 2.5 * var_cvar_multiplier, 
                         lower_is_better=True))
    
    print(colorize_output(results['cvar_95'], "Daily CVaR (95%):",
                         avg_loss_pct * 1.5 * var_cvar_multiplier,
                         avg_loss_pct * 3.0 * var_cvar_multiplier,
                         lower_is_better=True))

    # Market Capture Ratios (computed by _compute_capture_ratios before printing)
    if results.get('up_capture') is not None:
        # > 120% outpaces S&P in rallies; > 240% is unicorn
        print(colorize_output(results['up_capture'], "Up-Capture Ratio (%):",
                              good_threshold=120.0, bad_threshold=80.0,
                              unicorn_multiplier=2.0))
    if results.get('down_capture') is not None:
        # Negative = gains when market falls; unicorn triggers at value <= 50/10000 = 0.005
        print(colorize_output(results['down_capture'], "Down-Capture Ratio (%):",
                              good_threshold=50.0, bad_threshold=100.0,
                              lower_is_better=True, unicorn_multiplier=10000.0))

def print_trade_statistics(results):
    """Print trade statistics with colorized output."""
    print("\nTrade Statistics:")
    print(colorize_output(results['total_closed'], "Total Trades:", 50, 10))
    print(colorize_output(results['percent_profitable'], "Win Rate (after fees) %:", 60, 40))
    if results['gross_win_rate'] / results['percent_profitable'] > 1.01:
        print(colorize_output((results['gross_win_rate'] / results['percent_profitable']), "Fee Win rate diffrence (%):", 0.0001, 0.01))


    if 'gross_win_rate' in results:
        print(colorize_output(results['gross_win_rate'], "Win Rate (before fees) %:", 60, 40))
        print(colorize_output(results['commission_impact_pct'], "Commission Impact %:", 2.0, 5.0, lower_is_better=True))
        print(colorize_output(results['breakeven_threshold_pct'], "Breakeven Threshold %:", 4.0, 10.0, lower_is_better=True))

    # Dollar thresholds scale with account size — a $10K account shouldn't be punished for small positions
    _iv = results['initial_value']
    avg_win_good = max(_iv / 150, 30)   # ~$67 on $10K
    avg_win_bad  = max(_iv / 600, 10)   # ~$17 on $10K
    print(colorize_output(results['won_avg'], "Avg. Winning Trade ($):", avg_win_good, avg_win_bad))
    # Losing trade up to 85% of winning trade is fine at 60%+ win rate
    print(colorize_output(results['lost_avg'], "Avg. Losing Trade ($):", results['won_avg'] * 0.75, results['won_avg'] * 1.0, lower_is_better=True))
    # For 1-day hold: 0.4%+ avg win per trade is strong
    print(colorize_output(results['avg_win_pct'], "Avg. Winning Trade (%):", 0.4, 0.15))
    print(colorize_output(results['avg_loss_pct'], "Avg. Losing Trade (%):", results['avg_win_pct'] * 0.8, results['avg_win_pct'] * 1.1, lower_is_better=True))
    # Largest win: good = 5% of account, bad = 0.5% of account
    print(colorize_output(results['won_max'], "Largest Win ($):", _iv / 20, _iv / 200))
    # Largest loss: good ≤ 50% of largest win, bad ≥ 90%
    print(colorize_output(results['lost_max'], "Largest Loss ($):", results['won_max'] * 0.5, results['won_max'] * 0.9, lower_is_better=True))
    print(colorize_output(results['largest_win_pct'], "Largest Win (%):", 5.0, 2.0))
    print(colorize_output(results['largest_loss_pct'], "Largest Loss (%):", results['largest_win_pct'] * 0.5, results['largest_win_pct'] * 2.0, lower_is_better=True))
    print(colorize_output(results['avg_profit_per_trade'], "Avg. Trade P&L:", 50, 0))
    print(colorize_output(results['profit_factor'], "Profit Factor:", 2.5, 1.0))

    # EV Per Trade: win_rate * avg_win - loss_rate * avg_loss (do NOT divide by trade count)
    win_rate_dec = results['percent_profitable'] / 100
    results['Expected_Value_PerTrade'] = (win_rate_dec * results['won_avg']) - ((1 - win_rate_dec) * results['lost_avg'])
    ev_good = max(_iv / 500, 10)   # ~$20 on $10K
    ev_bad  = max(_iv / 2000, 3)   # ~$5  on $10K
    print(colorize_output(results['Expected_Value_PerTrade'], "EV Per Trade ($):", ev_good, ev_bad))

    print(colorize_output(results['net_profit_drawdown_ratio'], "Net Profit / Drawdown Ratio:", 3.0, 1.0))


def print_trade_management_metrics(results):
    """Print trade management metrics with colorized output."""
    print("\nTrade Management Metrics:")
    print(colorize_output(results['avg_trade_len'], "Avg. Holding Period (days):", 1, 5, lower_is_better=True))
    print(colorize_output(results['longest_trade'], "Longest Trade (days):", 15, 25, lower_is_better=True))
    print(colorize_output(results['shortest_trade'], "Shortest Trade (days):", 1, 5))
    print(colorize_output(results['max_consecutive_wins'], "Max Consecutive Wins:", 5, 3))
    print(colorize_output(results['max_consecutive_losses'], "Max Consecutive Losses:", max(1, results['max_consecutive_wins'] - 1), results['max_consecutive_wins'] + 1, lower_is_better=True))
    print(f"{'Current Streak:':<30}{results['current_streak'] if results['current_streak'] else 'None'}")
    print(colorize_output(results['win_loss_count_ratio'], "Win/Loss Count Ratio:", 1.5, 0.8))
    print(colorize_output(results['risk_reward_ratio'], "Risk/Reward Ratio:", 2.5, 1.0))
    print(colorize_output(results['kelly_percentage'], "Kelly %:", 20, 5))


def print_advanced_trade_metrics(results):
    """Print advanced trade quality metrics with colorized output."""
    print("\nAdvanced Trade Quality Metrics:")
    print(colorize_output(results['positive_days_pct'], "Percentage of Positive Days:", 50, 20))
    print(colorize_output(results['max_pos_streak'], "Max Pos Streak:", 5, 3))
    print(colorize_output(results['max_neg_streak'], "Max Neg streak:", results['max_pos_streak'], results['max_pos_streak'] * 10, lower_is_better=True))
    print(colorize_output(results['profit_per_day'], "Profit per Day ($):", 20, 5))


def print_signal_quality_metrics(results):
    """Print statistical validity and return-distribution quality metrics."""
    keys = ['psr', 'serenity_ratio', 'tail_ratio']
    if not any(results.get(k) is not None for k in keys):
        return

    print("\nStatistical Quality Metrics:")

    if results.get('psr') is not None:
        # P(true annualized SR > 1.0) corrected for skewness & kurtosis — > 99.91% is iron-clad
        print(colorize_output(results['psr'] * 100, "Probabilistic Sharpe Ratio (%):",
                              good_threshold=97.0, bad_threshold=90.0,
                              unicorn_multiplier=1.031))  # unicorn at >= 99.91%

    if results.get('serenity_ratio') is not None:
        # (Annual Return - RF) / Ulcer Index; penalises lingering drawdowns unlike Calmar
        print(colorize_output(results['serenity_ratio'], "Serenity Ratio:",
                              good_threshold=20.0, bad_threshold=5.0,
                              unicorn_multiplier=50.0))   # unicorn at >= 1000

    if results.get('tail_ratio') is not None:
        # P95 daily return / |P5 daily return|; > 1.0 = fat right tail, > 3.0 is unicorn
        print(colorize_output(results['tail_ratio'], "Tail Ratio (P95/|P5|):",
                              good_threshold=1.5, bad_threshold=0.9,
                              unicorn_multiplier=2.0))    # unicorn at >= 3.0



def calculate_period_returns(prices):
    """Calculate monthly and yearly returns from a price series."""
    # Convert to dataframe if it's a series
    if isinstance(prices, pd.Series):
        prices = pd.DataFrame(prices)
    
    # Make sure we have a datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Monthly returns
    monthly_returns = {}
    
    # Group by year and month
    monthly_grouped = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month])
    
    for (year, month), group in monthly_grouped:
        month_name = f"{year}-{month:02d}"
        # Calculate compounded return for the month
        monthly_return = ((1 + group.iloc[:, 0]).cumprod().iloc[-1] - 1) * 100
        monthly_returns[month_name] = monthly_return
    
    # Yearly returns
    yearly_returns = {}
    
    # Group by year
    yearly_grouped = daily_returns.groupby(daily_returns.index.year)
    
    for year, group in yearly_grouped:
        # Calculate compounded return for the year
        yearly_return = ((1 + group.iloc[:, 0]).cumprod().iloc[-1] - 1) * 100
        yearly_returns[str(year)] = yearly_return
    
    # Calculate annualized return
    total_days = (prices.index[-1] - prices.index[0]).days
    total_years = total_days / 365.25
    total_return = (prices.iloc[-1, 0] / prices.iloc[0, 0] - 1) * 100
    annualized_return = ((1 + total_return/100) ** (1/total_years) - 1) * 100
    
    return {
        'monthly_performance': monthly_returns,
        'yearly_performance': yearly_returns,
        'annualized_return': annualized_return
    }









def print_period_performance(results, start_date=None, end_date=None):
    """
    Print monthly and yearly performance metrics with REALISTIC market-beating expectations.
    
    Threshold Philosophy:
    - S&P 500 averages ~12% annually (~1% monthly)
    - Your strategy should consistently beat this or why bother?
    - 0% months are unacceptable - you're not beating cash
    - 1%+ monthly is where you should be as a minimum
    
    Parameters:
    -----------
    results : dict
        Dictionary containing performance metrics
    start_date : str or datetime
        Start date for market data (default: 2 years before today)
    end_date : str or datetime
        End date for market data (default: today)
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    if start_date is None:
        # Default to 2 years before end date
        if 'monthly_performance' in results and results['monthly_performance']:
            # Extract start date from the first month in results
            first_month = min(results['monthly_performance'].keys())
            year, month = map(int, first_month.split('-'))
            start_date = datetime(year, month, 1)
        else:
            start_date = end_date - timedelta(days=2*365)
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Download S&P 500 data for comparison
    try:
        sp500_data = yf.download(
            "^GSPC",
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        
        if len(sp500_data) == 0:
            print("Warning: No S&P 500 data available for the specified period.")
            market_results = None
        else:
            market_results = calculate_period_returns(sp500_data['Close'])
    except Exception as e:
        market_results = None
    
    # REALISTIC MONTHLY PERFORMANCE THRESHOLDS
    print("\nStrategy Monthly Performance (%) - Market-Beating Expectations:")
    print("Target: Consistently outperform S&P 500's ~1% monthly average")
    
    if results['monthly_performance']:
        months = sorted(results['monthly_performance'].keys())
        
        for month in months:
            perf = results['monthly_performance'][month]
        
            
            print(colorize_output(perf, f"{month}:", 
                                good_threshold=1.8,      # 1.8%+ is good (20%+ annualized)
                                bad_threshold=0.6,       # <0.6% is poor (7% annualized)
                                lower_is_better=False))
    
    # MARKET COMPARISON - Only show if we have market data
    if market_results and market_results['monthly_performance']:
        print("\nMonthly Excess Performance vs S&P 500 (%) - Alpha Generation:")
        print("Target: Consistent positive alpha (outperformance)")
        
        market_months = sorted(market_results['monthly_performance'].keys())
        
        for month in months:
            if month in market_results['monthly_performance']:
                strategy_perf = results['monthly_performance'][month]
                market_perf = market_results['monthly_performance'][month]
                relative_perf = strategy_perf - market_perf
                
                
                print(colorize_output(relative_perf, f"{month}:", 
                                    good_threshold=1.0,      # 1%+ monthly alpha is good
                                    bad_threshold=0.0,       # Negative alpha is poor
                                    lower_is_better=False))
    
    # REALISTIC YEARLY PERFORMANCE THRESHOLDS
    if results['yearly_performance']:
        print("\nStrategy Yearly Performance (%) - Annual Expectations:")
        print("Target: 20%+ annually to justify the complexity and risk")
        
        years = sorted(results['yearly_performance'].keys())
        for year in years:
            perf = results['yearly_performance'][year]
            
            
            print(colorize_output(perf, f"{year}:", 
                                good_threshold=22,       # 22%+ is good
                                bad_threshold=12,        # <12% is poor (market average)
                                lower_is_better=False))
        
        # Strategy annualized return
        print(colorize_output(results['annualized_return'], "Strategy Annualized Return:", 
                            good_threshold=22,           # 22%+ is good
                            bad_threshold=12,            # <12% is poor
                            lower_is_better=False))
    
    # YEARLY EXCESS RETURNS vs MARKET
    if market_results and market_results['yearly_performance']:
        print("\nYearly Excess Return vs S&P 500 (%) - Annual Alpha:")
        print("Target: 8%+ annual alpha to justify active management")
        
        market_years = sorted(market_results['yearly_performance'].keys())
        for year in years:
            if year in market_results['yearly_performance']:
                strategy_perf = results['yearly_performance'][year]
                market_perf = market_results['yearly_performance'][year]
                relative_perf = strategy_perf - market_perf
                
                
                print(colorize_output(relative_perf, f"{year}:", 
                                    good_threshold=10,       # 10%+ annual alpha is good
                                    bad_threshold=3,         # <3% annual alpha is poor
                                    lower_is_better=False))
        
        # Relative annualized return
        relative_annualized = results['annualized_return'] - market_results['annualized_return']
        print(colorize_output(relative_annualized, "Excess Annualized Return:", 
                            good_threshold=10,           # 10%+ annual alpha is good
                            bad_threshold=3,             # <3% annual alpha is poor
                            lower_is_better=False))

# ------------------------------------------------------------------------------
# Optional routines: plotting, buy signal check, and logging summary
# ------------------------------------------------------------------------------

def try_plot_results(cerebro, logger):
    """Attempt to plot the results if the dataset is small enough."""
    try:
        if len(cerebro.datas) <= 10:
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.facecolor'] = '#1e1e1e'
            plt.rcParams['grid.color'] = '#333333'
            
            cerebro.plot(style='candlestick',
                         barup='green',
                         bardown='red',
                         volup='green',
                         voldown='red',
                         grid=True,
                         subplot=True)
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

def create_results_summary(results):
    """Return a summary dictionary of the key backtest results."""
    summary = {
        'total_return': results['total_return'],
        'annualized_return': results['annualized_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'sortino_ratio': results['sortino_ratio'],
        'calmar_ratio': results['calmar_ratio'],
        'max_drawdown': results['max_dd'],
        'win_rate': results['percent_profitable'],
        'profit_factor': results['profit_factor'],
        'total_trades': results['total_closed'],
        'avg_trade_pnl': results['avg_profit_per_trade'],
        'risk_reward_ratio': results['risk_reward_ratio'],
        'sqn': results['sqn_value']
    }
    return summary



def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Sniper Trading Strategy")
    parser.add_argument("--sample", type=float, default=100, help="Percentage of random files to backtest (0-100)")
    parser.add_argument("--data_dir", default="Data/RFpredictions", help="Directory of per-ticker prediction parquets to backtest (read-only override; default = live signals). Use to validate a candidate model without clobbering live Data/RFpredictions.")
    parser.add_argument("--filter", type=float, default=0.01,help="Minimum UpProbability variance for stock filtering")
    parser.add_argument("--up_prob", type=float, default=0.68,help="UpProbability threshold for buy signals")
    parser.add_argument("--force", action='store_true', help="Force the script to run even if data is not up to last trading date")
    parser.add_argument("--recommend", action='store_true', default=False, help="Recommend basic system changes based on the backtest risk metrics")
    parser.add_argument("--best", action='store_true', default=False, help="Generate best buy signals for the current or last trading day")
    parser.add_argument("--num_signals", type=int, default=4, help="Number of best signals to generate (default: 4)")
    
    # Add new optimization related arguments
    parser.add_argument("--optimize", action='store_true', default=False, help="Run in optimization mode to find best parameters")
    parser.add_argument("--optimize_param", type=str, action='append', default=None, 
                       help="Parameters to optimize (can be used multiple times, e.g. --optimize_param up_prob_threshold --optimize_param max_positions)")
    parser.add_argument("--runs", type=int, default=10, help="Number of optimization runs (default: 10)")
    
    # Add individual parameter arguments for more granular control
    parser.add_argument("--max_positions", type=int, help="Maximum number of concurrent positions")
    parser.add_argument("--risk_per_trade", type=float, help="Risk per trade percentage")
    parser.add_argument("--stop_loss_atr", type=float, help="Stop loss ATR multiple")
    parser.add_argument("--trailing_stop_atr", type=float, help="Trailing stop ATR multiple")
    parser.add_argument("--take_profit", type=float, help="Take profit percentage")
    parser.add_argument("--position_timeout", type=int, help="Maximum days to hold a position")
    return parser.parse_args()



if __name__ == "__main__":
    clear_completed_trades()
    main()

