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
from colorama import Fore, Style, init

from finvizfinance.quote import finvizfinance
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from Util import *

# Initialize colorama
init(autoreset=True)

from Util import (
    STRATEGY_PARAMS_TUPLE as STRATEGY_PARAMS,  # Note we're using the tuple version for backtrader
)
import pickle

# ============ UTILITY FUNCTIONS ============
def parse_market_cap_string(cap_str):

    if cap_str is None or cap_str == '-':
        return None

    try:
        cap_str = str(cap_str).strip().upper()

        # Handle multipliers
        multipliers = {
            'T': 1_000_000_000_000,  # Trillion
            'B': 1_000_000_000,       # Billion
            'M': 1_000_000,           # Million
            'K': 1_000                # Thousand
        }

        for suffix, multiplier in multipliers.items():
            if cap_str.endswith(suffix):
                number_part = cap_str[:-1]
                return float(number_part) * multiplier

        # If no suffix, try to parse as plain number
        return float(cap_str)

    except (ValueError, AttributeError):
        return None

# ============ END UTILITY FUNCTIONS ============

class IBKRAdaptiveCommission(bt.CommInfoBase):

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

    # Step 5: Verify alignment
    start_dates = {df['Date'].dt.date.min() for _, df in aligned_data}
    lengths = {len(df) for _, df in aligned_data}

    if len(start_dates) != 1 or len(lengths) != 1:
        logging.warning(f"Imperfect alignment: {len(start_dates)} different start dates, {len(lengths)} different lengths")
        logging.warning(f"Start dates: {start_dates}")
        logging.warning(f"Lengths: {lengths}")

    return aligned_data


def read_trading_data():
    """Read the trading data parquet file."""
    file_path = '_Buy_Signals.parquet'
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
            'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
        ])
        df.to_parquet(file_path, index=False)
        return df
    
    return pd.read_parquet(file_path)


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
    df.to_parquet('_Buy_Signals.parquet', index=False)

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



class TradeRecorder:
    def __init__(self, filename='trade_history.parquet'):
        self.filename = filename
        self.trades = []
        
    def record_trade(self, trade_data):
        """Record a trade with detailed metadata."""
        self.trades.append(trade_data)
        
    def save_trades(self):
        """Save all recorded trades to a parquet file."""
        if not self.trades:
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




class StockSniperStrategy(bt.Strategy):
    params = STRATEGY_PARAMS
    
    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=self.p.atr_period)
            self.inds[d]['up_prob_ma3'] = bt.indicators.SMA(d.UpProbability, period=3)
            self.inds[d]['up_prob_ma5'] = bt.indicators.SMA(d.UpProbability, period=5)
            self.inds[d]['up_prob_roc'] = bt.indicators.ROC(d.UpProbability, period=3)
            self.inds[d]['up_prob'] = d.UpProbability

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

        self.trade_recorder = TradeRecorder('trade_history.parquet')

        self.open_positions = 0
        self.last_trade_date = None  # Track the last date a trade was placed
        self.days_without_trade = 0  # Track consecutive days without trades
        self.last_warning_day = -1  # Track last day we showed the warning

        self.correlation_df = pd.read_parquet('Correlations.parquet')

        if 'Ticker' in self.correlation_df.columns:
            self.correlation_df_by_ticker = self.correlation_df.copy()
            self.correlation_df.set_index('Ticker', inplace=True)
        else:
            logging.warning("'Ticker' column not found in correlation dataframe. Available columns: "
                           f"{list(self.correlation_df.columns)}")

        self.total_groups = self.correlation_df['Cluster'].nunique()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}

        try:
            with open('Data/fundamental_cache.pkl', 'rb') as f:
                fundamental_cache = pickle.load(f)

            self.market_cap_lookup = {}
            if 'market_caps' in fundamental_cache:
                for symbol, cap_str in fundamental_cache['market_caps'].items():
                    parsed_cap = parse_market_cap_string(cap_str)
                    if parsed_cap is not None:
                        self.market_cap_lookup[symbol] = parsed_cap
            else:
                logging.warning("'market_caps' key not found in fundamental cache")
                self.market_cap_lookup = {}
        except FileNotFoundError:
            logging.warning("fundamental_cache.pkl not found - market cap filtering will be disabled")
            self.market_cap_lookup = {}
        except Exception as e:
            logging.error(f"Error loading fundamental cache: {e}")
            self.market_cap_lookup = {}

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
            self.last_logged_date = current_date

        # Continue with normal position management
        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        if self.open_positions < self.p.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates or current_date == self.last_trading_date:
                self.process_buy_candidates(buy_candidates, current_date, verbose=False)

        current_equity = self.broker.getvalue()
        self.update_performance_tracking(current_equity, current_month, current_year)



    def update_performance_tracking(self, current_equity, current_month, current_year):
        # Monthly performance tracking
        if current_month != self.current_month:
            if self.current_month is not None and self.last_month_equity is not None:
                monthly_return = (current_equity / self.last_month_equity - 1) * 100
                self.monthly_performance[self.current_month] = monthly_return

            self.current_month = current_month
            self.last_month_equity = current_equity
            self.month_high_equity = current_equity
            self.month_low_equity = current_equity
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
            self.current_year = current_year
            self.last_year_equity = current_equity


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





    def can_buy2(self, data, current_date):
        """Original can_buy - preserved for revert."""

        DOLLAR_VOLUME_TIERS = {
            'VERY_LOW': 200_000,  
            'LOW': 1_000_000,     
            'MEDIUM': 5_000_000,  
            'HIGH': 20_000_000,   
        }

        CAP_TIERS = {
            'MICRO_MAX': 500_000_000,    # Hard exclusion (0.11 PF)
            'SMALL_MIN': 500_000_000,    # Sweet spot starts here
            'SMALL_MAX': 3_000_000_000,  # Sweet spot ends here
            'MID_MAX': 10_000_000_000,   # Mid cap upper bound
        }

        # ============ OPTIMIZED CORE CONFIGURATION ============
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        MIN_CLOSE_PRICE = 2.00
        MAX_CLOSE_PRICE = 1000.00
        MIN_DOLLAR_VOLUME = 1_500_000  # Kept as separate minimum threshold

        MAX_SINGLE_DAY_DROP = -0.15
        RECENT_DROP_LOOKBACK_DAYS = 10

        WEEK_52_HIGH_PROXIMITY_LIMIT = 0.99  # No restriction - top performer
        WEEK_52_LOOKBACK_DAYS = 252

        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.15
        MAX_MOMENTUM_LOSS = -0.075

        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.04
        VOLATILITY_LOOKBACK_DAYS = 20

        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 30

        SUFFICIENT_DATA_P_LOW = 90.0
        SUFFICIENT_DATA_P_HIGH = 98.0
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 98

        symbol = data._name

        # Strategy warmup period
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

        # Calculate current dollar volume
        current_dollar_volume = current_close * current_volume

        # ==== TIERED PERFORMANCE SCORING ====
        score = 0

        # Get market cap from cached lookup
        current_market_cap = self.market_cap_lookup.get(symbol, 0)

        # MARKET CAP SCORING
        if current_market_cap < CAP_TIERS['MICRO_MAX']:
            return False
        elif CAP_TIERS['SMALL_MIN'] <= current_market_cap < CAP_TIERS['SMALL_MAX']:
            score += 3
        elif CAP_TIERS['SMALL_MAX'] <= current_market_cap < CAP_TIERS['MID_MAX']:
            score += 0
        else:
            score -= 1

        # DOLLAR VOLUME SCORING (replacing share volume scoring)
        if current_dollar_volume < DOLLAR_VOLUME_TIERS['VERY_LOW']:
            score -= 2
        elif DOLLAR_VOLUME_TIERS['LOW'] <= current_dollar_volume < DOLLAR_VOLUME_TIERS['MEDIUM']:
            score += 2
        elif current_dollar_volume >= DOLLAR_VOLUME_TIERS['MEDIUM']:
            score += 1

        # PREMIUM COMBINATION BONUS
        if current_dollar_volume >= DOLLAR_VOLUME_TIERS['MEDIUM'] and current_market_cap >= CAP_TIERS['MID_MAX']:
            score += 2

        # FINAL SCORE CHECK
        if score < 0:
            return False

        # Original probability and position constraints
        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        # Probability bounds check
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

        # Price gates
        if current_close is None or not (MIN_CLOSE_PRICE <= current_close <= MAX_CLOSE_PRICE):
            return False

        # Liquidity check (dollar volume)
        if current_dollar_volume < MIN_DOLLAR_VOLUME:
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

            # Use optimized percentiles for limited data
            p_low = np.percentile(historic_probs, LIMITED_DATA_P_LOW) if len(historic_probs) > 1 else None
            p_high = np.percentile(historic_probs, LIMITED_DATA_P_HIGH) if len(historic_probs) > 1 else None
        else:
            # Use optimized percentiles for sufficient data
            p_low = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW) if len(historic_probs) > 1 else None
            p_high = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH) if len(historic_probs) > 1 else None

        if p_low is None or p_high is None:
            return False

        # OPTIMIZED: Relaxed 52-week high filter (now 1.00 = no restriction)
        try:
            closes_252 = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_252 and len(closes_252) > 0:
                highest_52w = max(closes_252)
                if highest_52w and (current_close / highest_52w) > WEEK_52_HIGH_PROXIMITY_LIMIT:
                    return False
        except Exception:
            pass

        # Momentum filter
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if not (MAX_MOMENTUM_LOSS <= ret_5d <= MAX_MOMENTUM_GAIN):
                    return False
        except Exception:
            pass

        # UPDATED: Volume spike detection using dollar volume
        try:
            # Get historical volumes and prices to calculate dollar volume
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            closes = data.close.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            
            if vols is not None and closes is not None and len(vols) > 0 and len(closes) > 0:
                # Calculate historical dollar volumes
                hist_dollar_vols = [v * c for v, c in zip(vols, closes) if v is not None and c is not None]
                
                if len(hist_dollar_vols) > 0:
                    avg_dollar_vol = np.mean(hist_dollar_vols)
                    if avg_dollar_vol and (current_dollar_volume / avg_dollar_vol) > VOLUME_SPIKE_MULTIPLIER:
                        # Only reject if also below medium liquidity threshold
                        if current_dollar_volume < DOLLAR_VOLUME_TIERS['MEDIUM']:
                            return False
        except Exception:
            pass

        # Volatility filter
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

        # RSI filter
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
                    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))

                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

        # Check for lower lows in history
        try:
            closes_history = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
            if closes_history is not None and len(closes_history) > 1:
                if current_close <= min(closes_history):
                    return False
        except Exception:
            pass

        # Check downtrend vs 50-day MA
        try:
            closes_for_ma = data.close.get(size=50)
            if closes_for_ma is not None and len(closes_for_ma) == 50:
                ma_50 = np.mean(closes_for_ma)
                if current_close < ma_50:
                    try:
                        closes_history = data.close.get(size=WEEK_52_LOOKBACK_DAYS)
                        if closes_history is not None and len(closes_history) > 1:
                            if current_close*1.1 <= min(closes_history):
                                return False
                    except Exception:
                        pass
        except Exception:
            pass

        # Core buy condition with optimized percentiles (90-98)
        return current_prob >= p_low and current_prob < p_high











    def can_buy(self, data, current_date):
        """
        Optimized can_buy based on parameter sweep results from 10_PredictorTest.ipynb.
        Key changes vs can_buy2:
          - Tighter stop loss (2%) and take profit (5%) favor quick trades
          - Trailing stop base 0.5% locks in gains faster
          - Hold period 1 day => tighter momentum/volatility filters
          - Relaxed 52-week high filter (removed - was rejecting top performers)
          - Tighter volatility ceiling (0.03 vs 0.04) to match 1-day hold
        To revert: change get_buy_candidates to call can_buy2 instead.
        """

        DOLLAR_VOLUME_TIERS = {
            'VERY_LOW': 200_000,
            'LOW': 1_000_000,
            'MEDIUM': 5_000_000,
            'HIGH': 20_000_000,
        }

        CAP_TIERS = {
            'MICRO_MAX': 500_000_000,
            'SMALL_MIN': 500_000_000,
            'SMALL_MAX': 3_000_000_000,
            'MID_MAX': 10_000_000_000,
        }

        # ============ OPTIMIZED CONFIGURATION (from sweep) ============
        MIN_DAYS_BEFORE_TRADING = 30
        TARGET_HISTORIC_PROB_COUNT = 45
        MAX_HISTORIC_LOOKBACK = 100
        MIN_HISTORIC_PROB_THRESHOLD = 30
        MIN_VIABLE_DATA_POINTS = 5

        UP_PROB_MIN_BOUND = 0.2
        UP_PROB_MAX_BOUND = 0.8

        MIN_CLOSE_PRICE = 2.00
        MAX_CLOSE_PRICE = 1000.00
        MIN_DOLLAR_VOLUME = 1_500_000

        MAX_SINGLE_DAY_DROP = -0.10       # Tighter: was -0.15
        RECENT_DROP_LOOKBACK_DAYS = 5     # Shorter: was 10 (1-day hold)

        MOMENTUM_LOOKBACK_DAYS = 5
        MAX_MOMENTUM_GAIN = 0.10          # Tighter: was 0.15
        MAX_MOMENTUM_LOSS = -0.05         # Tighter: was -0.075

        VOLUME_SPIKE_MULTIPLIER = 3.5
        VOLUME_AVG_LOOKBACK_DAYS = 20
        MAX_VOLATILITY_THRESHOLD = 0.03   # Tighter: was 0.04 (short hold = less vol tolerance)
        VOLATILITY_LOOKBACK_DAYS = 20

        RSI_PERIOD = 14
        MIN_RSI_THRESHOLD = 35            # Slightly higher: was 30

        SUFFICIENT_DATA_P_LOW = 90.0
        SUFFICIENT_DATA_P_HIGH = 98.0
        LIMITED_DATA_P_LOW = 90
        LIMITED_DATA_P_HIGH = 98

        symbol = data._name

        # Strategy warmup period
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

        current_dollar_volume = current_close * current_volume

        # ==== TIERED PERFORMANCE SCORING ====
        score = 0
        current_market_cap = self.market_cap_lookup.get(symbol, 0)

        # MARKET CAP SCORING
        if current_market_cap < CAP_TIERS['MICRO_MAX']:
            return False
        elif CAP_TIERS['SMALL_MIN'] <= current_market_cap < CAP_TIERS['SMALL_MAX']:
            score += 3
        elif CAP_TIERS['SMALL_MAX'] <= current_market_cap < CAP_TIERS['MID_MAX']:
            score += 0
        else:
            score -= 1

        # DOLLAR VOLUME SCORING
        if current_dollar_volume < DOLLAR_VOLUME_TIERS['VERY_LOW']:
            score -= 2
        elif DOLLAR_VOLUME_TIERS['LOW'] <= current_dollar_volume < DOLLAR_VOLUME_TIERS['MEDIUM']:
            score += 2
        elif current_dollar_volume >= DOLLAR_VOLUME_TIERS['MEDIUM']:
            score += 1

        # PREMIUM COMBINATION BONUS
        if current_dollar_volume >= DOLLAR_VOLUME_TIERS['MEDIUM'] and current_market_cap >= CAP_TIERS['MID_MAX']:
            score += 2

        if score < 0:
            return False

        # Position constraints
        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False

        # Probability bounds
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

        # Price gates
        if current_close is None or not (MIN_CLOSE_PRICE <= current_close <= MAX_CLOSE_PRICE):
            return False

        # Liquidity
        if current_dollar_volume < MIN_DOLLAR_VOLUME:
            return False

        # Build historical probability series
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

        # Limited data sanity
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

            p_low = np.percentile(historic_probs, LIMITED_DATA_P_LOW) if len(historic_probs) > 1 else None
            p_high = np.percentile(historic_probs, LIMITED_DATA_P_HIGH) if len(historic_probs) > 1 else None
        else:
            p_low = np.percentile(historic_probs, SUFFICIENT_DATA_P_LOW) if len(historic_probs) > 1 else None
            p_high = np.percentile(historic_probs, SUFFICIENT_DATA_P_HIGH) if len(historic_probs) > 1 else None

        if p_low is None or p_high is None:
            return False

        # Momentum filter (tighter bounds for 1-day hold)
        try:
            prev_close_5 = data.close[-MOMENTUM_LOOKBACK_DAYS]
            if prev_close_5 and prev_close_5 > 0:
                ret_5d = (current_close / prev_close_5) - 1
                if not (MAX_MOMENTUM_LOSS <= ret_5d <= MAX_MOMENTUM_GAIN):
                    return False
        except Exception:
            pass

        # Volume spike detection
        try:
            vols = data.volume.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            closes = data.close.get(size=VOLUME_AVG_LOOKBACK_DAYS)
            if vols is not None and closes is not None and len(vols) > 0 and len(closes) > 0:
                hist_dollar_vols = [v * c for v, c in zip(vols, closes) if v is not None and c is not None]
                if len(hist_dollar_vols) > 0:
                    avg_dollar_vol = np.mean(hist_dollar_vols)
                    if avg_dollar_vol and (current_dollar_volume / avg_dollar_vol) > VOLUME_SPIKE_MULTIPLIER:
                        if current_dollar_volume < DOLLAR_VOLUME_TIERS['MEDIUM']:
                            return False
        except Exception:
            pass

        # Volatility filter (tighter for short hold)
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

        # RSI filter
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
                    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
                if rsi < MIN_RSI_THRESHOLD:
                    return False
        except Exception:
            pass

        # Check for 52-week lows (still reject)
        try:
            closes_history = data.close.get(size=252)
            if closes_history is not None and len(closes_history) > 1:
                if current_close <= min(closes_history):
                    return False
        except Exception:
            pass

        # Downtrend check vs 50-day MA
        try:
            closes_for_ma = data.close.get(size=50)
            if closes_for_ma is not None and len(closes_for_ma) == 50:
                ma_50 = np.mean(closes_for_ma)
                if current_close < ma_50:
                    try:
                        closes_history = data.close.get(size=252)
                        if closes_history is not None and len(closes_history) > 1:
                            if current_close * 1.1 <= min(closes_history):
                                return False
                    except Exception:
                        pass
        except Exception:
            pass

        # Core buy condition
        return current_prob >= p_low and current_prob < p_high













    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    ##===============================[ SELLING ]==================================##
    
    # Optional: Add a method to track blacklisted stocks
    def update_blacklist(self):

        if not hasattr(self, 'stock_blacklist'):
            self.stock_blacklist = {}
        
        current_date = self.datetime.date()
        
        # Clean up old blacklist entries (older than 30 days)
        for symbol in list(self.stock_blacklist.keys()):
            if (current_date - self.stock_blacklist[symbol]['date']).days > 110:
                del self.stock_blacklist[symbol]
    
    
    def force_best_signal_for_current_day(self, data=None):
        current_date = self.datetime.date()

        candidates = self.get_buy_candidates(current_date)

        if candidates:
            # Save the signals using existing method
            self.save_best_buy_signals(candidates)
            return candidates
        else:
            logging.warning("No stocks passed the can_buy() criteria")
            return []
    
    def process_buy_candidates(self, buy_candidates, current_date, verbose=False):
        """
        Process buys and log signals - FIXED VERSION
        Intelligently determines signal target date based on backtest position vs real time
        """
        buy_candidates = self.sort_buy_candidates(buy_candidates)

        self.save_best_buy_signals(buy_candidates)

        if buy_candidates:
            signals = []
            real_current_date = datetime.now().date()

            # Calculate next trading day from the BACKTEST date
            next_trading_day_from_backtest = get_next_trading_day(current_date)

            # Calculate actual next trading day from TODAY
            actual_next_trading_day = get_next_trading_day(real_current_date)

            # Determine which target date to use
            days_from_backtest_to_now = (real_current_date - current_date).days

            # If backtest is within 10 calendar days of real time, use ACTUAL next trading day
            # This handles cases where backtest data is slightly stale (e.g., ends on Oct 28 but today is Nov 2)
            if days_from_backtest_to_now <= 10:
                signal_target_date = actual_next_trading_day
            else:
                signal_target_date = next_trading_day_from_backtest

            for d, size, correlation in buy_candidates[:self.p.max_positions]:
                price = d.close[0]
                atr = self.inds[d]['atr'][0] if d in self.inds and 'atr' in self.inds[d] else price * 0.02

                signal = {
                    'Symbol': d._name,
                    'Price': price,
                    'UpProbability': d.UpProbability[0],
                    'ATR': atr,
                    'Quality': d.UpProbability[0],
                    'DollarVolume': d.volume[0] * price,
                    'ThresholdValue': d.UpProbability[0],
                }
                signals.append(signal)

            # Check if we should save based on the chosen target date
            days_until_signal_date = (signal_target_date - real_current_date).days

            # Save if signals are for today or near future (within 5 days)
            # Allow -1 to catch "tomorrow" scenarios
            if -1 <= days_until_signal_date <= 5:
                print(f"SAVING {len(signals)} signals for target date {signal_target_date}")
                result = save_guaranteed_signals_to_parquet(signals, signal_target_date)
                if result:
                    print(f" Successfully saved signals to Data/0__Signals.parquet")
                else:
                    print(f"Failed to save signals!")
            else:
                verbose = False
                if verbose:
                    print(f"NOT SAVING - Signal target date {signal_target_date} is {days_until_signal_date} days away (outside -1 to +5 day window)")
        else:
            print(f"No buy candidates found for {current_date}")

        # Execute buys
        for d, size, _ in buy_candidates:
            if self.open_positions < self.p.max_positions:
                if self.check_group_allocation(d):
                    self.execute_buy(d, size, current_date)
            
    def sort_buy_candidates(self, buy_candidates):
        sorted_candidates = sorted(buy_candidates, key=lambda x: x[0].UpProbability[0], reverse=False)
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
                verbose = False
                if verbose:
                    logging.warning(f"Ticker {candidate_ticker} not found in correlation data")
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
            return False

        if group is None:
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
        # Optimized from parameter sweep (10_PredictorTest.ipynb)
        # Original: take_profit_percent = 20, trailing_stop_percent = 3.0
        take_profit_percent = 5
        trailing_stop_percent = 0.5

        target_price = current_price * (1 + take_profit_percent / 100.0)

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
                return self.exit_position(data)

            if days_held >= self.p.position_timeout:
                return self.exit_position(data)

            # Probability drop check (only after 3+ days)
            if days_held >= 3:
                recent_probs = [float(data.UpProbability[-i]) for i in range(1, min(11, len(data)))
                               if data.UpProbability[-i] is not None]

                if recent_probs and max(recent_probs) > 0.55 and current_prob < 0.48:
                    return self.exit_position(data)

        except (IndexError, AttributeError, TypeError) as e:
            logging.warning(f"No probability data for {symbol} on {current_date}: {e}")


    def exit_position(self, data):
        """Close position and clean up bracket tracking."""
        symbol = data._name
        position = self.getposition(data)

        if position.size <= 0:
            return

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



    def save_best_buy_signals(self, buy_candidates):
        """Save the best buy signals to 0__Signals.parquet using the new SIGNALS_SCHEMA."""

        # Use REAL-WORLD current date, not backtest date
        current_real_date = datetime.now().date()
        backtest_date = self.datetime.date()

        # Add data freshness check using real world date
        try:
            market_last_date = get_last_trading_date()
        except Exception as e:
            pass

        # Calculate next trading day from REAL-WORLD date
        try:
            next_trading_day = get_next_trading_day(current_real_date)
        except Exception as e:
            # Fallback to a simple calculation
            next_trading_day = current_real_date + timedelta(days=1)

        # Create signals using new SIGNALS_SCHEMA
        signal_data = []
        candidates_to_process = buy_candidates[:self.p.max_positions]

        current_datetime = datetime.now()

        for i, (d, size, correlation) in enumerate(candidates_to_process):
            symbol = str(d._name)

            try:
                # Get price and volume data
                price = round(d.close[0], 3)
                volume = float(d.volume[0]) if len(d.volume) > 0 else 0.0
                up_probability = float(d.UpProbability[0])

                # Calculate ATR if available
                try:
                    atr = float(d.atr[0]) if hasattr(d, 'atr') and len(d.atr) > 0 else np.nan
                except:
                    atr = np.nan

                # Get market cap data from Finviz
                cap_millions = np.nan
                cap_bucket = ""
                try:
                    from finvizfinance.quote import finvizfinance
                    stock = finvizfinance(symbol)
                    fundamentals = stock.ticker_fundament()

                    if fundamentals and 'Market Cap' in fundamentals:
                        market_cap_str = fundamentals['Market Cap']
                        # Parse market cap string (e.g., "1.23B", "456.78M")
                        if market_cap_str and market_cap_str != '-':
                            market_cap_str = market_cap_str.upper().replace(',', '')
                            if 'B' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('B', '')) * 1000
                            elif 'M' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('M', ''))
                            elif 'K' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('K', '')) / 1000

                            # Determine cap bucket
                            if cap_millions >= 10000:
                                cap_bucket = "Large"
                            elif cap_millions >= 2000:
                                cap_bucket = "Mid"
                            elif cap_millions >= 300:
                                cap_bucket = "Small"
                            else:
                                cap_bucket = "Micro"
                except Exception as e:
                    pass

                # Create signal record using SIGNALS_SCHEMA
                signal_record = {
                    'Symbol': symbol,
                    'Status': 'Pending',
                    'SignalDate': pd.Timestamp(current_datetime),
                    'TargetDate': pd.Timestamp(next_trading_day),
                    'SignalPrice': price,
                    'CurrentPrice': price,
                    'EntryDate': pd.NaT,
                    'EntryPrice': np.nan,
                    'PositionSize': np.nan,
                    'StopPrice': np.nan,
                    'TargetPrice': np.nan,
                    'ExitDate': pd.NaT,
                    'ExitPrice': np.nan,
                    'ExitReason': '',
                    'PnL': np.nan,
                    'PnLPct': np.nan,
                    'UpProbability': up_probability,
                    'Sentiment': np.nan,  # Will be filled by separate sentiment analysis if needed
                    'ATR': atr,
                    'Volume': volume,
                    'CapMillions': cap_millions,
                    'CapBucket': cap_bucket,
                    'ConsecutiveLosses': 0,
                    'LastUpdate': pd.Timestamp(current_datetime),
                    'SignalStrength': up_probability,  # Use probability as signal strength
                    'CreatedDate': pd.Timestamp(current_datetime),
                    'LastUpdated': pd.Timestamp(current_datetime)
                }

                signal_data.append(signal_record)

            except Exception as e:
                continue

        if signal_data:
            try:
                # Create DataFrame with new signals
                new_signals_df = pd.DataFrame(signal_data)

                # Read existing signals
                try:
                    existing_signals_df = read_signals()
                except:
                    existing_signals_df = pd.DataFrame()

                # Remove old pending signals for the same target date to avoid duplicates
                if not existing_signals_df.empty:
                    existing_signals_df = existing_signals_df[
                        ~((existing_signals_df['Status'] == 'Pending') &
                          (existing_signals_df['TargetDate'] == pd.Timestamp(next_trading_day)))
                    ]

                # Combine existing and new signals
                final_signals_df = pd.concat([existing_signals_df, new_signals_df], ignore_index=True)

                # Write signals using Util.py write_signals function (handles schema and column ordering)
                write_signals(final_signals_df)

            except Exception as e:
                logging.error(f"Error writing signals: {str(e)}")
                raise


    def notify_order(self, order):

        if order.status in [order.Completed, order.Partial]:
            self.handle_order_execution(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.handle_order_failure(order)
    

    def handle_order_execution(self, order):

        if order.isbuy():
            self.handle_buy_execution(order)
        elif order.issell():
            self.handle_sell_execution(order)
    
    def handle_buy_execution(self, order):

        data = order.data
        symbol = data._name

        mark_position_as_bought(symbol, order.executed.size)

        # Update last trade date tracking
        self.last_trade_date = self.datetime.date()
        self.days_without_trade = 0
    

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
                
                # Record using TradeRecorder for trade_history.parquet
                self.trade_recorder.record_trade(trade_data)
                
                # ALSO DIRECTLY ADD TO COMPLETED TRADES FILE
                # Convert to format expected by Util.add_completed_trade
                from Util import add_completed_trade

                # Get Volume and CapMillions data
                try:
                    volume = float(data.volume[0]) if len(data.volume) > 0 else np.nan
                except:
                    volume = np.nan

                # Calculate market cap if price and shares data available
                cap_millions = np.nan
                cap_bucket = ""
                try:
                    # Try to get market cap from Finviz at entry time
                    from finvizfinance.quote import finvizfinance
                    stock = finvizfinance(symbol)
                    fundamentals = stock.ticker_fundament()

                    if fundamentals and 'Market Cap' in fundamentals:
                        market_cap_str = fundamentals['Market Cap']
                        if market_cap_str and market_cap_str != '-':
                            market_cap_str = market_cap_str.upper().replace(',', '')
                            if 'B' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('B', '')) * 1000
                            elif 'M' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('M', ''))
                            elif 'K' in market_cap_str:
                                cap_millions = float(market_cap_str.replace('K', '')) / 1000

                            # Determine cap bucket
                            if cap_millions >= 10000:
                                cap_bucket = "Large"
                            elif cap_millions >= 2000:
                                cap_bucket = "Mid"
                            elif cap_millions >= 300:
                                cap_bucket = "Small"
                            else:
                                cap_bucket = "Micro"
                except:
                    pass  # Use np.nan if not available

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
                    'Volume': volume,
                    'CapMillions': cap_millions,
                    'CapBucket': cap_bucket,
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

            # Check if we should show a warning about days without trades
            if self.last_trade_date is not None:
                days_gap = (exit_date - self.last_trade_date).days
                if days_gap > 5:
                    print(f"{Fore.YELLOW}⚠ WARNING: {days_gap} days since last trade{Style.RESET_ALL}")

            # Print simplified trade completion (just ticker, colored percentage, days)
            profit_pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0
            if profit_pct > 0:
                print(f"{symbol} | {Fore.GREEN}+{profit_pct:.2f}%{Style.RESET_ALL} | {days_held} days")
            elif profit_pct < 0:
                print(f"{symbol} | {Fore.RED}{profit_pct:.2f}%{Style.RESET_ALL} | {days_held} days")
            else:
                print(f"{symbol} | 0.00% | {days_held} days")
        

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
    
    def stop(self):
        self.progress_bar.close()
        self.trade_recorder.save_trades()





##===========================================================[Control]=========================================================##




##===========================================================[Control]=========================================================##




##===========================================================[Control]=========================================================##




##===========================================================[Control]=========================================================##





def save_guaranteed_signals_to_parquet(signals, next_trading_day=None):
    """
    Save signals to parquet file using the Util.py schema and functions.
    This ensures compatibility with the rest of the system.
    """
    logger = logging.getLogger(__name__)

    if not signals:
        logger.error("CRITICAL: No signals to save! Check your data pipeline.")
        return False

    # Load sentiment model once for all signals
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")
        tokenizer = None
        model = None

    def get_finviz_fundamentals(ticker):
        """Get all available Finviz fundamental data for a ticker"""
        try:
            stock = finvizfinance(ticker)
            fundamentals = stock.ticker_fundament()

            return fundamentals if fundamentals else {}

        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {ticker}: {e}")
            return {}

    def get_sentiment_score(ticker):
        """Get sentiment score from news using FinBERT"""
        if tokenizer is None or model is None:
            return None

        try:
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()

            if news_df is None or news_df.empty:
                return None

            headlines = ' '.join(news_df.head(10)['Title'].tolist())
            inputs = tokenizer(headlines, return_tensors="pt", truncation=True,
                             max_length=512, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = predictions[0].cpu().numpy()
            positive = probs[0]
            negative = probs[1]
            neutral = probs[2]

            # Calculate composite score
            score = (positive + (neutral * 0.5)) / (positive + negative + neutral)
            return float(score)

        except Exception as e:
            logger.warning(f"Could not get sentiment for {ticker}: {e}")
            return None

    try:
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")

        # Read existing signals or create empty DataFrame with proper schema
        try:
            existing_signals_df = read_signals()
            logger.info(f"Read existing signals file with {len(existing_signals_df)} records")
        except:
            existing_signals_df = pd.DataFrame()
            logger.info("No existing signals file found, creating new one")

        signal_data = []
        current_datetime = datetime.now()

        for signal in signals:
            symbol = str(signal['Symbol']).upper()
            up_prob = float(signal['UpProbability']) if signal['UpProbability'] is not None else 0.0
            signal_price = float(signal.get('Price', 0.0))

            # Get Finviz fundamentals
            fundamentals = get_finviz_fundamentals(symbol)

            # Get sentiment score
            sentiment_score = get_sentiment_score(symbol)

            # Build signal record using SIGNALS_SCHEMA from Util.py
            signal_record = {
                'Symbol': symbol,
                'Status': 'Pending',  # New signals are always pending
                'SignalDate': pd.Timestamp(current_datetime),
                'TargetDate': pd.Timestamp(next_trading_day),
                'SignalPrice': signal_price,
                'CurrentPrice': signal_price,  # Initially same as signal price
                'EntryDate': pd.NaT,
                'EntryPrice': np.nan,
                'PositionSize': np.nan,
                'StopPrice': np.nan,
                'TargetPrice': np.nan,
                'ExitDate': pd.NaT,
                'ExitPrice': np.nan,
                'ExitReason': '',
                'PnL': np.nan,
                'PnLPct': np.nan,
                'UpProbability': up_prob,
                'ATR': np.nan,
                'ConsecutiveLosses': 0,
                'LastUpdate': pd.Timestamp(current_datetime)
            }

            # Store fundamentals in a separate column as JSON or just log them
            # (since SIGNALS_SCHEMA doesn't have columns for all Finviz data)
            signal_record['Sentiment'] = sentiment_score

            signal_data.append(signal_record)

            # Log what we captured
            market_cap = fundamentals.get('Market Cap', 'N/A')
            pe_ratio = fundamentals.get('P/E', 'N/A')
            sentiment_str = f"{sentiment_score:.3f}" if sentiment_score is not None else "N/A"

            logger.info(f"SIGNAL CREATED: {symbol} | UpProb: {up_prob:.3f} | "
                       f"Sentiment: {sentiment_str} | Market Cap: {market_cap} | P/E: {pe_ratio}")

        # Create DataFrame with new signals
        new_signals_df = pd.DataFrame(signal_data)

        # Merge with existing signals (ACCUMULATE MODE - save_guaranteed_signals_to_parquet)
        if not existing_signals_df.empty:
            # Check for exact duplicates only (same symbol, same target date, same signal price)
            # This allows accumulating multiple signals for the same symbol if they have different parameters
            new_signals_to_add = []

            for idx, new_signal in new_signals_df.iterrows():
                symbol = new_signal['Symbol']
                target_date = pd.Timestamp(next_trading_day)
                signal_price = new_signal['SignalPrice']

                # Check if this exact signal already exists
                is_duplicate = (
                    (existing_signals_df['Symbol'] == symbol) &
                    (existing_signals_df['Status'] == 'Pending') &
                    (existing_signals_df['TargetDate'] == target_date) &
                    (abs(existing_signals_df['SignalPrice'] - signal_price) < 0.01)  # Price tolerance
                ).any()

                if not is_duplicate:
                    new_signals_to_add.append(new_signal)
                else:
                    logger.info(f"Skipping duplicate signal for {symbol} at ${signal_price:.2f} for {target_date}")

            # Convert new signals to dataframe
            if new_signals_to_add:
                new_signals_filtered_df = pd.DataFrame(new_signals_to_add)
                # Combine existing and new signals (ACCUMULATE)
                final_signals_df = pd.concat([existing_signals_df, new_signals_filtered_df], ignore_index=True)
                logger.info(f"ACCUMULATE MODE: Added {len(new_signals_to_add)} new signals, kept {len(existing_signals_df)} existing signals")
            else:
                final_signals_df = existing_signals_df
                logger.info(f"ACCUMULATE MODE: No new signals to add (all were duplicates)")
        else:
            final_signals_df = new_signals_df
            logger.info(f"ACCUMULATE MODE: First run - added {len(new_signals_df)} signals")

        # Use Util.py write_signals function which handles schema validation and formatting
        write_signals(final_signals_df)

        logger.info(f"SUCCESS: Wrote {len(new_signals_df)} new signals to Data/0__Signals.parquet")
        logger.info(f"Total signals in file: {len(final_signals_df)}")
        logger.info(f"Signals ready for analysis on {next_trading_day}")

        # Verification
        try:
            verification_df = read_signals()
            logger.info(f"VERIFICATION: File contains {len(verification_df)} total signals")

            # Show pending signals for target date
            pending_for_target = verification_df[
                (verification_df['Status'] == 'Pending') &
                (verification_df['TargetDate'].dt.date == pd.Timestamp(next_trading_day).date())
            ]
            logger.info(f"Pending signals for {next_trading_day}: {len(pending_for_target)}")

            # Show sample of newly created signals
            for _, row in new_signals_df.head(3).iterrows():
                sentiment_disp = f"{row['Sentiment']:.3f}" if pd.notna(row.get('Sentiment')) else "N/A"
                logger.info(f"  {row['Symbol']}: UpProb={row['UpProbability']:.3f}, "
                           f"Sentiment={sentiment_disp}, "
                           f"Status={row['Status']}, TargetDate={row['TargetDate'].date()}")

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
    Save best signals to the 0__Signals.parquet file using the Util.py schema.
    This function appends new signals to existing ones.
    """
    logger = logging.getLogger(__name__)

    if not signals:
        logger.warning("No signals to save")
        return False

    try:
        # Get next trading day if not provided
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")

        # Read existing signals or create empty DataFrame with proper schema
        try:
            existing_signals_df = read_signals()
            logger.info(f"Read existing signals file with {len(existing_signals_df)} records")
        except:
            existing_signals_df = pd.DataFrame()
            logger.info("No existing signals file found, creating new one")

        # Create new signal data using SIGNALS_SCHEMA
        signal_data = []
        current_datetime = datetime.now()

        for signal in signals:
            symbol = str(signal['Symbol']).upper()
            signal_price = float(signal.get('Price', 0.0))
            up_prob = float(signal.get('UpProbability', 0.0))

            signal_record = {
                'Symbol': symbol,
                'Status': 'Pending',  # New signals are always pending
                'SignalDate': pd.Timestamp(current_datetime),
                'TargetDate': pd.Timestamp(next_trading_day),
                'SignalPrice': signal_price,
                'CurrentPrice': signal_price,
                'EntryDate': pd.NaT,
                'EntryPrice': np.nan,
                'PositionSize': np.nan,
                'StopPrice': np.nan,
                'TargetPrice': np.nan,
                'ExitDate': pd.NaT,
                'ExitPrice': np.nan,
                'ExitReason': '',
                'PnL': np.nan,
                'PnLPct': np.nan,
                'UpProbability': up_prob,
                'ATR': np.nan,
                'ConsecutiveLosses': 0,
                'LastUpdate': pd.Timestamp(current_datetime)
            }
            signal_data.append(signal_record)

        # Create new signals DataFrame
        new_signals_df = pd.DataFrame(signal_data)

        # Merge with existing signals (ACCUMULATE MODE - save_best_signals_to_parquet)
        if not existing_signals_df.empty:
            # Check for exact duplicates only (same symbol, same target date, same signal price)
            # This allows accumulating multiple signals for the same symbol if they have different parameters
            new_signals_to_add = []

            for idx, new_signal in new_signals_df.iterrows():
                symbol = new_signal['Symbol']
                target_date = pd.Timestamp(next_trading_day)
                signal_price = new_signal['SignalPrice']

                # Check if this exact signal already exists
                is_duplicate = (
                    (existing_signals_df['Symbol'] == symbol) &
                    (existing_signals_df['Status'] == 'Pending') &
                    (existing_signals_df['TargetDate'] == target_date) &
                    (abs(existing_signals_df['SignalPrice'] - signal_price) < 0.01)  # Price tolerance
                ).any()

                if not is_duplicate:
                    new_signals_to_add.append(new_signal)
                else:
                    logger.info(f"Skipping duplicate signal for {symbol} at ${signal_price:.2f} for {target_date}")

            # Convert new signals to dataframe
            if new_signals_to_add:
                new_signals_filtered_df = pd.DataFrame(new_signals_to_add)
                # Combine existing and new signals (ACCUMULATE)
                final_signals_df = pd.concat([existing_signals_df, new_signals_filtered_df], ignore_index=True)
                logger.info(f"ACCUMULATE MODE: Added {len(new_signals_to_add)} new signals, kept {len(existing_signals_df)} existing signals")
            else:
                final_signals_df = existing_signals_df
                logger.info(f"ACCUMULATE MODE: No new signals to add (all were duplicates)")
        else:
            final_signals_df = new_signals_df
            logger.info(f"ACCUMULATE MODE: First run - added {len(new_signals_df)} signals")

        # Use Util.py write_signals function which handles schema validation and formatting
        write_signals(final_signals_df)

        logger.info(f"SUCCESS: Wrote {len(new_signals_to_add) if 'new_signals_to_add' in locals() and new_signals_to_add else len(new_signals_df)} new buy signals to Data/0__Signals.parquet")
        logger.info(f"Total signals in file: {len(final_signals_df)}")

        return True

    except Exception as e:
        logger.error(f"Error saving best signals to parquet: {e}")
        logger.error(traceback.format_exc())
        return False





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


def clear_pending_signals():
    """
    Clear all pending signals from the signals file.
    Keeps Active and Completed signals intact.
    """
    logger = logging.getLogger(__name__)

    try:
        # Read existing signals
        signals_df = read_signals()

        if signals_df.empty:
            logger.info("No signals file found or file is empty. Nothing to clear.")
            return True

        # Count pending signals before clearing
        pending_count = len(signals_df[signals_df['Status'] == 'Pending'])

        if pending_count == 0:
            logger.info("No pending signals to clear.")
            return True

        # Keep only Active and Completed signals
        filtered_df = signals_df[signals_df['Status'] != 'Pending']

        # Write back to file
        write_signals(filtered_df)

        logger.info(f"SUCCESS: Cleared {pending_count} pending signals. Kept {len(filtered_df)} Active/Completed signals.")
        print(f"\n✓ Cleared {pending_count} pending signals")

        return True

    except Exception as e:
        logger.error(f"Error clearing pending signals: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# ------------------------------------------------------------------------------
# Main function and setup routines
# ------------------------------------------------------------------------------

def main():
    """Modified main function that ensures you get signals"""
    logger = get_logger(script_name="5__NightlyBackTester")
    start_time = time.time()

    try:
        args = arg_parser()

        # Clear pending signals if requested
        if args.clear_signals:
            logger.info("Clearing pending signals as requested via --clear_signals flag")
            print("\n" + "="*80)
            print("CLEARING PENDING SIGNALS")
            print("="*80)
            clear_pending_signals()
            print("="*80 + "\n")

        cerebro, data_feeds = setup_backtest_environment(args, logger)
        
        if not data_feeds:
            logger.error("No data feeds available. Exiting.")
            return None
            
        # Run backtest
        strategies = cerebro.run()
        if not strategies:
            logger.error("No strategies were executed.")
            return None
            
        # Process results
        first_strategy = strategies[0]
        results = extract_backtest_results(first_strategy, cerebro, logger)
        
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
    
    # Get data files
    data_dir = 'Data/RFpredictions'
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
    logger.info(f"Initalizing analyzers...")
    # Add analyzers
    add_analyzers(cerebro, logger)
    
    # Add strategy with parameters
    strategy_params = {}
    
    if hasattr(args, 'up_prob') and args.up_prob is not None:
        strategy_params['up_prob_threshold'] = args.up_prob
        strategy_params['up_prob_min_trigger'] = args.up_prob + 0.02
    
    cerebro.addstrategy(StockSniperStrategy, **strategy_params)
    logger.info(f"Adding Stratagy...")
    return cerebro, aligned_data



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
        (bt.analyzers.PyFolio, {"_name": "PyFolio"})
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
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
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
        equity_curve = [results['initial_value'] * (1 + r / 100) for r in np.cumsum(daily_returns)]
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
        
        # VaR and CVaR
        if len(daily_returns) > 5:
            results['var_95'] = np.percentile(daily_returns, 5) * 100
            cvar_values = [r for r in daily_returns if r < results['var_95'] / 100]
            if cvar_values and results['var_95'] < 0:
                results['cvar_95'] = np.mean(cvar_values) * 100
        

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


##evaluates to gain to pain ratio because the gain/loss metrics are the same after the prob is not being calculated and set to a defult
def calculate_omega_ratio(daily_returns, threshold=0.0002380):  # Changed to 10% annual (0.10/252)

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
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
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
        equity_curve = [results['initial_value'] * (1 + r / 100) for r in np.cumsum(daily_returns)]
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

    ##SQN CURRENTLY BROKERN - FIX LATER also the std on the postive returns is making this low when it should be high

    if not results['vwr'] == 0.0 or results['vwr'] == None:
        print(colorize_output(results['vwr'], "Variability-Weighted Return:", 5, 0.5))
    
    # Add enhanced SQN if available
    if 'enhanced_modified_sqn' in results:
        print(colorize_output(results['enhanced_modified_sqn'], "Modified SQN (% normalized):", 3.0, 1.6))
        
    # Risk metrics
    print_risk_metrics(results)
    
    # Trade statistics
    print_trade_statistics(results)
    
    # Trade management metrics
    print_trade_management_metrics(results)

    # Monthly and yearly performance
    print_period_performance(results)
    
    # Execution time and trade data notice
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("Trade data saved to trade_history.parquet for further analysis")


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

def print_trade_statistics(results):
    """Print trade statistics with colorized output."""
    print("\nTrade Statistics:")
    print(colorize_output(results['total_closed'], "Total Trades:", 50, 10))
    print(colorize_output(results['percent_profitable'], "Win Rate (after fees) %:", 60, 40))
    if results['gross_win_rate'] / results['percent_profitable'] > 1.01:
        print(colorize_output((results['gross_win_rate'] / results['percent_profitable']), "Fee Win rate diffrence (%):", 0.0001, 0.01))


    if 'gross_win_rate' in results:
        print(colorize_output(results['gross_win_rate'], "Win Rate (before fees) %:", 60, 40))
        print(colorize_output(results['commission_impact_pct'], "Commission Impact %:", 0.5, 2.0, lower_is_better=True))
        print(colorize_output(results['breakeven_threshold_pct'], "Breakeven Threshold %:", 1.0, 4.0, lower_is_better=True))
    
    print(colorize_output(results['won_avg'], "Avg. Winning Trade ($):", 100, 50))
    print(colorize_output(results['lost_avg'], "Avg. Losing Trade ($):", results['won_avg'] * 0.5, results['won_avg'] * 0.75, lower_is_better=True))
    print(colorize_output(results['avg_win_pct'], "Avg. Winning Trade (%):", 1.0, 0.5))
    print(colorize_output(results['avg_loss_pct'], "Avg. Losing Trade (%):", results['avg_win_pct'] * 0.5, results['avg_win_pct'] * 0.75, lower_is_better=True))
    print(colorize_output(results['won_max'], "Largest Win ($):", results['initial_value'] / 4, 200))
    print(colorize_output(results['lost_max'], "Largest Loss ($):", results['won_max'] * 0.25, results['won_max'] * 0.5, lower_is_better=True))
    print(colorize_output(results['largest_win_pct'], "Largest Win (%):", 5.0, 2.0))
    print(colorize_output(results['largest_loss_pct'], "Largest Loss (%):", results['largest_win_pct'] * 0.5, results['largest_win_pct'] * 2.0, lower_is_better=True))
    print(colorize_output(results['avg_profit_per_trade'], "Avg. Trade P&L:", 50, 0))
    print(colorize_output(results['profit_factor'], "Profit Factor:", 2.5, 1.0))

    ##expected value = Win Rate * Avg Win - Loss Rate * Avg Loss
    results['Expected_Value_PerTrade'] = (results['percent_profitable'] * results['won_avg'] - (100 - results['percent_profitable']) * results['lost_avg']) / results['total_closed']
    print(colorize_output(results['Expected_Value_PerTrade'], "EV Per Trade:", 100.0, 20.0))
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
    print(colorize_output(results['positive_days_pct'], "Percentage of Positive Days:", 50, 20))
    print(colorize_output(results['max_pos_streak'], "Max Pos Streak:", 5, 3))
    print(colorize_output(results['max_neg_streak'], "Max Neg streak:", results['max_pos_streak'], results['max_pos_streak'] * 10, lower_is_better=True))
    print(colorize_output(results['profit_per_day'], "Profit per Day ($):", 20, 5))


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
    parser.add_argument("--filter", type=float, default=0.01,help="Minimum UpProbability variance for stock filtering")
    parser.add_argument("--up_prob", type=float, default=0.68,help="UpProbability threshold for buy signals")
    parser.add_argument("--force", action='store_true', help="Force the script to run even if data is not up to last trading date")
    parser.add_argument("--recommend", action='store_true', default=False, help="Recommend basic system changes based on the backtest risk metrics")
    parser.add_argument("--best", action='store_true', default=False, help="Generate best buy signals for the current or last trading day")
    parser.add_argument("--num_signals", type=int, default=4, help="Number of best signals to generate (default: 4)")
    parser.add_argument("--clear_signals", action='store_true', default=False, help="Clear all pending signals before generating new ones (use to start fresh)")
    
    # Add new optimization related arguments
    parser.add_argument("--optimize", action='store_true', default=False, help="Run in optimization mode to find best parameters")
    parser.add_argument("--optimize_param", type=str, action='append', default=None, help="Parameters to optimize (can be used multiple times, e.g. --optimize_param up_prob_threshold --optimize_param max_positions)")
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
