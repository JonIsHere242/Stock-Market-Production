#!/usr/bin/env python
"""
0__Util.py - Consolidated trading system utilities using the new file structure
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time as Secondary_time_import_carefully_use_with_caution
import pandas_market_calendars as mcal
import traceback
import math
import warnings
from pytz import timezone as pytz_timezone
import inspect
import hashlib
from pathlib import Path
import threading
import atexit
from zoneinfo import ZoneInfo
#from backtrader_ib_insync import IBStore
import ib_insync as ibi
import platform
import subprocess
import argparse
import exchange_calendars as ec
import backtrader as bt
import json
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
if hasattr(pd, 'errors') and hasattr(pd.errors, 'PerformanceWarning'):
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

nyse = ec.get_calendar('XNYS')

try:
    from backtrader_ib_insync import IBStore
except ImportError:
    IBStore = None


# File paths for consolidated data files
SIGNALS_FILE = 'Data/0__signals.parquet'  # canonical signals file (recent-day candidate pool)
COMPLETED_TRADES_FILE = 'Data/TradeHistory.parquet'
BACKTEST_METRICS_FILE = 'Data/0__BacktestMetrics.parquet'

# Mechanical signal pre-filter (FilterRubric Step-1 non-web hard exclusions:
# price, market cap, weekly volatility, RSI death-zone). Impl lives in
# signal_filter.py; re-exported here so the rest of the pipeline has one home.
# Adds audit columns to the signals file (MechRSI14, MechWeeklyVolPct,
# MechExclude, MechReasons); does NOT drop rows.
try:
    from signal_filter import (
        annotate_signals_mechanical_filter,
        prefilter_signals_file,
        compute_rsi14,
        compute_weekly_vol_pct,
    )
except Exception:  # keep Util.py importable even if signal_filter is missing
    annotate_signals_mechanical_filter = None
    prefilter_signals_file = None
    compute_rsi14 = None
    compute_weekly_vol_pct = None

# Schemas for data files
SIGNALS_SCHEMA = {
    'Symbol': 'string',
    'Status': 'string',  # "Pending", "Active", "Completed"
    'SignalDate': 'datetime64[ns]',
    'TargetDate': 'datetime64[ns]',
    'SignalPrice': 'float64',
    'CurrentPrice': 'float64',
    'EntryDate': 'datetime64[ns]',
    'EntryPrice': 'float64',
    'PositionSize': 'float64',
    'StopPrice': 'float64',
    'TargetPrice': 'float64',
    'ExitDate': 'datetime64[ns]',
    'ExitPrice': 'float64',
    'ExitReason': 'string',
    'PnL': 'float64',
    'PnLPct': 'float64',
    'UpProbability': 'float64',
    'ATR': 'float64',
    'ConsecutiveLosses': 'int64',
    'LastUpdate': 'datetime64[ns]'
}

COMPLETED_TRADES_SCHEMA = {
    'Symbol': 'string',
    'EntryDate': 'datetime64[ns]',
    'ExitDate': 'datetime64[ns]',
    'EntryPrice': 'float64',
    'ExitPrice': 'float64',
    'PositionSize': 'float64',
    'PnL': 'float64',
    'PnLPct': 'float64',
    'DaysHeld': 'int64',
    'Commission': 'float64',
    'Slippage': 'float64',
    'TradeType': 'string',
    'ExitReason': 'string',
    'ATR': 'float64',
    'UpProbability': 'float64',
    'AccountValue': 'float64',
    'Source': 'string'  # "Backtest", "Live", "Paper"
}

# Strategy parameters
STRATEGY_PARAMS = {
    # Signal generation parameters
    'up_prob_threshold': 0.60,          # Probability threshold for buy signals  
    'up_prob_min_trigger': 0.70,        # Minimum probability to trigger buy
    
    # Position management parameters 
    'max_positions': 10,                # Maximum concurrent positions (4->8->10 2026-06-08; joint seed x config x book grid: k10 most robust marginally). NOTE: 9_SuperFastBroker sizes from its OWN hardcoded PositionSizer(max_positions=...) ~line 73 — kept in sync at 10.
    'reserve_percent': 0.10,            # Cash reserve percentage
    'max_group_allocation': 0.45,       # Maximum allocation to a group
    
    # Risk management parameters
    'risk_per_trade_pct': 2.0,          # Risk per trade as percentage
    'max_position_pct': 20.0,           # Maximum position size as percentage
    'min_position_pct': 20,            # Minimum position size as percentage
    'atr_period': 14,                   # ATR calculation period
    
    # Stop loss and take profit parameters
    'stop_loss_atr_multiple': 0.75,     # Stop loss ATR multiplier 
    'trailing_stop_atr_multiple': 2.0,  # Trailing stop ATR multiplier
    'take_profit_percent': 20.0,        # Take profit threshold percentage
    
    # Position timeout and evaluation parameters
    'position_timeout': 5,              # Maximum days to hold a position
    'min_daily_return': 1.0,            # Minimum expected daily return
    
    # Volume filter parameters
    'min_dollar_volume': 10000,       # Minimum dollar volume ($10M)
    'min_avg_volume': 1000,

    # Other system parameters
    'lockup_days': 3,                   # Trading lockup period
    'rule_201_threshold': -9.99,        # Rule 201 threshold
    'rule_201_cooldown': 1,             # Rule 201 cooldown period
    'stop_loss_percent': 5.0,           # Standard stop loss percentage
    'expected_profit_per_day_percentage': 0.25 # Expected profit per day
}

# Strategy parameters in tuple format for backtrader
STRATEGY_PARAMS_TUPLE = tuple((k, v) for k, v in STRATEGY_PARAMS.items())

#=================================================#
# Enhanced Logging System
#=================================================#

class Colors:
    """ANSI color codes for colored terminal output"""
    RESET = "\033[0m"
    INFO = "\033[38;2;100;149;237m"  # Cornflower blue
    WARN = "\033[38;2;220;220;0m"    # Yellow
    ERROR = "\033[38;2;220;0;0m"     # Red
    DETAIL = "\033[38;2;70;130;180m" # Steel blue
    DEBUG = "\033[38;2;0;180;180m"   # Cyan
    SUCCESS = "\033[38;2;50;220;50m" # Bright Green
    TRACE = "\033[38;2;180;180;180m" # Gray
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    TIMESTAMP = "\033[38;2;150;150;150m" # Gray for timestamps

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""
    COLOR_MAP = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARN,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.ERROR,
    }

    def format(self, record):
        # Get color from extra or level
        color = getattr(record, 'color', self.COLOR_MAP.get(record.levelno, Colors.RESET))
        
        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)
        colored_ts = f"{Colors.TIMESTAMP}{timestamp}{Colors.RESET}"
        
        # Format level name
        levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Include file and line info in debug mode
        file_info = ""
        if record.levelno <= logging.DEBUG:
            filename = os.path.basename(record.pathname)
            file_info = f"{Colors.TRACE}[{filename}:{record.lineno}]{Colors.RESET} "
        
        # Build formatted message
        return f"{colored_ts} - {levelname} - {file_info}{record.getMessage()}"

# Global registry to track initialized loggers
_LOGGER_REGISTRY = {}
_LOG_LOCK = threading.RLock()

def get_script_name():
    """Get the name of the calling script, handling both direct execution and imports"""
    # Start from one frame up to skip this function
    for frame in inspect.stack()[1:]:
        module = inspect.getmodule(frame[0])
        # Skip frames from this module
        if module and module.__name__ != __name__:
            # Get the file path
            file_path = Path(frame.filename)
            # If it's a .py file, return its stem
            if file_path.suffix.lower() == '.py':
                return file_path.stem
            # For Jupyter notebooks, create a stable name
            elif file_path.suffix.lower() == '.ipynb':
                # Hash the full path to create a stable identifier
                hash_obj = hashlib.md5(str(file_path).encode())
                return f"jupyter_{hash_obj.hexdigest()[:8]}"
    
    # Fallback to the main script name
    return Path(sys.argv[0]).stem if sys.argv[0] else "unknown"

def setup_logging(log_dir='Data/logging', console=True, debug=False, script_name=None):
    """
    Set up logging with consistent configuration.
    
    Args:
        log_dir: Directory for log files
        console: Whether to output to console
        debug: Whether to enable debug mode
        script_name: Override script name detection (optional)
        
    Returns:
        The configured logger
    """
    with _LOG_LOCK:
        # Determine the script name if not provided
        if script_name is None:
            script_name = get_script_name()
        
        # Check if this script already has a logger
        if script_name in _LOGGER_REGISTRY:
            return _LOGGER_REGISTRY[script_name]
        
        # Create a unique logger for this script
        logger = logging.getLogger(script_name)
        # Clear any existing handlers
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # Set up the log file path
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{script_name}.log"
        
        # File handler - one per script
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler with colors
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
            console_handler.setFormatter(ColoredFormatter())
            logger.addHandler(console_handler)
        
        # Set the logger's level
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Store in registry
        _LOGGER_REGISTRY[script_name] = logger
        
        # Register cleanup on exit
        atexit.register(lambda: logger.handlers.clear())
        
        # Log initialization
        logger.info(f"Logging initialized for {script_name}")
        if debug:
            logger.debug("Debug mode enabled")
        
        return logger

def get_logger(script_name=None, **kwargs):
    """
    Set up logging with consistent configuration including subprocess detection and colored output.
    
    Args:
        script_name: Name of script so the logging file will be created in the logging folder
        kwargs: Additional kwargs currently only debug and info by default but other methods like trace and success and failure
        
    Returns:
        The configured logger
    """
    is_subprocess = "multiprocessing" in sys.modules and sys.modules["multiprocessing"].current_process().name != "MainProcess"

    # If inside a subprocess make a more simple logger that does not need to initialize every single time
    if is_subprocess:
        logger = logging.getLogger("subprocess_logger")
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG if kwargs.get('debug', False) else logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(ColoredFormatter())
            logger.addHandler(handler)
        return logger

    if script_name is None:
        script_name = get_script_name()
    
    if script_name in _LOGGER_REGISTRY:
        return _LOGGER_REGISTRY[script_name]
    
    return setup_logging(script_name=script_name, **kwargs)

def dprint(message, level="INFO", show_timestamp=True, indent=0, logger=None):
    """
    Enhanced debug print with colors and logging integration
    
    Args:
        message: The message to print
        level: One of "INFO", "WARN", "ERROR", "DETAIL", "DEBUG", "SUCCESS", "TRACE"
        show_timestamp: Whether to include timestamp
        indent: Number of spaces to indent
        logger: Optional logger instance to also log the message
    """
    color = getattr(Colors, level, Colors.INFO)
    level_str = f"[{level}]".ljust(8)
    indent_str = " " * indent

    # Console output
    if show_timestamp:
        timestamp = f"{Colors.TIMESTAMP}{datetime.now().strftime('%H:%M:%S.%f')[:-3]}{Colors.RESET} "
    else:
        timestamp = ""
    
    print(f"{timestamp}{indent_str}{color}{level_str}{Colors.RESET} {message}")

    # File logging - try to get logger if not provided
    if logger is None and _LOGGER_REGISTRY:
        script_name = get_script_name()
        if script_name in _LOGGER_REGISTRY:
            logger = _LOGGER_REGISTRY[script_name]
    
    if logger:
        level_map = {
            'INFO': logging.INFO,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
            'DEBUG': logging.DEBUG,
            'SUCCESS': logging.INFO,
            'DETAIL': logging.DEBUG,
            'TRACE': logging.DEBUG
        }
        log_level = level_map.get(level, logging.INFO)
        logger.log(log_level, f"{indent_str}{message}", extra={'color': color})

class LogPerformance:
    """Context manager for timing and logging performance metrics"""
    
    def __init__(self, operation_name, logger=None, level="INFO"):
        self.operation_name = operation_name
        self.logger = logger if logger else get_logger()
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(
            getattr(logging, self.level) if hasattr(logging, self.level) else logging.INFO,
            f"Starting: {self.operation_name}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type:
            self.logger.error(f"Failed: {self.operation_name} after {duration} - {exc_val}")
        else:
            self.logger.log(
                getattr(logging, self.level) if hasattr(logging, self.level) else logging.INFO,
                f"Completed: {self.operation_name} in {duration}"
            )

def log_progress(iterable, logger=None, every=None, total=None, description="Processing"):
    """
    Log progress through an iterable
    
    Args:
        iterable: The iterable to process
        logger: Logger to use (will auto-detect if None)
        every: Log every N items (will auto-calculate if None)
        total: Total items (will use len() if None and available)
        description: Description to include in log messages
        
    Returns:
        Generator yielding items from the original iterable
    """
    if logger is None:
        logger = get_logger()
    
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None
    
    if every is None:
        if total and total > 0:
            # Log approximately 10 times during the process
            every = max(1, total // 10)
        else:
            every = 100
    
    start_time = datetime.now()
    last_log_time = start_time
    
    for i, item in enumerate(iterable, 1):
        yield item
        
        if i % every == 0 or (total and i == total):
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            # Only log if it's been at least 0.5 seconds since last log
            if (current_time - last_log_time).total_seconds() >= 0.5:
                if total:
                    percent = (i / total) * 100
                    msg = f"{description}: {i}/{total} ({percent:.1f}%) in {elapsed}"
                    
                    # Estimate time remaining
                    if i > 0:
                        items_per_sec = i / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                        if items_per_sec > 0:
                            remaining_items = total - i
                            est_remaining_sec = remaining_items / items_per_sec
                            est_completion = current_time + timedelta(seconds=est_remaining_sec)
                            msg += f", est. completion at {est_completion.strftime('%H:%M:%S')}"
                else:
                    msg = f"{description}: {i} items in {elapsed}"
                
                logger.info(msg)
                last_log_time = current_time




#=================================================#
# File context for easy file info
#=================================================#


def load(input_data):
    """
    Loads tabular data from a parquet file or CSV file.
    If a DataFrame is passed directly, it is returned unchanged.
    
    Parameters:
        input_data: A pandas DataFrame or a file path (string or Path) to a .parquet or .csv file.
        
    Returns:
        df: A pandas DataFrame containing the loaded data.
    """

    # If input is already a DataFrame, return it directly
    if isinstance(input_data, pd.DataFrame):
        return input_data

    # Validate file path
    if input_data is None or not os.path.exists(input_data):
        raise ValueError("File path is None or does not exist")

    file_path = Path(input_data)

    # Load data based on file extension
    if file_path.suffix.lower() == '.parquet':
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(f"Error reading parquet file: {e}")
    elif file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    else:
        raise ValueError("Unsupported file type. Only .parquet and .csv are supported.")
    
    return df



def context(input_data):
    """
    Takes in a file path (txt, csv, parquet) or pandas DataFrame 
    and returns various information about the data
    """
    # Check if input is already a DataFrame
    if isinstance(input_data, pd.DataFrame):
        df = input_data
        data_source = "DataFrame (passed directly)"
    else:
        # Handle file path input
        if input_data is None or not os.path.exists(input_data):
            raise ValueError("File path is None or does not exist")

        file_path = Path(input_data)
        data_source = file_path.name
        
        # Read the data based on file extension
        if file_path.suffix.lower() == '.parquet':
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                raise ValueError(f"Error reading parquet file: {str(e)}")
        elif file_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {str(e)}")
        elif file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r') as f:
                    data = f.readlines()
                df = pd.DataFrame(data, columns=['Line'])
            except Exception as e:
                raise ValueError(f"Error reading text file: {str(e)}")
        else:
            raise ValueError("Unsupported file type. Only .parquet, .csv, and .txt are supported.")
    
    # Print various information about the DataFrame
    dprint(f"Data source: {data_source}")
    dprint(f"Columns:")
    dprint(df.columns)
    dprint(f"Head:")
    dprint(df.head())
    dprint(f"Describe (numeric):")
    dprint(df.describe())
    dprint(f"Describe (all):")
    dprint(df.describe(include='all'))
    dprint(f"Info:")
    dprint(df.info())
    dprint(f"Data types:")
    dprint(df.dtypes)
    dprint(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")




#=================================================#
# Trading Calendar Functions  
#=================================================#

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
            today_market_open = today_market_open.replace(tzinfo=pytz_timezone('UTC'))
        
        now_utc = datetime.now(pytz_timezone('UTC'))
        
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

def is_market_open():
    """Check if the market is currently open."""
    nyse = mcal.get_calendar('NYSE')
    now = pd.Timestamp.now(tz='America/New_York')
    today_date = now.date()
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today_date, end_date=today_date)
    if schedule.empty:
        return False, "Not a trading day"
    
    market_open = schedule.iloc[0]['market_open'].tz_convert('America/New_York')
    market_close = schedule.iloc[0]['market_close'].tz_convert('America/New_York')
    
    if now < market_open:
        return False, "Market not yet open"
    elif now > market_close:
        return False, "Market closed for the day"
    else:
        return True, "Market open"
    



def is_nyse_open(manual_override=False):
    """
    Returns True if the NYSE is currently open, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        
        return True
        
    tz_nyse = ZoneInfo('America/New_York')
    now_nyse = pd.Timestamp.now(tz_nyse)
    current_date = now_nyse.date()

    if not nyse.is_session(current_date):
        
        return False

    try:
        market_open = nyse.session_open(current_date)
        market_close = nyse.session_close(current_date)
    except Exception as e:
        
        return False

    if market_open <= now_nyse <= market_close:
        
        return True
    else:
       
        return False





def wait_for_market_open(tz=ZoneInfo('America/New_York'), max_wait_minutes=180, manual_override=False):
    """
    Blocks execution until the NYSE is open or until max_wait_minutes is reached.
    Returns True if the market opened within that time, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        return True
        

    now = pd.Timestamp.now(tz)

    current_date = now.date()

    if nyse.is_session(current_date):
        try:
            market_open = nyse.session_open(current_date)
            market_close = nyse.session_close(current_date)
        except Exception as e:
            return False

        if market_open <= now <= market_close:
            return True

    next_open = nyse.next_open(now)
    if not next_open:
        return False


    max_wait_seconds = max_wait_minutes * 60
    wait_seconds = 0
    interval = 30  # Check every 30 seconds

    # Wait loop
    while wait_seconds < max_wait_seconds:
        now = pd.Timestamp.now(tz)
        if now >= next_open:
            return True
        Secondary_time_import_carefully_use_with_caution.sleep(interval)

        wait_seconds += interval

    return False


#=================================================#
# New Trading Data Functions - Signals
#=================================================#

def ensure_dir(file_path):
    """Ensure directory exists for a given file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)





def read_signals(status_filter=None):
    """
    Read signals data with optional status filtering
    
    Parameters:
    -----------
    status_filter : str or list, optional
        Filter by status: "Pending", "Active", "Completed", or a list of these
        
    Returns:
    --------
    DataFrame: Signals data
    """
    logger = get_logger()
    try:
        # Check if signals file exists
        if os.path.exists(SIGNALS_FILE):
            df = pd.read_parquet(SIGNALS_FILE)
            
            # Apply status filter if provided
            if status_filter:
                if isinstance(status_filter, str):
                    df = df[df['Status'] == status_filter]
                elif isinstance(status_filter, list):
                    df = df[df['Status'].isin(status_filter)]
            
            logger.debug(f"Read {len(df)} signals from {SIGNALS_FILE}" + 
                       (f" with filter {status_filter}" if status_filter else ""))
            return df
        else:
            logger.info(f"Signals file {SIGNALS_FILE} not found. Creating empty DataFrame.")
            return create_empty_signals_df()
    except Exception as e:
        logger.error(f"Error reading signals: {str(e)}")
        logger.debug(traceback.format_exc())
        return create_empty_signals_df()



def create_empty_signals_df():
    """Create an empty signals DataFrame with the correct schema"""
    df = pd.DataFrame(columns=list(SIGNALS_SCHEMA.keys()))
    
    # Set proper dtypes
    for col, dtype in SIGNALS_SCHEMA.items():
        if dtype == 'datetime64[ns]':
            df[col] = pd.Series(dtype='datetime64[ns]')
        else:
            df[col] = pd.Series(dtype=dtype)
    
    return df



def write_signals(df):
    """
    Write signals data to file
    
    Parameters:
    -----------
    df : DataFrame
        Signals data to write
    """
    logger = get_logger()
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure required fields exist
        for col, dtype in SIGNALS_SCHEMA.items():
            if col not in df.columns:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.Series(dtype='datetime64[ns]')
                elif dtype == 'float64':
                    df[col] = pd.Series(dtype='float64')
                elif dtype == 'int64':
                    df[col] = pd.Series(dtype='int64')
                elif dtype == 'string':
                    df[col] = pd.Series(dtype='string')
        
        # Ensure data types
        for col, dtype in SIGNALS_SCHEMA.items():
            if col in df.columns:
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
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Update LastUpdate timestamp
        df['LastUpdate'] = datetime.now()
        
        # Handle NaT for datetime columns before saving
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[col] = df[col].astype('object').where(df[col].notnull(), None)
        
        # Ensure directory exists
        ensure_dir(SIGNALS_FILE)
        
        # Write to file
        df.to_parquet(SIGNALS_FILE, index=False)
        #logger.info(f"Successfully wrote {len(df)} signals to {SIGNALS_FILE}")
        
    except Exception as e:
        logger.error(f"Error writing signals: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def get_buy_signals(target_date=None, min_probability=0.60):
    """
    Get pending buy signals for a specific target date
    
    Parameters:
    -----------
    target_date : datetime.date, optional
        Target date for signals (default: next trading day)
    min_probability : float, optional
        Minimum probability threshold for signals
        
    Returns:
    --------
    list: Buy signals as dictionaries
    """
    logger = get_logger()
    try:
        # Get all pending signals
        signals_df = read_signals(status_filter="Pending")
        
        if signals_df.empty:
            logger.info("No pending buy signals found")
            return []
        
        # Determine target date if not provided
        if target_date is None:
            current_date = datetime.now().date()
            target_date = get_next_trading_day(current_date)
        
        # Filter signals for target date and minimum probability
        filtered_signals = signals_df[
            (signals_df['TargetDate'].dt.date == target_date) &
            (signals_df['UpProbability'] >= min_probability)
        ]
        
        if filtered_signals.empty:
            logger.info(f"No buy signals found for {target_date} with probability >= {min_probability}")
            return []
        
        # Convert to list of dictionaries
        signals_list = filtered_signals.to_dict('records')
        logger.info(f"Found {len(signals_list)} buy signals for {target_date}")
        
        return signals_list
        
    except Exception as e:
        logger.error(f"Error getting buy signals: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def update_signal_status(symbol, new_status, **kwargs):
    """
    Update a signal's status and associated data
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to update
    new_status : str
        New status: "Pending", "Active", or "Completed"
    **kwargs : dict
        Additional fields to update
    """
    logger = get_logger()
    try:
        df = read_signals()
        
        # Check if symbol exists
        if symbol not in df['Symbol'].values:
            logger.warning(f"Symbol {symbol} not found in signals data")
            return False
        
        # Update status
        df.loc[df['Symbol'] == symbol, 'Status'] = new_status
        
        # Update additional fields
        for key, value in kwargs.items():
            if key in df.columns:
                df.loc[df['Symbol'] == symbol, key] = value
        
        # Update LastUpdate timestamp
        df.loc[df['Symbol'] == symbol, 'LastUpdate'] = datetime.now()
        
        # Write updated DataFrame
        write_signals(df)
        logger.info(f"Updated {symbol} status to {new_status}")
        
        # If completing a trade, add to completed trades
        if new_status == "Completed":
            row = df[df['Symbol'] == symbol].iloc[0]
            
            # Only add to completed trades if we have entry and exit data
            if not pd.isna(row['EntryDate']) and not pd.isna(row['ExitDate']):
                add_completed_trade_from_signal(row)
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating signal status for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def add_signal(symbol, signal_price, target_date, up_probability, atr=None):
    """
    Add a new pending signal
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    signal_price : float
        Signal price 
    target_date : datetime.date
        Target date for execution
    up_probability : float
        Prediction probability
    atr : float, optional
        Average True Range for volatility estimation
    """
    logger = get_logger()
    try:
        df = read_signals()
        
        # Check if signal already exists
        if symbol in df['Symbol'].values:
            existing = df[df['Symbol'] == symbol]
            
            # If already active or completed, don't override
            if any(existing['Status'].isin(["Active", "Completed"])):
                logger.info(f"Symbol {symbol} already has active or completed status. Not adding as new signal.")
                return False
            
            # If pending, update it
            df.loc[df['Symbol'] == symbol, 'SignalPrice'] = signal_price
            df.loc[df['Symbol'] == symbol, 'TargetDate'] = pd.Timestamp(target_date)
            df.loc[df['Symbol'] == symbol, 'UpProbability'] = up_probability
            df.loc[df['Symbol'] == symbol, 'SignalDate'] = pd.Timestamp(datetime.now())
            df.loc[df['Symbol'] == symbol, 'Status'] = "Pending"
            if atr is not None:
                df.loc[df['Symbol'] == symbol, 'ATR'] = atr
            
        else:
            # Create new row
            new_row = {
                'Symbol': symbol,
                'Status': "Pending",
                'SignalDate': pd.Timestamp(datetime.now()),
                'TargetDate': pd.Timestamp(target_date),
                'SignalPrice': signal_price,
                'UpProbability': up_probability,
                'ConsecutiveLosses': 0,
                'CurrentPrice': signal_price  # Set initial current price to signal price
            }
            
            if atr is not None:
                new_row['ATR'] = atr
                
            # Add new row to DataFrame
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
        
        # Write updated DataFrame
        write_signals(df)
        logger.info(f"Added new signal for {symbol} with target date {target_date}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding signal for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def mark_signal_as_active(symbol, position_size, entry_price, entry_date=None, stop_price=None, target_price=None):
    """
    Mark a signal as active (position opened)
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    position_size : float
        Number of shares/contracts
    entry_price : float
        Actual entry price
    entry_date : datetime, optional
        Entry date (default: now)
    stop_price : float, optional
        Initial stop loss price
    target_price : float, optional
        Take profit target price
    """
    if entry_date is None:
        entry_date = datetime.now()
        
    update_data = {
        'EntryDate': pd.Timestamp(entry_date),
        'EntryPrice': entry_price,
        'PositionSize': position_size,
        'CurrentPrice': entry_price
    }
    
    if stop_price is not None:
        update_data['StopPrice'] = stop_price
        
    if target_price is not None:
        update_data['TargetPrice'] = target_price
    
    return update_signal_status(symbol, "Active", **update_data)

def mark_signal_as_completed(symbol, exit_price, exit_date=None, exit_reason=None):
    """
    Mark a signal as completed (position closed)
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    exit_price : float
        Exit price
    exit_date : datetime, optional
        Exit date (default: now)
    exit_reason : str, optional
        Reason for exit (e.g., "Stop Loss", "Take Profit", "Max Hold Time")
    """
    logger = get_logger()
    if exit_date is None:
        exit_date = datetime.now()
    
    # Get current data
    df = read_signals()
    if symbol not in df['Symbol'].values:
        logger.warning(f"Symbol {symbol} not found in signals data")
        return False
    
    row = df[df['Symbol'] == symbol].iloc[0]
    
    # Calculate P&L
    entry_price = row.get('EntryPrice')
    position_size = row.get('PositionSize')
    
    if pd.isna(entry_price) or pd.isna(position_size):
        pnl = None
        pnl_pct = None
    else:
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price / entry_price - 1) * 100
    
    update_data = {
        'ExitDate': pd.Timestamp(exit_date),
        'ExitPrice': exit_price,
        'PnL': pnl,
        'PnLPct': pnl_pct,
        'CurrentPrice': exit_price
    }
    
    if exit_reason is not None:
        update_data['ExitReason'] = exit_reason
    
    return update_signal_status(symbol, "Completed", **update_data)

#=================================================#
# New Trading Data Functions - Completed Trades
#=================================================#

def read_completed_trades():
    """Read completed trades data"""
    logger = get_logger()
    try:
        if os.path.exists(COMPLETED_TRADES_FILE):
            df = pd.read_parquet(COMPLETED_TRADES_FILE)
            logger.debug(f"Read {len(df)} completed trades from {COMPLETED_TRADES_FILE}")
            return df
        else:
            logger.info(f"Completed trades file {COMPLETED_TRADES_FILE} not found. Creating empty DataFrame.")
            # Create empty DataFrame with correct schema
            df = pd.DataFrame(columns=list(COMPLETED_TRADES_SCHEMA.keys()))
            
            # Set proper dtypes
            for col, dtype in COMPLETED_TRADES_SCHEMA.items():
                if dtype == 'datetime64[ns]':
                    df[col] = pd.Series(dtype='datetime64[ns]')
                else:
                    df[col] = pd.Series(dtype=dtype)
            
            return df
            
    except Exception as e:
        logger.error(f"Error reading completed trades: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame(columns=list(COMPLETED_TRADES_SCHEMA.keys()))

def write_completed_trades(df):
    """Write completed trades data to file"""
    logger = get_logger()
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure required fields exist
        for col, dtype in COMPLETED_TRADES_SCHEMA.items():
            if col not in df.columns:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.Series(dtype='datetime64[ns]')
                elif dtype in ['float64', 'int64']:
                    df[col] = pd.Series(dtype=dtype)
                elif dtype == 'string':
                    df[col] = pd.Series(dtype='string')
        
        # Ensure data types
        for col, dtype in COMPLETED_TRADES_SCHEMA.items():
            if col in df.columns:
                if dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                elif dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'string':
                    if df[col].dtype != 'string':
                        df[col] = df[col].astype('string')
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle NaT for datetime columns before saving
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[col] = df[col].astype('object').where(df[col].notnull(), None)
        
        # Ensure directory exists
        ensure_dir(COMPLETED_TRADES_FILE)
        
        # Write to file
        df.to_parquet(COMPLETED_TRADES_FILE, index=False)
        #logger.info(f"Successfully wrote {len(df)} completed trades to {COMPLETED_TRADES_FILE}")
        
    except Exception as e:
        logger.error(f"Error writing completed trades: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def add_completed_trade(trade_data):
    """Add a completed trade to the historical record"""
    logger = get_logger()
    try:
        df = read_completed_trades()
        
        # Create new row
        new_df = pd.DataFrame([trade_data])
        
        # Concatenate with existing data
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Write updated DataFrame
        write_completed_trades(df)
        verbose = False
        if verbose:
            logger.info(f"Added completed trade for {trade_data.get('Symbol', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding completed trade: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def add_completed_trade_from_signal(signal_row):
    """Convert a signal row to a completed trade and add it"""
    logger = get_logger()
    try:
        # Skip if missing required data
        if pd.isna(signal_row['EntryDate']) or pd.isna(signal_row['ExitDate']):
            logger.warning(f"Cannot add completed trade for {signal_row['Symbol']}: Missing entry or exit data")
            return False
        
        # Calculate days held
        entry_date = pd.to_datetime(signal_row['EntryDate'])
        exit_date = pd.to_datetime(signal_row['ExitDate'])
        days_held = (exit_date - entry_date).days
        
        # Create trade data
        trade_data = {
            'Symbol': signal_row['Symbol'],
            'EntryDate': entry_date,
            'ExitDate': exit_date,
            'EntryPrice': signal_row['EntryPrice'],
            'ExitPrice': signal_row['ExitPrice'],
            'PositionSize': signal_row['PositionSize'],
            'PnL': signal_row.get('PnL'),
            'PnLPct': signal_row.get('PnLPct'),
            'DaysHeld': days_held,
            'Commission': 0.0,  # Default values
            'Slippage': 0.0,    # Default values
            'TradeType': 'Long',  # Assuming long trades only
            'ExitReason': signal_row.get('ExitReason', 'Unknown'),
            'ATR': signal_row.get('ATR'),
            'UpProbability': signal_row.get('UpProbability'),
            'AccountValue': np.nan,  # Will need to be updated separately
            'Source': 'Live'
        }
        
        return add_completed_trade(trade_data)
        
    except Exception as e:
        logger.error(f"Error converting signal to completed trade: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

#=================================================#
# New Trading Data Functions - Backtest Metrics
#=================================================#

def read_backtest_metrics():
    """Read backtest metrics data"""
    logger = get_logger()
    try:
        if os.path.exists(BACKTEST_METRICS_FILE):
            df = pd.read_parquet(BACKTEST_METRICS_FILE)
            logger.debug(f"Read {len(df)} backtest metrics from {BACKTEST_METRICS_FILE}")
            return df
        else:
            logger.info(f"Backtest metrics file {BACKTEST_METRICS_FILE} not found. Creating empty DataFrame.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error reading backtest metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

def write_backtest_metrics(df):
    """Write backtest metrics data to file"""
    logger = get_logger()
    try:
        # Ensure directory exists
        ensure_dir(BACKTEST_METRICS_FILE)
        
        # Write to file
        df.to_parquet(BACKTEST_METRICS_FILE, index=False)
        #logger.info(f"Successfully wrote backtest metrics to {BACKTEST_METRICS_FILE}")
        
    except Exception as e:
        logger.error(f"Error writing backtest metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def add_backtest_run(metrics, strategy_params=None, run_id=None):
    """Add metrics from a backtest run"""
    logger = get_logger()
    try:
        df = read_backtest_metrics()
        
        # Add run ID and timestamp
        metrics['run_id'] = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metrics['timestamp'] = datetime.now()
        
        # Add strategy parameters
        if strategy_params:
            for key, value in strategy_params.items():
                metrics[f"param_{key}"] = value
        
        # Create new row
        new_df = pd.DataFrame([metrics])
        
        # Concatenate with existing data
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Write updated DataFrame
        write_backtest_metrics(df)
        logger.info(f"Added new backtest run with ID {metrics['run_id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding backtest run: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

#=================================================#
# Position Management Functions
#=================================================#

def get_active_positions():
    """
    Get all currently active positions
    
    Returns:
    --------
    list: Active positions as dictionaries
    """
    logger = get_logger()
    try:
        # Get active positions
        positions_df = read_signals(status_filter="Active")
        
        if positions_df.empty:
            logger.info("No active positions found")
            return []
        
        # Convert to list of dictionaries
        positions = positions_df.to_dict('records')
        logger.debug(f"Found {len(positions)} active positions")
        
        return positions
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def get_position_count():
    """
    Get the current number of active positions
    
    Returns:
    --------
    int: Number of active positions
    """
    positions = get_active_positions()
    return len(positions)

def get_position_sizes():
    """
    Get the sizes of all active positions
    
    Returns:
    --------
    dict: Dictionary mapping symbols to position sizes
    """
    positions = get_active_positions()
    return {p['Symbol']: p.get('PositionSize', 0) for p in positions}

def calculate_position_value(symbol=None):
    """
    Calculate the current value of positions
    
    Parameters:
    -----------
    symbol : str, optional
        If provided, calculate value for this symbol only
        
    Returns:
    --------
    float: Total value of position(s)
    """
    logger = get_logger()
    try:
        # Get positions
        positions_df = read_signals(status_filter="Active")
        
        if symbol:
            positions_df = positions_df[positions_df['Symbol'] == symbol]
        
        if positions_df.empty:
            return 0.0
        
        # Calculate value for each position
        total_value = 0.0
        for _, position in positions_df.iterrows():
            if pd.notna(position.get('CurrentPrice')) and pd.notna(position.get('PositionSize')):
                position_value = position['CurrentPrice'] * position['PositionSize']
                total_value += position_value
        
        return total_value
    except Exception as e:
        logger.error(f"Error calculating position value: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0.0

def update_position_prices(symbols_with_prices):
    """
    Update current prices for multiple positions
    
    Parameters:
    -----------
    symbols_with_prices : dict
        Dictionary mapping symbols to current prices
        
    Returns:
    --------
    bool: Success status
    """
    logger = get_logger()
    try:
        # Get active positions
        positions_df = read_signals(status_filter="Active")
        
        if positions_df.empty:
            logger.info("No active positions to update")
            return True
        
        # Track whether any positions were updated
        updated = False
        
        # Update each position
        for symbol, price in symbols_with_prices.items():
            if symbol in positions_df['Symbol'].values:
                positions_df.loc[positions_df['Symbol'] == symbol, 'CurrentPrice'] = price
                
                # Calculate PnL if we have entry price
                entry_price = positions_df.loc[positions_df['Symbol'] == symbol, 'EntryPrice'].iloc[0]
                if pd.notna(entry_price):
                    pnl_pct = (price / entry_price - 1) * 100
                    positions_df.loc[positions_df['Symbol'] == symbol, 'PnLPct'] = pnl_pct
                
                updated = True
                logger.debug(f"Updated price for {symbol} to ${price:.2f}")
        
        # If no positions were updated, we're done
        if not updated:
            return True
        
        # Write updated positions back to file
        positions_df['LastUpdate'] = datetime.now()
        
        # Get all signals to ensure we update properly
        all_signals = read_signals()
        
        # Update the active positions in the full signals dataframe
        for symbol in symbols_with_prices.keys():
            if symbol in all_signals['Symbol'].values:
                mask = all_signals['Symbol'] == symbol
                
                # Only update if status is active
                if all_signals.loc[mask, 'Status'].iloc[0] == "Active":
                    all_signals.loc[mask, 'CurrentPrice'] = symbols_with_prices[symbol]
                    all_signals.loc[mask, 'LastUpdate'] = datetime.now()
                    
                    # Update PnL if we have entry price
                    entry_price = all_signals.loc[mask, 'EntryPrice'].iloc[0]
                    if pd.notna(entry_price):
                        pnl_pct = (symbols_with_prices[symbol] / entry_price - 1) * 100
                        all_signals.loc[mask, 'PnLPct'] = pnl_pct
        
        # Write updated signals
        write_signals(all_signals)
        logger.info(f"Updated prices for {len(symbols_with_prices)} positions")
        
        return True
    except Exception as e:
        logger.error(f"Error updating position prices: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

#=================================================#
# Risk Management Functions
#=================================================#






def should_sell(current_price, entry_price, entry_date, current_date, 
               stop_loss_percent, take_profit_percent, position_timeout, 
               expected_profit_per_day_percentage, verbose=False):
    """
    Determine if a position should be sold based on various criteria.
    
    Parameters:
    - current_price: Current price of the security
    - entry_price: Entry price of the position
    - entry_date: Date when the position was entered
    - current_date: Current date
    - stop_loss_percent: Stop-loss percentage
    - take_profit_percent: Take-profit percentage
    - position_timeout: Maximum number of days to hold the position
    - expected_profit_per_day_percentage: Expected minimum profit per day
    - verbose: Whether to print detailed information
    
    Returns:
    - Boolean indicating whether to sell the position and the reason
    """
    logger = get_logger()
    try:
        # Calculate metrics
        profit_pct = ((current_price / entry_price) - 1) * 100
        days_held = (current_date - entry_date).days
        
        # Calculate stop-loss and take-profit thresholds
        stop_loss_threshold = entry_price * (1 - stop_loss_percent / 100)
        take_profit_threshold = entry_price * (1 + take_profit_percent / 100)
        
        # Calculate minimum expected profit
        min_expected_profit = days_held * expected_profit_per_day_percentage
        
        # Check conditions
        stop_loss_triggered = current_price <= stop_loss_threshold
        take_profit_triggered = current_price >= take_profit_threshold
        timeout_triggered = days_held >= position_timeout
        poor_performance = days_held > 2 and profit_pct < min_expected_profit
        
        # Log details if verbose
        if verbose:
            logger.info(f"Sell Analysis - Profit: {profit_pct:.2f}%, Days Held: {days_held}")
            logger.info(f"Stop Loss: {stop_loss_triggered} (Threshold: {stop_loss_threshold:.2f})")
            logger.info(f"Take Profit: {take_profit_triggered} (Threshold: {take_profit_threshold:.2f})")
            logger.info(f"Timeout: {timeout_triggered} (Max: {position_timeout} days)")
            logger.info(f"Performance: {poor_performance} (Min Expected: {min_expected_profit:.2f}%)")
        
        # Determine if any sell condition is met
        should_sell_flag = stop_loss_triggered or take_profit_triggered or timeout_triggered or poor_performance
        reason = None
        
        if should_sell_flag:
            if stop_loss_triggered:
                reason = "Stop Loss"
            elif take_profit_triggered:
                reason = "Take Profit"
            elif timeout_triggered:
                reason = "Position Timeout"
            elif poor_performance:
                reason = "Poor Performance"
            
            if verbose:
                logger.info(f"Sell signal triggered: {reason}")
        
        return should_sell_flag, reason
    
    except Exception as e:
        logger.error(f"Error in should_sell: {str(e)}")
        logger.debug(traceback.format_exc())
        return False, "Error"
    





    

def check_position_exit(symbol, current_price):
    """
    Check if a position should be exited based on rules
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to check
    current_price : float
        Current market price
        
    Returns:
    --------
    tuple: (should_exit, reason)
    """
    logger = get_logger()
    try:
        # Get position data
        positions_df = read_signals(status_filter="Active")
        position = positions_df[positions_df['Symbol'] == symbol]
        
        if position.empty:
            logger.warning(f"No active position found for {symbol}")
            return False, None
        
        position = position.iloc[0]
        
        # Check stop loss
        if not pd.isna(position['StopPrice']) and current_price <= position['StopPrice']:
            return True, "Stop Loss"
        
        # Check take profit
        if not pd.isna(position['TargetPrice']) and current_price >= position['TargetPrice']:
            return True, "Take Profit"
        
        # Check max hold time
        if not pd.isna(position['EntryDate']):
            days_held = (datetime.now().date() - position['EntryDate'].date()).days
            max_days = STRATEGY_PARAMS.get('position_timeout', 5)
            
            if days_held >= max_days:
                return True, "Max Hold Time"
        
        # Check poor performance
        if not pd.isna(position['EntryPrice']):
            profit_pct = (current_price / position['EntryPrice'] - 1) * 100
            days_held = (datetime.now().date() - position['EntryDate'].date()).days
            
            if days_held > 2:
                min_expected_daily = STRATEGY_PARAMS.get('expected_profit_per_day_percentage', 0.25)
                min_expected_profit = days_held * min_expected_daily
                
                if profit_pct < min_expected_profit:
                    return True, "Poor Performance"
        
        # No exit signals triggered
        return False, None
        
    except Exception as e:
        logger.error(f"Error checking position exit for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False, "Error"

#=================================================#
# Sizer Class
#=================================================#


class PositionSizer__lot__size:

    """
    Production-ready position sizer that maintains cash buffer and equal allocation per position.
    
    Key Logic:
    1. Always keep cash_buffer_pct of account value in cash as buffer
    2. Divide remaining % equally among max_positions (default 4)
    3. Buy as many whole shares as possible with that allocation
    4. Never violate the cash buffer rule
    """
    
    def __init__(self, cash_buffer_pct=10.0, max_positions=4, debug=False):
        """
        Initialize the position sizer.
        
        Args:
            cash_buffer_pct: Percentage of account to keep in cash (default 10%)
            max_positions: Maximum number of positions to hold (default 4)
            debug: Enable detailed debug logging (default False for production)
        """
        self.cash_buffer_pct = cash_buffer_pct
        self.max_positions = max_positions
        self.debug = debug
        
        # Set up logging if available (graceful fallback if logger not available)
        try:
            import logging
            self.logger = logging.getLogger(__name__)
        except:
            self.logger = None
            
    def _log(self, message, level="INFO"):
        """Production logging - only logs if logger available and debug enabled."""
        if self.debug and self.logger:
            if level == "ERROR":
                self.logger.error(message)
            elif level == "WARN":
                self.logger.warning(message)
            elif level == "SUCCESS":
                self.logger.info(message)
            else:
                self.logger.info(message)
        elif self.debug:
            # Fallback to print if no logger available but debug is on
            print(f"[{level}] {message}")
    
    def calculate_position_size(self, account_value, current_cash, price, 
                              current_positions=0, symbol=""):
        """
        Calculate position size based on simple allocation rules.
        
        Args:
            account_value: Total account value including positions
            current_cash: Available cash in account  
            price: Stock price per share
            current_positions: Number of existing positions (optional)
            symbol: Stock symbol for logging (optional)
            
        Returns:
            int: Number of shares to buy (0 if cannot buy)
        """
        
        # Production logging - minimal unless debug enabled
        if self.debug:
            self._log(f"=== POSITION SIZER DEBUG for {symbol} ===")
            self._log(f"Inputs: Account=${account_value:.2f}, Cash=${current_cash:.2f}, Price=${price:.4f}")
        
        # === ROBUST INPUT VALIDATION ===
        if account_value <= 0:
            self._log(f"{symbol} - Invalid account value: ${account_value:.2f}", "ERROR")
            return 0
            
        if current_cash < 0:
            self._log(f"{symbol} - Invalid cash value: ${current_cash:.2f}", "ERROR")
            return 0
            
        if price <= 0:
            self._log(f"{symbol} - Invalid price: ${price:.4f}", "ERROR")
            return 0
            
        # Check for NaN values (robust handling)
        try:
            import numpy as np
            if np.isnan(account_value) or np.isnan(current_cash) or np.isnan(price):
                self._log(f"{symbol} - NaN detected in inputs", "ERROR")
                return 0
        except ImportError:
            # If numpy not available, do basic checks
            if not isinstance(account_value, (int, float)) or not isinstance(current_cash, (int, float)) or not isinstance(price, (int, float)):
                self._log(f"{symbol} - Invalid data types in inputs", "ERROR")
                return 0
        
        # Logical validation - warn but continue if cash > account (might be normal in some cases)
        if current_cash > account_value * 1.1:  # Allow 10% tolerance
            self._log(f"{symbol} - Warning: Cash (${current_cash:.2f}) significantly > Account (${account_value:.2f})", "WARN")
        
        # === CASH BUFFER CALCULATION ===
        required_cash_buffer = account_value * (self.cash_buffer_pct / 100.0)
        
        if self.debug:
            self._log(f"Required cash buffer: ${required_cash_buffer:.2f} ({self.cash_buffer_pct}%)")
        
        # === AVAILABLE CASH ABOVE BUFFER ===
        available_cash_above_buffer = current_cash - required_cash_buffer
        
        if self.debug:
            self._log(f"Available cash above buffer: ${available_cash_above_buffer:.2f}")
        
        if available_cash_above_buffer <= 0:
            if self.debug:
                self._log(f"{symbol} - Insufficient cash above buffer (need ${required_cash_buffer:.2f}, have ${current_cash:.2f})", "WARN")
            else:
                # Always log insufficient cash in production
                self._log(f"{symbol} - Insufficient cash: need ${required_cash_buffer:.2f} buffer, have ${current_cash:.2f} total")
            return 0
        
        # === WORKING CAPITAL AND ALLOCATION ===
        working_capital = account_value * (1.0 - self.cash_buffer_pct / 100.0)
        allocation_per_position = working_capital / self.max_positions
        
        if self.debug:
            self._log(f"Working capital: ${working_capital:.2f}")
            self._log(f"Allocation per position: ${allocation_per_position:.2f}")
        
        # === SHARE CALCULATIONS ===
        # Calculate maximum shares by allocation constraint
        max_shares_by_allocation = int(allocation_per_position / price)
        
        # Calculate maximum shares by available cash constraint
        max_shares_by_cash = int(available_cash_above_buffer / price)
        
        # Take the more restrictive constraint
        proposed_shares = min(max_shares_by_allocation, max_shares_by_cash)
        
        if self.debug:
            self._log(f"Max by allocation: {max_shares_by_allocation}, Max by cash: {max_shares_by_cash}")
            self._log(f"Proposed shares: {proposed_shares}")
        
        # === FINAL VALIDATION ===
        # Double-check that we won't violate the cash buffer
        proposed_cost = proposed_shares * price
        remaining_cash_after_purchase = current_cash - proposed_cost
        
        if remaining_cash_after_purchase < required_cash_buffer:
            # This shouldn't happen with our logic, but be extra safe
            max_safe_cost = current_cash - required_cash_buffer
            adjusted_shares = int(max_safe_cost / price)
            
            self._log(f"{symbol} - Buffer safety adjustment: {proposed_shares} → {adjusted_shares} shares", "WARN")
            proposed_shares = adjusted_shares
            proposed_cost = proposed_shares * price
        
        # Ensure non-negative result
        final_shares = max(0, proposed_shares)
        final_cost = final_shares * price
        
        # === PRODUCTION LOGGING ===
        if final_shares > 0:
            if self.debug:
                allocation_used_pct = (final_cost / allocation_per_position) * 100
                self._log(f"✅ {symbol}: {final_shares} shares @ ${price:.2f} = ${final_cost:.2f} ({allocation_used_pct:.1f}% of allocation)")
            else:
                # Minimal production logging
                self._log(f"{symbol}: Calculated {final_shares} shares (${final_cost:.2f})")
        else:
            if self.debug:
                self._log(f"❌ {symbol}: No shares purchased - insufficient funds or allocation")
            else:
                # Always log zero results in production for troubleshooting
                self._log(f"{symbol}: No shares calculated (price too high or insufficient funds)")
        
        return final_shares
    
    def get_allocation_per_position(self, account_value):
        """Get the dollar allocation per position for planning purposes."""
        working_capital = account_value * (1.0 - self.cash_buffer_pct / 100.0)
        return working_capital / self.max_positions
    
    def get_cash_buffer_requirement(self, account_value):
        """Get the required cash buffer amount."""
        return account_value * (self.cash_buffer_pct / 100.0)
    
    def can_afford_stock(self, account_value, current_cash, price):
        """Check if we can afford at least 1 share of a stock."""
        required_buffer = self.get_cash_buffer_requirement(account_value)
        available_cash = current_cash - required_buffer
        return available_cash >= price
    
    def enable_debug(self):
        """Enable debug logging for troubleshooting."""
        self.debug = True
        
    def disable_debug(self):
        """Disable debug logging for production."""
        self.debug = False
        
    def get_position_summary(self, account_value, current_cash):
        """Get a summary of position sizing parameters for diagnostics."""
        buffer_required = self.get_cash_buffer_requirement(account_value)
        available_cash = current_cash - buffer_required
        allocation_per_position = self.get_allocation_per_position(account_value)
        
        return {
            'account_value': account_value,
            'current_cash': current_cash,
            'buffer_required': buffer_required,
            'available_cash': available_cash,
            'allocation_per_position': allocation_per_position,
            'max_positions': self.max_positions,
            'cash_buffer_pct': self.cash_buffer_pct
        }









class PositionSizer:
    """
    Production-ready position sizer that maintains cash buffer and equal allocation per position.
    Now includes smart lot sizing to avoid odd lots and improve execution.
    
    Key Logic:
    1. Always keep cash_buffer_pct of account value in cash as buffer
    2. Divide remaining % equally among max_positions (default 4)
    3. Apply lot sizing: 10% tolerance to round down, 5% to round up
    4. Buy as many whole shares as possible with that allocation
    5. Buffer can be slightly reduced for better lot sizing
    """
    
    def __init__(self, cash_buffer_pct=10.0, max_positions=4, 
                 enable_lot_sizing=True, round_down_tolerance=0.10, 
                 round_up_tolerance=0.05, debug=False):
        """
        Initialize the position sizer.
        
        Args:
            cash_buffer_pct: Percentage of account to keep in cash (default 10%)
            max_positions: Maximum number of positions to hold (default 4)
            enable_lot_sizing: Whether to apply lot sizing rules (default True)
            round_down_tolerance: Tolerance for rounding down to lots (default 10%)
            round_up_tolerance: Tolerance for rounding up to lots (default 5%)
            debug: Enable detailed debug logging (default False for production)
        """
        self.cash_buffer_pct = cash_buffer_pct
        self.max_positions = max_positions
        self.enable_lot_sizing = enable_lot_sizing
        self.round_down_tolerance = round_down_tolerance
        self.round_up_tolerance = round_up_tolerance
        self.debug = debug
        
        # Set up logging if available (graceful fallback if logger not available)
        try:
            import logging
            self.logger = logging.getLogger(__name__)
        except:
            self.logger = None
            
    def _log(self, message, level="INFO"):
        """Production logging - only logs if logger available and debug enabled."""
        if self.debug and self.logger:
            if level == "ERROR":
                self.logger.error(message)
            elif level == "WARN":
                self.logger.warning(message)
            elif level == "SUCCESS":
                self.logger.info(message)
            else:
                self.logger.info(message)
        elif self.debug:
            # Fallback to print if no logger available but debug is on
            print(f"[{level}] {message}")
    
    def apply_lot_sizing(self, shares, symbol=""):
        """
        Apply lot sizing rules to avoid odd lots and mixed lots.
        
        Rules:
        - 10% tolerance to round DOWN to nearest lot
        - 5% tolerance to round UP to nearest lot
        - Warn about odd lots (1-99 shares)
        
        Args:
            shares: Original calculated share count
            symbol: Stock symbol for logging
            
        Returns:
            int: Adjusted share count
        """
        if not self.enable_lot_sizing:
            return shares
            
        LOT_SIZE = 100
        
        # Handle odd lots (less than 100 shares)
        if shares < LOT_SIZE:
            if shares >= LOT_SIZE * (1 - self.round_up_tolerance):  # 95+ shares
                adjusted_shares = LOT_SIZE
                if self.debug:
                    self._log(f"{symbol} - Rounding up from odd lot: {shares} -> {adjusted_shares}")
            else:
                # Keep as odd lot but warn
                adjusted_shares = shares
                if shares > 0 and self.debug:
                    self._log(f"{symbol} - WARNING: Odd lot order ({shares} shares) - may have execution issues", "WARN")
            return adjusted_shares
        
        # Handle round lots and mixed lots
        lower_lot = (shares // LOT_SIZE) * LOT_SIZE
        upper_lot = lower_lot + LOT_SIZE
        
        # Check round-up first (tighter 5% tolerance)
        if shares >= upper_lot * (1 - self.round_up_tolerance):
            adjusted_shares = upper_lot
            if self.debug:
                self._log(f"{symbol} - Rounding up to lot: {shares} -> {adjusted_shares}")
        # Then check round-down (looser 10% tolerance)
        elif lower_lot > 0 and shares <= lower_lot * (1 + self.round_down_tolerance):
            adjusted_shares = lower_lot
            if self.debug and adjusted_shares != shares:
                self._log(f"{symbol} - Rounding down to lot: {shares} -> {adjusted_shares}")
        else:
            # Mixed lot - warn about odd portion
            adjusted_shares = shares
            odd_portion = shares % LOT_SIZE
            if self.debug:
                self._log(f"{symbol} - Mixed lot: {shares} shares ({odd_portion} odd shares)", "WARN")
                
        return adjusted_shares
    
    def calculate_position_size(self, account_value, current_cash, price, 
                              current_positions=0, symbol=""):
        """
        Calculate position size based on simple allocation rules with smart lot sizing.
        
        Args:
            account_value: Total account value including positions
            current_cash: Available cash in account  
            price: Stock price per share
            current_positions: Number of existing positions (optional)
            symbol: Stock symbol for logging (optional)
            
        Returns:
            int: Number of shares to buy (0 if cannot buy)
        """
        
        # Production logging - minimal unless debug enabled
        if self.debug:
            self._log(f"=== POSITION SIZER DEBUG for {symbol} ===")
            self._log(f"Inputs: Account=${account_value:.2f}, Cash=${current_cash:.2f}, Price=${price:.4f}")
        
        # === ROBUST INPUT VALIDATION ===
        if account_value <= 0:
            self._log(f"{symbol} - Invalid account value: ${account_value:.2f}", "ERROR")
            return 0
            
        if current_cash < 0:
            self._log(f"{symbol} - Invalid cash value: ${current_cash:.2f}", "ERROR")
            return 0
            
        if price <= 0:
            self._log(f"{symbol} - Invalid price: ${price:.4f}", "ERROR")
            return 0
            
        # Check for NaN values (robust handling)
        try:
            import numpy as np
            if np.isnan(account_value) or np.isnan(current_cash) or np.isnan(price):
                self._log(f"{symbol} - NaN detected in inputs", "ERROR")
                return 0
        except ImportError:
            # If numpy not available, do basic checks
            if not isinstance(account_value, (int, float)) or not isinstance(current_cash, (int, float)) or not isinstance(price, (int, float)):
                self._log(f"{symbol} - Invalid data types in inputs", "ERROR")
                return 0
        
        # Logical validation - warn but continue if cash > account (might be normal in some cases)
        if current_cash > account_value * 1.1:  # Allow 10% tolerance
            self._log(f"{symbol} - Warning: Cash (${current_cash:.2f}) significantly > Account (${account_value:.2f})", "WARN")
        
        # === CASH BUFFER CALCULATION ===
        required_cash_buffer = account_value * (self.cash_buffer_pct / 100.0)
        
        if self.debug:
            self._log(f"Required cash buffer: ${required_cash_buffer:.2f} ({self.cash_buffer_pct}%)")
        
        # === AVAILABLE CASH ABOVE BUFFER ===
        available_cash_above_buffer = current_cash - required_cash_buffer
        
        if self.debug:
            self._log(f"Available cash above buffer: ${available_cash_above_buffer:.2f}")
        
        if available_cash_above_buffer <= 0:
            if self.debug:
                self._log(f"{symbol} - Insufficient cash above buffer (need ${required_cash_buffer:.2f}, have ${current_cash:.2f})", "WARN")
            else:
                # Always log insufficient cash in production
                self._log(f"{symbol} - Insufficient cash: need ${required_cash_buffer:.2f} buffer, have ${current_cash:.2f} total")
            return 0
        
        # === WORKING CAPITAL AND ALLOCATION ===
        working_capital = account_value * (1.0 - self.cash_buffer_pct / 100.0)
        allocation_per_position = working_capital / self.max_positions
        
        if self.debug:
            self._log(f"Working capital: ${working_capital:.2f}")
            self._log(f"Allocation per position: ${allocation_per_position:.2f}")
        
        # === SHARE CALCULATIONS ===
        # Calculate maximum shares by allocation constraint
        max_shares_by_allocation = int(allocation_per_position / price)
        
        # Calculate maximum shares by available cash constraint
        max_shares_by_cash = int(available_cash_above_buffer / price)
        
        # Take the more restrictive constraint
        proposed_shares = min(max_shares_by_allocation, max_shares_by_cash)
        
        if self.debug:
            self._log(f"Max by allocation: {max_shares_by_allocation}, Max by cash: {max_shares_by_cash}")
            self._log(f"Proposed shares (before lot sizing): {proposed_shares}")
        
        # === APPLY LOT SIZING ===
        proposed_shares = self.apply_lot_sizing(proposed_shares, symbol)
        
        if self.debug:
            self._log(f"Proposed shares (after lot sizing): {proposed_shares}")
        
        # === RELAXED BUFFER VALIDATION ===
        # We're more flexible with buffer since positions are only 1/4 of total
        proposed_cost = proposed_shares * price
        remaining_cash_after_purchase = current_cash - proposed_cost
        
        # Only reject if we're eating into more than 20% of the buffer
        # (since we have 4 positions and 10% buffer, this gives us cushion)
        min_acceptable_buffer = required_cash_buffer * 0.8
        
        if remaining_cash_after_purchase < min_acceptable_buffer:
            # Try to find a better lot size that preserves more buffer
            max_safe_cost = current_cash - min_acceptable_buffer
            max_safe_shares = int(max_safe_cost / price)
            
            # Round down to nearest lot if possible
            if max_safe_shares >= 100:
                adjusted_shares = (max_safe_shares // 100) * 100
            else:
                adjusted_shares = max_safe_shares
                
            if self.debug:
                self._log(f"{symbol} - Buffer protection: {proposed_shares} -> {adjusted_shares} shares", "WARN")
            proposed_shares = adjusted_shares
            proposed_cost = proposed_shares * price
        
        # Ensure non-negative result
        final_shares = max(0, proposed_shares)
        final_cost = final_shares * price
        
        # === PRODUCTION LOGGING ===
        if final_shares > 0:
            if self.debug:
                allocation_used_pct = (final_cost / allocation_per_position) * 100
                buffer_remaining = current_cash - final_cost
                buffer_pct = (buffer_remaining / account_value) * 100
                self._log(f"SUCCESS: {symbol}: {final_shares} shares @ ${price:.2f} = ${final_cost:.2f}")
                self._log(f"         Allocation used: {allocation_used_pct:.1f}%, Buffer remaining: ${buffer_remaining:.2f} ({buffer_pct:.1f}%)")
            else:
                # Minimal production logging
                self._log(f"{symbol}: Calculated {final_shares} shares (${final_cost:.2f})")
        else:
            if self.debug:
                self._log(f"SKIP: {symbol}: No shares purchased - insufficient funds or allocation")
            else:
                # Always log zero results in production for troubleshooting
                self._log(f"{symbol}: No shares calculated (price too high or insufficient funds)")
        
        return final_shares
    
    def get_allocation_per_position(self, account_value):
        """Get the dollar allocation per position for planning purposes."""
        working_capital = account_value * (1.0 - self.cash_buffer_pct / 100.0)
        return working_capital / self.max_positions
    
    def get_cash_buffer_requirement(self, account_value):
        """Get the required cash buffer amount."""
        return account_value * (self.cash_buffer_pct / 100.0)
    
    def can_afford_stock(self, account_value, current_cash, price):
        """Check if we can afford at least 1 share of a stock."""
        required_buffer = self.get_cash_buffer_requirement(account_value)
        available_cash = current_cash - required_buffer
        return available_cash >= price
    
    def enable_debug(self):
        """Enable debug logging for troubleshooting."""
        self.debug = True
        
    def disable_debug(self):
        """Disable debug logging for production."""
        self.debug = False
        
    def get_position_summary(self, account_value, current_cash):
        """Get a summary of position sizing parameters for diagnostics."""
        buffer_required = self.get_cash_buffer_requirement(account_value)
        available_cash = current_cash - buffer_required
        allocation_per_position = self.get_allocation_per_position(account_value)
        
        return {
            'account_value': account_value,
            'current_cash': current_cash,
            'buffer_required': buffer_required,
            'available_cash': available_cash,
            'allocation_per_position': allocation_per_position,
            'max_positions': self.max_positions,
            'cash_buffer_pct': self.cash_buffer_pct
        }






#=================================================#
# Basic Comisssions
#=================================================#

class FixedCommissionScheme(bt.CommInfoBase):
    """Fixed commission scheme for IB."""
    params = (
        ('commission', 3.0),  # Fixed commission per trade
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission  # Return fixed commission

#=================================================#
# CLI Interface
#=================================================#

def cache_terminal(line_count=100, label=None, cache_dir='Data/logging/cache'):
    """
    Cache terminal output to a file using clipboard
    
    Args:
        line_count: Number of lines to capture (default 100)
        label: Optional label to identify this cache entry
        cache_dir: Directory to save cache files
        
    Returns:
        Path to the cached file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_str = f"_{label}" if label else ""
    filename = f"terminal_cache_{timestamp}{label_str}.log"
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Full path to the cache file
    cache_file = cache_path / filename
    
    # Header for the cache file
    header = f"""
==============================================================
= TERMINAL OUTPUT CACHE - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
= Captured {line_count} lines{f' - {label}' if label else ''}
= System: {platform.system()} {platform.release()}
= Python: {sys.version.split()[0]}
==============================================================

"""
    
    # Try to use clipboard content (requires manual copy)
    try:
        # Check if pyperclip or pywin32 is available
        clipboard_content = None
        try:
            import pyperclip
            clipboard_content = pyperclip.paste()
        except ImportError:
            try:
                # Use win32clipboard if available; ensure pywin32 is installed: pip install pywin32
                import win32clipboard

                win32clipboard.OpenClipboard()
                if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_TEXT):
                    clipboard_content = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT).decode('utf-8')
                win32clipboard.CloseClipboard()
            except ImportError:
                clipboard_content = None
        
        if clipboard_content and len(clipboard_content) > 10:
            # Split by lines and take the last 'line_count' lines
            lines = clipboard_content.split('\n')
            if len(lines) > line_count:
                lines = lines[-line_count:]
            terminal_output = '\n'.join(lines)
            
            # Write to the cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(terminal_output)
            
            print(f"Cached {len(lines)} lines of terminal output from clipboard to {cache_file}")
            print("Note: For best results, select and copy (Ctrl+A, Ctrl+C) terminal content before running this command")
            return str(cache_file)
    except Exception as e:
        print(f"Error using clipboard: {str(e)}")
    
    # Fallback: just capture command history
    terminal_output = ""
    try:
        # Get command history
        history_cmd = f"Get-History -Count {line_count} | Format-Table Id, CommandLine -AutoSize | Out-String"
        history_result = subprocess.run(
            ["powershell", "-Command", history_cmd],
            capture_output=True, text=True, encoding='utf-8'
        )
        terminal_output = "=== COMMAND HISTORY ===\n\n" + history_result.stdout
        
        # Add system info
        sys_cmd = """
        Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First 5 | 
        Format-Table -Property Name, CPU, WorkingSet -AutoSize | Out-String
        """
        sys_result = subprocess.run(
            ["powershell", "-Command", sys_cmd],
            capture_output=True, text=True, encoding='utf-8'
        )
        terminal_output += "\n=== SYSTEM INFORMATION ===\n\n" + sys_result.stdout
        
        # Add message to instruct on better capture
        terminal_output += "\n=== NOTE ===\n"
        terminal_output += "For better terminal capture:\n"
        terminal_output += "1. Install pyperclip package: pip install pyperclip\n"
        terminal_output += "2. Select all terminal text (Ctrl+A)\n"
        terminal_output += "3. Copy to clipboard (Ctrl+C)\n"
        terminal_output += "4. Run this command again"
        
    except Exception as e:
        terminal_output = f"Error capturing command history: {str(e)}"
    
    # Write to the cache file
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(terminal_output)
    
    print(f"Cached terminal output to {cache_file}")
    print("Note: For best results, install pyperclip (pip install pyperclip)")
    print("      Then select and copy (Ctrl+A, Ctrl+C) terminal content before running this command")
    
    return str(cache_file)

def show_cache_entries(count=5, cache_dir='Data/logging/cache'):
    """
    Show the most recent terminal cache entries
    
    Args:
        count: Number of recent entries to show
        cache_dir: Directory with cache files
        
    Returns:
        List of recent cache files
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist.")
        return []
    
    # Get all cache files and sort by modification time (newest first)
    cache_files = list(cache_path.glob("terminal_cache_*.log"))
    cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Show the most recent entries
    print(f"\nRecent terminal cache entries ({min(count, len(cache_files))} of {len(cache_files)}):")
    for i, file in enumerate(cache_files[:count]):
        size_kb = file.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        # Try to extract the label if present
        label = file.stem.split('_', 3)[-1] if len(file.stem.split('_')) > 3 else ""
        label_str = f" - {label}" if label else ""
        print(f"{i+1}. {file.name}{label_str} ({size_kb:.1f} KB, {mod_time})")
    
    return cache_files[:count]

def read_cache(cache_file=None, index=None, cache_dir='Data/logging/cache'):
    """
    Read and display a cached terminal output file
    
    Args:
        cache_file: Path to the cache file to read
        index: Index of the recent cache file to read (1-based)
        cache_dir: Directory with cache files
        
    Returns:
        Content of the cache file
    """
    if cache_file is None and index is None:
        # If no arguments, show recent files and prompt for index
        files = show_cache_entries(cache_dir=cache_dir)
        if not files:
            return "No cache files found."
        
        try:
            index = int(input("\nEnter number to read (or press Enter to cancel): "))
            if index < 1 or index > len(files):
                return "Invalid index."
            cache_file = files[index-1]
        except ValueError:
            return "Operation cancelled."
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif index is not None:
        # Get file by index
        files = show_cache_entries(count=index, cache_dir=cache_dir)
        if not files or index > len(files):
            return f"No cache file found at index {index}."
        cache_file = files[index-1]
    
    # Read the cache file
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n===== Content of {Path(cache_file).name} =====")
        print(content)
        print("="*40)
        return content
    except Exception as e:
        return f"Error reading cache file: {str(e)}"

def check_market_status():
    """Check and display current market status"""
    is_open, status = is_market_open()
    
    # Get the style based on status
    if is_open:
        status_style = "\033[38;2;0;200;0m"  # Green for open
    elif "not yet open" in status.lower():
        status_style = "\033[38;2;220;220;0m"  # Yellow for not yet open
    else:
        status_style = "\033[38;2;150;150;150m"  # Gray for closed
        
    print(f"Market Status: {status_style}{status}\033[0m")
    
    # Get additional date information
    today = datetime.now().date()
    try:
        last_trading_date = get_last_trading_date()
        next_trading_date = get_next_trading_day(today)
        
        print(f"Last Trading Day: \033[38;2;100;149;237m{last_trading_date}\033[0m")
        print(f"Next Trading Day: \033[38;2;100;149;237m{next_trading_date}\033[0m")
    except Exception as e:
        print(f"Error retrieving trading dates: {str(e)}")
    
    return is_open

def check_market_time():
    """Check and display time remaining in the current market session or until next session"""
    nyse = mcal.get_calendar('NYSE')
    now = pd.Timestamp.now(tz='America/New_York')
    today_date = now.date()
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today_date, end_date=today_date)
    
    if schedule.empty:
        # Today is not a trading day, find the next trading day
        future_schedule = nyse.schedule(start_date=today_date + timedelta(days=1), 
                                        end_date=today_date + timedelta(days=10))
        if future_schedule.empty:
            print("\033[38;2;220;0;0mError: Unable to find next trading day\033[0m")
            return
            
        next_market_day = future_schedule.index[0].date()
        next_market_open = future_schedule.iloc[0]['market_open'].tz_convert('America/New_York')
        
        # Calculate time until market opens
        time_until_open = next_market_open - now
        days = time_until_open.days
        hours = time_until_open.seconds // 3600
        minutes = (time_until_open.seconds % 3600) // 60
        
        print(f"\033[38;2;150;150;150mMarket is currently closed\033[0m")
        print(f"Next market session: \033[38;2;100;149;237m{next_market_day.strftime('%A, %B %d')}\033[0m")
        print(f"Time until market opens: \033[38;2;100;149;237m{days} days, {hours} hours, {minutes} minutes\033[0m")
        
    else:
        # Today is a trading day
        market_open = schedule.iloc[0]['market_open'].tz_convert('America/New_York')
        market_close = schedule.iloc[0]['market_close'].tz_convert('America/New_York')
        
        if now < market_open:
            # Market not yet open
            time_until_open = market_open - now
            hours = time_until_open.seconds // 3600
            minutes = (time_until_open.seconds % 3600) // 60
            
            print(f"\033[38;2;220;220;0mMarket will open today at {market_open.strftime('%H:%M:%S')}\033[0m")
            print(f"Time until market opens: \033[38;2;220;220;0m{hours} hours, {minutes} minutes\033[0m")
            
        elif now > market_close:
            # Market already closed
            future_schedule = nyse.schedule(start_date=today_date + timedelta(days=1), 
                                          end_date=today_date + timedelta(days=10))
            if future_schedule.empty:
                print("\033[38;2;220;0;0mError: Unable to find next trading day\033[0m")
                return
                
            next_market_day = future_schedule.index[0].date()
            next_market_open = future_schedule.iloc[0]['market_open'].tz_convert('America/New_York')
            
            # Calculate time until market opens
            time_until_open = next_market_open - now
            days = time_until_open.days
            hours = time_until_open.seconds // 3600
            minutes = (time_until_open.seconds % 3600) // 60
            
            print(f"\033[38;2;150;150;150mMarket closed at {market_close.strftime('%H:%M:%S')}\033[0m")
            print(f"Next market session: \033[38;2;100;149;237m{next_market_day.strftime('%A, %B %d')}\033[0m")
            print(f"Time until market opens: \033[38;2;100;149;237m{days} days, {hours} hours, {minutes} minutes\033[0m")
            
        else:
            # Market is open
            time_remaining = market_close - now
            hours = time_remaining.seconds // 3600
            minutes = (time_remaining.seconds % 3600) // 60
            
            print(f"\033[38;2;0;200;0mMarket is open\033[0m")
            print(f"Market closes at: \033[38;2;0;200;0m{market_close.strftime('%H:%M:%S')}\033[0m")
            print(f"Time remaining in session: \033[38;2;0;200;0m{hours} hours, {minutes} minutes\033[0m")






def display_signals(status=None, count=10):
    """
    Display current signals with optional status filtering
    
    Parameters:
    -----------
    status : str or list, optional
        Filter by status ("Pending", "Active", "Completed")
    count : int
        Maximum number of signals to display
    """
    df = read_signals(status_filter=status)
    
    if df.empty:
        print(f"\033[38;2;220;220;0mNo signals found{' with status ' + status if status else ''}\033[0m")
        return
    
    status_display = status if status else "All"
    print(f"\n\033[1mSignals ({status_display}) - Showing {min(count, len(df))} of {len(df)}:\033[0m")
    
    # Sort by appropriate column based on status
    if status == "Pending":
        df = df.sort_values(['TargetDate', 'UpProbability'], ascending=[True, False])
    elif status == "Active":
        df = df.sort_values(['EntryDate'], ascending=[False])
    elif status == "Completed":
        df = df.sort_values(['ExitDate'], ascending=[False])
    else:
        df = df.sort_values(['LastUpdate'], ascending=[False])
    
    # Select columns based on status
    if status == "Pending":
        cols = ['Symbol', 'TargetDate', 'SignalPrice', 'UpProbability']
        df_display = df[cols].head(count)
        
        # Format display
        df_display = df_display.copy()
        df_display['TargetDate'] = df_display['TargetDate'].dt.strftime('%Y-%m-%d')
        df_display['SignalPrice'] = df_display['SignalPrice'].map('${:.2f}'.format)
        df_display['UpProbability'] = df_display['UpProbability'].map('{:.1%}'.format)
        
        # Print with headers
        headers = ["Symbol", "Target Date", "Signal Price", "Probability"]
        print(f"{headers[0]:<8} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10}")
        print("-" * 45)
        
        for _, row in df_display.iterrows():
            print(f"{row['Symbol']:<8} {row['TargetDate']:<12} {row['SignalPrice']:<12} {row['UpProbability']:<10}")
            
    elif status == "Active":
        cols = ['Symbol', 'EntryDate', 'EntryPrice', 'CurrentPrice', 'PositionSize', 'PnLPct']
        df_display = df[cols].head(count)
        
        # Format display
        df_display = df_display.copy()
        df_display['EntryDate'] = df_display['EntryDate'].dt.strftime('%Y-%m-%d')
        df_display['EntryPrice'] = df_display['EntryPrice'].map('${:.2f}'.format)
        df_display['CurrentPrice'] = df_display['CurrentPrice'].map('${:.2f}'.format)
        
        # Calculate and color-code PnL
        df_display['PnL_Display'] = df_display['PnLPct'].apply(
            lambda x: f"\033[38;2;0;200;0m{x:.2f}%\033[0m" if pd.notna(x) and x >= 0 
            else f"\033[38;2;220;0;0m{x:.2f}%\033[0m" if pd.notna(x)
            else "N/A"
        )
        
        # Print with headers
        headers = ["Symbol", "Entry Date", "Entry Price", "Current Price", "Size", "P&L"]
        print(f"{headers[0]:<8} {headers[1]:<12} {headers[2]:<12} {headers[3]:<14} {headers[4]:<10} {headers[5]:<8}")
        print("-" * 70)
        
        for _, row in df_display.iterrows():
            print(f"{row['Symbol']:<8} {row['EntryDate']:<12} {row['EntryPrice']:<12} {row['CurrentPrice']:<14} " +
                  f"{row['PositionSize']:<10.0f} {row['PnL_Display']}")
            
    elif status == "Completed":
        cols = ['Symbol', 'EntryDate', 'ExitDate', 'EntryPrice', 'ExitPrice', 'PnLPct', 'ExitReason']
        df_display = df[cols].head(count)
        
        # Format display
        df_display = df_display.copy()
        df_display['EntryDate'] = df_display['EntryDate'].dt.strftime('%Y-%m-%d')
        df_display['ExitDate'] = df_display['ExitDate'].dt.strftime('%Y-%m-%d')
        df_display['EntryPrice'] = df_display['EntryPrice'].map('${:.2f}'.format)
        df_display['ExitPrice'] = df_display['ExitPrice'].map('${:.2f}'.format)
        
        # Calculate and color-code PnL
        df_display['PnL_Display'] = df_display['PnLPct'].apply(
            lambda x: f"\033[38;2;0;200;0m{x:.2f}%\033[0m" if pd.notna(x) and x >= 0 
            else f"\033[38;2;220;0;0m{x:.2f}%\033[0m" if pd.notna(x)
            else "N/A"
        )
        
        # Print with headers
        headers = ["Symbol", "Entry Date", "Exit Date", "Entry", "Exit", "P&L", "Reason"]
        print(f"{headers[0]:<8} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10} {headers[4]:<10} {headers[5]:<8} {headers[6]:<15}")
        print("-" * 80)
        
        for _, row in df_display.iterrows():
            print(f"{row['Symbol']:<8} {row['EntryDate']:<12} {row['ExitDate']:<12} {row['EntryPrice']:<10} " +
                  f"{row['ExitPrice']:<10} {row['PnL_Display']:<15} {row['ExitReason']}")
    else:
        # Display mixed status
        cols = ['Symbol', 'Status', 'SignalDate', 'EntryPrice', 'ExitPrice', 'PnLPct']
        df_display = df[cols].head(count)
        
        # Format display
        df_display = df_display.copy()
        df_display['SignalDate'] = df_display['SignalDate'].dt.strftime('%Y-%m-%d')
        df_display['EntryPrice'] = df_display['EntryPrice'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        df_display['ExitPrice'] = df_display['ExitPrice'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        
        # Calculate and color-code status
        df_display['Status_Display'] = df_display['Status'].apply(
            lambda x: f"\033[38;2;220;220;0m{x}\033[0m" if x == "Pending"
            else f"\033[38;2;0;200;0m{x}\033[0m" if x == "Active"
            else f"\033[38;2;150;150;150m{x}\033[0m" if x == "Completed"
            else x
        )
        
        # Calculate and color-code PnL
        df_display['PnL_Display'] = df_display['PnLPct'].apply(
            lambda x: f"\033[38;2;0;200;0m{x:.2f}%\033[0m" if pd.notna(x) and x >= 0 
            else f"\033[38;2;220;0;0m{x:.2f}%\033[0m" if pd.notna(x)
            else "N/A"
        )
        
        # Print with headers
        headers = ["Symbol", "Status", "Date", "Entry", "Exit", "P&L"]
        print(f"{headers[0]:<8} {headers[1]:<10} {headers[2]:<12} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
        print("-" * 65)
        
        for _, row in df_display.iterrows():
            print(f"{row['Symbol']:<8} {row['Status_Display']:<19} {row['SignalDate']:<12} {row['EntryPrice']:<10} " +
                  f"{row['ExitPrice']:<10} {row['PnL_Display']}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    status_counts = df['Status'].value_counts()
    
    for status_type in ["Pending", "Active", "Completed"]:
        count = status_counts.get(status_type, 0)
        if status_type == "Pending":
            color = "\033[38;2;220;220;0m"  # Yellow
        elif status_type == "Active":
            color = "\033[38;2;0;200;0m"    # Green
        else:
            color = "\033[38;2;150;150;150m"  # Gray
        
        print(f"  {color}{status_type}: {count}\033[0m")
    
    if "Completed" in status_counts:
        completed = df[df['Status'] == "Completed"]
        if not completed.empty and 'PnLPct' in completed.columns:
            winners = len(completed[completed['PnLPct'] > 0])
            losers = len(completed[completed['PnLPct'] <= 0])
            win_rate = (winners / len(completed)) * 100 if len(completed) > 0 else 0
            
            print(f"  Win Rate: {winners}/{len(completed)} ({win_rate:.1f}%)")



def read_trading_data(is_live=False):
    """
    Read trading data from legacy file format
    
    Parameters:
    -----------
    is_live : bool
        Whether to read from live trading data file or backtest file
    
    Returns:
    --------
    DataFrame: Trading data
    """
    logger = get_logger()
    try:
        # Determine file path based on mode
        # Non-live per-ticker state ledger RETIRED: routed to a retired file so it
        # never overwrites _Buy_Signals.parquet (now the broker's narrowed book).
        file_path = 'Data/Production/LiveTradingData/pending_signals.parquet' if is_live else 'Data/_retired_trading_ledger.parquet'
        
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.debug(f"Read {len(df)} trading data records from {file_path}")
            return df
        else:
            logger.info(f"Trading data file {file_path} not found. Creating empty DataFrame.")
            # Create empty DataFrame with basic schema
            return pd.DataFrame(columns=['Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 
                                         'IsCurrentlyBought', 'ConsecutiveLosses', 
                                         'LastTradedDate', 'UpProbability', 'PositionSize'])
    except Exception as e:
        logger.error(f"Error reading trading data: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame(columns=['Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 
                                     'IsCurrentlyBought', 'ConsecutiveLosses', 
                                     'LastTradedDate', 'UpProbability', 'PositionSize'])

def write_trading_data(df, is_live=False):
    """
    Write trading data to legacy file format
    
    Parameters:
    -----------
    df : DataFrame
        Trading data to write
    is_live : bool
        Whether to write to live trading data file or backtest file
    """
    logger = get_logger()
    try:
        # Determine file path based on mode
        # Non-live per-ticker state ledger RETIRED: routed to a retired file so it
        # never overwrites _Buy_Signals.parquet (now the broker's narrowed book).
        file_path = 'Data/Production/LiveTradingData/pending_signals.parquet' if is_live else 'Data/_retired_trading_ledger.parquet'
        
        # Ensure required columns exist
        required_cols = ['Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 
                         'IsCurrentlyBought', 'ConsecutiveLosses', 
                         'LastTradedDate', 'UpProbability', 'PositionSize']
        
        for col in required_cols:
            if col not in df.columns:
                if 'Date' in col:
                    df[col] = pd.Series(dtype='datetime64[ns]')
                elif col in ['LastBuySignalPrice', 'PositionSize', 'UpProbability']:
                    df[col] = pd.Series(dtype='float64')
                elif col == 'ConsecutiveLosses':
                    df[col] = pd.Series(dtype='int64')
                elif col == 'IsCurrentlyBought':
                    df[col] = pd.Series(dtype='bool')
                else:
                    df[col] = pd.Series(dtype='string')
        
        # Ensure directory exists
        ensure_dir(file_path)
        
        # Write to file
        df.to_parquet(file_path, index=False)
        #logger.info(f"Successfully wrote {len(df)} trading data records to {file_path}")
        
    except Exception as e:
        logger.error(f"Error writing trading data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def sync_trading_data():
    """
    Synchronize trading data between backtester signals and live trading.
    This copies new signals from the backtester to the live trading system.
    """
    logger = get_logger()
    try:
        # Check for consolidated signals file
        if os.path.exists(SIGNALS_FILE):
            logger.info("Found consolidated signals file, using new format")
            # Get pending signals from consolidated file
            signals_df = read_signals(status_filter="Pending")
            
            # Get current live trading data
            live_df = read_trading_data(is_live=True)
            
            # For each signal in signals_df that's not in live_df, add it
            for _, signal in signals_df.iterrows():
                symbol = signal['Symbol']
                
                # Skip if already in live data
                if symbol in live_df['Symbol'].values:
                    continue
                
                # Create new row for live data
                new_row = {
                    'Symbol': symbol,
                    'LastBuySignalDate': signal.get('SignalDate', pd.NaT),
                    'LastBuySignalPrice': signal.get('SignalPrice', 0),
                    'IsCurrentlyBought': False,
                    'ConsecutiveLosses': 0,
                    'UpProbability': signal.get('UpProbability', 0),
                    'TargetDate': signal.get('TargetDate', pd.NaT),
                    'LastTradedDate': pd.NaT
                }
                
                # Add to live data
                live_df = pd.concat([live_df, pd.DataFrame([new_row])], ignore_index=True)
                logger.info(f"Added new signal for {symbol} to live trading data")
            
            # Write back to live file
            write_trading_data(live_df, is_live=True)
            logger.info("Successfully synchronized signals to live trading")
            
        else:
            # Legacy sync
            logger.info("Using legacy sync method")
            # Read backtest signals
            backtest_path = 'Data/Production/BacktestData/signals.parquet'
            
            if not os.path.exists(backtest_path):
                logger.warning(f"Backtest signals file {backtest_path} not found")
                return
                
            backtest_df = pd.read_parquet(backtest_path)
            
            # Read current live data
            live_df = read_trading_data(is_live=True)
            
            # Synchronize signals
            for _, signal in backtest_df.iterrows():
                symbol = signal['Symbol']
                
                # Skip if already in live data
                if symbol in live_df['Symbol'].values:
                    continue
                
                # Add to live data
                live_df = pd.concat([live_df, pd.DataFrame([signal])], ignore_index=True)
                logger.info(f"Added legacy signal for {symbol} to live trading data")
            
            # Write back to live file
            write_trading_data(live_df, is_live=True)
            logger.info("Successfully synchronized legacy signals to live trading")
            
    except Exception as e:
        logger.error(f"Error synchronizing trading data: {str(e)}")
        logger.debug(traceback.format_exc())




def update_trade_data(symbol, action, price, date=None, probability=0, is_live=False):
    """
    Update trading data for a specific symbol
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to update
    action : str
        'buy' or 'sell'
    price : float
        Transaction price
    date : datetime, optional
        Transaction date (default: now)
    probability : float, optional
        Probability for buy signals
    is_live : bool
        Whether to update live trading data or backtest data
    """
    logger = get_logger()
    
    if date is None:
        date = datetime.now()
        
    try:
        # Read current data
        df = read_trading_data(is_live=is_live)
        
        # Check if symbol exists
        if symbol not in df['Symbol'].values:
            logger.warning(f"Symbol {symbol} not found in trading data")
            
            # If buy action, add new row
            if action.lower() == 'buy':
                new_row = {
                    'Symbol': symbol,
                    'LastBuySignalDate': pd.Timestamp(date),
                    'LastBuySignalPrice': price,
                    'IsCurrentlyBought': True,
                    'ConsecutiveLosses': 0,
                    'LastTradedDate': pd.Timestamp(date),
                    'UpProbability': probability,
                    'PositionSize': 0  # To be set separately
                }
                
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                logger.info(f"Added new symbol {symbol} to trading data with buy signal")
            else:
                logger.warning(f"Cannot update {symbol} with {action} - symbol not found")
                return
        else:
            # Update existing symbol
            if action.lower() == 'buy':
                df.loc[df['Symbol'] == symbol, 'LastBuySignalDate'] = pd.Timestamp(date)
                df.loc[df['Symbol'] == symbol, 'LastBuySignalPrice'] = price
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
                df.loc[df['Symbol'] == symbol, 'UpProbability'] = probability
                logger.info(f"Updated {symbol} with buy signal at ${price:.2f}")
            elif action.lower() == 'sell':
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(date)
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                logger.info(f"Updated {symbol} with sell signal at ${price:.2f}")
        
        # Write updated DataFrame
        write_trading_data(df, is_live=is_live)
        
    except Exception as e:
        logger.error(f"Error updating trade data for {symbol}: {str(e)}")
        logger.debug(traceback.format_exc())


def display_trades(count=10):
    """
    Display completed trades
    
    Parameters:
    -----------
    count : int
        Maximum number of trades to display
    """
    df = read_completed_trades()
    
    if df.empty:
        print("\033[38;2;220;220;0mNo completed trades found\033[0m")
        return
    
    print(f"\n\033[1mCompleted Trades - Showing {min(count, len(df))} of {len(df)}:\033[0m")
    
    # Sort by exit date (most recent first)
    df = df.sort_values('ExitDate', ascending=False)
    
    # Select relevant columns
    cols = ['Symbol', 'EntryDate', 'ExitDate', 'EntryPrice', 'ExitPrice', 'PnL', 'PnLPct', 'DaysHeld', 'ExitReason', 'Source']
    df_display = df[cols].head(count)
    
    # Format display
    df_display = df_display.copy()
    df_display['EntryDate'] = df_display['EntryDate'].dt.strftime('%Y-%m-%d')
    df_display['ExitDate'] = df_display['ExitDate'].dt.strftime('%Y-%m-%d')
    df_display['EntryPrice'] = df_display['EntryPrice'].map('${:.2f}'.format)
    df_display['ExitPrice'] = df_display['ExitPrice'].map('${:.2f}'.format)
    
    # Calculate and color-code PnL
    df_display['PnL_Display'] = df_display.apply(
        lambda row: f"\033[38;2;0;200;0m${row['PnL']:.2f} ({row['PnLPct']:.2f}%)\033[0m" if pd.notna(row['PnL']) and row['PnL'] >= 0 
        else f"\033[38;2;220;0;0m${row['PnL']:.2f} ({row['PnLPct']:.2f}%)\033[0m" if pd.notna(row['PnL'])
        else "N/A",
        axis=1
    )
    
    # Color-code source
    df_display['Source_Display'] = df_display['Source'].apply(
        lambda x: f"\033[38;2;0;200;0m{x}\033[0m" if x == "Live"
        else f"\033[38;2;220;220;0m{x}\033[0m" if x == "Paper"
        else f"\033[38;2;150;150;150m{x}\033[0m" if x == "Backtest"
        else x
    )
    
    # Print with headers
    headers = ["Symbol", "Entry Date", "Exit Date", "Entry", "Exit", "P&L", "Days", "Reason", "Source"]
    print(f"{headers[0]:<8} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10} {headers[4]:<10} " +
          f"{headers[5]:<25} {headers[6]:<6} {headers[7]:<15} {headers[8]:<10}")
    print("-" * 110)
    
    for _, row in df_display.iterrows():
        print(f"{row['Symbol']:<8} {row['EntryDate']:<12} {row['ExitDate']:<12} {row['EntryPrice']:<10} " +
              f"{row['ExitPrice']:<10} {row['PnL_Display']:<30} {row['DaysHeld']:<6.0f} {row['ExitReason']:<15} {row['Source_Display']}")
    
    # Print summary statistics
    print("\nTrade Performance Summary:")
    
    # Calculate statistics
    total_trades = len(df)
    winners = len(df[df['PnL'] > 0])
    losers = len(df[df['PnL'] <= 0])
    win_rate = (winners / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = df[df['PnL'] > 0]['PnLPct'].mean() if winners > 0 else 0
    avg_loss = df[df['PnL'] <= 0]['PnLPct'].mean() if losers > 0 else 0
    
    profit_factor = abs(df[df['PnL'] > 0]['PnL'].sum() / df[df['PnL'] < 0]['PnL'].sum()) if df[df['PnL'] < 0]['PnL'].sum() != 0 else float('inf')
    
    avg_days_held = df['DaysHeld'].mean()
    
    # Print statistics
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {winners}/{total_trades} ({win_rate:.1f}%)")
    print(f"  Average Win: {avg_win:.2f}%")
    print(f"  Average Loss: {avg_loss:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Average Days Held: {avg_days_held:.1f}")

def display_metrics(count=5):
    """
    Display backtest metrics
    
    Parameters:
    -----------
    count : int
        Maximum number of backtest runs to display
    """
    df = read_backtest_metrics()
    
    if df.empty:
        print("\033[38;2;220;220;0mNo backtest metrics found\033[0m")
        return
    
    print(f"\n\033[1mBacktest Metrics - Showing {min(count, len(df))} of {len(df)}:\033[0m")
    
    # Sort by timestamp (most recent first)
    df = df.sort_values('timestamp', ascending=False)
    
    # Get the most important metrics
    key_metrics = ['run_id', 'timestamp', 'roi', 'roi_annualized', 'sharpe_ratio', 
                   'max_drawdown', 'total_trades', 'win_rate']
    
    # Keep only columns that exist in the DataFrame
    existing_metrics = [col for col in key_metrics if col in df.columns]
    
    # If important metrics aren't available, take whatever is there
    if len(existing_metrics) < 3:
        existing_metrics = list(df.columns)[:min(8, len(df.columns))]
    
    df_display = df[existing_metrics].head(count)
    
    # Format display
    df_display = df_display.copy()
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Format percentages
    for col in ['roi', 'roi_annualized', 'max_drawdown', 'win_rate']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Print the DataFrame
    print(df_display.to_string(index=False))
    
    # If parameters are stored, show the parameters for the most recent run
    param_cols = [col for col in df.columns if col.startswith('param_')]
    if param_cols and not df.empty:
        most_recent = df.iloc[0]
        most_recent_id = most_recent.get('run_id', 'Unknown')
        
        print(f"\nParameters for most recent run ({most_recent_id}):")
        for col in param_cols:
            param_name = col[6:]  # Remove 'param_' prefix
            param_value = most_recent[col]
            print(f"  {param_name}: {param_value}")


def pstock(ticker, data_folder="Data/PriceData", log_scale=True, width=900, height=700, 
           days=None, warn_stale_days=3, show_gaps=True):
    """
    Robust stock chart plotter with data quality checks
    
    Args:
        ticker: Stock ticker (string or variable, quotes optional)
        data_folder: Path to parquet files
        log_scale: Use log scale for price (default True)
        width, height: Chart dimensions
        days: Number of recent days to show (None = all data)
        warn_stale_days: Warn if data is older than this many days
        show_gaps: Show warnings for data gaps
    """
    
    # Convert ticker to string and clean
    ticker_str = str(ticker).upper().strip()
    
    def load_and_validate_data(ticker, data_folder):
        file_path = os.path.join(data_folder, f"{ticker}.parquet")
        
        # Check if file exists
        if not os.path.exists(file_path):
            available_files = [f.replace('.parquet', '') for f in os.listdir(data_folder) 
                             if f.endswith('.parquet')]
            print(f"❌ No data found for {ticker}")
            print(f"💡 Available tickers: {', '.join(available_files[:10])}{'...' if len(available_files) > 10 else ''}")
            return None
        
        try:
            data = pd.read_parquet(file_path)
        except Exception as e:
            print(f"❌ Error reading {ticker}.parquet: {e}")
            return None
        
        # Standardize index
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        data.index = pd.to_datetime(data.index)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"❌ Missing required columns for {ticker}: {missing_cols}")
            return None
        
        # Add Volume if missing
        if 'Volume' not in data.columns:
            print(f"⚠️  No volume data for {ticker}")
            data['Volume'] = 0
        
        # Clean data
        original_len = len(data)
        data = data.dropna(subset=required_cols)
        for col in required_cols + ['Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)
        
        if len(data) < original_len:
            print(f"⚠️  Cleaned {original_len - len(data)} rows with missing/invalid data")
        
        return data
    
    def check_data_quality(data, ticker):
        """Check for gaps and staleness"""
        if data is None or len(data) == 0:
            return
        
        # Sort by date
        data = data.sort_index()
        
        # Check data freshness
        latest_date = data.index[-1]
        days_old = (datetime.now() - latest_date).days
        
        if days_old > warn_stale_days:
            print(f"🕒 Data is {days_old} days old (last: {latest_date.strftime('%Y-%m-%d')})")
        
        # Check for gaps (missing trading days)
        if show_gaps and len(data) > 1:
            date_diff = data.index.to_series().diff().dt.days
            gaps = date_diff[date_diff > 3]  # More than 3 days gap (weekend + holiday)
            
            if len(gaps) > 0:
                large_gaps = gaps[gaps > 7]  # Week+ gaps
                if len(large_gaps) > 0:
                    print(f"⚠️  Found {len(large_gaps)} data gaps > 7 days")
        
        # Basic data validation
        if len(data) > 0:
            # Check for impossible prices (negative, or extreme ratios)
            bad_data = (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
            if bad_data.sum() > 0:
                print(f"⚠️  Found {bad_data.sum()} rows with invalid prices")
            
            # Check for data integrity (High >= Low, etc.)
            integrity_issues = (data['High'] < data['Low']).sum()
            if integrity_issues > 0:
                print(f"⚠️  Found {integrity_issues} rows where High < Low")
    
    # Load and validate data
    data = load_and_validate_data(ticker_str, data_folder)
    if data is None:
        return None
    
    # Check data quality
    check_data_quality(data, ticker_str)
    
    # Filter by days if specified
    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        data = data[data.index >= cutoff_date]
        if len(data) == 0:
            print(f"❌ No data found for {ticker_str} in last {days} days")
            return None
    
    # Create chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'],
                    name=ticker_str), row=1, col=1)
    
    # Volume bars (handle zero volume)
    if data['Volume'].sum() > 0:
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                             name='Volume', showlegend=False), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=data.index, y=[0]*len(data), 
                                name='No Volume Data', showlegend=False), row=2, col=1)
    
    # Update layout
    days_suffix = f" ({days}d)" if days else ""
    fig.update_layout(
        title=f'{ticker_str} Stock Chart{days_suffix}',
        xaxis_rangeslider_visible=False,
        width=width,
        height=height,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    # Set scale
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    
    # Labels
    fig.update_yaxes(title_text="Price (Log)" if log_scale else "Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    #fig.show()
    
    # Print summary stats
    if len(data) > 0:
        latest = data.iloc[-1]
        first = data.iloc[0]
        pct_change = ((latest['Close'] - first['Close']) / first['Close']) * 100
        
        print(f"📊 {ticker_str} | Latest: ${latest['Close']:.2f} | "
              f"Period Return: {pct_change:+.1f}% | "
              f"Volume: {latest['Volume']:,.0f} | "
              f"Data Points: {len(data):,}")
    
    return fig




def colorize_output(value, label, good_threshold, bad_threshold, lower_is_better=False, reverse=False, unicorn_multiplier=20.0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{label:<30}\033[38;2;150;150;150mN/A        \033[0m[\033[38;2;150;150;150mNo Data\033[0m]"
    
    def get_color_code(normalized_value, is_unicorn=False):
        if is_unicorn:
            return "\033[38;2;100;149;237m"  # Cornflower blue
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
        r = int(colors[index][0] * (1-t) + colors[index+1][0] * t)
        g = int(colors[index][1] * (1-t) + colors[index+1][1] * t)
        b = int(colors[index][2] * (1-t) + colors[index+1][2] * t)
        return f"\033[38;2;{r};{g};{b}m"
    
    def get_quality_tag(normalized_value, is_unicorn=False):
        if is_unicorn:
            return "Unicorn"
            
        if normalized_value <= 0.15:
            return "Excellent"
        elif normalized_value <= 0.3:
            return "Very Good"
        elif normalized_value <= 0.45:
            return "Good"
        elif normalized_value <= 0.6:
            return "Average"
        elif normalized_value <= 0.75:
            return "Below Average"
        elif normalized_value <= 0.9:
            return "Poor"
        else:
            return "Unacceptable"

    is_unicorn = False
    if not lower_is_better and value >= good_threshold * unicorn_multiplier:
        is_unicorn = True
    elif lower_is_better and value <= good_threshold / unicorn_multiplier:
        is_unicorn = True

    if reverse:
        good_threshold, bad_threshold = bad_threshold, good_threshold

    try:
        if is_unicorn:
            normalized_value = 0  # Best value
        elif lower_is_better:
            if value <= good_threshold:
                normalized_value = 0  # Best value (Green)
            elif value >= bad_threshold:
                normalized_value = 1  # Worst value (Red)
            else:
                normalized_value = (value - good_threshold) / (bad_threshold - good_threshold)
        else:
            if value >= good_threshold:
                normalized_value = 0  # Best value (Green)
            elif value <= bad_threshold:
                normalized_value = 1  # Worst value (Red)
            else:
                normalized_value = (good_threshold - value) / (good_threshold - bad_threshold)
    except Exception as e:
        return f"{label:<30}\033[38;2;150;150;150mError      \033[0m[\033[38;2;150;150;150mCalculation Error\033[0m]"

    color_code = get_color_code(normalized_value, is_unicorn)
    quality_tag = get_quality_tag(normalized_value, is_unicorn)
    
    if isinstance(value, float):
        value_str = f"{value:.2f}"
    else:
        value_str = str(value)
    
    return f"{label:<30}{color_code}{value_str:<10}\033[0m[{color_code}{quality_tag}\033[0m]"






def clear_completed_trades():
    """Clear the completed trades file to start fresh with a new backtest."""
    logger = get_logger()

    try:
        trades_df = pd.read_parquet(COMPLETED_TRADES_FILE)
    except Exception as e:
        logger.error(f"Error reading completed trades file: {str(e)}")
        logger.debug(traceback.format_exc())
        return False
    
    trades_df = trades_df.iloc[0:0]

    trades_df.to_parquet(COMPLETED_TRADES_FILE, index=False)











def create_ib_connection(host='127.0.0.1', port=7497, max_attempts=3, timeout=20.0, debug=True):
    """
    Create a robust IB connection with proper disconnection handling.
    """
    dprint = globals().get('dprint', print)
    dprint("Starting IB connection process")
    
    for attempt in range(1, max_attempts + 1):
        # Generate a unique client ID for each attempt
        client_id = random.randint(10000, 99999)
        dprint(f"Connection attempt {attempt}/{max_attempts} with clientId={client_id}")
        
        try:
            # First try a direct connection to verify IB is responsive
            test_ib = ibi.IB()
            test_ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout/2)
            
            if test_ib.isConnected():
                dprint(f"Test connection successful with clientId={client_id}")
                server_time = test_ib.reqCurrentTime()
                dprint(f"Server time: {server_time}")
                
                # Disconnect test connection
                test_ib.disconnect()
                dprint("Test connection disconnected")
                
                # Now create the real connection via IBStore
                dprint(f"Creating IBStore with clientId={client_id}")
                # Use backtrader's built-in IBStore
                
                store = IBStore(
                    host=host,
                    port=port,
                    clientId=client_id,
                    reconnect=3,
                    timeout=timeout,
                    notifyall=True,
                )
                
                # Get IB instance from store
                ib = store.ib
                
                # Verify the store connection is working
                if ib.isConnected():
                    dprint(f"IBStore connection successful")
                    return store, ib
                else:
                    dprint("IBStore connection failed")
                    if store:
                        try:
                            store.stop()
                        except:
                            pass
            else:
                dprint("Test connection failed - IB may not be running or accessible")
                
        except Exception as e:
            dprint(f"Connection error: {type(e).__name__}: {e}")
            
            # Clean up any resources
            if 'test_ib' in locals() and test_ib.isConnected():
                try:
                    test_ib.disconnect()
                except:
                    pass
                    
            if 'store' in locals() and store:
                try:
                    store.stop()
                except:
                    pass
                
        # Wait before next attempt
        if attempt < max_attempts:
            dprint(f"Waiting before next connection attempt...")
            Secondary_time_import_carefully_use_with_caution.sleep(2)
    
    dprint(f"Failed to connect after {max_attempts} attempts")
    return None, None








def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Consolidated trading system utilities")
    
    # Main commands (subparsers for different functions)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Cache terminal output command
    cache_parser = subparsers.add_parser("cache", help="Cache terminal output")
    cache_parser.add_argument("lines", type=int, nargs='?', default=100, 
                             help="Number of lines to capture")
    cache_parser.add_argument("-l", "--label", help="Label for this cache")
    
    # Show recent cache entries command
    show_parser = subparsers.add_parser("show", help="Show recent cache entries")
    show_parser.add_argument("count", type=int, nargs='?', default=5, 
                            help="Number of entries to show")
    
    # Read cache entry command
    read_parser = subparsers.add_parser("read", help="Read a cache entry")
    read_parser.add_argument("index", type=int, nargs="?", help="Index of the cache entry to read")
    
    # Market info command
    subparsers.add_parser("market", help="Show current market status")
    
    # Market time command
    subparsers.add_parser("markettime", help="Show time remaining in market session or until next session")
    
    # Signals command
    signals_parser = subparsers.add_parser("signals", help="Show signal information")
    signals_parser.add_argument("--status", choices=["Pending", "Active", "Completed"],
                              help="Filter by signal status")
    signals_parser.add_argument("--count", type=int, default=10,
                               help="Number of signals to show")
    
    # Trades command
    trades_parser = subparsers.add_parser("trades", help="Show completed trades")
    trades_parser.add_argument("--count", type=int, default=10,
                              help="Number of trades to show")
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Show backtest metrics")
    metrics_parser.add_argument("--count", type=int, default=5,
                               help="Number of metrics to show")
    
    # Migration command
    subparsers.add_parser("migrate", help="Migrate from legacy data files")
    
    # Commands for managing signals
    add_signal_parser = subparsers.add_parser("add-signal", help="Add a new pending signal")
    add_signal_parser.add_argument("symbol", help="Stock symbol")
    add_signal_parser.add_argument("price", type=float, help="Signal price")
    add_signal_parser.add_argument("--probability", type=float, default=0.7, help="Up probability")
    add_signal_parser.add_argument("--target-date", help="Target date (YYYY-MM-DD), defaults to next trading day")
    add_signal_parser.add_argument("--atr", type=float, help="Average True Range for volatility estimation")

    # Activate signal command
    activate_parser = subparsers.add_parser("activate", help="Mark a signal as active (position opened)")
    activate_parser.add_argument("symbol", help="Stock symbol")
    activate_parser.add_argument("price", type=float, help="Entry price")
    activate_parser.add_argument("size", type=float, help="Position size (number of shares)")
    activate_parser.add_argument("--stop", type=float, help="Stop loss price")
    activate_parser.add_argument("--target", type=float, help="Target price")
    
    # Complete signal command
    complete_parser = subparsers.add_parser("complete", help="Mark a signal as completed (position closed)")
    complete_parser.add_argument("symbol", help="Stock symbol")
    complete_parser.add_argument("price", type=float, help="Exit price")
    complete_parser.add_argument("--reason", help="Exit reason", default="Manual")
    
    # Update signal command
    update_parser = subparsers.add_parser("update", help="Update an existing signal")
    update_parser.add_argument("symbol", help="Stock symbol")
    update_parser.add_argument("--status", choices=["Pending", "Active", "Completed"], help="New status")
    update_parser.add_argument("--current-price", type=float, help="Current price")
    update_parser.add_argument("--stop", type=float, help="Stop loss price")
    update_parser.add_argument("--target", type=float, help="Target price")
    
    return parser.parse_args()

def main():
    """Main command line interface function"""
    args = parse_args()
    logger = get_logger()
    
    # If no command provided, show help
    if not args.command:
        parse_args.__globals__['parser'].print_help()
        return
    
    # Handle different commands
    if args.command == "cache":
        cache_terminal(args.lines, args.label)
    
    elif args.command == "show":
        show_cache_entries(args.count)
    
    elif args.command == "read":
        read_cache(index=args.index)
    
    elif args.command == "market":
        check_market_status()
        
    elif args.command == "markettime":
        check_market_time()
    
    elif args.command == "signals":
        display_signals(status=args.status, count=args.count)
    
    elif args.command == "trades":
        display_trades(count=args.count)
    
    elif args.command == "metrics":
        display_metrics(count=args.count)
    
    
    elif args.command == "add-signal":
        # Determine target date
        target_date = None
        if args.target_date:
            target_date = datetime.strptime(args.target_date, "%Y-%m-%d").date()
        else:
            # Default to next trading day
            target_date = get_next_trading_day(datetime.now().date())
        
        # Add signal
        success = add_signal(
            symbol=args.symbol,
            signal_price=args.price,
            target_date=target_date,
            up_probability=args.probability,
            atr=args.atr
        )
        
        if success:
            print(f"\033[38;2;0;200;0mAdded signal for {args.symbol} at ${args.price:.2f} for {target_date}\033[0m")
        else:
            print(f"\033[38;2;220;0;0mFailed to add signal for {args.symbol}\033[0m")
    
    elif args.command == "activate":
        # Activate signal (mark as active)
        success = mark_signal_as_active(
            symbol=args.symbol,
            position_size=args.size,
            entry_price=args.price,
            stop_price=args.stop,
            target_price=args.target
        )
        
        if success:
            print(f"\033[38;2;0;200;0mActivated position for {args.symbol} at ${args.price:.2f}\033[0m")
        else:
            print(f"\033[38;2;220;0;0mFailed to activate position for {args.symbol}\033[0m")
    
    elif args.command == "complete":
        # Complete signal (mark as completed)
        success = mark_signal_as_completed(
            symbol=args.symbol,
            exit_price=args.price,
            exit_reason=args.reason
        )
        
        if success:
            print(f"\033[38;2;0;200;0mCompleted position for {args.symbol} at ${args.price:.2f}\033[0m")
        else:
            print(f"\033[38;2;220;0;0mFailed to complete position for {args.symbol}\033[0m")
    
    elif args.command == "update":
        # Update signal with provided fields
        update_fields = {}
        
        if args.current_price is not None:
            update_fields['CurrentPrice'] = args.current_price
        
        if args.stop is not None:
            update_fields['StopPrice'] = args.stop
            
        if args.target is not None:
            update_fields['TargetPrice'] = args.target
        
        success = update_signal_status(
            symbol=args.symbol,
            new_status=args.status if args.status else None,
            **update_fields
        )
        
        if success:
            print(f"\033[38;2;0;200;0mUpdated signal for {args.symbol}\033[0m")
        else:
            print(f"\033[38;2;220;0;0mFailed to update signal for {args.symbol}\033[0m")
    
    else:
        print(f"\033[38;2;220;0;0mUnknown command: {args.command}\033[0m")

#=================================================#
# Performance Analysis Functions
#=================================================#

def calculate_performance_metrics(start_date=None, end_date=None, source=None):
    """
    Calculate performance metrics from completed trades
    
    Parameters:
    -----------
    start_date : datetime.date, optional
        Start date for filtering trades
    end_date : datetime.date, optional
        End date for filtering trades
    source : str, optional
        Filter by source ("Live", "Paper", "Backtest")
        
    Returns:
    --------
    dict: Performance metrics
    """
    logger = get_logger()
    try:
        # Get completed trades
        trades_df = read_completed_trades()
        
        if trades_df.empty:
            logger.warning("No completed trades found for analysis")
            return {}
        
        # Apply filters
        if start_date:
            trades_df = trades_df[trades_df['ExitDate'].dt.date >= start_date]
            
        if end_date:
            trades_df = trades_df[trades_df['ExitDate'].dt.date <= end_date]
            
        if source:
            trades_df = trades_df[trades_df['Source'] == source]
        
        if trades_df.empty:
            logger.warning("No trades match the filter criteria")
            return {}
        
        # Sort by exit date
        trades_df = trades_df.sort_values('ExitDate')
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades) if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = winning_trades['PnL'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['PnL'].sum()) if not losing_trades.empty else 0
        
        net_profit = total_profit - total_loss
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Calculate average metrics
        avg_win = winning_trades['PnLPct'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['PnLPct'].mean() if not losing_trades.empty else 0
        
        avg_trade_pnl = trades_df['PnLPct'].mean()
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Calculate holding period metrics
        avg_days_held = trades_df['DaysHeld'].mean()
        max_days_held = trades_df['DaysHeld'].max()
        
        # Calculate drawdown metrics
        if 'AccountValue' in trades_df.columns and not trades_df['AccountValue'].isna().all():
            # Use account value if available
            equity_series = trades_df['AccountValue']
            
            # Calculate drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdown = abs(drawdown.min())
        else:
            # Estimate drawdown from trade P&L
            max_drawdown = 0  # Can't calculate without account value
        
        # Calculate consecutive wins/losses
        trades_df['IsWin'] = trades_df['PnL'] > 0
        
        # Initialize variables
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        current_is_win = None
        
        # Iterate through trades
        for is_win in trades_df['IsWin']:
            if current_is_win is None:
                current_is_win = is_win
            elif current_is_win == is_win:
                current_streak += 1
            else:
                # Streak ended
                if current_is_win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                
                current_streak = 1
                current_is_win = is_win
        
        # Check final streak
        if current_is_win is not None:
            if current_is_win:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Compile all metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate * 100,  # as percentage
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'avg_trade_pct': avg_trade_pnl,
            'expectancy': expectancy,
            'avg_days_held': avg_days_held,
            'max_days_held': max_days_held,
            'max_drawdown': max_drawdown,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'start_date': trades_df['EntryDate'].min(),
            'end_date': trades_df['ExitDate'].max(),
        }
        
        # Calculate SQN (System Quality Number)
        if total_trades >= 30:  # Need sufficient trades for meaningful SQN
            pnl_series = trades_df['PnLPct']
            sqn = (pnl_series.mean() / pnl_series.std()) * np.sqrt(total_trades) if pnl_series.std() > 0 else 0
            metrics['sqn'] = sqn
            
            # SQN interpretation
            if sqn < 1.6:
                metrics['sqn_quality'] = "Poor"
            elif sqn < 2.0:
                metrics['sqn_quality'] = "Below Average"
            elif sqn < 2.5:
                metrics['sqn_quality'] = "Average"
            elif sqn < 3.0:
                metrics['sqn_quality'] = "Good"
            elif sqn < 5.0:
                metrics['sqn_quality'] = "Excellent"
            elif sqn < 7.0:
                metrics['sqn_quality'] = "Superb"
            else:
                metrics['sqn_quality'] = "Holy Grail"
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}

def display_performance_metrics(metrics=None, start_date=None, end_date=None, source=None):
    """
    Display performance metrics
    
    Parameters:
    -----------
    metrics : dict, optional
        Pre-calculated metrics (if None, will calculate from trades)
    start_date : datetime.date, optional
        Start date for filtering trades
    end_date : datetime.date, optional
        End date for filtering trades
    source : str, optional
        Filter by source ("Live", "Paper", "Backtest")
    """
    if metrics is None:
        metrics = calculate_performance_metrics(start_date, end_date, source)
    
    if not metrics:
        print("\033[38;2;220;220;0mNo performance metrics available\033[0m")
        return
    
    # Get date range for header
    date_range = ""
    if 'start_date' in metrics and 'end_date' in metrics:
        start_str = metrics['start_date'].strftime('%Y-%m-%d') if metrics['start_date'] else "Unknown"
        end_str = metrics['end_date'].strftime('%Y-%m-%d') if metrics['end_date'] else "Unknown"
        date_range = f" ({start_str} to {end_str})"
    
    source_str = f" - {source}" if source else ""
    
    # Print header
    print(f"\n\033[1mPerformance Metrics{source_str}{date_range}:\033[0m")
    
    # Print trade metrics
    print("\nTrade Metrics:")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    win_rate = metrics.get('win_rate', 0)
    win_rate_color = "\033[38;2;0;200;0m" if win_rate >= 50 else "\033[38;2;220;220;0m"
    print(f"  Win Rate: {win_rate_color}{win_rate:.1f}%\033[0m ({metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)})")
    
    print(f"  Avg Win: \033[38;2;0;200;0m{metrics.get('avg_win_pct', 0):.2f}%\033[0m")
    print(f"  Avg Loss: \033[38;2;220;0;0m{metrics.get('avg_loss_pct', 0):.2f}%\033[0m")
    
    expectancy = metrics.get('expectancy', 0)
    expectancy_color = "\033[38;2;0;200;0m" if expectancy > 0 else "\033[38;2;220;0;0m"
    print(f"  Expectancy: {expectancy_color}{expectancy:.2f}%\033[0m")
    
    # Print profit metrics
    print("\nProfit Metrics:")
    net_profit = metrics.get('net_profit', 0)
    profit_color = "\033[38;2;0;200;0m" if net_profit >= 0 else "\033[38;2;220;0;0m"
    print(f"  Net Profit: {profit_color}${net_profit:.2f}\033[0m")
    
    profit_factor = metrics.get('profit_factor', 0)
    pf_color = "\033[38;2;0;200;0m" if profit_factor >= 1.5 else "\033[38;2;220;220;0m"
    print(f"  Profit Factor: {pf_color}{profit_factor:.2f}\033[0m")
    
    print(f"  Avg Trade P&L: {metrics.get('avg_trade_pct', 0):.2f}%")
    
    # Print time metrics
    print("\nTime Metrics:")
    print(f"  Avg Days Held: {metrics.get('avg_days_held', 0):.1f}")
    print(f"  Max Days Held: {metrics.get('max_days_held', 0):.1f}")
    
    # Print streak metrics
    print("\nStreak Metrics:")
    print(f"  Max Win Streak: \033[38;2;0;200;0m{metrics.get('max_win_streak', 0)}\033[0m")
    print(f"  Max Loss Streak: \033[38;2;220;0;0m{metrics.get('max_loss_streak', 0)}\033[0m")
    
    # Print risk metrics
    print("\nRisk Metrics:")
    max_dd = metrics.get('max_drawdown', 0)
    dd_color = "\033[38;2;0;200;0m" if max_dd < 10 else "\033[38;2;220;220;0m" if max_dd < 20 else "\033[38;2;220;0;0m"
    print(f"  Max Drawdown: {dd_color}{max_dd:.2f}%\033[0m")
    
    # Print SQN if available
    if 'sqn' in metrics:
        sqn = metrics.get('sqn', 0)
        sqn_quality = metrics.get('sqn_quality', 'Unknown')
        
        # Color based on quality
        if sqn_quality in ["Holy Grail", "Superb"]:
            sqn_color = "\033[38;2;0;200;0m"  # Bright Green
        elif sqn_quality in ["Excellent", "Good"]:
            sqn_color = "\033[38;2;180;255;180m"  # Light Green
        elif sqn_quality == "Average":
            sqn_color = "\033[38;2;220;220;0m"  # Yellow
        elif sqn_quality == "Below Average":
            sqn_color = "\033[38;2;255;180;0m"  # Orange
        else:
            sqn_color = "\033[38;2;220;0;0m"  # Red
        
        print(f"\nSystem Quality:")
        print(f"  SQN: {sqn_color}{sqn:.2f}\033[0m ({sqn_color}{sqn_quality}\033[0m)")

if __name__ == "__main__":
    main()