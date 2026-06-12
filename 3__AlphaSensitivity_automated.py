#!/root/root/miniconda4/envs/tf/bin/python
import os
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
from scipy.stats import linregress, entropy, gaussian_kde, skew, kurtosis
import logging
import argparse
import traceback
from pykalman import KalmanFilter
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, argrelextrema, detrend
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, jit
from tqdm import tqdm
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import requests
import random
from ts2vg import NaturalVG
import networkx as nx
import asyncio
from ib_insync import IB, Stock, util, Contract
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



CONFIG = {
    'input_directory': 'Data/PriceData',
    # AUTOMATED variant: writes to a SEPARATE folder so it never touches the live
    # Data/ProcessedData used by baseline retrains. Run this, then point
    # 4__Predictor.py --input_dir Data/ProcessedData_automated at it.
    'output_directory': 'Data/ProcessedData_automated',
    'index_temp_dir': 'Data/Indexes',
    'log_lines_to_read': 500,
    'core_count_division': True,
}

GLOBAL_INDEX_DATA = {}

INDEXES = {
    'SPY': {
        'name': 'S&P 500 ETF',
        'secType': 'STK',
        'exchange': 'SMART',
        'primaryExchange': 'ARCA'
    },
    'QQQ': {
        'name': 'Nasdaq 100 ETF',
        'secType': 'STK',
        'exchange': 'SMART',
        'primaryExchange': 'NASDAQ'
    },
    'IWM': {
        'name': 'Russell 2000 ETF',
        'secType': 'STK',
        'exchange': 'SMART',
        'primaryExchange': 'ARCA'
    },
    'DIA': {
        'name': 'Dow Jones ETF',
        'secType': 'STK',
        'exchange': 'SMART',
        'primaryExchange': 'ARCA'
    },
    'VIX': {
        'name': 'CBOE Volatility Index',
        'secType': 'IND',
        'exchange': 'CBOE',
        'primaryExchange': None
    },
}


def get_logger(script_name="Data/logging/3__Indicators"):
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(f"{script_name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

logger = get_logger()






# Data pre-processing utilities
def interpolate_columns(df, max_gap_fill=50, mark_interpolated=False):
    df_result = df.copy()
    
    if mark_interpolated:
        interpolation_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for column in df.columns:
        # Check if the column is a DataFrame or a Series with a non-numeric dtype
        try:
            col_data = df[column]
            if isinstance(col_data, pd.DataFrame) or not np.issubdtype(col_data.dtype, np.number):
                continue
                
            # Identify consecutive NaN stretches
            consec_nan_count = col_data.isna().astype(int).groupby(col_data.notna().astype(int).cumsum()).cumsum()
            interp_mask = (consec_nan_count <= max_gap_fill) & col_data.isna()
            
            if interp_mask.any():
                if column in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                    # Time-based interpolation for price data
                    temp_series = col_data.copy()
                    df_result.loc[interp_mask, column] = temp_series.interpolate(
                        method='time', limit=max_gap_fill, limit_direction='forward'
                    )[interp_mask]
                elif column == 'Volume':
                    # Linear interpolation for volume instead of nearest
                    temp_series = col_data.copy()
                    df_result.loc[interp_mask, column] = temp_series.interpolate(
                        method='linear', limit=max_gap_fill, limit_direction='forward'
                    )[interp_mask]
                else:
                    # Linear interpolation for other numeric columns
                    temp_series = col_data.copy()
                    df_result.loc[interp_mask, column] = temp_series.interpolate(
                        method='linear', limit=max_gap_fill, limit_direction='forward'
                    )[interp_mask]
                
                if mark_interpolated:
                    interpolation_mask.loc[interp_mask, column] = True
            
            # Handle any remaining NaNs with forward fill only
            remaining_nas = df_result[column].isna()
            if remaining_nas.any():
                df_result[column] = df_result[column].ffill(limit=max_gap_fill)
                # Removed bfill to prevent look-ahead bias
                
                if mark_interpolated:
                    newly_filled = remaining_nas & ~df_result[column].isna()
                    interpolation_mask.loc[newly_filled, column] = True
        except (AttributeError, TypeError):
            # Skip columns that don't support these operations
            continue
    
    if mark_interpolated:
        for column in df.columns:
            try:
                if isinstance(df[column], pd.Series) and np.issubdtype(df[column].dtype, np.number):
                    df_result[f'{column}_interpolated'] = interpolation_mask[column]
            except (AttributeError, TypeError):
                continue
    
    return df_result











def squash_col_outliers(df, col_name=None, num_std_dev=3):
    if col_name:
        columns_to_process = [col_name]
    else:
        columns_to_process = df.select_dtypes(include=['float64']).columns

    for col in columns_to_process:
        if col not in df.columns or df[col].dtype != 'float64':
            continue
        rolled_means = df[col][df[col] != 0].rolling(window=282, min_periods=1).mean()
        rolled_stds = df[col][df[col] != 0].rolling(window=282, min_periods=1).std()
        lower_bounds = rolled_means - num_std_dev * rolled_stds
        upper_bounds = rolled_means + num_std_dev * rolled_stds
        df[col] = df[col].clip(lower=lower_bounds, upper=upper_bounds)
    return df

def clean_and_interpolate_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
    df.ffill(inplace=True)
    return df

def safe_divide(a, b, fill_value=0):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, fill_value=fill_value)
    elif isinstance(a, pd.Series):
        b = pd.Series(b, index=a.index)
    elif isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, pd.Series):
            result = result.where((b != 0) & (b.notna()), fill_value)
        else:
            result = np.where((b != 0) & (~np.isnan(b)), result, fill_value)
    return result

def safe_log(x, epsilon=1e-14):
    return np.log(np.maximum(x, epsilon))

# Basic indicator calculations
@njit
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_cov = np.sum((x - x_mean) * (y - y_mean))
    xx_cov = np.sum((x - x_mean) ** 2)
    slope = xy_cov / xx_cov
    intercept = y_mean - slope * x_mean
    return slope, intercept

def find_best_fit_line(x, y):
    try:
        slope, intercept, _, _, _ = linregress(x, y)
        return slope, intercept
    except ValueError:
        return np.nan, np.nan

def hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def rolling_hurst_exponent(series, window_size):
    series_clean = series.dropna()
    def hurst_window(window):
        return hurst_exponent(window)
    return series_clean.rolling(window=window_size).apply(hurst_window, raw=True)


# ── GP-discovered cross-sectional alpha features ──────────────────────────────
# compute_gp_input_features() runs inside indicators() (per-ticker, time-series).
# add_gp_cross_sectional_features() runs after the parallel loop (cross-ticker).

from scipy.stats import rankdata as _gp_rankdata

GP_REQUIRED_COLS = [
    'Return', 'High_Low', 'High_Close', 'Low_Close',
    'time_reversal_asymmetry', 'time_between_extremes', 'volume_entropy',
    'information_rate_of_change',
    'emd_features_IMF_1_Energy', 'emd_features_IMF_1_Std',
    'emd_features_IMF_2_Energy', 'emd_features_IMF_2_Std',
    'entropy', 'noise_WhiteNoise', 'noise_PinkNoise',
    'higuchi_fractal_dimension', 'lyapunov_exponent', 'dfa_alpha',
    'spectral_entropy', 'autocorrelation', 'hurst_exponent',
    'higher_order_moments_Skewness', 'higher_order_moments_Kurtosis',
]

def _gp_cs_rank(arr):
    r = _gp_rankdata(arr, method='average')
    return r / (len(arr) + 1) * 2.0 - 1.0

def _gp_conditional(x, y):
    return np.where(x > 0, y, -y)

def _gp_safe_div(x, y):
    safe_y = np.where(np.abs(y) > 1e-10, y, 1.0)
    return np.where(np.abs(y) > 1e-10, x / safe_y, 0.0)

def _gp_safe_sqrt(x):
    return np.sqrt(np.abs(x))

def _gp_apply_cs_rank(signal, groups):
    out = np.zeros_like(signal, dtype=np.float64)
    for idx in groups:
        s = signal[idx]
        mask = np.isfinite(s)
        if mask.sum() >= 2:
            tmp = np.zeros(len(s))
            tmp[mask] = _gp_cs_rank(s[mask])
            out[idx] = tmp
    return out

def _gp_apply_cs_zscore(signal, groups):
    out = np.zeros_like(signal, dtype=np.float64)
    for idx in groups:
        s = signal[idx]
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        out[idx] = (s - mu) / (sigma + 1e-10)
    return out


def compute_gp_input_features(df, window=60):
    """
    Compute per-ticker rolling inputs (GP_REQUIRED_COLS) for the cross-sectional
    GP feature pass. Called inside indicators() on each single-ticker DataFrame.
    All new columns are batched into a dict and joined once via pd.concat to avoid
    the PerformanceWarning that results from repeated in-place column insertion.
    """
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']
    prev_c = close.shift(1)
    ret    = close.pct_change().fillna(0)
    min_p  = max(window // 2, 10)

    new_cols = {}

    new_cols['Return']     = ret
    new_cols['High_Low']   = (high - low) / (close + 1e-10)
    new_cols['High_Close'] = (high - prev_c) / (prev_c + 1e-10)
    new_cols['Low_Close']  = (low  - prev_c) / (prev_c + 1e-10)

    # time_reversal_asymmetry: mean((x_t^2 * x_{t-1}) - (x_{t-1}^2 * x_t))
    new_cols['time_reversal_asymmetry'] = (
        ret ** 2 * ret.shift(1) - ret.shift(1) ** 2 * ret
    ).rolling(window, min_periods=min_p).mean()

    # time_between_extremes: (argmax - argmin) / window, in (-1, 1)
    def _tbe(x):
        return (np.argmax(x) - np.argmin(x)) / max(len(x) - 1, 1)
    new_cols['time_between_extremes'] = close.rolling(window, min_periods=min_p).apply(_tbe, raw=True)

    # volume_entropy: normalized Shannon entropy of volume within window
    def _vol_ent(x):
        x = x[x > 0]
        if len(x) < 4:
            return 0.0
        p = x / x.sum()
        return float(-np.sum(p * np.log(p + 1e-10)))
    new_cols['volume_entropy'] = volume.rolling(window, min_periods=min_p).apply(_vol_ent, raw=True)

    # information_rate_of_change: mean absolute first difference of returns
    new_cols['information_rate_of_change'] = ret.diff().abs().rolling(window, min_periods=min_p).mean()

    # EMD approximations: IMF1 = close - fast_ma (high-freq), IMF2 = fast_ma - slow_ma (mid-freq)
    fast_ma = close.rolling(5, min_periods=2).mean()
    slow_ma = close.rolling(20, min_periods=5).mean()
    imf1 = close - fast_ma
    imf2 = fast_ma - slow_ma
    new_cols['emd_features_IMF_1_Energy'] = (imf1 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols['emd_features_IMF_1_Std']    = imf1.rolling(window, min_periods=min_p).std()
    new_cols['emd_features_IMF_2_Energy'] = (imf2 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols['emd_features_IMF_2_Std']    = imf2.rolling(window, min_periods=min_p).std()

    # entropy: Shannon entropy over 10 equal-width return buckets
    def _ent(x):
        counts, _ = np.histogram(x, bins=10)
        p = counts / (counts.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    new_cols['entropy'] = ret.rolling(window, min_periods=min_p).apply(_ent, raw=True)

    # noise_WhiteNoise / noise_PinkNoise: proximity to spectral slope 0 or -1
    def _spectral_slope(x):
        if len(x) < 8:
            return 0.0
        vals = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x))[1:]
        vals = vals[1:]
        if len(freqs) < 2:
            return 0.0
        slope, _ = np.polyfit(np.log(freqs + 1e-10), np.log(vals + 1e-10), 1)
        return float(slope)
    slopes = ret.rolling(window, min_periods=min_p).apply(_spectral_slope, raw=True)
    new_cols['noise_WhiteNoise'] = np.exp(-(slopes ** 2) / 0.5)
    new_cols['noise_PinkNoise']  = np.exp(-((slopes + 1.0) ** 2) / 0.5)

    # higuchi_fractal_dimension
    def _higuchi(x, kmax=6):
        n = len(x)
        if n < kmax * 2:
            return 1.5
        lm = []
        for k in range(1, kmax + 1):
            lk = 0.0
            for m in range(1, k + 1):
                idxs = np.arange(m - 1, n, k)
                if len(idxs) < 2:
                    continue
                lmk = np.sum(np.abs(np.diff(x[idxs]))) * (n - 1) / (k * (len(idxs) - 1))
                lk += lmk
            lm.append(lk / k)
        if len(lm) < 2:
            return 1.5
        slope, _ = np.polyfit(np.log(range(1, len(lm) + 1)), np.log(np.array(lm) + 1e-10), 1)
        return float(slope)
    new_cols['higuchi_fractal_dimension'] = close.rolling(window, min_periods=min_p).apply(
        _higuchi, raw=True
    )

    # lyapunov_exponent: mean log of absolute price differences
    def _lyap(x):
        d = np.abs(np.diff(x))
        d = d[d > 1e-10]
        return float(np.mean(np.log(d))) if len(d) > 0 else 0.0
    new_cols['lyapunov_exponent'] = close.rolling(window, min_periods=min_p).apply(_lyap, raw=True)

    # dfa_alpha: detrended fluctuation analysis scaling exponent
    def _dfa(x):
        n = len(x)
        if n < 16:
            return 0.5
        y = np.cumsum(x - np.mean(x))
        max_scale = max(4, n // 4)
        scales = np.unique(np.round(np.logspace(1, np.log10(max_scale), 6)).astype(int))
        scales = scales[scales >= 4]
        flucts = []
        for s in scales:
            n_segs = n // s
            if n_segs < 1:
                continue
            f2 = 0.0
            for i in range(n_segs):
                seg = y[i * s:(i + 1) * s]
                t = np.arange(len(seg), dtype=float)
                p = np.polyfit(t, seg, 1)
                f2 += np.mean((seg - np.polyval(p, t)) ** 2)
            flucts.append(np.sqrt(f2 / n_segs))
        if len(flucts) < 2:
            return 0.5
        valid = [(scales[i], flucts[i]) for i in range(len(flucts)) if flucts[i] > 0]
        if len(valid) < 2:
            return 0.5
        ls, lf = zip(*valid)
        alpha, _ = np.polyfit(np.log(ls), np.log(lf), 1)
        return float(np.clip(alpha, -1, 3))
    new_cols['dfa_alpha'] = ret.rolling(window, min_periods=min_p).apply(_dfa, raw=True)

    # spectral_entropy: FFT power-spectrum entropy of returns
    def _spec_ent(x):
        if len(x) < 8:
            return 0.0
        ps = np.abs(np.fft.rfft(x - x.mean())) ** 2
        ps = ps / (ps.sum() + 1e-10)
        return float(-np.sum(ps * np.log(ps + 1e-10)))
    new_cols['spectral_entropy'] = ret.rolling(window, min_periods=min_p).apply(_spec_ent, raw=True)

    # autocorrelation: lag-1 autocorrelation of returns
    def _ac1(x):
        if len(x) < 3:
            return 0.0
        cc = np.corrcoef(x[:-1], x[1:])
        return float(cc[0, 1]) if np.isfinite(cc[0, 1]) else 0.0
    new_cols['autocorrelation'] = ret.rolling(window, min_periods=min_p).apply(_ac1, raw=True)

    # hurst_exponent: reuse existing rolling implementation
    new_cols['hurst_exponent'] = rolling_hurst_exponent(close, window_size=window)

    new_cols['higher_order_moments_Skewness'] = ret.rolling(window, min_periods=min_p).skew()
    new_cols['higher_order_moments_Kurtosis'] = ret.rolling(window, min_periods=min_p).kurt()

    # Single concat — avoids the PerformanceWarning from repeated in-place insertion
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def compute_gp_features(df, date_col='Date'):
    """
    Compute GP-discovered cross-sectional alpha features.

    Operates on a panel DataFrame (multiple tickers stacked with a date column).
    All expressions rank within each date cross-sectionally.

    Returns DataFrame with columns:
        F1_PinkNoise, F3_Return_Rev, F4_Price_Ratio,
        N1_WN_LC, N4_SpectralEnt, composite
    """
    missing = [c for c in GP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"GP features: missing columns {missing}")
    if date_col not in df.columns:
        raise ValueError(f"GP features: date_col '{date_col}' not found")

    df = df.copy().reset_index(drop=True)

    groups = [
        grp.index.to_numpy()
        for _, grp in df.groupby(date_col, sort=False)
    ]

    # Cross-sectionally rank all 23 inputs within each date → x[0..22]
    x = {}
    for i, col in enumerate(GP_REQUIRED_COLS):
        raw = np.nan_to_num(df[col].values.astype(np.float64), nan=0.0)
        x[i] = _gp_apply_cs_rank(raw, groups)

    # F1 — seed 1017: x14 + 0.993 * x3  (PinkNoise + Low_Close)
    # conditional(|1.77| + x19, ...) always takes positive branch since 1.77 + x19 > 0
    f1_raw = x[14] - _gp_conditional(
        np.abs(1.77026646253569) + x[19],
        -0.9931974192883872 * x[3]
    )

    # F3 — seed 1004: -0.073 * x0  (return reversal)
    f3_raw = -0.07320318705361156 * x[0]

    # F4 — seed 1002: (|0.177| + x2) / (x3 + (x7 + 2.694²) + x8)
    f4_raw = _gp_safe_div(
        np.abs(0.17726009244632013) + x[2],
        x[3] + (x[7] + 2.6940490745579684 ** 2) + x[8]
    )

    # N1 — seed 3003: x13 + x3  (WhiteNoise + Low_Close)
    n1_raw = x[13] + x[3]

    # N4 — seed 3029: sqrt(|x3| * ((-1.710 + x18 + x0) / 4.978))
    _denom_n4 = np.abs(-1.2577672553937247 * 3.959124395312415)
    n4_raw = _gp_safe_sqrt(
        np.abs(x[3]) * _gp_safe_div(-1.7102360477772514 + x[18] + x[0], _denom_n4)
    )

    # Per-feature cross-sectional transforms
    F1 = _gp_apply_cs_rank(f1_raw, groups)
    F3 = _gp_apply_cs_rank(f3_raw, groups)
    F4 = _gp_apply_cs_zscore(f4_raw, groups)
    N1 = _gp_apply_cs_rank(n1_raw, groups)
    N4 = _gp_apply_cs_rank(n4_raw, groups)

    def _safe(a):
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    F1, F3, F4, N1, N4 = _safe(F1), _safe(F3), _safe(F4), _safe(N1), _safe(N4)

    # ridge_interactions composite (test IC=+0.0591, WF IC=+0.0523, IR=6.74)
    # Dominant nonlinear term: F3*N4 (+2.39) — return reversal × spectral entropy
    composite = (
        -0.034224 * F1 + 1.296387 * F3 + 0.004473 * F4
        + 0.012287 * N1 + 0.009157 * N4
        - 0.274113 * F1 * F3 + 0.085361 * F1 * F4
        + 0.021151 * F1 * N1 - 0.181167 * F1 * N4
        - 1.295755 * F3 * F4 + 0.355264 * F3 * N1
        + 2.387638 * F3 * N4 - 0.044057 * F4 * N1
        - 0.000961 * F4 * N4 + 0.048930 * N1 * N4
        + 0.002355
    )

    return pd.DataFrame({
        'F1_PinkNoise':   F1,
        'F3_Return_Rev':  F3,
        'F4_Price_Ratio': F4,
        'N1_WN_LC':       N1,
        'N4_SpectralEnt': N4,
        'composite':      _safe(composite),
    }, index=df.index)


def add_gp_cross_sectional_features(output_dir):
    """
    Post-processing cross-sectional pass.
    Loads all per-ticker processed parquet files, computes GP features
    (F1_PinkNoise … composite) cross-sectionally per date, and saves back.
    Called once in process_data_files() after the parallel processing loop.

    Uses pyarrow + ThreadPoolExecutor for both load and save (matches the
    fast-load pattern in 4__Predictor.py — ~10s for 4000 files on NVMe).
    Verbose: prints progress and timing for each phase.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "=" * 70)
    print("GP CROSS-SECTIONAL PASS")
    print("=" * 70)

    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.parquet')
    ]
    if not files:
        print("  no processed parquet files found, skipping")
        return

    # ---- Phase 1: parallel load via pyarrow + ThreadPool ----
    print(f"\n[1/4] Loading {len(files):,} parquet files (16 threads)...")
    t0 = time.time()
    dfs = []
    n_fail = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_fp = {ex.submit(pq.read_table, fp): fp for fp in files}
        with tqdm(total=len(files), desc="  loading", unit="file") as pbar:
            for fut in as_completed(future_to_fp):
                fp = future_to_fp[fut]
                try:
                    tmp = fut.result().to_pandas()
                    tmp['_gp_src'] = fp
                    dfs.append(tmp)
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 5:
                        print(f"    load fail {os.path.basename(fp)}: {e}")
                pbar.update(1)
    print(f"  loaded {len(dfs):,} files in {time.time()-t0:.1f}s "
          f"({len(dfs)/(time.time()-t0):.0f} files/s, {n_fail} failed)")

    if not dfs:
        print("  no files loaded successfully, skipping")
        return

    # ---- Phase 2: concat ----
    print(f"\n[2/4] Concatenating {len(dfs):,} frames...")
    t0 = time.time()
    combined = pd.concat(dfs, ignore_index=True)
    del dfs  # free memory before the big compute
    mem_gb = combined.memory_usage(deep=True).sum() / 1e9
    print(f"  combined: {len(combined):,} rows × {len(combined.columns)} cols, "
          f"{mem_gb:.2f} GB ({time.time()-t0:.1f}s)")

    missing = [c for c in GP_REQUIRED_COLS if c not in combined.columns]
    if missing:
        print(f"  MISSING REQUIRED COLUMNS, skipping: {missing}")
        return

    # ---- Phase 3: compute GP cross-sectional features ----
    print(f"\n[3/4] Computing GP cross-sectional features "
          f"(F1-N4 + composite)...")
    t0 = time.time()
    try:
        gp_feats = compute_gp_features(combined, date_col='Date')
    except Exception as e:
        import traceback
        print(f"  GP FEATURE COMPUTE FAILED: {e}")
        traceback.print_exc()
        return
    gp_cols = list(gp_feats.columns)
    for col in gp_cols:
        combined[col] = gp_feats[col].values
    del gp_feats
    print(f"  added {len(gp_cols)} columns in {time.time()-t0:.1f}s: {gp_cols}")

    # ---- Phase 4: parallel write back ----
    print(f"\n[4/4] Writing back {len(files):,} parquet files (16 threads)...")
    t0 = time.time()

    # Pre-split combined by source file. groupby on the _gp_src column is one
    # vectorized pass — far cheaper than per-file boolean masking in a loop.
    print("  splitting combined by source file...")
    grouped = {fp: g.drop(columns=['_gp_src']).reset_index(drop=True)
               for fp, g in combined.groupby('_gp_src', sort=False)}
    print(f"  split into {len(grouped):,} subsets in {time.time()-t0:.1f}s")

    def _write_one(item):
        fp, subset = item
        try:
            subset.to_parquet(fp, index=False)
            return True, fp, None
        except Exception as e:
            return False, fp, str(e)

    t1 = time.time()
    n_ok = n_bad = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_write_one, item) for item in grouped.items()]
        with tqdm(total=len(futures), desc="  writing", unit="file") as pbar:
            for fut in as_completed(futures):
                ok, fp, err = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    if n_bad <= 5:
                        print(f"    save fail {os.path.basename(fp)}: {err}")
                pbar.update(1)
    print(f"  wrote {n_ok:,} files in {time.time()-t1:.1f}s "
          f"({n_ok/(time.time()-t1):.0f} files/s, {n_bad} failed)")

    print(f"\nGP cross-sectional pass complete: {gp_cols}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interaction-conjunction cross-sectional features (added 2026-05-31, automated
# variant only). From the exhaustive interaction search (interaction_search.py):
# the top tail-consistent, range-orthogonal, low-redundancy elite discriminators
# that survived the temporal-holdout gate. Each = min over 3 ingredients of their
# per-day CROSS-SECTIONAL z-score ("all three elevated vs the universe today").
# Computed here (cross-ticker) — per-day z is unavailable in the per-ticker pass.
# Lookahead-safe: every ingredient is known at close of day t; z is within-day.
# These are CANDIDATES pending Phase-B gating (OOS top-1% precision + backtest).
_IX_CONJUNCTIONS = {
    # A: pattern + gap energy + volume (cleanest: range_corr 0.05, redund 0.18)
    'IX_pattern_energy_min3': ['hammer_pattern', 'gap_in_atr_terms', 'dollar_volume_ma_10'],
    # B: over-extension AVOID signal (most orthogonal: range_corr 0.007) — negative
    'IX_overextension_min3': ['VWAP%_from_high', 'HC_Predict_Regime_norm', 'cv_50d_percentile'],
    # C: structure + trading signal + momentum (highest tail_ir 0.75)
    'IX_struct_signal_min3': ['mp_d4_b185_sym_trace', 'trading_signal_composite',
                              'G_Momentum_Confluence_Indicator'],
}


def add_interaction_conjunction_features(output_dir):
    """Cross-sectional MIN-conjunction features (see _IX_CONJUNCTIONS)."""
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
             if f.endswith('.parquet')]
    if not files:
        print("  [IX] no parquet files, skipping")
        return
    print(f"\n[IX] interaction conjunctions: loading {len(files):,} files...")
    dfs = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        fut = {ex.submit(pq.read_table, fp): fp for fp in files}
        for f in as_completed(fut):
            try:
                tmp = f.result().to_pandas()
                tmp['_ix_src'] = fut[f]
                dfs.append(tmp)
            except Exception as e:
                print(f"    [IX] load fail {os.path.basename(fut[f])}: {e}")
    if not dfs:
        print("  [IX] nothing loaded, skipping")
        return
    combined = pd.concat(dfs, ignore_index=True)
    del dfs

    needed = sorted({c for cols in _IX_CONJUNCTIONS.values() for c in cols})
    present = [c for c in needed if c in combined.columns]
    g = combined.groupby('Date')
    zmap = {}
    for c in present:
        s = combined[c].replace([np.inf, -np.inf], np.nan)
        mu = g[c].transform('mean')
        sd = g[c].transform('std')
        zmap[c] = (s - mu) / (sd + 1e-9)

    added = []
    for name, cols in _IX_CONJUNCTIONS.items():
        missing = [c for c in cols if c not in zmap]
        if missing:
            print(f"  [IX] skip {name}: missing ingredients {missing}")
            continue
        combined[name] = np.minimum.reduce([zmap[c].values for c in cols])
        added.append(name)
    if not added:
        print("  [IX] no conjunctions added (missing ingredients), skipping write")
        return
    print(f"  [IX] added {len(added)}: {added}")

    grouped = {fp: grp.drop(columns=['_ix_src']).reset_index(drop=True)
               for fp, grp in combined.groupby('_ix_src', sort=False)}

    def _w(item):
        fp, subset = item
        try:
            subset.to_parquet(fp, index=False)
            return True
        except Exception as e:
            print(f"    [IX] save fail {os.path.basename(fp)}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = sum(ex.map(_w, grouped.items()))
    print(f"  [IX] wrote {ok:,}/{len(grouped):,} files")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────


def fetch_index(symbol, start_date=None, end_date=None, cache_dir='Data/Indexes'):

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{symbol}.parquet')
    
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
        
    if start_date is None:
        start_date = end_date - timedelta(days=365*2)
    else:
        start_date = pd.to_datetime(start_date)
    
    # Check cache
    if os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        
        need_update = (
            start_date < cached.index.min() or 
            end_date > cached.index.max()
        )
        
        if not need_update:
            filtered = cached.loc[start_date:end_date]
            print(f"  {symbol}: Using cache ({len(filtered)} days)")
            return filtered
        
        print(f"  {symbol}: Updating cache...")
    else:
        print(f"  {symbol}: Downloading from IBKR...")
    
    # Download from IBKR
    async def download():
        ib = IB()
        
        try:
            client_id = random.randint(1, 1000)
            await ib.connectAsync('127.0.0.1', 7496, clientId=client_id, timeout=5)
            
            config = INDEXES[symbol]
            contract = Contract(
                symbol=symbol,
                secType=config['secType'],
                exchange=config['exchange'],
                currency='USD'
            )
            
            if config['primaryExchange']:
                contract.primaryExchange = config['primaryExchange']
            
            bars = await ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime='',
                durationStr='2 Y',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            df = util.df(bars)
            
            if df.empty:
                return None
            
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df['Adj Close'] = df['Close']
            df['Volume'] = df['Volume'].fillna(0).astype(np.int64)
            
            if (df['Volume'] == 0).all() and symbol == 'VIX':
                np.random.seed(42)
                df['Volume'] = np.random.randint(5000000, 15000000, size=len(df))
            
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            return df
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            return None
        
        finally:
            if ib.isConnected():
                ib.disconnect()
    
    try:
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(download())
        
        if data is not None and not data.empty:
            data.to_parquet(cache_file)
            print(f"    Saved {len(data)} days to {cache_file}")
            return data
        
    except Exception as e:
        print(f"    Download failed: {str(e)}")
    
    if os.path.exists(cache_file):
        print(f"    Falling back to cached data")
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        mask = (cached.index >= start_date) & (cached.index <= end_date)
        return cached.loc[mask]
    
    return None


def update_all_indexes(start_date=None, end_date=None):

    print("\n" + "="*70)
    print("UPDATING MARKET INDEXES")
    print("="*70)
    
    results = {}
    success = 0
    failed = 0
    
    for symbol, config in INDEXES.items():
        print(f"\n{symbol} ({config['name']}):")
        
        data = fetch_index(symbol, start_date, end_date)
        
        if data is not None:
            results[symbol] = data
            success += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success} successful, {failed} failed")
    print("="*70)
    
    if results:
        print(f"\nData saved in: Data/Indexes/")
        print("\nAvailable indexes:")
        for symbol in results.keys():
            print(f"  - {symbol}.parquet")
    
    return results


def load_index(symbol, cache_dir='Data/Indexes'):

    cache_file = os.path.join(cache_dir, f'{symbol}.parquet')
    
    if not os.path.exists(cache_file):
        print(f"Index {symbol} not found. Run update_all_indexes() first.")
        return None
    
    df = pd.read_parquet(cache_file)
    df.index = pd.to_datetime(df.index)
    
    return df


def load_all_indexes(cache_dir='Data/Indexes'):


    if not os.path.exists(cache_dir):
        print(f"No indexes found in {cache_dir}")
        print("Run update_all_indexes() first.")
        return {}
    
    results = {}
    
    for symbol in INDEXES.keys():
        df = load_index(symbol, cache_dir)
        if df is not None:
            results[symbol] = df
    
    print(f"Loaded {len(results)} indexes")
    return results





def load_indexes_for_worker():
    """Load all indexes from disk into GLOBAL_INDEX_DATA for this worker process."""
    global GLOBAL_INDEX_DATA
    
    if GLOBAL_INDEX_DATA:  # Already loaded
        return
    
    index_dir = 'Data/Indexes'
    if not os.path.exists(index_dir):
        return
    
    for index_file in os.listdir(index_dir):
        if index_file.endswith('.parquet'):
            index_name = index_file.replace('.parquet', '')
            try:
                index_df = pd.read_parquet(os.path.join(index_dir, index_file))
                index_df.index = pd.to_datetime(index_df.index)
                GLOBAL_INDEX_DATA[index_name] = index_df
            except Exception as e:
                logging.error(f"Error loading {index_name}: {e}")



def add_top_stress_indicators(df, v6_window=10, v7_window=15):
    df = df.sort_values('Date').copy()
    open_to_low_dd = (df['Low'] - df['Open']) / df['Open'] * 100
    
    # stress_v6: Simple EMA on -3% events
    alpha_v6 = 2 / (v6_window + 1)
    df['stress_v6'] = (open_to_low_dd <= -3).ewm(alpha=alpha_v6, min_periods=1).mean()
    
    # Common masks for v7 indicators
    moderate_mask = open_to_low_dd <= -2
    severe_mask = open_to_low_dd <= -4
    extreme_mask = open_to_low_dd <= -6
    
    # stress_v7: Multi-level frequency
    moderate_freq = moderate_mask.rolling(v7_window, min_periods=1).mean()
    severe_freq = severe_mask.rolling(v7_window, min_periods=1).mean()
    extreme_freq = extreme_mask.rolling(v7_window, min_periods=1).mean()
    df['stress_v7'] = (moderate_freq * 0.3 + severe_freq * 0.5 + extreme_freq * 0.2).clip(0, 1)
    
    # stress_vol_v7: Volume-enhanced (if Volume exists)
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20, min_periods=1).mean()
        vol_ratio = df['Volume'] / vol_ma
        moderate_stress = ((moderate_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        severe_stress = ((severe_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        extreme_stress = ((extreme_mask * vol_ratio)).rolling(v7_window, min_periods=1).mean()
        alpha_v7 = 2 / (v7_window + 1)
        ema_moderate = ((moderate_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        ema_severe = ((severe_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        ema_extreme = ((extreme_mask * vol_ratio)).ewm(alpha=alpha_v7, min_periods=1).mean()
        df['stress_vol_v7'] = (
            (moderate_stress + ema_moderate) * 0.2 + 
            (severe_stress + ema_severe) * 0.4 + 
            (extreme_stress + ema_extreme) * 0.4
        ).clip(0, 2)
    
    return df








def calculate_beta(df, window=60, min_periods=30):
    """
    Calculate rolling beta, correlation, and alpha vs all market indexes.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with at least ['Date', 'Close']
    window : int
        Rolling window size in days (default: 60)
    min_periods : int
        Minimum observations per window (default: 30)

    Returns
    -------
    pd.DataFrame
        Original OHLCV data + new columns:
        - beta_<index>, corr_<index>, alpha_<index>
    """

    global GLOBAL_INDEX_DATA

    result_df = df.copy()

    # Ensure Date column exists
    if 'Date' not in result_df.columns:
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df = result_df.reset_index()
        else:
            print(" No Date column found, skipping beta calculation.")
            return result_df

    result_df['Date'] = pd.to_datetime(result_df['Date'])

    if not GLOBAL_INDEX_DATA:
        print(" GLOBAL_INDEX_DATA not loaded, skipping beta calculation.")
        return result_df

    # Compute stock log returns (temporary, not stored)
    stock_close = result_df.set_index('Date')['Close']
    stock_returns = np.log(stock_close / stock_close.shift(1))

    # Work on a copy to safely merge new columns
    final_df = result_df.set_index('Date')

    # Loop through each index (e.g., SPY, QQQ, etc.)
    for index_name, index_df in GLOBAL_INDEX_DATA.items():
        try:
            index_close = index_df['Close']
            index_returns = np.log(index_close / index_close.shift(1))

            # Align on dates
            aligned_stock, aligned_index = stock_returns.align(index_returns, join='inner')

            if len(aligned_stock) < min_periods:
                print(f" Insufficient data for {index_name} beta calculation.")
                continue

            # Rolling stats
            rolling_cov = aligned_stock.rolling(window, min_periods=min_periods).cov(aligned_index)
            rolling_var = aligned_index.rolling(window, min_periods=min_periods).var()

            beta = rolling_cov / rolling_var
            corr = aligned_stock.rolling(window, min_periods=min_periods).corr(aligned_index)

            stock_mean = aligned_stock.rolling(window, min_periods=min_periods).mean()
            index_mean = aligned_index.rolling(window, min_periods=min_periods).mean()
            alpha = stock_mean - (beta * index_mean)

            # Clip outliers for stability
            beta = beta.clip(-5, 5)
            corr = corr.clip(-1, 1)

            # Merge into final DataFrame
            final_df[f'beta_{index_name}'] = beta
            final_df[f'corr_{index_name}'] = corr
            final_df[f'alpha_{index_name}'] = alpha

        except Exception as e:
            print(f" Error calculating beta for {index_name}: {e}")
            continue

    # Reset index back to columns
    final_df = final_df.reset_index()

    # Keep only original columns + new metrics
    keep_cols = list(df.columns) + [
        col for col in final_df.columns if col.startswith(('beta_', 'corr_', 'alpha_'))
    ]
    final_df = final_df[keep_cols]

    return final_df






def calculate_vix_features(df):    
    result_df = df.copy()
    
    # Ensure df has a Date column
    if 'Date' not in result_df.columns:
        if isinstance(result_df.index, pd.DatetimeIndex):
            result_df = result_df.reset_index()
        else:
            raise ValueError("DataFrame must have a 'Date' column or datetime index")
    
    # Make sure Date is datetime
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    
    # Use the global INDEX data - get VIX from the dict
    global GLOBAL_INDEX_DATA
    if not GLOBAL_INDEX_DATA or 'VIX' not in GLOBAL_INDEX_DATA:
        raise ValueError("Global VIX data not initialized. Run update_all_indexes() first.")
    
    # Get VIX data from the index dict
    vix_data = GLOBAL_INDEX_DATA['VIX'].copy()
    
    # Ensure VIX data has a datetime index
    if not isinstance(vix_data.index, pd.DatetimeIndex):
        raise ValueError("Global VIX data must have a datetime index")
    
    # Create a resampled daily VIX dataframe with Close price
    vix_daily = vix_data['Close'].resample('D').last().ffill()
    vix_daily = pd.DataFrame({'VIX_Close': vix_daily})
    vix_daily = vix_daily.reset_index()
    vix_daily.rename(columns={'index': 'Date'}, inplace=True)
    
    # Sort data by date to ensure chronological processing
    result_df = result_df.sort_values('Date')
    
    # Merge VIX data with main dataframe
    # Only merge VIX data that's available BEFORE or ON the current date
    result_df = pd.merge_asof(
        result_df, 
        vix_daily.sort_values('Date'), 
        on='Date', 
        direction='backward'  # Use the last known VIX value on or before the current date
    )
    
    # Forward fill any missing VIX values (only using past values)
    result_df['VIX_Close'] = result_df['VIX_Close'].ffill()
    
    # 1. VIX REGIME DETECTION
    # Create numeric VIX regimes: 0=Low, 1=Normal, 2=High, 3=Extreme
    result_df['VIX_Regime_Numeric'] = np.select(
        [
            result_df['VIX_Close'] < 15,
            (result_df['VIX_Close'] >= 15) & (result_df['VIX_Close'] < 20),
            (result_df['VIX_Close'] >= 20) & (result_df['VIX_Close'] < 30),
            result_df['VIX_Close'] >= 30
        ],
        [0, 1, 2, 3],
        default=1
    )
    
    # 2. VIX RELATIVE LEVELS AND CHANGES
    # Calculating percentiles WITHOUT including current point in the window
    # This way, we're only using past data to determine percentile
    for window in [30, 60, 252]:
        result_df[f'VIX_Percentile_{window}d'] = np.nan
        
        for i in range(len(result_df)):
            if i >= window:  # Only calculate if we have enough history
                lookback_values = result_df.iloc[i-window:i]['VIX_Close'].values
                if len(lookback_values) > 0:
                    current_value = result_df.iloc[i]['VIX_Close']
                    result_df.iloc[i, result_df.columns.get_loc(f'VIX_Percentile_{window}d')] = (
                        stats.percentileofscore(lookback_values, current_value) / 100
                    )
    
    # Calculate rate of change for VIX over different periods
    for period in [1, 5, 10, 20]:
        if len(result_df) > period:
            result_df[f'VIX_Change_{period}d'] = result_df['VIX_Close'].pct_change(period, fill_method=None)
    
    # Calculate VIX momentum (smoothed rate of change)
    result_df['VIX_Momentum'] = np.nan
    for i in range(len(result_df)):
        if i >= 10:  # Need at least 10 days of history
            mean_10d = result_df.iloc[i-10:i]['VIX_Close'].mean()
            if mean_10d > 0:  # Avoid division by zero
                result_df.iloc[i, result_df.columns.get_loc('VIX_Momentum')] = (
                    result_df.iloc[i]['VIX_Close'] / mean_10d - 1
                )
    
    # VIX Acceleration (change in momentum)
    result_df['VIX_Acceleration'] = np.nan
    # Properly calculate the acceleration using only past data
    if 'VIX_Change_5d' in result_df.columns:
        for i in range(len(result_df)):
            if i >= 10:  # Need at least 10 days for 5d change + 5 more for diff
                if pd.notna(result_df.iloc[i]['VIX_Change_5d']) and pd.notna(result_df.iloc[i-5]['VIX_Change_5d']):
                    result_df.iloc[i, result_df.columns.get_loc('VIX_Acceleration')] = (
                        result_df.iloc[i]['VIX_Change_5d'] - result_df.iloc[i-5]['VIX_Change_5d']
                    )
    

    # 3. VIX MOVING AVERAGES AND CROSSOVERS
    for window in [10, 20, 50]:
        result_df[f'VIX_MA{window}'] = result_df['VIX_Close'].shift(1).rolling(window, min_periods=window).mean()
    
    # VIX crossover signals - Only calculate if we have both MAs
    if 'VIX_MA10' in result_df.columns and 'VIX_MA50' in result_df.columns:
        result_df['VIX_Cross_10_50'] = np.where(
            (result_df['VIX_MA10'].notna()) & (result_df['VIX_MA50'].notna()),
            np.where(result_df['VIX_MA10'] > result_df['VIX_MA50'], 1, -1),
            np.nan
        )
    
    if 'VIX_MA20' in result_df.columns and 'VIX_MA50' in result_df.columns:
        result_df['VIX_Cross_20_50'] = np.where(
            (result_df['VIX_MA20'].notna()) & (result_df['VIX_MA50'].notna()),
            np.where(result_df['VIX_MA20'] > result_df['VIX_MA50'], 1, -1),
            np.nan
        )
    
    # 4. VIX MEAN REVERSION SIGNALS
    for window in [20, 50, 100]:
        if len(result_df) >= window:
            mean_col = f'VIX_Mean_{window}d'
            std_col = f'VIX_Std_{window}d'
            z_col = f'VIX_Z_{window}d'
            
            result_df[mean_col] = result_df['VIX_Close'].rolling(window, min_periods=window).mean()
            result_df[std_col] = result_df['VIX_Close'].rolling(window, min_periods=window).std()
            
            # Add epsilon to avoid division by zero
            result_df[z_col] = np.where(
                (result_df[mean_col].notna()) & (result_df[std_col].notna()) & (result_df[std_col] > 0),
                (result_df['VIX_Close'] - result_df[mean_col]) / (result_df[std_col] + 1e-10),
                np.nan
            )
    
    # Create mean reversion signals using Z-score
    if 'VIX_Z_50d' in result_df.columns:
        result_df['VIX_Extreme_High'] = np.where(result_df['VIX_Z_50d'] > 2, 1, 0)
        result_df['VIX_Extreme_Low'] = np.where(result_df['VIX_Z_50d'] < -1, 1, 0)
    
    # 5. VOLATILITY OF VOLATILITY
    # Calculate volatility of VIX (properly using only past data)
    result_df['VIX_of_VIX'] = np.nan
    for i in range(len(result_df)):
        if i >= 21:  # Need at least 21 days (20 for rolling + 1 for current day)
            # First handle any NaN values, then calculate percentage change with fill_method=None
            vix_slice = result_df.iloc[i-20:i]['VIX_Close'].ffill()
            vix_returns = vix_slice.pct_change(fill_method=None).dropna()
            if len(vix_returns) >= 5:  # Ensure we have enough data points
                result_df.iloc[i, result_df.columns.get_loc('VIX_of_VIX')] = (
                    vix_returns.std() * np.sqrt(252)
                )
    
    # Z-score of VIX volatility (properly using only past data)
    result_df['VIX_of_VIX_Z'] = np.nan
    for i in range(len(result_df)):
        if i >= 100 and pd.notna(result_df.iloc[i]['VIX_of_VIX']):
            vix_vol_history = result_df.iloc[i-100:i]['VIX_of_VIX'].dropna()
            if len(vix_vol_history) >= 30:  # Ensure enough data
                mean_vol = vix_vol_history.mean()
                std_vol = vix_vol_history.std()
                if std_vol > 0:
                    result_df.iloc[i, result_df.columns.get_loc('VIX_of_VIX_Z')] = (
                        (result_df.iloc[i]['VIX_of_VIX'] - mean_vol) / (std_vol + 1e-10)
                    )
    
    # 6. VIX-ADJUSTED PRICE INDICATORS
    # Calculate log returns (today vs yesterday)
    result_df['Log_Return'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
    
    # Calculate realized volatility over rolling windows
    for window in [10, 21]:
        result_df[f'Realized_Vol_{window}d'] = result_df['Log_Return'].rolling(
            window, min_periods=window
        ).std() * np.sqrt(252)
    
    # Calculate VIX-to-Realized Vol ratio (risk premium)
    if 'Realized_Vol_21d' in result_df.columns:
        result_df['VIX_vs_Realized_21d'] = np.where(
            result_df['Realized_Vol_21d'] > 0,
            result_df['VIX_Close'] / (100 * result_df['Realized_Vol_21d']),
            np.nan
        )
    
    # VIX-adjusted ATR calculations
    high_low = result_df['High'] - result_df['Low']
    high_close = np.abs(result_df['High'] - result_df['Close'].shift(1))
    low_close = np.abs(result_df['Low'] - result_df['Close'].shift(1))
    
    # Calculate true range
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR with proper rolling window
    result_df['ATR'] = true_range.rolling(14, min_periods=14).mean()
    
    # Calculate VIX-adjusted ATR
    result_df['VIX_Adjusted_ATR'] = np.where(
        result_df['ATR'].notna() & (result_df['VIX_Close'] > 0),
        result_df['ATR'] * np.sqrt(result_df['VIX_Close'] / 20),
        np.nan
    )
    
    # 7. DYNAMIC LOOKBACK PERIODS
    # Calculate VIX factor for dynamic lookback (avoiding division by zero)
    result_df['VIX_Factor'] = np.where(
        result_df['VIX_Close'] > 0,
        np.clip(20 / result_df['VIX_Close'], 0.5, 2),
        1.0  # Default to 1.0 if VIX is zero or negative
    )
    
    # Add 'Ticker' column if it doesn't exist
    if 'Ticker' not in result_df.columns:
        result_df['Ticker'] = 'Unknown'
    
    # Initialize Dynamic_RSI column
    result_df['Dynamic_RSI'] = np.nan
    
    # Calculate Dynamic RSI for each ticker separately
    tickers = result_df['Ticker'].unique()
    
    for ticker in tickers:
        # Get data for this ticker only
        ticker_data = result_df[result_df['Ticker'] == ticker].copy()
        
        # Process each row chronologically
        for i in range(len(ticker_data)):
            if i < 50:  # Skip initial rows with insufficient history
                continue
                
            current_idx = ticker_data.index[i]
            
            # Calculate dynamic lookback period based on VIX
            if pd.notna(ticker_data.iloc[i]['VIX_Factor']):
                dynamic_period = int(14 * ticker_data.iloc[i]['VIX_Factor'])
                # Constrain between 5 and 30 days
                dynamic_period = max(min(dynamic_period, 30), 5)
                
                # Get historical data up to but not including current point
                hist_data = ticker_data.iloc[i-dynamic_period:i].copy()
                
                if len(hist_data) >= 5:  # Ensure we have enough data
                    # Calculate dynamic RSI on historical data
                    deltas = hist_data['Close'].diff().dropna()
                    gains = deltas.where(deltas > 0, 0)
                    losses = -deltas.where(deltas < 0, 0)
                    
                    avg_gain = gains.mean()
                    avg_loss = losses.mean()
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        result_df.loc[current_idx, 'Dynamic_RSI'] = 100 - (100 / (1 + rs))
                    elif avg_gain > 0:
                        # All positive changes, no losses
                        result_df.loc[current_idx, 'Dynamic_RSI'] = 100
                    else:
                        # No change
                        result_df.loc[current_idx, 'Dynamic_RSI'] = 50
    
    # 8. RISK ADJUSTMENT FEATURES
    result_df['VIX_Risk_Multiplier'] = result_df['VIX_Close'] / 20
    
    # Calculate percentage change in Close price
    result_df['percent_change_Close'] = result_df['Close'].pct_change(fill_method=None)
    
    ##OLD VERSION
    # 
    # # Calculate risk-adjusted returns (avoid division by zero)
    ##result_df['Risk_Adjusted_Return'] = np.where(
    ##    result_df['VIX_Close'] > 0,
    ##    result_df['percent_change_Close'] / (result_df['VIX_Close'] / 100),
    ##    np.nan
    ##)
    
    # Improved version
    #epsilon = 1e-10
    #result_df['Risk_Adjusted_Return'] = result_df['percent_change_Close'] / ((result_df['VIX_Close'] / 100) + epsilon)


    # 9. MARKET REGIME CLASSIFICATION
    # Calculate 50-day SMA
    result_df['SMA50'] = result_df['Close'].rolling(50, min_periods=50).mean()
    
    # Determine trend direction
    result_df['Trend_Direction'] = np.where(
        result_df['SMA50'].notna(),
        np.where(result_df['Close'] > result_df['SMA50'], 1, -1),
        np.nan
    )
    
    # Create market regime based on VIX and trend
    # 0: Low vol + Uptrend, 1: Low vol + Downtrend, 2: High vol + Uptrend, 3: High vol + Downtrend
    result_df['Market_Regime'] = np.where(
        result_df['Trend_Direction'].notna(),
        np.where(
            result_df['VIX_Close'] < 20,
            np.where(result_df['Trend_Direction'] > 0, 0, 1),  # Low vol regimes
            np.where(result_df['Trend_Direction'] > 0, 2, 3)   # High vol regimes
        ),
        np.nan
    )
    
    # 10. VIX PATTERN DETECTION
    # Properly detect patterns using only past data, shifting correctly
    result_df['VIX_Spike'] = np.nan
    result_df['VIX_Bottom'] = np.nan
    
    for i in range(len(result_df)):
        if i >= 25:  # Need at least 25 days (5 for first shift, 1 for current, and 20 for rolling mean)
            # Get the relevant values
            vix_5days_ago = result_df.iloc[i-5]['VIX_Close']
            vix_1day_ago = result_df.iloc[i-1]['VIX_Close']
            vix_today = result_df.iloc[i]['VIX_Close']
            
            # Calculate the 20-day mean EXCLUDING the current day
            vix_mean_20d = result_df.iloc[i-20:i]['VIX_Close'].mean()
            
            # Check for spike pattern (VIX was rising, now falling, and recently high)
            if (vix_5days_ago < vix_1day_ago) and (vix_1day_ago > vix_today) and (vix_1day_ago > vix_mean_20d * 1.5):
                result_df.iloc[i, result_df.columns.get_loc('VIX_Spike')] = 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('VIX_Spike')] = 0
                
            # Check for bottom pattern (VIX was falling, now rising, and recently low)
            if (vix_5days_ago > vix_1day_ago) and (vix_1day_ago < vix_today) and (vix_1day_ago < vix_mean_20d * 0.8):
                result_df.iloc[i, result_df.columns.get_loc('VIX_Bottom')] = 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('VIX_Bottom')] = 0
    
    # Fill NaN values with forward fill only (using only past data)
    numeric_cols = result_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Use ffill() instead of bfill() to avoid look-ahead bias
        result_df[col] = result_df[col].ffill()
    
    # Final pass: fill any remaining NaNs with zeros
    # This only affects the beginning of the series where no past data exists
    result_df = result_df.fillna(0)
    
    # Reset index to make sure it's consecutive after all operations
    result_df = result_df.reset_index(drop=True)
    
    return result_df



def calculate_dvamr_probability(df, 
                               lookback_window=60, 
                               momentum_window=20, 
                               probability_window=120):
    """
    Dynamic Volatility Adjusted Mean Reversion Probability (DVAMR)
    
    Calculates a probability score (0-1) indicating confidence in mean reversion opportunities
    during volatile market conditions. Higher values indicate higher confidence in potential
    reversion trades based on VIX dynamics, market stress, and volatility regimes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain columns: ['Date', 'Close', 'Mean_Reversion_Z_Score_90_std_1', 
                              'VIX_Acceleration', 'VIX_Close']
    lookback_window : int, default=60
        Rolling window for calculating market conditions
    momentum_window : int, default=20  
        Rolling window for momentum signals
    probability_window : int, default=120
        Rolling window for probability normalization
    
    Returns:
    --------
    pandas.DataFrame with added column:
        - 'dvamr_probability': Probability score (0-1) indicating reversion confidence
    """
    
    # Validate required columns
    required_cols = ['Date', 'Close', 'Mean_Reversion_Z_Score_90_std_1', 'VIX_Acceleration', 'VIX_Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create working copy
    result_df = df.copy()
    result_df['daily_return'] = result_df['Close'].pct_change()
    
    # Initialize output
    n_rows = len(result_df)
    raw_signal = np.zeros(n_rows)
    
    # Rolling window calculation to avoid data leakage
    for i in range(n_rows):
        
        # Calculate base GP signal (can use current row)
        numerator = result_df['Mean_Reversion_Z_Score_90_std_1'].iloc[i] - result_df['VIX_Acceleration'].iloc[i]
        denominator = result_df['VIX_Acceleration'].iloc[i] + result_df['VIX_Close'].iloc[i] + result_df['VIX_Close'].iloc[i]
        
        if denominator != 0:
            base_signal = numerator / denominator
        else:
            base_signal = 0.0
        
        # Skip enhancement calculations until we have enough history
        if i < lookback_window:
            raw_signal[i] = base_signal
            continue
            
        # Get lookback window (only past data to avoid leakage)
        window_df = result_df.iloc[i-lookback_window:i].copy()
        
        # === DYNAMIC ADJUSTMENT FACTOR CALCULATION ===
        adjustment_factor = 1.0
        
        # 1. Volatility factor (higher volatility = boost signal)
        current_volatility = window_df['daily_return'].std() * np.sqrt(252)
        volatility_factor = np.clip(current_volatility / 0.6, 0.5, 2.0)
        adjustment_factor *= (0.8 + 0.4 * volatility_factor)
        
        # 2. Up days factor (lower up days = boost signal)
        current_up_days_pct = (window_df['daily_return'] > 0).mean()
        up_days_factor = np.clip(1.6 - current_up_days_pct * 2, 0.6, 1.4)
        adjustment_factor *= up_days_factor
        
        # 3. Mean reversion factor (lower MR score = boost signal)
        current_mr_score = window_df['Mean_Reversion_Z_Score_90_std_1'].mean()
        mr_factor = np.clip(1.2 - current_mr_score * 0.5, 0.7, 1.5)
        adjustment_factor *= mr_factor
        
        # 4. Drawdown factor (higher drawdown = boost signal)
        window_peak = window_df['Close'].expanding().max()
        current_drawdown = ((window_df['Close'] - window_peak) / window_peak).min()
        drawdown_factor = np.clip(1.0 + abs(current_drawdown) * 1.5, 0.8, 1.8)
        adjustment_factor *= drawdown_factor
        
        # 5. Price stability factor (lower stability = boost signal)
        price_cv = window_df['Close'].std() / window_df['Close'].mean()
        current_price_stability = 1 / (price_cv + 1e-8)
        stability_factor = np.clip(1.5 - current_price_stability * 0.1, 0.7, 1.3)
        adjustment_factor *= stability_factor
        
        # Bound the adjustment factor
        adjustment_factor = np.clip(adjustment_factor, 0.3, 3.0)
        
        # === VIX REGIME FILTER ===
        current_vix = result_df['VIX_Close'].iloc[i-1]  # Use previous day to avoid leakage
        if pd.isna(current_vix):
            current_vix = 20
            
        if current_volatility > 0.4 and current_vix > 15:
            vix_regime = 1.0
        elif current_volatility < 0.25 or current_vix < 12:
            vix_regime = 0.3
        else:
            vix_regime = 0.7
        
        # === DISTRESS AMPLIFIER ===
        stock_return = (window_df['Close'].iloc[-1] / window_df['Close'].iloc[0] - 1)
        distress_amp = 1.0
        
        if current_vix > 25:
            distress_amp *= 1.5
        elif current_vix > 35:
            distress_amp *= 2.0
            
        if stock_return < -0.1 and current_vix > 20:
            distress_amp *= 1.3
            
        if stock_return < -0.3:
            distress_amp *= 1.6
            
        distress_amp = np.clip(distress_amp, 0.5, 2.5)
        
        # === FINAL ENHANCED SIGNAL ===
        enhanced_signal = base_signal * adjustment_factor * vix_regime * distress_amp
        raw_signal[i] = enhanced_signal
    
    # === CONVERT TO PROBABILITY (0-1) ===
    # Use rolling percentile rank to create probability-like output
    probability_signal = np.zeros(n_rows)
    
    for i in range(probability_window, n_rows):
        # Get rolling window of signals
        window_signals = raw_signal[i-probability_window:i]
        
        # Calculate percentile rank of current signal
        current_signal = raw_signal[i]
        
        if len(window_signals) > 0:
            # Count how many past signals are less than current signal
            percentile_rank = np.sum(window_signals <= current_signal) / len(window_signals)
            
            # Apply sigmoid transformation for smoother probability
            # This emphasizes extreme values and compresses middle values
            sigmoid_prob = 1 / (1 + np.exp(-5 * (percentile_rank - 0.5)))
            
            probability_signal[i] = sigmoid_prob
        else:
            probability_signal[i] = 0.5  # Neutral when no history
    
    # For early periods without enough history, use simple normalization
    for i in range(probability_window):
        if i > 0:
            # Simple min-max normalization on available data
            window_signals = raw_signal[:i+1]
            if np.std(window_signals) > 0:
                normalized = (raw_signal[i] - np.min(window_signals)) / (np.max(window_signals) - np.min(window_signals))
                probability_signal[i] = np.clip(normalized, 0, 1)
            else:
                probability_signal[i] = 0.5
    
    # Add only the probability signal to output
    result_df['dvamr_probability'] = probability_signal
    
    # Drop intermediate columns, keep only original columns + dvamr_probability
    columns_to_keep = list(df.columns) + ['dvamr_probability']
    result_df = result_df[columns_to_keep]
    
    return result_df






def calculate_moving_average_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Create all columns in a dictionary first
    new_columns = {}
    
    # Add 14-day moving average with min_periods=1
    new_columns['14ma'] = close.rolling(window=14, min_periods=1).mean()
    
    # Calculate percentage difference from 14-day MA
    # Using epsilon to avoid division by zero
    epsilon = 1e-10
    new_columns['14ma%'] = ((close - new_columns['14ma']) / (new_columns['14ma'] + epsilon)) * 100
    
    # Standard deviation with proper min_periods
    new_columns['std_14'] = close.rolling(window=14, min_periods=1).std()
    
    # Percentage change of 14ma% with proper handling of NAs
    new_columns['14ma%_change'] = new_columns['14ma%'].pct_change(fill_method=None)
    
    # Count of positive days in 14-day window
    new_columns['14ma%_count'] = new_columns['14ma%'].gt(0).rolling(window=14, min_periods=1).sum()
    
    # True Range calculation using shift(1) to ensure we only use past data
    close_shift_1 = close.shift(1)
    true_range = np.maximum(
        high - low, 
        np.maximum(
            np.abs(high - close_shift_1), 
            np.abs(low - close_shift_1)
        )
    )
    
    # Calculate ATR using rolling mean with min_periods=1
    new_columns['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    
    # ATR as percentage of price
    new_columns['ATR%'] = (new_columns['ATR'] / (close + epsilon)) * 100
    
    # Add all columns at once to minimize DataFrame operations
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)




def calculate_vol_fade_signal(df):
    """Volatility fade signal: shorts vol expansion after upward compression, longs downward."""
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    kappa = 0.2
    lookback = 20
    norm_lookback = 40
    
    close_shift_1 = close.shift(1)
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - close_shift_1), np.abs(low - close_shift_1))
    )
    
    atr = true_range.rolling(window=norm_lookback, min_periods=1).mean()
    norm_range = (high - low) / (atr + 1e-10)
    
    # Hawkes process smoothing
    alpha = np.exp(-kappa)
    hawkes_vol = np.zeros(len(df))
    hawkes_vol[:] = np.nan
    norm_range_arr = norm_range.values
    
    for i in range(1, len(df)):
        if np.isnan(hawkes_vol[i - 1]):
            hawkes_vol[i] = norm_range_arr[i]
        else:
            hawkes_vol[i] = hawkes_vol[i - 1] * alpha + norm_range_arr[i]
    
    hawkes_vol = hawkes_vol * kappa
    hawkes_series = pd.Series(hawkes_vol, index=df.index)
    
    # Shift to avoid lookahead
    hawkes_shifted = hawkes_series.shift(1)
    
    q05 = hawkes_shifted.rolling(window=lookback, min_periods=lookback//2).quantile(0.05)
    q95 = hawkes_shifted.rolling(window=lookback, min_periods=lookback//2).quantile(0.95)
    
    signal = np.zeros(len(df))
    last_compression_idx = -1
    current_signal = 0
    
    close_arr = close.values
    hawkes_arr = hawkes_shifted.values
    q05_arr = q05.values
    q95_arr = q95.values
    
    for i in range(1, len(df)):
        if np.isnan(q05_arr[i]) or np.isnan(q95_arr[i]) or np.isnan(hawkes_arr[i]):
            signal[i] = current_signal
            continue
        
        if hawkes_arr[i] < q05_arr[i]:
            last_compression_idx = i
            current_signal = 0
        
        # Vol expansion after compression - fade the move
        if (hawkes_arr[i] > q95_arr[i] and 
            hawkes_arr[i-1] <= q95_arr[i-1] and
            last_compression_idx > 0):
            
            price_change = close_arr[i] - close_arr[last_compression_idx]
            current_signal = -1 if price_change > 0 else 1
        
        signal[i] = current_signal
    
    new_columns = {'vol_fade_signal': signal}
    
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def add_genetic_info_decay_3d(df, epsilon=1e-8):

    df = df.copy()
    
    # Step 1: Calculate base genetic price differential
    df['_temp_genetic'] = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    
    # Step 2: Calculate 3-day lagged version
    lagged_genetic = df['_temp_genetic'].shift(3)
    current_genetic = df['_temp_genetic']
    
    # Step 3: Calculate rolling 20-day correlation between current and lagged
    rolling_autocorr = lagged_genetic.rolling(window=20, min_periods=10).corr(current_genetic)
    
    # Step 4: Information decay = 1 - autocorrelation
    df['G_PDA_Info_Decay_3d'] = 1 - rolling_autocorr
    
    # Clean up temporary column
    df.drop('_temp_genetic', axis=1, inplace=True)
    
    return df

def add_genetic_autocorr_3d(df, epsilon=1e-8):

    df = df.copy()
    
    # Step 1: Calculate base genetic price differential
    df['_temp_genetic'] = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    
    # Step 2: Calculate 3-day lagged version
    lagged_genetic = df['_temp_genetic'].shift(3)
    current_genetic = df['_temp_genetic']
    
    # Step 3: Calculate rolling 20-day correlation
    df['G_PDA_Autocorr_3d'] = lagged_genetic.rolling(window=20, min_periods=10).corr(current_genetic)
    
    # Clean up temporary column
    df.drop('_temp_genetic', axis=1, inplace=True)
    
    return df

def add_volume_spectral_splatter(df, window=50, epsilon=1e-8):

    df = df.copy()
    
    def calculate_spectral_splatter(volume_series, window_size):
        """Calculate spectral splatter for a volume series."""
        results = []
        
        for i in range(len(volume_series)):
            if i < window_size:
                results.append(np.nan)
                continue
                
            # Get window of volume data
            vol_window = volume_series.iloc[i-window_size:i].values
            
            # Handle zero/negative volumes
            vol_window = np.maximum(vol_window, epsilon)
            
            # Log transform to stabilize variance
            log_vol = np.log(vol_window)
            
            # Remove trend (detrend)
            detrended = detrend(log_vol)
            
            # Apply window function to reduce spectral leakage
            windowed = detrended * np.hanning(len(detrended))
            
            # Compute FFT
            fft_vals = fft(windowed)
            power_spectrum = np.abs(fft_vals) ** 2
            
            # Normalize power spectrum
            total_power = np.sum(power_spectrum)
            if total_power > epsilon:
                normalized_power = power_spectrum / total_power
            else:
                normalized_power = np.ones_like(power_spectrum) / len(power_spectrum)
            
            # Calculate spectral entropy (measure of signal dispersion)
            # Higher entropy = more dispersed/chaotic signal
            spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + epsilon))
            
            # Normalize to 0-1 range approximately
            max_entropy = np.log(len(normalized_power))
            if max_entropy > 0:
                normalized_entropy = spectral_entropy / max_entropy
            else:
                normalized_entropy = 0
            
            results.append(normalized_entropy)
        
        return pd.Series(results, index=volume_series.index)
    
    # Calculate spectral splatter
    df['Volume_Spectral_Splatter'] = calculate_spectral_splatter(df['Volume'], window)
    
    return df

def add_genetic_log_zscore_20d(df, epsilon=1e-8):

    df = df.copy()
    # Step 1: Calculate base genetic price differential
    genetic_feature = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    
    # Step 2: Apply log scaling (preserve sign)
    log_genetic = np.log(np.abs(genetic_feature) + epsilon) * np.sign(genetic_feature)
    
    # Step 3: Rolling z-score normalization (20-day window)
    rolling_mean = log_genetic.rolling(window=20, min_periods=10).mean()
    rolling_std = log_genetic.rolling(window=20, min_periods=10).std()
    
    df['G_PDA_Log_Zscore_20d'] = (log_genetic - rolling_mean) / (rolling_std + epsilon)
    
    return df



def add_genetic_signal_strength(df, epsilon=1e-8):

    df = df.copy()
    
    # Step 1: Calculate base genetic price differential
    genetic_feature = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    
    # Step 2: Calculate rolling statistics for signal detection
    rolling_mean = genetic_feature.rolling(20, min_periods=10).mean()
    rolling_std = genetic_feature.rolling(20, min_periods=10).std()
    
    # Step 3: Generate individual signals
    
    # Z-score based extreme signals (weight: 3 each)
    z_score = (genetic_feature - rolling_mean) / (rolling_std + epsilon)
    extreme_high = (z_score > 2).astype(int) * 3
    extreme_low = (z_score < -2).astype(int) * 3
    
    # Momentum signals (weight: 2 each)
    genetic_roc_3d = genetic_feature.pct_change(3)
    roc_rolling_80th = genetic_roc_3d.rolling(10, min_periods=5).quantile(0.8)
    roc_rolling_20th = genetic_roc_3d.rolling(10, min_periods=5).quantile(0.2)
    
    momentum_accelerating = (genetic_roc_3d > roc_rolling_80th).astype(int) * 2
    momentum_decelerating = (genetic_roc_3d < roc_rolling_20th).astype(int) * 2
    
    # Trend cross signals (weight: 2 each)
    genetic_sma_short = genetic_feature.rolling(5, min_periods=3).mean()
    genetic_sma_long = genetic_feature.rolling(20, min_periods=10).mean()
    
    bullish_cross = ((genetic_sma_short > genetic_sma_long) & 
                     (genetic_sma_short.shift(1) <= genetic_sma_long.shift(1))).astype(int) * 2
    bearish_cross = ((genetic_sma_short < genetic_sma_long) & 
                     (genetic_sma_short.shift(1) >= genetic_sma_long.shift(1))).astype(int) * 2
    
    # Volatility regime signals (weight: 1 each)
    rolling_vol = genetic_feature.rolling(20, min_periods=10).std()
    vol_75th = rolling_vol.rolling(60, min_periods=30).quantile(0.75)
    vol_25th = rolling_vol.rolling(60, min_periods=30).quantile(0.25)
    
    high_vol_regime = (rolling_vol > vol_75th).astype(int) * 1
    low_vol_regime = (rolling_vol < vol_25th).astype(int) * 1
    
    # Step 4: Combine into weighted signal strength
    df['G_PDA_Weighted_Signal_Strength'] = (
        extreme_high + extreme_low + 
        momentum_accelerating + momentum_decelerating +
        bullish_cross + bearish_cross +
        high_vol_regime + low_vol_regime
    )
    
    return df








def add_price_differential_ratio(df, epsilon=1e-6):
    df = df.copy()
    df['Price_Differential_Ratio'] = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    return df

def add_rolling_log_scaled_features(df, base_column, windows=[10, 20, 50], methods=['zscore', 'minmax', 'rank'], epsilon=1e-6):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    def apply_log_scaling(series, window, method):
        clean_series = series.replace([np.inf, -np.inf], np.nan)
        abs_series = np.abs(clean_series)
        adaptive_epsilon = max(epsilon, abs_series.median() * 1e-6) if abs_series.median() > 0 else epsilon
        
        with np.errstate(invalid='ignore', divide='ignore'):
            log_abs = np.log(abs_series + adaptive_epsilon)
            log_series = log_abs * np.sign(clean_series)
            log_series = pd.Series(log_series, index=series.index).replace([np.inf, -np.inf], np.nan)
        
        if method == 'zscore':
            rolling_mean = log_series.rolling(window=window, min_periods=max(1, window//2)).mean()
            rolling_std = log_series.rolling(window=window, min_periods=max(1, window//2)).std()
            rolling_std = rolling_std.fillna(1.0).replace(0, 1.0)
            result = (log_series - rolling_mean) / rolling_std
        elif method == 'minmax':
            rolling_min = log_series.rolling(window=window, min_periods=max(1, window//2)).min()
            rolling_max = log_series.rolling(window=window, min_periods=max(1, window//2)).max()
            rolling_range = rolling_max - rolling_min
            rolling_range = rolling_range.replace(0, 1.0)
            result = (log_series - rolling_min) / rolling_range
        elif method == 'rank':
            result = log_series.rolling(window=window, min_periods=max(1, window//2)).rank(pct=True)
        
        return result
    
    for window in windows:
        for method in methods:
            feature_name = f'{base_column}_LogScale_{method.title()}_{window}d'
            df[feature_name] = apply_log_scaling(df[base_column], window, method)
    
    return df

def add_rate_of_change_features(df, base_column, periods=[1, 3, 5, 10]):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    for period in periods:
        df[f'{base_column}_PctChange_{period}d'] = df[base_column].pct_change(period)
        df[f'{base_column}_Diff_{period}d'] = df[base_column].diff(period)
        
        series = df[base_column]
        abs_series = np.abs(series)
        adaptive_epsilon = max(1e-8, abs_series.median() * 1e-8) if abs_series.median() > 0 else 1e-8
        
        with np.errstate(invalid='ignore', divide='ignore'):
            log_series = np.log(abs_series + adaptive_epsilon) * np.sign(series)
            log_series = pd.Series(log_series, index=series.index).replace([np.inf, -np.inf], np.nan)
            df[f'{base_column}_LogDiff_{period}d'] = log_series.diff(period)
    
    if f'{base_column}_PctChange_3d' in df.columns:
        df[f'{base_column}_Acceleration_3d'] = df[f'{base_column}_PctChange_3d'].pct_change(1)
    if f'{base_column}_PctChange_5d' in df.columns:
        df[f'{base_column}_Acceleration_5d'] = df[f'{base_column}_PctChange_5d'].pct_change(1)
    
    return df

def add_z_score_signals(df, base_column, window=20, epsilon=1e-6):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    feature = df[base_column]
    rolling_mean = feature.rolling(window).mean()
    rolling_std = feature.rolling(window).std()
    rolling_std = rolling_std.fillna(epsilon).replace(0, epsilon)
    
    z_score = (feature - rolling_mean) / rolling_std
    df[f'{base_column}_ZScore'] = z_score
    df[f'{base_column}_ExtremeHigh'] = (z_score > 2).astype(int)
    df[f'{base_column}_ExtremeLow'] = (z_score < -2).astype(int)
    
    return df

def add_momentum_signals(df, base_column, window=10):
    df = df.copy()
    roc_col = f'{base_column}_PctChange_3d'
    if roc_col not in df.columns:
        df = add_rate_of_change_features(df, base_column, [3])
    
    if roc_col in df.columns:
        roc_3d = df[roc_col]
        df[f'{base_column}_MomentumAccelerating'] = (roc_3d > roc_3d.rolling(window).quantile(0.8)).astype(int)
        df[f'{base_column}_MomentumDecelerating'] = (roc_3d < roc_3d.rolling(window).quantile(0.2)).astype(int)
    
    return df

def add_trend_reversal_signals(df, base_column, short_window=5, long_window=20):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    feature = df[base_column]
    feature_sma_short = feature.rolling(short_window).mean()
    feature_sma_long = feature.rolling(long_window).mean()
    
    df[f'{base_column}_BullishCross'] = ((feature_sma_short > feature_sma_long) & 
                                        (feature_sma_short.shift(1) <= feature_sma_long.shift(1))).astype(int)
    df[f'{base_column}_BearishCross'] = ((feature_sma_short < feature_sma_long) & 
                                        (feature_sma_short.shift(1) >= feature_sma_long.shift(1))).astype(int)
    
    return df

def add_volatility_regime_signals(df, base_column, vol_window=20, regime_window=60):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    feature = df[base_column]
    rolling_vol = feature.rolling(vol_window).std()
    vol_threshold_high = rolling_vol.rolling(regime_window).quantile(0.75)
    vol_threshold_low = rolling_vol.rolling(regime_window).quantile(0.25)
    
    df[f'{base_column}_HighVolatilityRegime'] = (rolling_vol > vol_threshold_high).astype(int)
    df[f'{base_column}_LowVolatilityRegime'] = (rolling_vol < vol_threshold_low).astype(int)

    return df

def add_signal_persistence_features(df, base_column, window=20, epsilon=1e-6):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    feature = df[base_column]
    rolling_mean = feature.rolling(window).mean()
    feature_direction = np.sign(feature - rolling_mean)
    direction_changes = (feature_direction != feature_direction.shift(1)).astype(int)
    persistence = direction_changes.cumsum()
    df[f'{base_column}_SignalPersistence'] = persistence.groupby(persistence).cumcount() + 1
    
    return df

def add_composite_signal_strength(df, base_column):
    df = df.copy()
    signal_endings = ['ExtremeHigh', 'ExtremeLow', 'MomentumAccelerating', 
                     'MomentumDecelerating', 'BullishCross', 'BearishCross']
    
    signal_cols = []
    for ending in signal_endings:
        col_name = f'{base_column}_{ending}'
        if col_name in df.columns:
            signal_cols.append(col_name)
    
    if signal_cols:
        df[f'{base_column}_SignalStrength'] = df[signal_cols].sum(axis=1)
        
        weights = {
            'ExtremeHigh': 3, 'ExtremeLow': 3, 'MomentumAccelerating': 2,
            'MomentumDecelerating': 2, 'BullishCross': 2, 'BearishCross': 2,
        }
        
        weighted_strength = pd.Series(0, index=df.index)
        for col in signal_cols:
            signal_type = col.split('_')[-1]
            weight = weights.get(signal_type, 1)
            weighted_strength += df[col] * weight
        
        df[f'{base_column}_WeightedSignalStrength'] = weighted_strength
    
    return df

def add_information_efficiency_features(df, base_column, window=20, epsilon=1e-6):
    df = df.copy()
    if base_column not in df.columns:
        raise ValueError(f"Column '{base_column}' not found in dataframe")
    
    feature_returns = df[base_column].pct_change()
    rolling_mean = feature_returns.rolling(window).mean()
    rolling_std = feature_returns.rolling(window).std()
    rolling_std = rolling_std.fillna(epsilon).replace(0, epsilon)
    df[f'{base_column}_InformationRatio'] = rolling_mean / rolling_std
    
    for lag in [1, 3, 5]:
        lagged_feature = df[base_column].shift(lag)
        current_feature = df[base_column]
        rolling_autocorr = lagged_feature.rolling(window, min_periods=max(1, window//2)).corr(current_feature)
        df[f'{base_column}_Autocorr_{lag}d'] = rolling_autocorr
        autocorr_clamped = rolling_autocorr.clip(0, 1).fillna(0)
        df[f'{base_column}_InfoDecay_{lag}d'] = 1 - autocorr_clamped
    
    return df





def create_gap_zscore_predictors(df):
    """
    Create all Gap ZScore predictor variations from OHLCV data.
    
    Takes a DataFrame with OHLCV columns and returns the same DataFrame
    with additional prediction columns added.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        Optional: 'Date' for time-based effects
    
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added prediction columns:
        - Gap_ZScore_Base: The base gap z-score signal
        - Gap_ZScore_Volume_Enhanced: Volume-confirmed gap signals  
        - Gap_ZScore_Regime_Aware: Volatility regime-adjusted signals
        - Gap_ZScore_Time_Conditional: Day-of-week adjusted signals
        - Gap_ZScore_Reversion_Momentum: Market state conditional signals
        - Gap_ZScore_Multi_Timeframe: Multi-timeframe combined signals
        - Gap_ZScore_Adaptive_Threshold: Self-adapting effectiveness signals
    """
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Work on a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate basic returns for internal use
    returns = result_df['Close'].pct_change()
    volume_change = result_df['Volume'].pct_change()
    
    
    
    # ============================================================================
    # 1. BASE GAP ZSCORE CALCULATION
    # ============================================================================
    
    
    # Calculate overnight gaps
    gaps = result_df['Open'] - result_df['Close'].shift(1)
    gap_pct = gaps / result_df['Close'].shift(1)
    
    # Rolling statistics for z-score
    rolling_gap_mean = gap_pct.rolling(window=20).mean()
    rolling_gap_std = gap_pct.rolling(window=20).std()
    gap_zscore_base = (gap_pct - rolling_gap_mean) / rolling_gap_std
    
    result_df['Gap_ZScore_Base'] = gap_zscore_base.fillna(0)
    
    # ============================================================================
    # 2. VOLUME ENHANCED VARIATION
    # ============================================================================
    
    
    gap_zscore_volume_enhanced = pd.Series(np.nan, index=result_df.index)
    
    # Volume z-score for confirmation
    vol_rolling_mean = result_df['Volume'].rolling(window=20).mean()
    vol_rolling_std = result_df['Volume'].rolling(window=20).std()
    volume_zscore = (result_df['Volume'] - vol_rolling_mean) / vol_rolling_std
    
    for i in range(len(result_df)):
        gap_z = gap_zscore_base.iloc[i]
        vol_z = volume_zscore.iloc[i]
        
        if not (np.isnan(gap_z) or np.isnan(vol_z)):
            # Volume-enhanced gap signal
            volume_multiplier = min(max(vol_z / 2.0, 0.1), 2.0)  # Scale 0.1 to 2.0
            
            if abs(gap_z) > 1.0:
                gap_zscore_volume_enhanced.iloc[i] = gap_z * volume_multiplier
            else:
                gap_zscore_volume_enhanced.iloc[i] = gap_z * 0.5  # Weak signal for small gaps
        else:
            gap_zscore_volume_enhanced.iloc[i] = 0
    
    result_df['Gap_ZScore_Volume_Enhanced'] = gap_zscore_volume_enhanced.fillna(0)
    
    # ============================================================================
    # 3. REGIME AWARE VARIATION
    # ============================================================================
    
    
    gap_zscore_regime_aware = pd.Series(np.nan, index=result_df.index)
    
    # Calculate volatility regime
    rolling_vol = returns.rolling(window=20).std()
    long_vol = returns.rolling(window=50).std()
    vol_regime = rolling_vol / long_vol  # >1 = high vol regime
    
    for i in range(len(result_df)):
        gap_z = gap_zscore_base.iloc[i]
        vol_reg = vol_regime.iloc[i]
        
        if not (np.isnan(gap_z) or np.isnan(vol_reg)):
            # Adjust signal based on volatility regime - continuous scaling
            if vol_reg > 1.2:  # High volatility regime
                regime_multiplier = 0.7  # Reduce sensitivity in high vol
            elif vol_reg < 0.8:  # Low volatility regime  
                regime_multiplier = 1.3  # Increase sensitivity in low vol
            else:
                regime_multiplier = 1.0
            
            # Apply regime adjustment to gap signal
            gap_zscore_regime_aware.iloc[i] = gap_z * regime_multiplier
        else:
            gap_zscore_regime_aware.iloc[i] = 0
    
    result_df['Gap_ZScore_Regime_Aware'] = gap_zscore_regime_aware.fillna(0)
    
    # ============================================================================
    # 4. TIME CONDITIONAL VARIATION
    # ============================================================================
    
    
    gap_zscore_time_conditional = pd.Series(np.nan, index=result_df.index)
    
    # Add day of week if we have dates
    if 'Date' in result_df.columns:
        try:
            day_of_week = pd.to_datetime(result_df['Date']).dt.dayofweek
        except:
            # Fallback if date parsing fails
            day_of_week = pd.Series(np.arange(len(result_df)) % 5, index=result_df.index)
    else:
        # Create dummy day of week cycling through 0-4 (Mon-Fri)
        day_of_week = pd.Series(np.arange(len(result_df)) % 5, index=result_df.index)
    
    for i in range(len(result_df)):
        gap_z = gap_zscore_base.iloc[i]
        day = day_of_week.iloc[i]
        
        if not np.isnan(gap_z):
            # Different multipliers by day (continuous scaling)
            if day == 0:  # Monday
                day_multiplier = 1.2  # Higher signal strength
            elif day == 4:  # Friday  
                day_multiplier = 0.8  # Lower signal strength
            else:  # Tue-Thu
                day_multiplier = 1.0
            
            # Apply day-of-week adjustment
            gap_zscore_time_conditional.iloc[i] = gap_z * day_multiplier
        else:
            gap_zscore_time_conditional.iloc[i] = 0
    
    result_df['Gap_ZScore_Time_Conditional'] = gap_zscore_time_conditional.fillna(0)
    
    # ============================================================================
    # 5. REVERSION/MOMENTUM VARIATION
    # ============================================================================
    
    
    gap_zscore_reversion_momentum = pd.Series(np.nan, index=result_df.index)
    
    # Detect trend vs range using recent price action
    price_momentum = result_df['Close'].pct_change(5)
    momentum_strength = np.abs(price_momentum)
    
    # Rolling correlation of price with time (trend indicator)
    time_series = np.arange(len(result_df))
    rolling_trend = pd.Series(np.nan, index=result_df.index)
    
    for i in range(20, len(result_df)):
        price_window = result_df['Close'].iloc[i-20:i]
        time_window = time_series[i-20:i]
        if len(price_window) == 20:
            correlation = np.corrcoef(price_window, time_window)[0,1]
            rolling_trend.iloc[i] = abs(correlation) if not np.isnan(correlation) else 0
    
    for i in range(len(result_df)):
        gap_z = gap_zscore_base.iloc[i]
        trend_strength = rolling_trend.iloc[i]
        momentum = momentum_strength.iloc[i]
        
        if not (np.isnan(gap_z) or np.isnan(trend_strength) or np.isnan(momentum)):
            
            # Create continuous multipliers based on market state
            if trend_strength > 0.3 and momentum > 0.02:  # Strong trending market
                # In trends, moderate gaps continue, extreme gaps reverse
                if abs(gap_z) < 1.5:
                    regime_multiplier = 1.0  # Momentum signal
                else:
                    regime_multiplier = 0.5  # Reduced signal for extreme gaps
                    
            else:  # Ranging/low momentum market
                # In ranges, all gaps get some reversal weighting
                regime_multiplier = 0.8  # Slight mean reversion bias
            
            # Apply regime-based adjustment
            gap_zscore_reversion_momentum.iloc[i] = gap_z * regime_multiplier
        else:
            gap_zscore_reversion_momentum.iloc[i] = 0
    
    result_df['Gap_ZScore_Reversion_Momentum'] = gap_zscore_reversion_momentum.fillna(0)
    
    # ============================================================================
    # 6. MULTI-TIMEFRAME VARIATION
    # ============================================================================
    
    
    
    # Short-term z-score
    short_mean = gap_pct.rolling(window=10).mean()
    short_std = gap_pct.rolling(window=10).std()
    gap_zscore_short = (gap_pct - short_mean) / short_std
    
    # Long-term z-score  
    long_mean = gap_pct.rolling(window=50).mean()
    long_std = gap_pct.rolling(window=50).std()
    gap_zscore_long = (gap_pct - long_mean) / long_std
    
    # Combined signal - weighted average with agreement bonus
    gap_zscore_multi_timeframe = pd.Series(np.nan, index=result_df.index)
    
    for i in range(len(result_df)):
        short_z = gap_zscore_short.iloc[i]
        long_z = gap_zscore_long.iloc[i]
        
        if not (np.isnan(short_z) or np.isnan(long_z)):
            # Weight short-term more heavily but add agreement bonus
            base_signal = 0.7 * short_z + 0.3 * long_z
            
            # Agreement bonus when both point same direction
            if (short_z > 0 and long_z > 0) or (short_z < 0 and long_z < 0):
                agreement_multiplier = 1.0 + min(abs(short_z * long_z) * 0.1, 0.5)
                gap_zscore_multi_timeframe.iloc[i] = base_signal * agreement_multiplier
            else:
                gap_zscore_multi_timeframe.iloc[i] = base_signal * 0.8  # Penalty for disagreement
        else:
            gap_zscore_multi_timeframe.iloc[i] = 0
    
    result_df['Gap_ZScore_Multi_Timeframe'] = gap_zscore_multi_timeframe.fillna(0)
    
    # ============================================================================
    # 7. ADAPTIVE THRESHOLD VARIATION
    # ============================================================================
    
    
    gap_zscore_adaptive_threshold = pd.Series(np.nan, index=result_df.index)
    
    # Calculate future returns for effectiveness measurement
    future_returns = returns.shift(-1)
    
    # Rolling effectiveness of gap signals
    rolling_effectiveness = pd.Series(0.5, index=result_df.index)  # Start with neutral effectiveness
    
    for i in range(100, len(result_df)-1):
        # Look back at recent gap signals and their effectiveness
        recent_gaps = gap_zscore_base.iloc[i-100:i]
        recent_returns = future_returns.iloc[i-100:i]
        
        # Calculate how well gaps predicted direction (continuous measure)
        valid_mask = ~(np.isnan(recent_gaps) | np.isnan(recent_returns))
        if valid_mask.sum() > 10:  # Need minimum occurrences
            correlation = np.corrcoef(recent_gaps[valid_mask], recent_returns[valid_mask])[0,1]
            if not np.isnan(correlation):
                # Convert correlation to effectiveness score (0 to 1)
                effectiveness = (correlation + 1) / 2  # Scale -1,1 to 0,1
                rolling_effectiveness.iloc[i] = effectiveness
    
    # Generate adaptive signals
    for i in range(len(result_df)):
        gap_z = gap_zscore_base.iloc[i]
        effectiveness = rolling_effectiveness.iloc[i]
        
        if not np.isnan(gap_z):
            # Adjust signal strength based on recent effectiveness
            effectiveness_multiplier = 0.5 + effectiveness  # Range: 0.5 to 1.5
            
            # Apply effectiveness-based adjustment
            gap_zscore_adaptive_threshold.iloc[i] = gap_z * effectiveness_multiplier
        else:
            gap_zscore_adaptive_threshold.iloc[i] = 0
    
    result_df['Gap_ZScore_Adaptive_Threshold'] = gap_zscore_adaptive_threshold.fillna(0)
    

    return result_df

# Price and volume based indicators
def calculate_price_volume_indicators(df):
    """Fixed version without data leakage"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Create all columns in a dictionary first
    new_columns = {}
    
    # FIX: Use expanding max instead of cummax to avoid future data
    expanding_max = close.expanding().max()
    new_columns['percent_from_high'] = ((close - expanding_max) / expanding_max) * 100
    new_columns['new_high'] = (close == expanding_max)
    
    # Days since high calculation - FIXED
    new_columns['days_since_high'] = (~new_columns['new_high']).cumsum() - \
                                   (~new_columns['new_high']).cumsum().where(new_columns['new_high']).ffill().fillna(0)
    
    new_columns['percent_range'] = (high - low) / close * 100
    
    # High-Close Ratio - this is okay as it uses same-day data
    new_columns['High_Close_Ratio'] = (high - close) / (close + 1e-10)
    
    # FIX: Use shifted data for normalization
    shifted_hc_ratio = new_columns['High_Close_Ratio'].shift(1)
    new_columns['High_Close_Ratio_norm'] = (
        new_columns['High_Close_Ratio'] - shifted_hc_ratio.rolling(50).mean()
    ) / shifted_hc_ratio.rolling(50).std()
    new_columns['High_Close_Ratio_norm'] = new_columns['High_Close_Ratio_norm'].clip(-3, 3)
    
    # FIX: VWAP calculations using shifted data
    typical_price_shifted = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    volume_shifted = volume.shift(1)
    
    new_columns['VWAP'] = (
        (typical_price_shifted * volume_shifted).rolling(window=14).sum() / 
        volume_shifted.rolling(window=14).sum()
    )
    
    new_columns['VWAP_std14'] = new_columns['VWAP'].rolling(window=14).std()
    new_columns['VWAP_std200'] = new_columns['VWAP'].rolling(window=20).std()
    new_columns['VWAP%'] = ((close - new_columns['VWAP']) / new_columns['VWAP']) * 100
    
    # FIX: VWAP from high using expanding max
    new_columns['VWAP%_from_high'] = ((new_columns['VWAP'] - expanding_max) / expanding_max) * 100
    
    # OBV calculation - this uses shifted close which is correct
    close_shift_1 = close.shift(1)
    obv_condition = close > close_shift_1
    new_columns['OBV'] = np.where(obv_condition, volume, -volume).cumsum()
    
    # Volume metrics - FIX: use shifted volume for rolling calculations
    new_columns['Volume_rolling_28'] = volume.shift(1).rolling(window=28).mean()
    new_columns['Volume_rolling_90'] = volume.shift(1).rolling(window=90).mean()
    new_columns['Volume%'] = ((volume - new_columns['Volume_rolling_28']) / new_columns['Volume_rolling_28']) * 100
    new_columns['Volume%_rolling_90'] = ((volume - new_columns['Volume_rolling_90']) / new_columns['Volume_rolling_90']) * 100
    new_columns['Volume_std'] = volume.shift(1).rolling(window=28).std()
    new_columns['Volume_lag_1'] = volume.shift(1)
    
    # Weighted velocity - FIX: use shifted price changes
    window = 10
    price_change = close.diff().shift(1).fillna(0)  # Shift the price changes
    weights = np.linspace(1, 0, window)
    weights /= np.sum(weights)
    weighted_velocity = price_change.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    new_columns['Weighted_Close_Change_Velocity'] = weighted_velocity
    
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)



def add_vg_indicators(df: pd.DataFrame, 
                      price_col: str = 'Close', 
                      volume_col: str = 'Volume',
                      lookback: int = 12,
                      volume_trend_threshold: float = 0.70,
                      volume_lookback: int = 50) -> pd.DataFrame:
    """
    Add visibility graph indicators and volume filter to dataframe.
    
    Args:
        df: Input dataframe with price and volume data
        price_col: Name of price column (default: 'Close')
        volume_col: Name of volume column (default: 'Volume') 
        lookback: Lookback period for VG calculation (default: 12)
        volume_trend_threshold: Threshold for volume trend filter (default: 0.70)
        volume_lookback: Lookback for volume trend calculation (default: 50)
    
    Returns:
        Enhanced dataframe with VG indicators and signals
    """
    
    df = df.copy()
    close_prices = df[price_col].to_numpy()
    
    # Detrend the price data for stationarity
    log_prices = np.log(close_prices)
    x = np.arange(len(log_prices))
    coeffs = np.polyfit(x, log_prices, 1)
    trend = coeffs[0] * x + coeffs[1]
    detrended_prices = log_prices - trend
    
    # Calculate volume trend filter
    if volume_col in df.columns:
        volume = df[volume_col].fillna(0)
        df['volume_trend'] = np.nan
        df['volume_trend_signal'] = 1
        
        for i in range(volume_lookback, len(volume)):
            volume_window = volume.iloc[i-volume_lookback:i+1]
            if len(volume_window) > 1 and volume_window.std() > 0:
                time_index = np.arange(len(volume_window))
                trend_corr = np.corrcoef(time_index, volume_window)[0, 1]
                df.loc[i, 'volume_trend'] = trend_corr
                
                if trend_corr > volume_trend_threshold:
                    df.loc[i, 'volume_trend_signal'] = 0
    else:
        print(f"Warning: {volume_col} column not found, skipping volume filter")
        df['volume_trend'] = np.nan
        df['volume_trend_signal'] = 1
    
    # Calculate visibility graph metrics
    pos_path = np.full(len(close_prices), np.nan)
    neg_path = np.full(len(close_prices), np.nan)
    
    for i in range(lookback, len(close_prices)):
        window_data = detrended_prices[i - lookback + 1: i + 1]
        
        try:
            pos_vg = NaturalVG()
            pos_vg.build(window_data)
            pos_nx = pos_vg.as_networkx()
            pos_path[i] = nx.average_shortest_path_length(pos_nx)
            
            neg_vg = NaturalVG()
            neg_vg.build(-window_data)
            neg_nx = neg_vg.as_networkx()
            neg_path[i] = nx.average_shortest_path_length(neg_nx)
            
        except:
            pos_path[i] = np.nan
            neg_path[i] = np.nan
    
    # Add VG indicators
    df['vg_pos'] = pos_path
    df['vg_neg'] = neg_path
    df['vg_diff'] = df['vg_pos'] - df['vg_neg']
    df['vg_abs_diff'] = np.abs(df['vg_diff'])
    
    # Add returns
    df['log_return'] = np.log(df[price_col]).diff().shift(-1)
    
    # Generate trading signals with volume filter
    df['vg_long_signal'] = 0
    df['vg_short_signal'] = 0
    
    long_condition = (df['vg_pos'] > df['vg_neg']) & (df['volume_trend_signal'] == 1)
    short_condition = (df['vg_pos'] < df['vg_neg']) & (df['volume_trend_signal'] == 1)
    
    df.loc[long_condition, 'vg_long_signal'] = 1
    df.loc[short_condition, 'vg_short_signal'] = -1
    df['vg_combined_signal'] = df['vg_long_signal'] + df['vg_short_signal']
    
    # Add returns per signal
    df['vg_long_return'] = df['vg_long_signal'] * df['log_return']
    df['vg_short_return'] = df['vg_short_signal'] * df['log_return']
    df['vg_combined_return'] = df['vg_combined_signal'] * df['log_return']
    
    return df



# Genetic indicators group
def calculate_genetic_indicators(df):
    epsilon = 1e-10
    
    for i in range(1, 8):
        df[f'High_Lag{i}'] = df['High'].shift(i) + epsilon
        df[f'Low_Lag{i}'] = df['Low'].shift(i) + epsilon
        df[f'Volume_Lag{i}'] = df['Volume'].shift(i) + epsilon
        df[f'Open_Lag{i}'] = df['Open'].shift(i) + epsilon

    df['G_Momentum_Confluence_Indicator'] = safe_divide(df['High_Lag2'], df['High_Lag2'] * df['Open'])
    df['G_Price_Gap_Analyzer'] = safe_divide(safe_log(df['Open_Lag2']), df['High_Lag1'])
    df['G_Triple_High_Trend_Indicator'] = safe_divide(df['High_Lag2'], df['High_Lag1'] * df['High'])
    df['G_Cyclical_Price_Oscillator'] = safe_divide(
        safe_divide(np.cos(safe_divide(df['High_Lag2'], df['High'])), df['High_Lag1']),
        np.sqrt(df['Open_Lag1'] + epsilon)
    )
    df['G_Volume_Adjusted_Price_Indicator'] = safe_divide(df['High_Lag2'], safe_divide(df['Volume'], df['Low_Lag1']))
    df['G_Adjusted_Close_Tracker'] = df['Adj Close']
    df['G_Volume_Weighted_High_Ratio'] = safe_divide(safe_divide(df['High'], df['High_Lag1']), safe_log(df['Volume'] + 1))
    df['G_High_Price_Momentum_Indicator'] = safe_divide(df['High'], (df['High_Lag1'] + df['High_Lag2']) / 2)
    df['G_Advanced_Trend_Synthesizer'] = (
        safe_log(safe_divide(df['High_Lag1'] + df['High_Lag5'], df['High_Lag1'])) *
        np.abs(safe_log(safe_divide(df['High_Lag2'], df['High_Lag2'])) - df['High_Lag7']) *
        safe_divide(df['Open_Lag2'], df['Close'])
    )
    df['G_Price_Volatility_Gauge'] = safe_divide(np.abs(df['High_Lag2'] - df['High']), df['Open'])
    df['G_Multi_Point_Price_Analyzer'] = np.abs(
        safe_divide(
            safe_divide(safe_log(safe_divide(df['High'], df['High_Lag2'])), safe_divide(df['Close'], df['High_Lag2'])),
            safe_divide(df['Close'], df['High_Lag2']) * safe_divide(df['Close'], df['High_Lag1'])
        )
    )
    df['G_Logarithmic_Trend_Detector'] = -safe_log(safe_divide(df['High_Lag2'], df['High_Lag4']))
    df['G_Complex_Price_Pattern_Indicator'] = safe_log(
        safe_divide(
            np.sqrt(np.sqrt(np.sqrt(np.sqrt(df['High_Lag7'] * df['High_Lag4'] + epsilon)))),
            df['Close']
        )
    )
    df['G_Log_Scaled_Price_Ratio'] = safe_log(safe_divide(df['High'], (df['High_Lag1'] + df['High_Lag2']) / 2))
    df['G_Volume_Price_Impact_Indicator'] = safe_divide(
        -df['High_Lag1'] + safe_divide(df['High_Lag3'], df['Close']),
        df['Volume']
    )
    df['G_Volume_Trend_Analyzer'] = safe_log(safe_divide(df['Volume'], df['Volume_Lag1']))
    df['G_Price_Open_Ratio_Indicator'] = safe_divide(
        safe_divide(safe_log(safe_divide(df['High_Lag2'], df['High'])), safe_divide(df['High'], df['Open_Lag2'])),
        safe_divide(df['High'], df['Open_Lag2'])
    )
    df['G_Price_Differential_Analyzer'] = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    df['G_Price_Volatility_Trend_Measure'] = 0.278 - np.abs(safe_divide(df['Low'], df['High']) / safe_divide(df['High_Lag5'], df['Low_Lag5']))
    df['G_Lagged_Price_Volume_Convergence'] = safe_divide(df['Low_Lag2'], (df['High_Lag2'] + safe_divide(0.791, df['Low'] * df['High'])))
    df['G_Price_Volume_Disparity_Index'] = np.abs(safe_divide(df['Low_Lag2'], df['High_Lag2'])) / (safe_divide(df['High_Lag5'], df['Low_Lag5']) / -0.2831)

    for i in range(1, 8):
        df = df.drop(columns=[f'High_Lag{i}', f'Low_Lag{i}', f'Volume_Lag{i}', f'Open_Lag{i}'])

    return df

# Kalman filter based indicators
def calculate_kalman_indicators(df):
    try:
        n = len(df)
        close_prices = df['Close'].values
        
        # Pre-allocate arrays
        kalman_values = np.full(n, np.nan)
        minima_values = np.full(n, np.nan)
        maxima_values = np.full(n, np.nan)
        support_pct = np.full(n, np.nan)
        resistance_pct = np.full(n, np.nan)
        
        # Initialize Kalman filter parameters
        transition_matrix = np.array([[1]])
        observation_matrix = np.array([[1]])
        transition_covariance = np.array([[0.01]])
        observation_covariance = np.array([[1]])
        initial_state_mean = close_prices[0]
        initial_state_covariance = np.array([[1]])
        
        # Calculate Kalman filter values
        current_state_mean = initial_state_mean
        current_state_covariance = initial_state_covariance
        
        for i in range(n):
            # Prediction step
            predicted_state_mean = np.dot(transition_matrix, current_state_mean)
            predicted_state_covariance = np.dot(np.dot(transition_matrix, current_state_covariance), transition_matrix.T) + transition_covariance
            
            # Update step with current observation
            kalman_gain = np.dot(
                np.dot(predicted_state_covariance, observation_matrix.T),
                np.linalg.inv(np.dot(np.dot(observation_matrix, predicted_state_covariance), observation_matrix.T) + observation_covariance)
            )
            
            current_state_mean = predicted_state_mean + np.dot(kalman_gain, (close_prices[i] - np.dot(observation_matrix, predicted_state_mean)))
            current_state_covariance = predicted_state_covariance - np.dot(np.dot(kalman_gain, observation_matrix), predicted_state_covariance)
            
            # Store the current filtered value
            kalman_values[i] = current_state_mean[0]
        
        # Compute extrema and percentages
        window_size = 140
        min_data_points = 20
        for i in range(min_data_points, n):
            # Use only lookback window for extrema detection
            lookback = min(window_size, i)
            lookback_start = max(0, i - lookback + 1)
            historical_window = kalman_values[lookback_start:i+1]
            
            # Find local minima/maxima in historical window
            if len(historical_window) >= 3:
                min_indices = argrelextrema(historical_window, np.less_equal, order=1)[0]
                max_indices = argrelextrema(historical_window, np.greater_equal, order=1)[0]
                
                # Process minima
                if len(min_indices) > 0:
                    most_recent_min_idx = min_indices[-1] + lookback_start
                    if most_recent_min_idx < i:
                        minima_values[i] = kalman_values[most_recent_min_idx]
                    elif i > 0:
                        minima_values[i] = minima_values[i-1]
                elif i > 0:
                    minima_values[i] = minima_values[i-1]
                
                # Process maxima
                if len(max_indices) > 0:
                    most_recent_max_idx = max_indices[-1] + lookback_start
                    if most_recent_max_idx < i:
                        maxima_values[i] = kalman_values[most_recent_max_idx]
                    elif i > 0:
                        maxima_values[i] = maxima_values[i-1]
                elif i > 0:
                    maxima_values[i] = maxima_values[i-1]
            elif i > 0:
                minima_values[i] = minima_values[i-1]
                maxima_values[i] = maxima_values[i-1]
            
            # Calculate percentages
            if not np.isnan(minima_values[i]) and minima_values[i] > 0:
                support_pct[i] = (close_prices[i] - minima_values[i]) / minima_values[i] * 100
                
            if not np.isnan(maxima_values[i]) and close_prices[i] > 0:
                resistance_pct[i] = (maxima_values[i] - close_prices[i]) / close_prices[i] * 100
        
        # Add to the original DataFrame
        df['Kalman'] = kalman_values
        df['minima'] = minima_values
        df['maxima'] = maxima_values
        df['Distance to Support (%)'] = support_pct
        df['Distance to Resistance (%)'] = resistance_pct
        
        # Additional Kalman calculations
        epsilon = 0.001
        df['Smoothed_Close'] = df['Kalman']
        df['Perturbed_Kalman'] = df['Kalman'] * (1 + epsilon)
        df['Divergence'] = np.abs(df['Perturbed_Kalman'] - df['Kalman'])
        df['Log_Divergence'] = np.log(df['Divergence'] + np.finfo(float).eps)
        df['Lyapunov_Exponent'] = df['Log_Divergence'].diff() / np.log(1 + epsilon)
        window_size = 14
        df['Lyapunov_Exponent_MA'] = df['Lyapunov_Exponent'].rolling(window=window_size).mean()
        
        # MA and percentage difference
        df['MA_200'] = df['Close'].rolling(window=200, min_periods=200).mean()
        df['Perc_Diff'] = (df['Kalman'] - df['MA_200']) / df['MA_200'] * 100
        
    except Exception as e:
        logging.error(f"Error in calculate_kalman_indicators: {str(e)}")
    
    return df

# Specialized volatility indicators
def calculate_volatility_indicators(df):
    # Mean reversion z-scores
    def add_multiple_mean_reversion_z_scores(data, price_column='Smoothed_Close', windows=[28, 90, 151], std_multipliers=[1, 1, 3]):
        # Create a dictionary to collect all values
        new_columns = {}
        
        for window, std_multiplier in zip(windows, std_multipliers):
            mean_col = f'Rolling_Mean_{window}'
            std_col = f'Rolling_Std_{window}'
            z_score_col = f'Mean_Reversion_Z_Score_{window}_std_{std_multiplier}'

            rolling_window = data[price_column].rolling(window=window)
            new_columns[mean_col] = rolling_window.mean()
            new_columns[std_col] = rolling_window.std()
            new_columns[z_score_col] = (data[price_column] - new_columns[mean_col]) / (new_columns[std_col] * std_multiplier)

        # Add all columns at once
        return pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    
    # Add complexity metrics
    def add_complexity_metrics(data, window_size=90):
        # Create a dictionary to collect all values
        new_columns = {}
        
        rolling_variance = data['Close'].rolling(window=window_size).var()
        new_columns['Complexity_Invariant_Distance'] = rolling_variance.diff().abs()
        new_columns['CID_Mean'] = new_columns['Complexity_Invariant_Distance'].rolling(window=window_size).mean()
        new_columns['CID_SD'] = new_columns['Complexity_Invariant_Distance'].rolling(window=window_size).std()
        
        # Add all columns at once
        return pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    
    # Calculate rolling indicators
    pct_change_close = df['Close'].pct_change()
    rolling_20 = df['Close'].rolling(window=20)
    rolling_14 = df['Close'].rolling(window=14)
    
    df['percent_change_Close'] = pct_change_close
    df['pct_change_std'] = rolling_20.std()
    df['percent_change_Close_lag_1'] = pct_change_close.shift(1)
    df['percent_change_Close_lag_5'] = pct_change_close.shift(5)
    df['percent_change_Close_lag_10'] = pct_change_close.shift(10)
    df['pct_change_std_rolling'] = rolling_20.mean()
    
    # Direction flipper calculations
    direction_flipper = (pct_change_close > 0).astype(int)
    df['direction_flipper_count5'] = direction_flipper.rolling(window=5).sum()
    df['direction_flipper_count_10'] = direction_flipper.rolling(window=10).sum()
    
    # Keltner Channel calculations
    keltner_central = df['Close'].ewm(span=20).mean()
    keltner_range = df['ATR'] * 1.5
    df['KC_UPPER%'] = ((keltner_central + keltner_range) - df['Close']) / df['Close'] * 100
    df['KC_LOWER%'] = (df['Close'] - (keltner_central - keltner_range)) / df['Close'] * 100
    
    # Apply additional indicator calculations
    if 'Smoothed_Close' in df.columns:
        df = add_multiple_mean_reversion_z_scores(df)
    df = add_complexity_metrics(df)
    
    return df

# Composite and significant indicators
def calculate_significant_indicators(df):
    df = df.copy()
    
    epsilon = 1e-10
    new_data = {}
    # Ensure we don't recreate columns that already exist
    if 'High_Close_Ratio' in df.columns:
        new_data['HC_Ratio'] = pd.Series((df['High'] - df['Close']) / (df['Close'] + epsilon), index=df.index)
    else:
        new_data['High_Close_Ratio'] = pd.Series((df['High'] - df['Close']) / (df['Close'] + epsilon), index=df.index)
    
    # Create columns for distances
    distance_from_peak = np.zeros(len(df))
    distance_from_trough = np.zeros(len(df))
    
    for ticker in df['Ticker'].unique():
        ticker_mask = df['Ticker'] == ticker
        ticker_indices = df.index[ticker_mask]
        
        min_max_window = 10
        
        # Process each point using only past data
        for i in range(min_max_window, len(ticker_indices)):
            current_idx = ticker_indices[i]
            # Get historical window up to and including current point
            historical_window_indices = ticker_indices[max(0, i-50):i+1]
            historical_prices = df.loc[historical_window_indices, 'Close'].values
            
            # Find peaks and troughs in the historical window only
            local_max_indices = argrelextrema(historical_prices, np.greater, order=min_max_window)[0]
            local_min_indices = argrelextrema(historical_prices, np.less, order=min_max_window)[0]
            
            # Translate indices within the window to dataframe indices
            local_max_df_indices = local_max_indices + max(0, i-50)
            local_min_df_indices = local_min_indices + max(0, i-50)
            
            # Make sure we don't consider the current point itself as a peak/trough
            local_max_df_indices = local_max_df_indices[local_max_df_indices < i]
            local_min_df_indices = local_min_df_indices[local_min_df_indices < i]
            
            current_price = df.loc[current_idx, 'Close']
            
            # Calculate distance from nearest peak
            if len(local_max_df_indices) > 0:
                nearest_peak_idx = ticker_indices[local_max_df_indices[-1]]
                peak_price = df.loc[nearest_peak_idx, 'Close']
                distance_from_peak[current_idx] = (current_price - peak_price) / peak_price
            
            # Calculate distance from nearest trough
            if len(local_min_df_indices) > 0:
                nearest_trough_idx = ticker_indices[local_min_df_indices[-1]]
                trough_price = df.loc[nearest_trough_idx, 'Close']
                distance_from_trough[current_idx] = (current_price - trough_price) / trough_price
    
    # Store distances in DataFrame as Series
    distance_from_peak_series = pd.Series(distance_from_peak, index=df.index)
    distance_from_trough_series = pd.Series(distance_from_trough, index=df.index)
    
    # Check which ratio to use
    ratio_column = 'High_Close_Ratio' if 'High_Close_Ratio' in new_data else 'HC_Ratio'
    
    # Calculate Pattern_Indicator
    new_data['Pattern_Indicator'] = pd.Series(-1 * distance_from_peak_series * new_data[ratio_column], index=df.index)
    
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        new_data['RSI'] = pd.Series(100 - (100 / (1 + rs)), index=df.index)
    else:
        new_data['RSI'] = df['RSI'].copy()

    # Calculate the composite indicators - ensuring we always have Series
    new_data['RSI_HC_Composite'] = pd.Series(new_data[ratio_column] * (100 - new_data['RSI']) / 50, index=df.index)
    
    # Create a temporary DataFrame to handle Series operations properly
    temp_df = pd.DataFrame(new_data, index=df.index)
    
    temp_df['RSI_HC_Composite_norm'] = (
        temp_df['RSI_HC_Composite'] - 
        temp_df['RSI_HC_Composite'].shift(1).rolling(50).mean()
    ) / temp_df['RSI_HC_Composite'].shift(1).rolling(50).std()
    
    temp_df['RSI_HC_Composite_norm'] = temp_df['RSI_HC_Composite_norm'].clip(-3, 3)
    
    if 'Lyapunov_Scaled' not in df.columns:
        lyapunov_data = {}
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            indices = df.index[ticker_mask]
            close_volatility = df.loc[ticker_mask, 'Close'].pct_change().rolling(50, min_periods=10).std()
            
            # Calculate Lyapunov with lagged statistics
            rolling_mean = close_volatility.shift(1).rolling(50, min_periods=10).mean()
            rolling_std = close_volatility.shift(1).rolling(50, min_periods=10).std()
            
            # Apply z-score normalization using only past data
            lyapunov_scaled = (close_volatility - rolling_mean) / rolling_std.replace(0, 1)
            
            # Clip and scale
            lyapunov_scaled = lyapunov_scaled.clip(-3, 3) / 3
            
            for idx, value in zip(indices, lyapunov_scaled):
                lyapunov_data[idx] = value
        
        # Convert to Series
        temp_df['Lyapunov_Scaled'] = pd.Series(lyapunov_data)
    else:
        temp_df['Lyapunov_Scaled'] = df['Lyapunov_Scaled'].copy()

    # Now we're using the DataFrame to ensure we have Series for shift operations
    temp_df['HC_Predict_Regime'] = np.where(
        temp_df['Lyapunov_Scaled'].shift(1) < -0.3,
        temp_df[ratio_column] * 1.5,
        np.where(
            temp_df['Lyapunov_Scaled'].shift(1) > 0.3,
            temp_df[ratio_column] * 0.5,
            temp_df[ratio_column]
        )
    )
    
    temp_df['HC_Predict_Regime_norm'] = (
        temp_df['HC_Predict_Regime'] - 
        temp_df['HC_Predict_Regime'].shift(1).rolling(50).mean()
    ) / temp_df['HC_Predict_Regime'].shift(1).rolling(50).std()
    
    temp_df['HC_Predict_Regime_norm'] = temp_df['HC_Predict_Regime_norm'].clip(-3, 3)
    
    if 'G_Momentum_Confluence_Indicator' in df.columns and 'G_Lagged_Price_Volume_Convergence' in df.columns:
        temp_df['HC_GP_Composite'] = (
            temp_df[ratio_column] * 
            df['G_Momentum_Confluence_Indicator'] * 
            (1 + df['G_Lagged_Price_Volume_Convergence'])
        )
    
    sig_indicators = ['Pattern_Indicator', 'RSI_HC_Composite', ratio_column, 'HC_Predict_Regime']
    sig_indicators = [i for i in sig_indicators if i in temp_df.columns]
    
    # Normalize each indicator before combining
    for indicator in sig_indicators:
        norm_name = f"{indicator}_norm"
        if norm_name not in temp_df.columns:
            temp_df[norm_name] = (
                temp_df[indicator] - 
                temp_df[indicator].shift(1).rolling(50).mean()
            ) / temp_df[indicator].shift(1).rolling(50).std()
            
            temp_df[norm_name] = temp_df[norm_name].clip(-3, 3)
    
    norm_columns = [f"{i}_norm" for i in sig_indicators if f"{i}_norm" in temp_df.columns]
    if len(norm_columns) > 0:
        temp_df['Significant_Indicators_Ensemble'] = temp_df[norm_columns].mean(axis=1)
    
    # Return the combined DataFrame
    return pd.concat([df, temp_df], axis=1)


    
# Main indicator calculation function



def add_trading_signal_features(df):
    """
    Add trading signal features based on can_buy logic to the indicators pipeline.
    These features allow the ML model to learn the filtering patterns.
    All features are numerical and relative rather than hardcoded thresholds.
    """
    
    if len(df) < 50:
        return df
    
    # Initialize new columns dictionary
    features = {}
    
    # === VIX REGIME FEATURES ===
    if 'VIX_Close' in df.columns:
        features.update(calculate_vix_regime_features(df))
    
    # === LIQUIDITY AND VOLUME FEATURES ===
    features.update(calculate_liquidity_features(df))
    
    # === VOLATILITY AND ATR FEATURES ===
    features.update(calculate_volatility_atr_features(df))
    
    # === PRICE MOMENTUM FEATURES ===
    features.update(calculate_price_momentum_features(df))
    
    # === VOLUME MOMENTUM FEATURES ===
    features.update(calculate_volume_momentum_features(df))
    
    # === MARKET REGIME FEATURES ===
    features.update(calculate_market_regime_features(df))
    
    # === PRICE ACTION AND GAP FEATURES ===
    features.update(calculate_price_action_features(df))
    
    # === COEFFICIENT OF VARIATION FEATURES ===
    features.update(calculate_volatility_cv_features(df))
    
    # Convert to DataFrame and add to original
    features_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, features_df], axis=1)


def calculate_vix_regime_features(df):
    """Extract VIX-based regime features"""
    features = {}
    
    current_vix = df['VIX_Close']
    
    # Rolling VIX statistics (using only past data)
    vix_ma_20 = current_vix.shift(1).rolling(20, min_periods=10).mean()
    vix_ma_50 = current_vix.shift(1).rolling(50, min_periods=20).mean()
    vix_std_20 = current_vix.shift(1).rolling(20, min_periods=10).std()
    
    # VIX percentile rank (relative measure)
    vix_window_252 = current_vix.shift(1).rolling(252, min_periods=50)
    features['vix_percentile_rank'] = vix_window_252.apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100 if len(x) > 0 else 50, raw=False
    )
    
    # VIX regime scaling factor (continuous)
    features['vix_regime_scale'] = np.where(
        features['vix_percentile_rank'] >= 80, 1.20,
        np.where(features['vix_percentile_rank'] >= 60, 1.10,
                np.where(features['vix_percentile_rank'] <= 20, 0.95, 1.0))
    )
    
    # VIX momentum and acceleration
    features['vix_momentum_5d'] = current_vix.pct_change(5)
    features['vix_acceleration'] = features['vix_momentum_5d'].diff()
    
    # VIX z-score
    features['vix_zscore'] = (current_vix - vix_ma_20) / (vix_std_20 + 1e-8)
    
    # VIX regime transitions
    features['vix_regime_low'] = (features['vix_percentile_rank'] <= 25).astype(float)
    features['vix_regime_high'] = (features['vix_percentile_rank'] >= 75).astype(float)
    
    return features






def calculate_liquidity_features(df):
    """Extract liquidity and volume-related features"""
    features = {}
    
    current_close = df['Close']
    current_volume = df['Volume']
    
    # Dollar volume calculations
    current_dollar_volume = current_volume * current_close
    
    # Historical dollar volume statistics (using shifted data)
    dollar_vol_shifted = current_dollar_volume.shift(1)
    features['dollar_volume_ma_10'] = dollar_vol_shifted.rolling(10, min_periods=5).mean()
    features['dollar_volume_ma_252'] = dollar_vol_shifted.rolling(252, min_periods=50).mean()
    features['dollar_volume_std_252'] = dollar_vol_shifted.rolling(252, min_periods=50).std()
    
    # Current dollar volume relative measures
    features['dollar_volume_ratio_10d'] = current_dollar_volume / (features['dollar_volume_ma_10'] + 1e-8)
    features['dollar_volume_ratio_252d'] = current_dollar_volume / (features['dollar_volume_ma_252'] + 1e-8)
    features['dollar_volume_zscore'] = (
        current_dollar_volume - features['dollar_volume_ma_252']
    ) / (features['dollar_volume_std_252'] + 1e-8)
    
    # Volume percentile rank
    vol_window_252 = dollar_vol_shifted.rolling(252, min_periods=50)
    features['dollar_volume_percentile'] = vol_window_252.apply(
        lambda x: (x <= current_dollar_volume.iloc[x.index[-1]]).mean() * 100 if len(x) > 0 else 50, 
        raw=False
    )
    
    # Volume trend analysis (5d vs 20d)
    features['volume_trend_5d_20d'] = (
        dollar_vol_shifted.rolling(5, min_periods=3).mean() / 
        (dollar_vol_shifted.rolling(20, min_periods=10).mean() + 1e-8)
    )
    
    # Short-term liquidity stress (3d vs median)
    features['liquidity_stress_3d'] = (
        dollar_vol_shifted.rolling(3, min_periods=2).mean() / 
        (dollar_vol_shifted.rolling(252, min_periods=50).median() + 1e-8)
    )
    
    # Volume spike detection (relative to recent average)
    vol_10d_avg = current_volume.shift(1).rolling(10, min_periods=5).mean()
    features['volume_spike_ratio'] = current_volume / (vol_10d_avg + 1e-8)
    
    # Sustained volume burst (count of high volume days in last 5)
    volume_burst_threshold = vol_10d_avg * 3
    recent_volumes = current_volume.rolling(5, min_periods=3)
    features['sustained_volume_burst_count'] = recent_volumes.apply(
        lambda x: (x > volume_burst_threshold.iloc[x.index[-1]]).sum() if len(x) > 0 else 0,
        raw=False
    )

    return features


def calculate_volatility_atr_features(df):
    """Extract ATR and volatility-related features"""
    features = {}
    
    # True Range calculation (using shifted close)
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    features['atr_14'] = true_range.rolling(14, min_periods=10).mean()
    features['atr_percentage'] = features['atr_14'] / df['Close']
    
    # ATR percentile rank
    atr_pct_shifted = features['atr_percentage'].shift(1)
    atr_window_252 = atr_pct_shifted.rolling(252, min_periods=50)
    features['atr_percentile_rank'] = atr_window_252.apply(
        lambda x: (x <= features['atr_percentage'].iloc[x.index[-1]]).mean() * 100 if len(x) > 0 else 50,
        raw=False
    )
    
    # Gap analysis
    current_open = df['Open']
    prev_close = df['Close'].shift(1)
    gap_pct = (current_open - prev_close) / (prev_close + 1e-8)
    features['gap_percentage'] = gap_pct
    features['gap_absolute'] = np.abs(gap_pct)
    
    # Gap in ATR terms
    prev_atr_pct = features['atr_percentage'].shift(1)
    features['gap_in_atr_terms'] = gap_pct / (prev_atr_pct + 1e-8)
    
    # ATR regime classification
    features['atr_regime_low'] = (features['atr_percentile_rank'] <= 25).astype(float)
    features['atr_regime_high'] = (features['atr_percentile_rank'] >= 75).astype(float)

    return features


def calculate_price_momentum_features(df):
    """Extract price momentum features"""
    features = {}
    
    close = df['Close']
    
    # Multi-timeframe returns
    for period in [3, 5, 10, 20]:
        features[f'return_{period}d'] = close.pct_change(period)
        features[f'return_{period}d_abs'] = np.abs(features[f'return_{period}d'])

    # Momentum strength indicators
    features['momentum_strength_5d'] = np.where(features['return_5d'] > 0.005, 1.0, 0.0)
    features['momentum_strength_continuous'] = np.tanh(features['return_5d'] * 100)  # Continuous version
    
    # Rapid price movement detection
    features['rapid_price_change_5d'] = close / close.shift(5) - 1
    features['rapid_price_change_zscore'] = (
        features['rapid_price_change_5d'] - 
        features['rapid_price_change_5d'].shift(1).rolling(50, min_periods=20).mean()
    ) / (features['rapid_price_change_5d'].shift(1).rolling(50, min_periods=20).std() + 1e-8)
    
    # Price acceleration
    features['price_acceleration'] = features['return_5d'] - features['return_5d'].shift(5)
    
    # Trend consistency (what % of last N days were positive)
    for window in [5, 10, 20]:
        daily_returns = close.pct_change()
        features[f'trend_consistency_{window}d'] = (
            daily_returns.rolling(window, min_periods=window//2).apply(
                lambda x: (x > 0).mean(), raw=True
            )
        )
    
    return features


def calculate_volume_momentum_features(df):
    """Extract volume momentum features"""
    features = {}
    
    volume = df['Volume']
    close = df['Close']
    dollar_volume = volume * close
    
    # Dollar volume momentum (5d vs 20d comparison)
    dv_5d = dollar_volume.shift(1).rolling(5, min_periods=3).mean()
    dv_20d = dollar_volume.shift(1).rolling(20, min_periods=10).mean()
    features['volume_momentum_ratio'] = dv_5d / (dv_20d + 1e-8)
    
    # Volume trend strength
    features['volume_trend_strength'] = np.where(
        features['volume_momentum_ratio'] > 1.0, 
        np.log(features['volume_momentum_ratio']), 
        -np.log(1.0 / (features['volume_momentum_ratio'] + 1e-8))
    )
    
    # Volume persistence (how many days has volume been above/below average)
    vol_ma_20 = volume.shift(1).rolling(20, min_periods=10).mean()
    vol_above_avg = volume > vol_ma_20
    features['volume_persistence'] = vol_above_avg.rolling(10, min_periods=5).sum()
    
    # Volume volatility
    vol_returns = volume.pct_change()
    features['volume_volatility'] = vol_returns.rolling(20, min_periods=10).std()
    
    return features


def calculate_market_regime_features(df):
    """Extract market regime features"""
    features = {}
    
    close = df['Close']
    
    # Price position in recent ranges
    for window in [20, 50, 100]:
        high_window = df['High'].rolling(window, min_periods=window//2).max()
        low_window = df['Low'].rolling(window, min_periods=window//2).min()
        price_range = high_window - low_window
        features[f'price_position_{window}d'] = (
            (close - low_window) / (price_range + 1e-8)
        )
    
    # Trend direction using multiple timeframes
    for window in [10, 20, 50]:
        sma = close.rolling(window, min_periods=window//2).mean()
        features[f'trend_direction_{window}d'] = np.where(close > sma, 1.0, -1.0)
        features[f'distance_from_sma_{window}d'] = (close - sma) / (sma + 1e-8)
    
    # Market regime composite (trend + volatility)
    vol_regime = df.get('vix_regime_scale', pd.Series(1.0, index=df.index))
    trend_regime = features.get('trend_direction_20d', pd.Series(0.0, index=df.index))
    features['market_regime_composite'] = vol_regime * trend_regime
    
    return features


def calculate_price_action_features(df):
    """Extract price action features"""
    features = {}
    
    close = df['Close']
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    
    # Price floor relative measure (distance from minimum viable price)
    features['price_quality_score'] = np.log(close / 1.10) if (close > 1.10).all() else np.log(close / close.min())
    
    # Intraday range characteristics
    features['intraday_range_pct'] = (high - low) / close
    features['open_close_ratio'] = (close - open_price) / (open_price + 1e-8)
    features['high_close_ratio'] = (high - close) / (close + 1e-8)
    features['low_close_ratio'] = (close - low) / (close + 1e-8)
    
    # Price action patterns
    features['doji_pattern'] = np.where(
        np.abs(features['open_close_ratio']) < 0.001, 1.0, 0.0
    )
    features['hammer_pattern'] = np.where(
        (features['low_close_ratio'] > features['high_close_ratio'] * 2) &
        (features['open_close_ratio'] > 0), 1.0, 0.0
    )

    return features


def calculate_volatility_cv_features(df):
    """Extract coefficient of variation features"""
    features = {}
    
    close = df['Close']
    
    # Coefficient of variation for different windows
    for window in [10, 20, 50]:
        close_window = close.rolling(window, min_periods=window//2)
        mean_price = close_window.mean()
        std_price = close_window.std()
        features[f'cv_{window}d'] = std_price / (mean_price + 1e-8)
        
        # Historical CV comparison
        cv_shifted = features[f'cv_{window}d'].shift(1)
        cv_window_252 = cv_shifted.rolling(252, min_periods=50)
        features[f'cv_{window}d_percentile'] = cv_window_252.apply(
            lambda x: (x <= features[f'cv_{window}d'].iloc[x.index[-1]]).mean() * 100 if len(x) > 0 else 50,
            raw=False
        )
        
        # CV regime classification
        features[f'cv_{window}d_regime_high'] = (features[f'cv_{window}d_percentile'] >= 80).astype(float)
        features[f'cv_{window}d_regime_low'] = (features[f'cv_{window}d_percentile'] <= 20).astype(float)

    return features


def create_trading_signal_composite_scores(df):
    """
    Create composite scores that summarize the trading signal strength.
    These can be used as high-level features for the ML model.
    """
    
    features = {}
    
    # Liquidity score (0-1, higher is better liquidity)
    liquidity_components = [
        df.get('dollar_volume_percentile', 50) / 100,
        np.clip(df.get('volume_trend_5d_20d', 1.0), 0, 2) / 2,
        1 - np.clip(df.get('liquidity_stress_3d', 1.0), 0, 2) / 2
    ]
    features['liquidity_score'] = np.mean(liquidity_components, axis=0)
    
    # Volatility score (0-1, higher indicates better trading conditions)
    vol_components = [
        df.get('vix_percentile_rank', 50) / 100,
        np.clip(df.get('atr_percentile_rank', 50), 20, 80) / 100,
        1 - np.clip(df.get('cv_20d_percentile', 50), 0, 90) / 100
    ]
    features['volatility_score'] = np.mean(vol_components, axis=0)
    
    # Momentum score (0-1, higher indicates strong momentum)
    momentum_components = [
        np.clip(df.get('momentum_strength_continuous', 0), -1, 1) / 2 + 0.5,
        np.clip(df.get('volume_momentum_ratio', 1.0), 0.5, 1.5) / 2,
        np.clip(df.get('trend_consistency_10d', 0.5), 0, 1)
    ]
    features['momentum_score'] = np.mean(momentum_components, axis=0)
    
    # Overall trading signal score
    features['trading_signal_composite'] = (
        features['liquidity_score'] * 0.3 +
        features['volatility_score'] * 0.4 +
        features['momentum_score'] * 0.3
    )
    
    return features


# Integration function to add to your indicators pipeline
def add_can_buy_features_to_indicators(df):
    """
    Main function to add all trading signal features to your indicators pipeline.
    Call this at the end of your indicators() function.
    """
    
    # Add the basic trading signal features
    df = add_trading_signal_features(df)
    
    # Add composite scores
    composite_features = create_trading_signal_composite_scores(df)
    composite_df = pd.DataFrame(composite_features, index=df.index)
    df = pd.concat([df, composite_df], axis=1)
    
    # Fill any NaN values with sensible defaults
    trading_feature_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in [
            'vix_', 'dollar_volume_', 'atr_', 'gap_', 'return_', 'momentum_', 
            'volume_', 'trend_', 'price_', 'cv_', 'liquidity_', 'trading_signal_'
        ]
    )]
    
    for col in trading_feature_columns:
        if col in df.columns:
            # Fill with forward fill first, then with column median
            df[col] = df[col].ffill().fillna(df[col].median())
    
    return df



def gap_indicators(df, vol_percentile=75, extreme_pct=10, lookback=20, expanding_window=None):
    
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.sort_index()
    
    prev_close = df['Close'].shift(1)
    gap_pct = (df['Open'] - prev_close) / prev_close * 100
    
    daily_range = df['High'] - df['Low']
    close_to_low = df['Close'] - df['Low']
    close_position_raw = close_to_low / daily_range.replace(0, np.nan)
    df['close_position'] = close_position_raw.fillna(0.5).clip(0, 1)
    
    df['gap_volatility'] = gap_pct.rolling(lookback, min_periods=2).std()
    
    if expanding_window is not None:
        df['vol_threshold'] = df['gap_volatility'].shift(1).rolling(
            window=expanding_window, min_periods=expanding_window
        ).quantile(vol_percentile / 100)
        
        df['close_low_threshold'] = df['close_position'].shift(1).rolling(
            window=expanding_window, min_periods=expanding_window
        ).quantile(extreme_pct / 100)
        
        df['close_high_threshold'] = df['close_position'].shift(1).rolling(
            window=expanding_window, min_periods=expanding_window
        ).quantile(1 - extreme_pct / 100)
    else:
        df['vol_threshold'] = df['gap_volatility'].shift(1).expanding(
            min_periods=252
        ).quantile(vol_percentile / 100)
        
        df['close_low_threshold'] = df['close_position'].shift(1).expanding(
            min_periods=252
        ).quantile(extreme_pct / 100)
        
        df['close_high_threshold'] = df['close_position'].shift(1).expanding(
            min_periods=252
        ).quantile(1 - extreme_pct / 100)
    
    df['high_vol'] = (df['gap_volatility'] > df['vol_threshold']).astype(int)
    
    df['long_signal'] = (
        (df['close_position'] < df['close_low_threshold']) & 
        (df['high_vol'] == 1)
    ).astype(int)
    
    df['short_signal'] = (
        (df['close_position'] > df['close_high_threshold']) & 
        (df['high_vol'] == 1)
    ).astype(int)
    
    return df





##data clean up and preciscion control
def final_cleanup(df):
    # Remove duplicate columns before final cleanup
    if len(df.columns) != len(df.columns.unique()):
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        
    # Final cleanup
    columns_to_drop = ['Adj Close', 'ATZ_Upper', 'ATZ_Lower', 'VWAP', '200DAY_ATR-', '200DAY_ATR', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = interpolate_columns(df, max_gap_fill=50)
    df = df.iloc[200:]
    df = df.drop(columns=columns_to_drop, axis=1)
    df = df.round(8)

    return df


def forward_fill(df):
    df = df.copy()
    
    # Forward-fill and convert data types
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Volume'] = df['Volume'].ffill()

    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    return df


# Matrix-power indicators
# 14 mp_* features specified by Data/matrix_power_spec.json. Each builds a DxD
# matrix M[t] from sign-weighted, panel-normalized OHLCV primitives over a
# rolling window, applies a variant (raw/sym/skew), computes expm(M[t]) via
# scaling-and-squaring + Taylor expansion, and extracts a scalar
# (trace, Frobenius norm, or top-left element).
#
# Output column names are `mp_<spec_name>` where <spec_name> matches the
# `name` field in the JSON spec, e.g. `mp_d4_b65_sym_frob`.
#
# Sanity-check IC reproduction with analysis_output/_validate_matrix_power_full.py.
import json as _mp_json

_MP_SPEC_PATH = "Data/matrix_power_spec.json"
_MP_SPEC_CACHE = None


def _mp_load_spec():
    global _MP_SPEC_CACHE
    if _MP_SPEC_CACHE is None:
        with open(_MP_SPEC_PATH, "r") as _f:
            _MP_SPEC_CACHE = _mp_json.load(_f)
    return _MP_SPEC_CACHE


def _mp_compute_primitives(df, window):
    """14 causal OHLCV primitives. Returns ndarray (N, 14) in spec order."""
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    C = df["Close"].astype(np.float64)
    V = df["Volume"].astype(np.float64)
    log_ret = np.log(C).diff()
    log_vol = np.log(V.replace(0, np.nan)).diff()
    range_pct = (H - L) / C
    body_pct = (C - O) / C
    upper_wick = (H - np.maximum(O, C)) / C
    lower_wick = (np.minimum(O, C) - L) / C
    rv = log_ret.rolling(window).std()
    z_close = (C - C.rolling(window).mean()) / (C.rolling(window).std() + 1e-9)
    z_vol = (V - V.rolling(window).mean()) / (V.rolling(window).std() + 1e-9)
    skew_p = log_ret.rolling(window).skew()
    kurt_p = log_ret.rolling(window).kurt()
    mom5 = np.log(C / C.shift(5))
    mom20 = np.log(C / C.shift(window))
    sign_last = np.sign(log_ret)
    # Column order MUST match spec['primitive_names'].
    return np.column_stack([
        log_ret.to_numpy(), log_vol.to_numpy(),
        range_pct.to_numpy(), body_pct.to_numpy(),
        upper_wick.to_numpy(), lower_wick.to_numpy(),
        rv.to_numpy(), z_close.to_numpy(), z_vol.to_numpy(),
        skew_p.to_numpy(), kurt_p.to_numpy(),
        mom5.to_numpy(), mom20.to_numpy(),
        sign_last.to_numpy(),
    ])


def _mp_expm_batch(M_batch, K=16):
    """Batched matrix exponential. M_batch: (N, D, D). Returns (N, D, D)."""
    N, D, _ = M_batch.shape
    norms = np.linalg.norm(M_batch, ord="fro", axis=(1, 2))
    nmax = float(norms.max()) if norms.size else 0.0
    s = max(0, int(np.ceil(np.log2(max(nmax, 1e-9)))))
    scale = 2.0 ** s
    M_s = M_batch / scale
    I = np.broadcast_to(np.eye(D, dtype=M_batch.dtype), (N, D, D)).copy()
    result = I.copy()
    term = I.copy()
    for k in range(1, K + 1):
        term = (term @ M_s) / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result


def add_matrix_power_features(df):
    """Append 14 mp_* features to df. Rows before the rolling window are NaN."""
    spec = _mp_load_spec()
    window = spec["window"]
    scales = np.asarray(spec["primitive_scales"], dtype=np.float64)
    scales = np.where(scales > 0, scales, 1.0)

    prim_arr = _mp_compute_primitives(df, window)  # (N, 14)
    # Rows where any primitive is non-finite must produce NaN output, otherwise
    # the zero-replaced values below would yield expm(0) = I — bogus signal.
    nan_mask = ~np.isfinite(prim_arr).all(axis=1)
    # Panel-normalize and bound entries to [-1, 1] via tanh (matches EDA).
    prim_norm = np.tanh(prim_arr / scales)
    prim_norm = np.nan_to_num(prim_norm, nan=0.0, posinf=0.0, neginf=0.0)

    N = prim_norm.shape[0]
    for feat in spec["features"]:
        D = int(feat["d"])
        idx_mat = np.asarray(feat["idx_matrix"], dtype=np.int64)
        sgn_mat = np.asarray(feat["sign_matrix"], dtype=np.float64)
        # M[t, i, j] = sgn_mat[i, j] * prim_norm[t, idx_mat[i, j]]
        M_all = sgn_mat[None, :, :] * prim_norm[:, idx_mat]
        variant = feat["variant"]
        if variant == "sym":
            M_all = 0.5 * (M_all + M_all.transpose(0, 2, 1))
        elif variant == "skew":
            M_all = 0.5 * (M_all - M_all.transpose(0, 2, 1))
        elif variant != "raw":
            raise ValueError(f"Unknown matrix-power variant: {variant!r}")
        try:
            E_all = _mp_expm_batch(M_all)
        except Exception:
            E_all = np.broadcast_to(np.eye(D, dtype=np.float64), (N, D, D)).copy()
        extractor = feat["extractor"]
        if extractor == "trace":
            vals = E_all.trace(axis1=1, axis2=2).real
        elif extractor == "frob":
            vals = np.linalg.norm(E_all, ord="fro", axis=(1, 2))
        elif extractor == "top_left":
            v00 = E_all[:, 0, 0]
            vals = v00.real if np.iscomplexobj(v00) else v00
        else:
            raise ValueError(f"Unknown matrix-power extractor: {extractor!r}")
        vals = np.where(nan_mask, np.nan, vals)
        df[f"mp_{feat['name']}"] = vals
    return df


def indicators(df):
    df = forward_fill(df)
    df = df.copy()  # defragment after forward_fill before adding ~300 new columns

    df = add_genetic_info_decay_3d(df)
    df = add_genetic_autocorr_3d(df)
    df = add_volume_spectral_splatter(df)
    df = add_genetic_log_zscore_20d(df)
    df = add_genetic_signal_strength(df)

    # Calculate all indicator groups
    df = calculate_genetic_indicators(df)
    df = calculate_parabolic_SAR(df)
    df = calculate_moving_average_indicators(df)
    df = calculate_price_volume_indicators(df)
    df = calculate_kalman_indicators(df)
    df = calculate_volatility_indicators(df)
    df = calculate_significant_indicators(df)

    df = add_price_differential_ratio(df, epsilon = 1e-10)
    df = add_top_stress_indicators(df)
    base_column = 'Price_Differential_Ratio'

    df = add_rolling_log_scaled_features(df, base_column)
    df = add_rate_of_change_features(df, base_column)
    df = add_z_score_signals(df, base_column)
    df = add_momentum_signals(df, base_column)
    df = add_trend_reversal_signals(df, base_column)
    df = add_volatility_regime_signals(df, base_column)
    df = add_signal_persistence_features(df, base_column)
    df = add_composite_signal_strength(df, base_column)
    df = add_information_efficiency_features(df, base_column)

    df = calculate_vix_features(df)
    df = calculate_dvamr_probability(df)
    #df = calculate_vol_fade_signal(df)

    df = calculate_beta(df, window=60, min_periods=30)
    df = df.copy()  # defragment mid-way before the final GP feature block

    df = add_can_buy_features_to_indicators(df)

    df = compute_gp_input_features(df)

    df = add_matrix_power_features(df)

    df = final_cleanup(df)
    return df









# Additional indicator calculation functions
def calculate_parabolic_SAR(df):
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Initialize SAR
    sar = low[0]
    ep = high[0]
    af = 0.02
    sar_values = [sar]

    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if close[i] > close[i - 1]:
            af = min(af + 0.02, 0.2)
        else:
            af = 0.02

        if close[i] > close[i - 1]:
            ep = max(high[i], ep)
        else:
            ep = min(low[i], ep)

        sar = min(sar, low[i], low[i - 1]) if close[i] > close[i - 1] else max(sar, high[i], high[i - 1])
        sar_values.append(sar)

    df['Parabolic_SAR'] = sar_values
    return df

# File processing functions
def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True

def DataQualityCheck(df, all_dfs=None):
    quality_issues = []
    
    if df is None or df.empty:
        logging.error("DataFrame is empty.")
        return None
    
    if len(df) < 201:
        quality_issues.append(f"Insufficient data points: {len(df)}/201 minimum required.")
    
    price_variance = df[['Open', 'High', 'Low', 'Close']].std().sum()
    if price_variance < 0.01:
        quality_issues.append(f"Data appears flat. Price variance: {price_variance:.6f}")
    
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            quality_issues.append("Date column contains invalid date formats.")
    
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    missing_pct = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().mean().mean() * 100
    if missing_pct > 20:
        quality_issues.append(f"Excessive missing data: {missing_pct:.1f}% of price/volume data is missing.")
    
    data_span_years = (df['Date'].max() - df['Date'].min()).days / 365.25
    if data_span_years < 1:
        quality_issues.append(f"Limited historical data: only spans {data_span_years:.1f} years.")
    
    price_jumps = df['Close'].pct_change().abs()
    extreme_moves = (price_jumps > 1.0).sum()
    if extreme_moves > 10:
        quality_issues.append(f"Detected {extreme_moves} extreme price movements (>50% daily).")
    
    if quality_issues:
        ticker = df.get('Ticker', ['Unknown'])[0] if 'Ticker' in df.columns else 'Unknown'

        #annoying error that clutters the terminal
        #logging.warning(f"Data quality issues for {ticker}: {'; '.join(quality_issues)}")
        
        if any("flat" in issue or "Insufficient" in issue or "missing" in issue for issue in quality_issues):
            return None
    
    return df

def SaveData(df, file_path, output_dir):
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, file_name)
    
    # Fix for duplicate columns - keep the last occurrence of each column
    if len(df.columns) != len(df.columns.unique()):
        # Identify duplicate columns
        duplicates = df.columns[df.columns.duplicated(keep=False)]
        if duplicates.any():
            logging.warning(f"Found duplicate columns: {list(duplicates.unique())}")
        
        # Keep only the last occurrence of each duplicate column
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        logging.info(f"Removed duplicate columns. DataFrame now has {len(df.columns)} columns.")
    
    df.to_parquet(output_file, index=False)
    del df

def clear_output_directory(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file))

def process_file(file_path, output_dir):
    try:
        df = pd.read_parquet(file_path)
        if 'Date' not in df.columns:
            if df.index.name == 'Date':
                df = df.reset_index()
            else:
                logging.error("The 'Date' column is missing from the DataFrame.")
                return False

        if df['Date'].dtype != 'datetime64[ns]':
            logging.info("Converting 'Date' column to datetime.")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
        if not validate_columns(df, ['Close', 'High', 'Low', 'Volume']):
            logging.error(f"File {file_path} does not contain all required columns.")
            return False

        df = DataQualityCheck(df)
        if df is None:
            logging.error(f"Data quality check failed for {file_path}.")
            return False

        df = indicators(df)
        df = clean_and_interpolate_data(df)
        SaveData(df, file_path, output_dir)
        return True

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        traceback_info = traceback.format_exc()
        logging.error(traceback_info)
        return False

def process_file_wrapper(file_path):
    return process_file(file_path, CONFIG['output_directory'])













def process_files(file_path, output_dir, index_temp_files=None):
    global GLOBAL_INDEX_DATA
    
    # Load indexes in this worker if not already loaded
    load_indexes_for_worker()
    
    try:
        df = pd.read_parquet(file_path)
        
        if 'Date' not in df.columns:
            if df.index.name == 'Date':
                df = df.reset_index()
            else:
                logging.error(f"'Date' column missing from {file_path}")
                return False

        if df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
        if not validate_columns(df, ['Close', 'High', 'Low', 'Volume']):
            logging.error(f"File {file_path} missing required columns")
            return False

        df = DataQualityCheck(df)
        if df is None:
            logging.error(f"Data quality check failed for {file_path}")
            return False

        df = indicators(df)
        df = clean_and_interpolate_data(df)
        SaveData(df, file_path, output_dir)
        
        return True

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        traceback_info = traceback.format_exc()
        logging.error(traceback_info)
        return False





def process_data_files(run_percent):
    """Main processing function with all market indexes."""
    
    print(f"Processing {run_percent}% of files from {CONFIG['input_directory']}")
    StartTimer = time.time()
    
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    clear_output_directory(CONFIG['output_directory'])

    # Update/download all indexes - this saves them to Data/Indexes
    print("Updating market indexes...")
    update_all_indexes()

    # Get files to process
    file_paths = [
        os.path.join(CONFIG['input_directory'], f) 
        for f in os.listdir(CONFIG['input_directory']) 
        if f.endswith('.parquet')
    ]
    
    files_to_process = file_paths[:int(len(file_paths) * (run_percent / 100))]
    num_workers = min(32, len(files_to_process))
    
    completed = 0
    failed = 0
    
    # Process files in parallel - each worker will load indexes independently
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file_path in files_to_process:
            future = executor.submit(
                process_files, 
                file_path, 
                CONFIG['output_directory'],
                None  # No longer need to pass temp files
            )
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Exception in worker process: {str(e)}")
                    failed += 1
                finally:
                    pbar.update(1)
                    pbar.set_description(
                        f"Processing files (Success: {completed}, Failed: {failed})"
                    )

    total_time = time.time() - StartTimer
    files_per_second = len(files_to_process) / total_time if total_time > 0 else 0
    
    print(f"\nProcessed {len(files_to_process)} files in {total_time:.2f} seconds")
    print(f"Files per second: {files_per_second:.2f}")
    print(f'Average time per file: {round(total_time / len(files_to_process), 2) if len(files_to_process) > 0 else 0} seconds')
    print(f'Successfully processed: {completed}')
    print(f'Failed to process: {failed}')

    add_gp_cross_sectional_features(CONFIG['output_directory'])
    add_interaction_conjunction_features(CONFIG['output_directory'])








if __name__ == "__main__":
    logger.info("Starting the process...")
    parser = argparse.ArgumentParser(description="Process financial market data files.")
    parser.add_argument('--runpercent', type=int, default=100, help="Percentage of files to process from the input directory.")
    args = parser.parse_args()

    process_data_files(args.runpercent)