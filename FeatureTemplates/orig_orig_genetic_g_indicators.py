import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_orig_genetic_g_indicators",
    "description": "Genetically-discovered OHLCV expression indicators (G_* family) ported verbatim from AlphaSensitivity.calculate_genetic_indicators",
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "G_Momentum_Confluence_Indicator",
        "G_Price_Gap_Analyzer",
        "G_Triple_High_Trend_Indicator",
        "G_Cyclical_Price_Oscillator",
        "G_Volume_Adjusted_Price_Indicator",
        "G_Adjusted_Close_Tracker",
        "G_Volume_Weighted_High_Ratio",
        "G_High_Price_Momentum_Indicator",
        "G_Advanced_Trend_Synthesizer",
        "G_Price_Volatility_Gauge",
        "G_Multi_Point_Price_Analyzer",
        "G_Logarithmic_Trend_Detector",
        "G_Complex_Price_Pattern_Indicator",
        "G_Log_Scaled_Price_Ratio",
        "G_Volume_Price_Impact_Indicator",
        "G_Volume_Trend_Analyzer",
        "G_Price_Open_Ratio_Indicator",
        "G_Price_Differential_Analyzer",
        "G_Price_Volatility_Trend_Measure",
        "G_Lagged_Price_Volume_Convergence",
        "G_Price_Volume_Disparity_Index",
    ],
    "tags":    ["genetic", "ohlcv", "alphasens"],
    "version": "1.0",
    "author":  "alphasens port",
}


# --- helpers copied verbatim from 3__AlphaSensitivity_pre_tsz_20260530.py ---
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


def compute(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-10

    # Adj Close alias: original used 'Adj Close'; fall back to Close if absent.
    if 'Adj Close' in df.columns:
        adj_close = df['Adj Close']
    else:
        adj_close = df['Close']

    # Build per-row lag cols High_Lag1..7, Low_Lag1..7, Volume_Lag1..7, Open_Lag1..7
    lag = {}
    for i in range(1, 8):
        lag[f'High_Lag{i}'] = df['High'].shift(i) + epsilon
        lag[f'Low_Lag{i}'] = df['Low'].shift(i) + epsilon
        lag[f'Volume_Lag{i}'] = df['Volume'].shift(i) + epsilon
        lag[f'Open_Lag{i}'] = df['Open'].shift(i) + epsilon

    High_Lag1 = lag['High_Lag1']; High_Lag2 = lag['High_Lag2']
    High_Lag3 = lag['High_Lag3']; High_Lag4 = lag['High_Lag4']
    High_Lag5 = lag['High_Lag5']; High_Lag7 = lag['High_Lag7']
    Low_Lag1 = lag['Low_Lag1']; Low_Lag2 = lag['Low_Lag2']; Low_Lag5 = lag['Low_Lag5']
    Volume_Lag1 = lag['Volume_Lag1']
    Open_Lag1 = lag['Open_Lag1']; Open_Lag2 = lag['Open_Lag2']

    High = df['High']; Low = df['Low']; Open = df['Open']
    Close = df['Close']; Volume = df['Volume']

    df['G_Momentum_Confluence_Indicator'] = safe_divide(High_Lag2, High_Lag2 * Open)
    df['G_Price_Gap_Analyzer'] = safe_divide(safe_log(Open_Lag2), High_Lag1)
    df['G_Triple_High_Trend_Indicator'] = safe_divide(High_Lag2, High_Lag1 * High)
    df['G_Cyclical_Price_Oscillator'] = safe_divide(
        safe_divide(np.cos(safe_divide(High_Lag2, High)), High_Lag1),
        np.sqrt(Open_Lag1 + epsilon)
    )
    df['G_Volume_Adjusted_Price_Indicator'] = safe_divide(High_Lag2, safe_divide(Volume, Low_Lag1))
    df['G_Adjusted_Close_Tracker'] = adj_close
    df['G_Volume_Weighted_High_Ratio'] = safe_divide(safe_divide(High, High_Lag1), safe_log(Volume + 1))
    df['G_High_Price_Momentum_Indicator'] = safe_divide(High, (High_Lag1 + High_Lag2) / 2)
    df['G_Advanced_Trend_Synthesizer'] = (
        safe_log(safe_divide(High_Lag1 + High_Lag5, High_Lag1)) *
        np.abs(safe_log(safe_divide(High_Lag2, High_Lag2)) - High_Lag7) *
        safe_divide(Open_Lag2, Close)
    )
    df['G_Price_Volatility_Gauge'] = safe_divide(np.abs(High_Lag2 - High), Open)
    df['G_Multi_Point_Price_Analyzer'] = np.abs(
        safe_divide(
            safe_divide(safe_log(safe_divide(High, High_Lag2)), safe_divide(Close, High_Lag2)),
            safe_divide(Close, High_Lag2) * safe_divide(Close, High_Lag1)
        )
    )
    df['G_Logarithmic_Trend_Detector'] = -safe_log(safe_divide(High_Lag2, High_Lag4))
    df['G_Complex_Price_Pattern_Indicator'] = safe_log(
        safe_divide(
            np.sqrt(np.sqrt(np.sqrt(np.sqrt(High_Lag7 * High_Lag4 + epsilon)))),
            Close
        )
    )
    df['G_Log_Scaled_Price_Ratio'] = safe_log(safe_divide(High, (High_Lag1 + High_Lag2) / 2))
    df['G_Volume_Price_Impact_Indicator'] = safe_divide(
        -High_Lag1 + safe_divide(High_Lag3, Close),
        Volume
    )
    df['G_Volume_Trend_Analyzer'] = safe_log(safe_divide(Volume, Volume_Lag1))
    df['G_Price_Open_Ratio_Indicator'] = safe_divide(
        safe_divide(safe_log(safe_divide(High_Lag2, High)), safe_divide(High, Open_Lag2)),
        safe_divide(High, Open_Lag2)
    )
    df['G_Price_Differential_Analyzer'] = (0.1673 / (High + epsilon) - Low) / (High + epsilon)
    df['G_Price_Volatility_Trend_Measure'] = 0.278 - np.abs(safe_divide(Low, High) / safe_divide(High_Lag5, Low_Lag5))
    df['G_Lagged_Price_Volume_Convergence'] = safe_divide(Low_Lag2, (High_Lag2 + safe_divide(0.791, Low * High)))
    df['G_Price_Volume_Disparity_Index'] = np.abs(safe_divide(Low_Lag2, High_Lag2)) / (safe_divide(High_Lag5, Low_Lag5) / -0.2831)

    return df
