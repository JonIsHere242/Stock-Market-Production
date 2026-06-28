"""
orig_orig_kalman.py -- faithful port of AlphaSensitivity.calculate_kalman_indicators
(_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py, L2588-2697).

OHLCV-only, deterministic (no randomness). Stateful 1-D constant-position Kalman
filter over Close, plus argrelextrema-based support/resistance, a chaos-proxy
Lyapunov suite, and an MA-200 percent-difference.

IMPORTANT (producer-before-consumer): this block emits Smoothed_Close, which
orig_mean_reversion_complexity consumes (price_column='Smoothed_Close'). It must
run BEFORE that block in topo order.

Self-contained: imports only numpy/pandas/scipy.signal.argrelextrema. The math
(windows, min_periods, epsilon, order of ops) is copied verbatim from the backup.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

METADATA = {
    "name":        "orig_orig_kalman",
    "description": ("Faithful port of AlphaSensitivity calculate_kalman_indicators: 1-D "
                    "constant-position Kalman filter over Close, argrelextrema "
                    "support/resistance distances, a Lyapunov-exponent chaos proxy, and "
                    "MA-200 percent difference. Produces Smoothed_Close consumed downstream "
                    "by orig_mean_reversion_complexity."),
    "requires":    ["Close"],
    "produces": [
        "Kalman",
        "minima",
        "maxima",
        "Distance to Support (%)",
        "Distance to Resistance (%)",
        "Smoothed_Close",
        "Perturbed_Kalman",
        "Divergence",
        "Log_Divergence",
        "Lyapunov_Exponent",
        "Lyapunov_Exponent_MA",
        "MA_200",
        "Perc_Diff",
    ],
    "tags":        ["kalman", "support_resistance", "chaos", "trend"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close_prices = df["Close"].values

    # Pre-allocate arrays
    kalman_values = np.full(n, np.nan)
    minima_values = np.full(n, np.nan)
    maxima_values = np.full(n, np.nan)
    support_pct = np.full(n, np.nan)
    resistance_pct = np.full(n, np.nan)

    if n == 0:
        # Degrade gracefully: emit empty columns of the right names.
        df["Kalman"] = kalman_values
        df["minima"] = minima_values
        df["maxima"] = maxima_values
        df["Distance to Support (%)"] = support_pct
        df["Distance to Resistance (%)"] = resistance_pct
        epsilon = 0.001
        df["Smoothed_Close"] = df["Kalman"]
        df["Perturbed_Kalman"] = df["Kalman"] * (1 + epsilon)
        df["Divergence"] = np.abs(df["Perturbed_Kalman"] - df["Kalman"])
        df["Log_Divergence"] = np.log(df["Divergence"] + np.finfo(float).eps)
        df["Lyapunov_Exponent"] = df["Log_Divergence"].diff() / np.log(1 + epsilon)
        df["Lyapunov_Exponent_MA"] = df["Lyapunov_Exponent"].rolling(window=14).mean()
        df["MA_200"] = df["Close"].rolling(window=200, min_periods=200).mean()
        df["Perc_Diff"] = (df["Kalman"] - df["MA_200"]) / df["MA_200"] * 100
        return df

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
        predicted_state_covariance = (
            np.dot(np.dot(transition_matrix, current_state_covariance), transition_matrix.T)
            + transition_covariance
        )

        # Update step with current observation
        kalman_gain = np.dot(
            np.dot(predicted_state_covariance, observation_matrix.T),
            np.linalg.inv(
                np.dot(np.dot(observation_matrix, predicted_state_covariance),
                       observation_matrix.T)
                + observation_covariance
            ),
        )

        current_state_mean = predicted_state_mean + np.dot(
            kalman_gain,
            (close_prices[i] - np.dot(observation_matrix, predicted_state_mean)),
        )
        current_state_covariance = predicted_state_covariance - np.dot(
            np.dot(kalman_gain, observation_matrix), predicted_state_covariance
        )

        # Store the current filtered value
        kalman_values[i] = current_state_mean[0]

    # Compute extrema and percentages
    window_size = 140
    min_data_points = 20
    for i in range(min_data_points, n):
        # Use only lookback window for extrema detection
        lookback = min(window_size, i)
        lookback_start = max(0, i - lookback + 1)
        historical_window = kalman_values[lookback_start:i + 1]

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
                    minima_values[i] = minima_values[i - 1]
            elif i > 0:
                minima_values[i] = minima_values[i - 1]

            # Process maxima
            if len(max_indices) > 0:
                most_recent_max_idx = max_indices[-1] + lookback_start
                if most_recent_max_idx < i:
                    maxima_values[i] = kalman_values[most_recent_max_idx]
                elif i > 0:
                    maxima_values[i] = maxima_values[i - 1]
            elif i > 0:
                maxima_values[i] = maxima_values[i - 1]
        elif i > 0:
            minima_values[i] = minima_values[i - 1]
            maxima_values[i] = maxima_values[i - 1]

        # Calculate percentages
        if not np.isnan(minima_values[i]) and minima_values[i] > 0:
            support_pct[i] = (close_prices[i] - minima_values[i]) / minima_values[i] * 100

        if not np.isnan(maxima_values[i]) and close_prices[i] > 0:
            resistance_pct[i] = (maxima_values[i] - close_prices[i]) / close_prices[i] * 100

    # Add to the original DataFrame
    df["Kalman"] = kalman_values
    df["minima"] = minima_values
    df["maxima"] = maxima_values
    df["Distance to Support (%)"] = support_pct
    df["Distance to Resistance (%)"] = resistance_pct

    # Additional Kalman calculations
    epsilon = 0.001
    df["Smoothed_Close"] = df["Kalman"]
    df["Perturbed_Kalman"] = df["Kalman"] * (1 + epsilon)
    df["Divergence"] = np.abs(df["Perturbed_Kalman"] - df["Kalman"])
    df["Log_Divergence"] = np.log(df["Divergence"] + np.finfo(float).eps)
    df["Lyapunov_Exponent"] = df["Log_Divergence"].diff() / np.log(1 + epsilon)
    window_size = 14
    df["Lyapunov_Exponent_MA"] = df["Lyapunov_Exponent"].rolling(window=window_size).mean()

    # MA and percentage difference
    df["MA_200"] = df["Close"].rolling(window=200, min_periods=200).mean()
    df["Perc_Diff"] = (df["Kalman"] - df["MA_200"]) / df["MA_200"] * 100

    return df
