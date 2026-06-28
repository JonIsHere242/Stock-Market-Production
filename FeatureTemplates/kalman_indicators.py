import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

METADATA = {
    "name":        "kalman_indicators",
    "description": "Manual 1-D Kalman filter on Close with extrema-based support/resistance, Lyapunov divergence, and MA-200 percentage difference",
    "requires":    ["Close"],
    "produces":    [
        "kalman",
        "kalman_minima",
        "kalman_maxima",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "smoothed_close",
        "perturbed_kalman",
        "divergence",
        "log_divergence",
        "lyapunov_exponent_kalman",
        "lyapunov_exponent_kalman_ma",
        "ma_200",
        "perc_diff",
    ],
    "tags":        ["trend", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "ported from 3__AlphaSensitivity.calculate_kalman_indicators",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
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
    df['kalman'] = kalman_values
    df['kalman_minima'] = minima_values
    df['kalman_maxima'] = maxima_values
    df['distance_to_support_pct'] = support_pct
    df['distance_to_resistance_pct'] = resistance_pct

    # Additional Kalman calculations
    epsilon = 0.001
    df['smoothed_close'] = df['kalman']
    df['perturbed_kalman'] = df['kalman'] * (1 + epsilon)
    df['divergence'] = np.abs(df['perturbed_kalman'] - df['kalman'])
    df['log_divergence'] = np.log(df['divergence'] + np.finfo(float).eps)
    df['lyapunov_exponent_kalman'] = df['log_divergence'].diff() / np.log(1 + epsilon)
    window_size = 14
    df['lyapunov_exponent_kalman_ma'] = df['lyapunov_exponent_kalman'].rolling(window=window_size).mean()

    # MA and percentage difference
    df['ma_200'] = df['Close'].rolling(window=200, min_periods=200).mean()
    df['perc_diff'] = (df['kalman'] - df['ma_200']) / df['ma_200'] * 100

    return df


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['kalman', 'divergence', 'log_divergence', 'perturbed_kalman']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
