import numpy as np
import pandas as pd

METADATA = {
    "name":        "orig_mean_reversion_complexity",
    "description": "AlphaSensitivity volatility-indicator port: rolling mean/std and "
                   "mean-reversion z-scores on Smoothed_Close (windows 28/90/151, std "
                   "multipliers 1/1/3), Complexity-Invariant-Distance on Close (90d var "
                   "diff), CID rolling mean/std, lagged percent-change of Close (lags "
                   "1/5/10), and Keltner-channel %% bands (KC_UPPER%%, KC_LOWER%%). "
                   "Depends on Smoothed_Close (orig_kalman) and ATR (orig_moving_average); "
                   "both are computed internally (verbatim formulas) when absent so the "
                   "block is self-contained for verification. Only the produced columns "
                   "are emitted; the Smoothed_Close/ATR intermediates are not.",
    "requires":    ["Close", "High", "Low"],
    "produces":    [
        "Rolling_Mean_28", "Rolling_Mean_90", "Rolling_Mean_151",
        "Rolling_Std_28", "Rolling_Std_90", "Rolling_Std_151",
        "Mean_Reversion_Z_Score_28_std_1", "Mean_Reversion_Z_Score_90_std_1",
        "Mean_Reversion_Z_Score_151_std_3",
        "Complexity_Invariant_Distance", "CID_Mean", "CID_SD",
        "percent_change_Close_lag_1", "percent_change_Close_lag_5",
        "percent_change_Close_lag_10",
        "KC_UPPER%", "KC_LOWER%",
    ],
    "tags":        ["volatility", "mean_reversion", "complexity", "keltner", "alphasens"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def _kalman_smoothed_close(close_prices):
    """Scalar Kalman filter over Close, verbatim from calculate_kalman_indicators
    (3__AlphaSensitivity_pre_tsz_20260530.py L2600-2627). Smoothed_Close = Kalman."""
    n = len(close_prices)
    kalman_values = np.zeros(n)

    transition_matrix = np.array([[1]])
    observation_matrix = np.array([[1]])
    transition_covariance = np.array([[0.01]])
    observation_covariance = np.array([[1]])
    initial_state_mean = close_prices[0]
    initial_state_covariance = np.array([[1]])

    current_state_mean = initial_state_mean
    current_state_covariance = initial_state_covariance

    for i in range(n):
        predicted_state_mean = np.dot(transition_matrix, current_state_mean)
        predicted_state_covariance = np.dot(
            np.dot(transition_matrix, current_state_covariance),
            transition_matrix.T,
        ) + transition_covariance

        kalman_gain = np.dot(
            np.dot(predicted_state_covariance, observation_matrix.T),
            np.linalg.inv(
                np.dot(np.dot(observation_matrix, predicted_state_covariance),
                       observation_matrix.T) + observation_covariance
            ),
        )

        current_state_mean = predicted_state_mean + np.dot(
            kalman_gain,
            (close_prices[i] - np.dot(observation_matrix, predicted_state_mean)),
        )
        current_state_covariance = predicted_state_covariance - np.dot(
            np.dot(kalman_gain, observation_matrix), predicted_state_covariance
        )

        kalman_values[i] = current_state_mean[0]

    return kalman_values


def _atr(close, high, low):
    """ATR (min_periods=1) verbatim from orig_moving_average / calculate_volatility
    intermediates: True Range with shift(1), 14d rolling mean."""
    close_shift_1 = close.shift(1)
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close_shift_1),
            np.abs(low - close_shift_1),
        ),
    )
    return true_range.rolling(window=14, min_periods=1).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- Mean-reversion z-scores on Smoothed_Close (guard preserved) ---
    # In production Smoothed_Close is supplied by orig_kalman; compute it here
    # (verbatim scalar Kalman) when absent so the block is self-contained.
    if "Smoothed_Close" not in df.columns:
        smoothed = pd.Series(
            _kalman_smoothed_close(df["Close"].values.astype(float)),
            index=df.index,
        )
    else:
        smoothed = df["Smoothed_Close"]

    new_columns = {}
    windows = [28, 90, 151]
    std_multipliers = [1, 1, 3]
    for window, std_multiplier in zip(windows, std_multipliers):
        mean_col = f"Rolling_Mean_{window}"
        std_col = f"Rolling_Std_{window}"
        z_score_col = f"Mean_Reversion_Z_Score_{window}_std_{std_multiplier}"

        rolling_window = smoothed.rolling(window=window)
        new_columns[mean_col] = rolling_window.mean()
        new_columns[std_col] = rolling_window.std()
        new_columns[z_score_col] = (
            (smoothed - new_columns[mean_col]) / (new_columns[std_col] * std_multiplier)
        )

    # --- Percent-change lags of Close ---
    pct_change_close = df["Close"].pct_change()
    new_columns["percent_change_Close_lag_1"] = pct_change_close.shift(1)
    new_columns["percent_change_Close_lag_5"] = pct_change_close.shift(5)
    new_columns["percent_change_Close_lag_10"] = pct_change_close.shift(10)

    # --- Keltner channel %% bands ---
    if "ATR" in df.columns:
        atr = df["ATR"]
    else:
        atr = _atr(df["Close"], df["High"], df["Low"])
    keltner_central = df["Close"].ewm(span=20).mean()
    keltner_range = atr * 1.5
    new_columns["KC_UPPER%"] = (
        ((keltner_central + keltner_range) - df["Close"]) / df["Close"] * 100
    )
    new_columns["KC_LOWER%"] = (
        (df["Close"] - (keltner_central - keltner_range)) / df["Close"] * 100
    )

    # --- Complexity metrics on Close (window_size=90) ---
    window_size = 90
    rolling_variance = df["Close"].rolling(window=window_size).var()
    cid = rolling_variance.diff().abs()
    new_columns["Complexity_Invariant_Distance"] = cid
    new_columns["CID_Mean"] = cid.rolling(window=window_size).mean()
    new_columns["CID_SD"] = cid.rolling(window=window_size).std()

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df
