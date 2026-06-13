import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from Util import get_logger

logger = get_logger(script_name="4__XGBoostPredictor")

try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    _ = cp.asnumpy(test_array)
    CUPY_AVAILABLE = True
    logging.info("CuPy detected and verified - GPU acceleration enabled")
except (ImportError, RuntimeError, Exception) as e:
    CUPY_AVAILABLE = False
    logging.warning(f"CuPy not available: {str(e)}")
    logging.info("Falling back to CPU mode")

def to_gpu_array(arr):
    if CUPY_AVAILABLE:
        try:
            return cp.asarray(arr)
        except Exception as e:
            logging.warning(f"GPU conversion failed: {str(e)}, using CPU")
            return arr
    return arr

def to_cpu_array(arr):
    if CUPY_AVAILABLE:
        try:
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except Exception:
            pass
    return arr

argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=50, help="Percentage of TIME PERIOD to use for training.")
argparser.add_argument("--calibpercent", type=int, default=20, help="Percentage of TIME PERIOD to use for calibration.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--nocalib", action='store_true', help="Flag to disable probability calibration.")
argparser.add_argument("--fixed_threshold", type=float, default=None, help="Use a fixed threshold instead of optimization (e.g., 0.70)")
argparser.add_argument("--target_precision", type=float, default=0.70, help="Target precision for UP predictions (default: 0.70)")
args = argparser.parse_args()

config = {
    "input_directory": "Data/ProcessedData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "calibration_output_directory": "Data/ModelData/CalibrationData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/xgboost_feature_importance.parquet",
    "calibration_plot_output": "Data/ModelData/calibration_plot.png",
    "log_file": "data/logging/4__XGBoostPredictor.log",
    "file_selection_percentage": args.runpercent,
    "calibration_percentage": args.calibpercent,
    "target_column": "percent_change_Close",
    "apply_calibration": not args.nocalib,

    "xgboost_params": {
        "booster": "dart",
        "rate_drop": 0.3,
        "skip_drop": 0.5,
        "one_drop": 1,
        "learning_rate": 0.05,
        "n_estimators": 256,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 3301,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "max_delta_step": 1
    },

    "training_params": {
        "early_stopping_rounds": 30,
        "verbose": 50
    },
    
    "random_state": 3301
}

logging.info(f"XGBoost running in CPU mode")

class CalibratedModel:
    def __init__(self, base_model, temperature, isotonic_regressor):
        self.base_model = base_model
        self.temperature = temperature
        self.isotonic = isotonic_regressor

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)
        base_probs_cpu = to_cpu_array(base_probs)
        temp_scaled = base_probs_cpu[:, 1] ** (1.0 / self.temperature)
        calibrated = self.isotonic.predict(temp_scaled)
        result = np.column_stack([1 - calibrated, calibrated])
        return result

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

class TimeSeriesCalibratedModel:
    """
    Time series aware calibrated model - prevents lookahead bias in calibration
    """
    def __init__(self, base_model, calibration_method='isotonic', calibration_data=None):
        """
        Parameters:
        -----------
        base_model : trained classifier
            The base XGBoost model
        calibration_method : str
            'isotonic' or 'temperature'
        calibration_data : tuple or None
            (X_calib, y_calib, date_column) for fitting final calibrator
        """
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibrator = None
        self.is_fitted = False

        # Fit final calibrator on entire calibration set if provided
        if calibration_data is not None:
            self._fit_final_calibrator(*calibration_data)

    def _fit_final_calibrator(self, X_calib, y_calib, date_column):
        """Fit final calibrator on entire calibration set (for production use)"""
        try:
            # Remove date column for prediction
            X_calib_no_date = X_calib.drop(columns=[date_column]) if date_column in X_calib.columns else X_calib

            # Get base probabilities
            base_probs = self.base_model.predict_proba(X_calib_no_date)
            base_probs = to_cpu_array(base_probs)[:, 1]

            if self.calibration_method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(base_probs, y_calib)
                logging.info("Final isotonic calibrator fitted on entire calibration set")

            elif self.calibration_method == 'temperature':
                def temperature_loss(t):
                    eps = 1e-7
                    scaled = np.clip(base_probs ** (1.0 / t), eps, 1 - eps)
                    loss = -np.mean(y_calib * np.log(scaled) + (1 - y_calib) * np.log(1 - scaled))
                    return loss

                result = minimize(temperature_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')

                if result.success:
                    self.calibrator = result.x[0]  # Store optimal temperature
                    logging.info(f"Final temperature calibrator fitted: T={self.calibrator:.4f}")
                else:
                    logging.warning("Temperature optimization failed, using T=1.0")
                    self.calibrator = 1.0

            self.is_fitted = True

        except Exception as e:
            logging.error(f"Failed to fit final calibrator: {str(e)}")
            self.calibrator = None
            self.is_fitted = False

    def predict_proba(self, X):
        """Predict probabilities using calibrated model"""
        if not self.is_fitted or self.calibrator is None:
            # Fallback to base model
            logging.warning("Calibrator not fitted, using base model")
            return self.base_model.predict_proba(X)

        base_probs = self.base_model.predict_proba(X)
        base_probs_cpu = to_cpu_array(base_probs)
        base_probs_1 = base_probs_cpu[:, 1]

        if self.calibration_method == 'isotonic':
            calibrated = self.calibrator.predict(base_probs_1)
        elif self.calibration_method == 'temperature':
            calibrated = base_probs_1 ** (1.0 / self.calibrator)
        else:
            calibrated = base_probs_1

        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        """Predict classes"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

def time_series_calibrate(clf, X_calib, y_calib, date_column, method='isotonic'):
    """
    Calibrate probabilities using expanding window to prevent lookahead bias

    This is the VALIDATION function - it shows what calibration would look like
    in a time series setting where we only use past data.

    Parameters:
    -----------
    clf : trained classifier
        The base XGBoost model
    X_calib : DataFrame
        Calibration features, must include date_column
    y_calib : Series
        Calibration labels (0, 1)
    date_column : str
        Name of the date column in X_calib
    method : str
        'isotonic' or 'temperature' scaling method

    Returns:
    --------
    calibrated_probs : ndarray
        Time series calibrated probabilities for class 1
    """
    # Validate inputs
    if date_column not in X_calib.columns:
        raise ValueError(f"Date column '{date_column}' not found in calibration data")

    if len(X_calib) != len(y_calib):
        raise ValueError("X_calib and y_calib must have same length")

    # Create copies to avoid modifying original data
    calib_dates = X_calib[date_column].copy()
    X_calib_vals = X_calib.drop(columns=[date_column]).copy()

    # Sort by date to ensure temporal order - CRITICAL STEP
    sort_idx = calib_dates.argsort()
    X_calib_vals = X_calib_vals.iloc[sort_idx] if hasattr(X_calib_vals, 'iloc') else X_calib_vals[sort_idx]
    y_calib_sorted = y_calib.iloc[sort_idx] if hasattr(y_calib, 'iloc') else y_calib[sort_idx]
    calib_dates_sorted = calib_dates.iloc[sort_idx] if hasattr(calib_dates, 'iloc') else calib_dates[sort_idx]

    # Get base probabilities from the model
    logging.info("Getting base probabilities from model...")
    base_probs = clf.predict_proba(X_calib_vals)
    base_probs = to_cpu_array(base_probs)[:, 1]  # Probability of class 1

    # Initialize output array
    calibrated_probs = np.zeros_like(base_probs)

    # Calculate minimum window size - critical for early data points
    min_window = max(1000, len(X_calib_vals) // 20)  # At least 1000 samples or 5% of data
    logging.info(f"Time series calibration: min_window = {min_window} samples")

    # Track calibration statistics
    calibration_updates = 0
    skipped_updates = 0

    # Main expanding window loop
    logging.info("Starting expanding window calibration (this may take a while)...")

    for i in tqdm(range(len(X_calib_vals)), desc="Time series calibration", ncols=100):
        if i < min_window:
            # For early points: use raw probabilities (not enough history)
            calibrated_probs[i] = base_probs[i]
            continue

        # Use expanding window up to current point (NO lookahead)
        train_probs = base_probs[:i]  # All points before current
        train_labels = y_calib_sorted.iloc[:i] if hasattr(y_calib_sorted, 'iloc') else y_calib_sorted[:i]

        if method == 'isotonic':
            try:
                # Check if we have both classes in training data
                unique_labels = np.unique(train_labels)
                if len(unique_labels) > 1:
                    iso_reg = IsotonicRegression(out_of_bounds='clip')
                    iso_reg.fit(train_probs, train_labels)

                    # Calibrate current probability
                    current_prob = base_probs[i]
                    calibrated_probs[i] = iso_reg.predict([current_prob])[0]
                    calibration_updates += 1
                else:
                    # Only one class - skip calibration for this point
                    calibrated_probs[i] = base_probs[i]
                    skipped_updates += 1

            except Exception as e:
                logging.warning(f"Isotonic calibration failed at index {i}: {str(e)}")
                calibrated_probs[i] = base_probs[i]

        elif method == 'temperature':
            try:
                # Check if we have both classes
                unique_labels = np.unique(train_labels)
                if len(unique_labels) > 1:
                    # Temperature scaling optimization
                    def temperature_loss(t):
                        eps = 1e-7
                        scaled = np.clip(train_probs ** (1.0 / t), eps, 1 - eps)
                        loss = -np.mean(
                            train_labels * np.log(scaled) +
                            (1 - train_labels) * np.log(1 - scaled)
                        )
                        return loss

                    # Find optimal temperature
                    result = minimize(
                        temperature_loss,
                        x0=1.0,
                        bounds=[(0.1, 10.0)],
                        method='L-BFGS-B'
                    )

                    if result.success:
                        optimal_temp = result.x[0]
                        calibrated_probs[i] = base_probs[i] ** (1.0 / optimal_temp)
                        calibration_updates += 1
                    else:
                        calibrated_probs[i] = base_probs[i]
                        skipped_updates += 1
                else:
                    calibrated_probs[i] = base_probs[i]
                    skipped_updates += 1

            except Exception as e:
                logging.warning(f"Temperature calibration failed at index {i}: {str(e)}")
                calibrated_probs[i] = base_probs[i]

    # Log calibration statistics
    total_points = len(X_calib_vals)
    calibration_rate = calibration_updates / total_points * 100 if total_points > 0 else 0
    logging.info(f"Time series calibration completed:")
    logging.info(f"  Total points: {total_points}")
    logging.info(f"  Calibrated points: {calibration_updates} ({calibration_rate:.1f}%)")
    logging.info(f"  Skipped points: {skipped_updates}")

    return calibrated_probs

def prepare_data_splits(input_directory, train_output_directory, calib_output_directory, 
                       file_selection_percentage, calibration_percentage, target_column, reuse, date_column):
    train_output_file = os.path.join(train_output_directory, 'training_data.parquet')
    calib_output_file = os.path.join(calib_output_directory, 'calibration_data.parquet')
    split_info_file = os.path.join(train_output_directory, 'temporal_split_info.json')
    
    os.makedirs(calib_output_directory, exist_ok=True)
    
    if reuse and os.path.exists(train_output_file) and os.path.exists(calib_output_file):
        logging.info("Reusing existing training and calibration data.")
        print("Reusing existing training and calibration data.")
        return pd.read_parquet(train_output_file), pd.read_parquet(calib_output_file)
    
    logging.info("=" * 80)
    logging.info("TEMPORAL DATA SPLITTING - Preventing Data Leakage")
    logging.info("=" * 80)
    
    cutoff_date = pd.to_datetime('2021-01-10')
    logging.info(f"Filtering out data before {cutoff_date.strftime('%Y-%m-%d')}")
    
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    logging.info(f"Found {len(all_files)} parquet files")
    
    logging.info("Step 1: Scanning all files to find global date range...")
    global_min_date = pd.Timestamp.max
    global_max_date = pd.Timestamp.min
    
    for file in tqdm(all_files, desc="Scanning dates", ncols=100):
        try:
            df = pd.read_parquet(os.path.join(input_directory, file))
            if date_column in df.columns and len(df) > 0:
                df[date_column] = pd.to_datetime(df[date_column])
                file_min = df[date_column].min()
                file_max = df[date_column].max()
                
                if file_min < global_min_date:
                    global_min_date = file_min
                if file_max > global_max_date:
                    global_max_date = file_max
        except Exception as e:
            logging.warning(f"Error reading {file}: {str(e)}")
            continue
    
    logging.info(f"Global date range: {global_min_date.date()} to {global_max_date.date()}")
    
    total_days = (global_max_date - global_min_date).days
    logging.info(f"Total time span: {total_days} days")
    
    train_days = int(total_days * file_selection_percentage / 100)
    calib_days = int(total_days * calibration_percentage / 100)
    
    train_end_date = global_min_date + pd.Timedelta(days=train_days)
    calib_end_date = train_end_date + pd.Timedelta(days=calib_days)
    
    logging.info("=" * 80)
    logging.info(f"TRAINING period:    {global_min_date.date()} to {train_end_date.date()} ({train_days} days)")
    logging.info(f"CALIBRATION period: {train_end_date.date()} to {calib_end_date.date()} ({calib_days} days)")
    logging.info(f"TEST period:        {calib_end_date.date()} to {global_max_date.date()} (unused, {total_days - train_days - calib_days} days)")
    logging.info("=" * 80)
    
    import json
    split_info = {
        'global_min_date': str(global_min_date),
        'global_max_date': str(global_max_date),
        'train_end_date': str(train_end_date),
        'calib_end_date': str(calib_end_date),
        'train_days': train_days,
        'calib_days': calib_days,
        'total_days': total_days
    }
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logging.info("Step 3: Processing files and splitting by temporal boundaries...")
    train_data = []
    calib_data = []
    
    train_rows = 0
    calib_rows = 0
    
    for file in tqdm(all_files, desc="Processing files", ncols=100):
        try:
            df = pd.read_parquet(os.path.join(input_directory, file))
            
            df[target_column] = df[target_column].shift(-1)
            df = df.iloc[:-1]
            df = df.iloc[1:]

            if df.shape[0] <= 50 or target_column not in df.columns or date_column not in df.columns:
                continue
            
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            
            df = df[df[date_column] >= cutoff_date]
            if len(df) < 30:
                continue
            
            columns_to_drop = [col for col in df.columns 
                              if col not in [date_column, target_column] and df[col].dtype == 'object']
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)

            df = df.dropna(subset=[target_column])
            df = df[(df[target_column] <= 1000) & (df[target_column] >= -1000)]
            
            feature_columns = [col for col in df.columns if col not in [date_column, target_column]]
            
            nan_threshold = 0.5
            high_nan_cols = []
            for col in feature_columns:
                nan_ratio = df[col].isnull().sum() / len(df)
                if nan_ratio > nan_threshold:
                    high_nan_cols.append(col)
            
            if high_nan_cols:
                df = df.drop(columns=high_nan_cols)
                feature_columns = [col for col in df.columns if col not in [date_column, target_column]]
            
            df = df.dropna(subset=feature_columns)
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 30:
                continue
            
            train_subset = df[df[date_column] < train_end_date].copy()
            calib_subset = df[(df[date_column] >= train_end_date) & 
                             (df[date_column] < calib_end_date)].copy()
            
            if len(train_subset) >= 10:
                train_data.append(train_subset)
                train_rows += len(train_subset)
            
            if len(calib_subset) >= 10:
                calib_data.append(calib_subset)
                calib_rows += len(calib_subset)
                
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
    
    logging.info(f"Collected {train_rows} training rows and {calib_rows} calibration rows")
    
    if calib_rows < 10000:
        logging.error(f"Calibration set too small: {calib_rows} rows collected")
        logging.error("Need at least 10,000 rows for reliable calibration")
        logging.error("Options:")
        logging.error(f"  1. Increase --calibpercent (currently {calibration_percentage}%)")
        logging.error("  2. Check that your data has enough rows per file")
        logging.error("  3. Verify date filtering isn't removing too much data")
        raise ValueError(f"Insufficient calibration data: {calib_rows} rows, need 10,000+")
    
    if len(train_data) == 0:
        raise ValueError("No valid training data found after temporal split")
    
    if len(calib_data) == 0:
        logging.warning("No calibration data found in specified time period")
        logging.warning("Using last 20% of training data for calibration")
        train_df = pd.concat(train_data)
        train_df = train_df.sort_values(by=date_column)
        split_idx = int(len(train_df) * 0.8)
        calib_df = train_df.iloc[split_idx:]
        train_df = train_df.iloc[:split_idx]
    else:
        train_df = pd.concat(train_data)
        calib_df = pd.concat(calib_data)
    
    train_df = train_df.sort_values(by=date_column).reset_index(drop=True)
    calib_df = calib_df.sort_values(by=date_column).reset_index(drop=True)
    
    if os.path.exists(train_output_file):
        os.remove(train_output_file)
    if os.path.exists(calib_output_file):
        os.remove(calib_output_file)
    
    train_df.to_parquet(train_output_file, index=False)
    calib_df.to_parquet(calib_output_file, index=False)
    
    logging.info("=" * 80)
    logging.info(f"TRAINING DATA:    {len(train_df):,} rows from {train_df[date_column].min().date()} to {train_df[date_column].max().date()}")
    logging.info(f"CALIBRATION DATA: {len(calib_df):,} rows from {calib_df[date_column].min().date()} to {calib_df[date_column].max().date()}")
    logging.info("=" * 80)
    logging.info("TEMPORAL SPLIT COMPLETE - No data leakage between train/calib")
    logging.info("=" * 80)
    
    return train_df, calib_df

def plot_calibration_curves(clf, X_calib, y_calib, calibrated_clf=None, output_path=None):
    plt.figure(figsize=(10, 8))

    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

    y_prob = clf.predict_proba(X_calib)
    y_prob = to_cpu_array(y_prob)[:, 1]

    prob_true, prob_pred = calibration_curve(y_calib, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, 's-', label='Original XGBoost model')

    if calibrated_clf is not None:
        try:
            calibrated_prob = calibrated_clf.predict_proba(X_calib)
            calibrated_prob = to_cpu_array(calibrated_prob)[:, 1]
            calib_prob_true, calib_prob_pred = calibration_curve(y_calib, calibrated_prob, n_bins=10)
            plt.plot(calib_prob_pred, calib_prob_true, 's-',
                     label='Calibrated model (time series aware)', linewidth=2)
        except Exception as e:
            logging.warning(f"Could not plot calibrated curve: {str(e)}")

    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve - XGBoost Model')
    plt.legend(loc='best')
    plt.grid(True)

    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logging.info(f"Calibration plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving calibration plot: {str(e)}")

    plt.close()

def plot_time_series_calibration_comparison(base_probs, ts_calibrated_probs, y_calib, output_path=None):
    """Plot comparison between base and time series calibrated probabilities"""
    plt.figure(figsize=(14, 10))

    # Plot 1: Calibration curves
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated', linewidth=2)

    # Base model calibration
    prob_true_base, prob_pred_base = calibration_curve(y_calib, base_probs, n_bins=10)
    plt.plot(prob_pred_base, prob_true_base, 's-', label='Base model', linewidth=2, markersize=8)

    # Time series calibrated
    prob_true_ts, prob_pred_ts = calibration_curve(y_calib, ts_calibrated_probs, n_bins=10)
    plt.plot(prob_pred_ts, prob_true_ts, 'o-', label='Time series calibrated', linewidth=2, markersize=8)

    plt.xlabel('Mean predicted probability', fontsize=11)
    plt.ylabel('Fraction of positives', fontsize=11)
    plt.title('Calibration Curve Comparison', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 2: Probability distributions
    plt.subplot(2, 2, 2)
    plt.hist(base_probs, bins=50, alpha=0.6, label='Base probs', density=True, color='blue')
    plt.hist(ts_calibrated_probs, bins=50, alpha=0.6, label='TS calibrated', density=True, color='red')
    plt.xlabel('Probability', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Probability Distributions', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 3: Scatter plot of base vs calibrated
    plt.subplot(2, 2, 3)
    plt.scatter(base_probs, ts_calibrated_probs, alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'k--', label='No change', linewidth=2)
    plt.xlabel('Base probability', fontsize=11)
    plt.ylabel('Calibrated probability', fontsize=11)
    plt.title('Base vs Calibrated Probabilities', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot 4: Time series of probabilities (sample)
    plt.subplot(2, 1, 2)
    n_plot = min(1000, len(base_probs))
    x_indices = np.arange(n_plot)
    plt.plot(x_indices, base_probs[:n_plot], 'b-', alpha=0.5, label='Base probs', linewidth=1)
    plt.plot(x_indices, ts_calibrated_probs[:n_plot], 'r-', alpha=0.7, label='TS calibrated', linewidth=1)
    plt.xlabel('Time index (sorted by date)', fontsize=11)
    plt.ylabel('Probability', fontsize=11)
    plt.title(f'Probability Time Series (First {n_plot} points)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"Time series calibration comparison plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving comparison plot: {str(e)}")

    plt.close()

def find_optimal_precision_threshold(probs, labels, min_samples=100, target_precision=0.70):
    thresholds = np.arange(0.50, 0.95, 0.01)
    
    results = []
    
    for threshold in thresholds:
        mask = probs >= threshold
        n_predictions = mask.sum()
        
        if n_predictions < min_samples:
            continue
        
        true_positives = (labels[mask] == 1).sum()
        false_positives = (labels[mask] == 0).sum()
        
        precision = true_positives / n_predictions if n_predictions > 0 else 0.0
        
        actual_positives = (labels == 1).sum()
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'n_predictions': n_predictions,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'above_target': precision >= target_precision
        })
    
    if not results:
        return None
    
    high_precision_results = [r for r in results if r['above_target']]
    
    if high_precision_results:
        best = max(high_precision_results, key=lambda x: x['n_predictions'])
    else:
        best = max(results, key=lambda x: x['precision'])
        logging.warning(f"Could not achieve target precision {target_precision:.2f}, best found: {best['precision']:.4f}")
    
    return best

def train_model(training_data, calibration_data, config):
    apply_calibration = config['apply_calibration']
    
    logging.info(f"Training XGBoost model with calibration={apply_calibration}.")
    logging.info("=" * 80)
    logging.info("LONG-ONLY STRATEGY: HIGH PRECISION UP PREDICTIONS")
    logging.info("Goal: Minimize false positives (predicting UP when stock goes DOWN)")
    logging.info("=" * 80)
    
    model_filename = "xgboost_model.joblib"
    model_output_path = os.path.join(config['model_output_directory'], model_filename)
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    training_data = training_data.sort_values('Date')
    calibration_data = calibration_data.sort_values('Date')
    
    X_train = training_data.drop(columns=[config['target_column']])
    y_train = training_data[config['target_column']]
    y_train = y_train.apply(lambda x: 0 if x < 0 else 1)
    
    X_calib = calibration_data.drop(columns=[config['target_column']])
    y_calib = calibration_data[config['target_column']]
    y_calib = y_calib.apply(lambda x: 0 if x < 0 else 1)
    
    datetime_columns_train = X_train.select_dtypes(include=['datetime64']).columns
    datetime_columns_calib = X_calib.select_dtypes(include=['datetime64']).columns
    
    X_train = X_train.drop(columns=datetime_columns_train)
    X_calib = X_calib.drop(columns=datetime_columns_calib)
    
    split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_train_final = y_train.iloc[:split_idx]
    y_val = y_train.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_final_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val)
    X_calib_scaled = scaler.transform(X_calib)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train_final
    )
    
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    logging.info(f"Class distribution in training: Class 0: {(y_train_final == 0).sum()}, Class 1: {(y_train_final == 1).sum()}")
    logging.info(f"Class weights: {class_weight_dict}")
    
    sample_weights_train = np.array([class_weight_dict[y] for y in y_train_final])
    sample_weights_val = np.array([class_weight_dict[y] for y in y_val])
    
    xgboost_params = config['xgboost_params'].copy()
    training_params = config['training_params'].copy()
    
    xgboost_params['scale_pos_weight'] = class_weight_dict[1] / class_weight_dict[0]
    xgboost_params['early_stopping_rounds'] = training_params['early_stopping_rounds']
    
    clf = XGBClassifier(**xgboost_params)
    
    try:
        clf.fit(
            X_train_final_scaled, y_train_final,
            eval_set=[(X_val_scaled, y_val)],
            sample_weight=sample_weights_train,
            sample_weight_eval_set=[sample_weights_val],
            verbose=training_params['verbose']
        )
        logging.info("XGBoost model trained successfully with class weights.")
    except Exception as e:
        logging.error(f"Error during XGBoost fitting: {str(e)}")
        raise
    
    calibrated_clf = None
    if apply_calibration:
        logging.info("=" * 80)
        logging.info("TIME SERIES AWARE CALIBRATION - Preventing Data Leakage")
        logging.info("Using expanding window: only past data used for each calibration")
        logging.info("=" * 80)

        # Prepare calibration data with Date column for time series aware calibration
        calibration_data_with_date = calibration_data.copy()
        X_calib_with_date = calibration_data_with_date.drop(columns=[config['target_column']])
        y_calib_sorted = calibration_data_with_date[config['target_column']].apply(lambda x: 0 if x < 0 else 1)

        # Scale features but keep Date column for sorting
        X_calib_features = X_calib_with_date.drop(columns=['Date'])
        X_calib_scaled_arr = scaler.transform(X_calib_features)

        # Create DataFrame with scaled features + Date
        X_calib_final = pd.DataFrame(
            X_calib_scaled_arr,
            columns=X_calib_features.columns
        )
        X_calib_final['Date'] = calibration_data_with_date['Date'].values

        try:
            # Perform time series aware calibration for validation/evaluation
            logging.info("Step 1: Performing expanding window time series calibration for validation...")
            logging.info("This shows how well calibration works with NO lookahead bias")

            # Get base probabilities for comparison
            base_probs = clf.predict_proba(X_calib_scaled)
            base_probs = to_cpu_array(base_probs)[:, 1]

            ts_calibrated_probs = time_series_calibrate(
                clf=clf,
                X_calib=X_calib_final,
                y_calib=y_calib_sorted,
                date_column='Date',
                method='isotonic'
            )

            # Create production calibrator using entire calibration set
            logging.info("=" * 80)
            logging.info("Step 2: Creating production calibrator on entire calibration set...")
            logging.info("This is used for future predictions (already separated by time)")
            logging.info("=" * 80)

            calibrated_clf = TimeSeriesCalibratedModel(
                base_model=clf,
                calibration_method='isotonic',
                calibration_data=(
                    X_calib_final,
                    y_calib_sorted,
                    'Date'
                )
            )

            if calibrated_clf.is_fitted:
                logging.info("Time series aware calibration completed successfully")

                # Plot comparison between base and time series calibrated
                try:
                    comparison_plot_path = config['calibration_plot_output'].replace('.png', '_ts_comparison.png')
                    plot_time_series_calibration_comparison(
                        base_probs=base_probs,
                        ts_calibrated_probs=ts_calibrated_probs,
                        y_calib=y_calib_sorted,
                        output_path=comparison_plot_path
                    )
                except Exception as e:
                    logging.error(f"Failed to create comparison plot: {str(e)}")

                # Plot standard calibration curves
                try:
                    plot_calibration_curves(
                        clf, X_calib_scaled, y_calib,
                        calibrated_clf=calibrated_clf,
                        output_path=config['calibration_plot_output']
                    )
                except Exception as e:
                    logging.error(f"Failed to create calibration plot: {str(e)}")
            else:
                logging.error("Time series calibration failed to fit properly")
                calibrated_clf = None

        except Exception as e:
            logging.error(f"Error during time series calibration: {str(e)}")
            logging.info("Falling back to standard calibration (WARNING: may have lookahead bias)...")
            try:
                y_calib_probs = clf.predict_proba(X_calib_scaled)
                y_calib_probs = to_cpu_array(y_calib_probs)[:, 1]

                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(y_calib_probs, y_calib)

                calibrated_clf = CalibratedModel(clf, 1.0, iso_reg)
                logging.info("Fallback calibration completed (isotonic regression)")
            except Exception as e2:
                logging.error(f"Fallback calibration also failed: {str(e2)}")
                calibrated_clf = None
    
    if calibrated_clf is not None:
        clf_for_prediction = calibrated_clf
        logging.info("Using calibrated model for predictions.")
    else:
        clf_for_prediction = clf
        logging.info("Using uncalibrated model for predictions.")
    
    y_pred_proba = clf_for_prediction.predict_proba(X_calib_scaled)
    y_pred_proba = to_cpu_array(y_pred_proba)

    logging.info("="*80)
    logging.info("HIGH-PRECISION THRESHOLD OPTIMIZATION")
    logging.info(f"Target: {args.target_precision*100:.0f}%+ precision on UP predictions")
    logging.info("="*80)
    
    logging.info(f"Probability stats: min={y_pred_proba[:, 1].min():.4f}, max={y_pred_proba[:, 1].max():.4f}, mean={y_pred_proba[:, 1].mean():.4f}")
    logging.info(f"  50th percentile: {np.percentile(y_pred_proba[:, 1], 50):.4f}")
    logging.info(f"  60th percentile: {np.percentile(y_pred_proba[:, 1], 60):.4f}")
    logging.info(f"  70th percentile: {np.percentile(y_pred_proba[:, 1], 70):.4f}")
    logging.info(f"  80th percentile: {np.percentile(y_pred_proba[:, 1], 80):.4f}")
    logging.info(f"  90th percentile: {np.percentile(y_pred_proba[:, 1], 90):.4f}")
    
    if args.fixed_threshold is not None:
        optimal_threshold_pos = args.fixed_threshold
        logging.info(f"\nUsing FIXED threshold: {optimal_threshold_pos:.4f} (from --fixed_threshold)")

        mask = y_pred_proba[:, 1] >= optimal_threshold_pos
        n_preds = mask.sum()
        true_positives = (y_calib[mask] == 1).sum()
        false_positives = (y_calib[mask] == 0).sum()
        precision = true_positives / n_preds if n_preds > 0 else 0.0

        logging.info(f"  Precision (Win Rate): {precision:.4f} ({precision*100:.2f}%)")
        logging.info(f"  Predictions: {n_preds:,} ({n_preds/len(y_calib)*100:.2f}% coverage)")
        logging.info(f"  False Positives (COST MONEY): {false_positives:,}")
    else:
        logging.info(f"\nSearching for high-precision threshold (target: {args.target_precision*100:.0f}%+)...")
        best_result = find_optimal_precision_threshold(
            y_pred_proba[:, 1], 
            y_calib, 
            min_samples=100,
            target_precision=args.target_precision
        )

        if best_result is None:
            optimal_threshold_pos = 0.75
            logging.warning(f"Could not find optimal threshold, using conservative default: {optimal_threshold_pos:.4f}")
        else:
            optimal_threshold_pos = best_result['threshold']
            logging.info(f"\nOptimal high-precision threshold found: {optimal_threshold_pos:.4f}")
            logging.info(f"  Precision (Win Rate): {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
            logging.info(f"  Recall: {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
            logging.info(f"  Predictions: {best_result['n_predictions']:,} ({best_result['n_predictions']/len(y_calib)*100:.2f}% coverage)")
            logging.info(f"  True Positives: {best_result['true_positives']:,}")
            logging.info(f"  False Positives (COST MONEY): {best_result['false_positives']:,}")

    logging.info("="*80)
    logging.info("APPLYING HIGH-PRECISION UP-ONLY STRATEGY")
    logging.info("Long-only: Only predict UP, never predict DOWN")
    logging.info("="*80)
    
    y_pred = np.full(len(y_calib), -1)
    
    up_mask = y_pred_proba[:, 1] >= optimal_threshold_pos
    y_pred[up_mask] = 1
    up_count = up_mask.sum()
    logging.info(f"  UP predictions: {up_count:,} ({up_count/len(y_pred)*100:.2f}%)")

    no_pred = (y_pred == -1).sum()
    logging.info(f"  No prediction (hold cash): {no_pred:,} ({no_pred/len(y_pred)*100:.2f}%)")

    mask_definitive = y_pred == 1
    y_calib_filtered = y_calib[mask_definitive]
    y_pred_filtered = y_pred[mask_definitive]
    
    logging.info(f"\nTotal UP predictions: {len(y_pred_filtered):,} out of {len(y_calib):,}")
    
    if len(y_pred_filtered) > 0:
        up_count = (y_pred_filtered == 1).sum()
        up_correct = ((y_pred_filtered == 1) & (y_calib_filtered == 1)).sum()
        up_incorrect = ((y_pred_filtered == 1) & (y_calib_filtered == 0)).sum()
        
        up_precision = up_correct / up_count if up_count > 0 else 0.0
        
        total_actual_positives = (y_calib == 1).sum()
        up_recall = up_correct / total_actual_positives if total_actual_positives > 0 else 0.0
        
        logging.info("="*80)
        logging.info(f"LONG-ONLY STRATEGY PERFORMANCE ON CALIBRATION SET")
        logging.info(f"  UP Predictions Made: {up_count:,}")
        logging.info(f"  Correct (True Positives): {up_correct:,}")
        logging.info(f"  WRONG (False Positives - COSTS MONEY): {up_incorrect:,}")
        logging.info(f"  Win Rate (Precision): {up_precision:.4f} ({up_precision*100:.2f}%)")
        logging.info(f"  False Positive Rate: {up_incorrect/up_count*100:.2f}%")
        logging.info(f"  Recall (captured opportunities): {up_recall:.4f} ({up_recall*100:.2f}%)")
        logging.info("="*80)
        
        expected_value = up_precision - 0.5
        logging.info(f"  Expected Value per trade: {expected_value*100:.2f}%")
        
        if up_precision < 0.55:
            logging.warning("WARNING: Win rate below 55% - model may not be profitable")
        elif up_precision >= args.target_precision:
            logging.info(f"SUCCESS: Win rate meets or exceeds target of {args.target_precision*100:.0f}%")
    else:
        logging.warning("No UP predictions made after applying threshold.")
    
    model_data = {
        'base_model': clf,
        'calibrated_model': clf_for_prediction if calibrated_clf is not None else None,
        'is_calibrated': calibrated_clf is not None,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist(),
        'threshold_pos': optimal_threshold_pos,
        'threshold_neg': None,
    }
    
    dump(model_data, model_output_path)
    logging.info(f"Model and threshold saved to {model_output_path}")
    
    try:
        feature_importance = clf.feature_importances_
        
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values(by='importance', ascending=False)
        
        feature_importances['importance'] = feature_importances['importance'].round(5)
        feature_importances.to_parquet(config['feature_importance_output'], index=False)
        logging.info(f"XGBoost feature importances saved to {config['feature_importance_output']}")
        
        logging.info(f"Top 10 most important features:")
        for idx, row in feature_importances.head(10).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        logging.error(f"Error saving feature importances: {str(e)}")
    
    return model_data

def predict_and_save(input_directory, model_path, output_directory, target_column, date_column):
    logging.info("Loading the trained XGBoost model for prediction.")
    
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    model_data = load(model_path)
    
    if isinstance(model_data, dict):
        if 'calibrated_model' in model_data and model_data['calibrated_model'] is not None:
            clf = model_data['calibrated_model']
            logging.info("Using calibrated XGBoost model for predictions.")
        elif 'base_model' in model_data:
            clf = model_data['base_model']
            logging.info("Using base XGBoost model for predictions.")
        else:
            clf = model_data
            logging.warning("Using model from legacy format.")
        
        scaler = model_data.get('scaler', None)
        model_features = model_data.get('feature_names', None)
        
        if 'threshold_pos' in model_data:
            threshold_pos = model_data['threshold_pos']
            threshold_neg = model_data.get('threshold_neg', None)
            logging.info(f"Using threshold for UP predictions: {threshold_pos:.4f}")
            if threshold_neg is not None:
                logging.info(f"Note: threshold_neg={threshold_neg} (not used in long-only strategy)")
        else:
            threshold_pos = 0.70
            threshold_neg = None
            logging.warning("Using default threshold 0.70")
    else:
        clf = model_data
        scaler = None
        model_features = None
        threshold_pos = 0.70
        threshold_neg = None
        logging.warning("Using default threshold 0.70 with legacy model format")
    
    if scaler is None:
        logging.error("No scaler found in model data. Scaling is required.")
        raise ValueError("No scaler found in model data")
    
    if model_features is None:
        logging.warning("Could not determine feature names from model, using all features.")
        sample_file = os.path.join(input_directory, os.listdir(input_directory)[0])
        sample_df = pd.read_parquet(sample_file)
        datetime_columns = sample_df.select_dtypes(include=['datetime64']).columns
        model_features = [col for col in sample_df.columns if col not in [date_column, target_column] + list(datetime_columns)]
    
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    total_predictions = 0
    up_predictions = 0
    
    for file in all_files:
        try:
            df = pd.read_parquet(os.path.join(input_directory, file))
            df[date_column] = pd.to_datetime(df[date_column])
            
            if df.shape[0] < 252:
                pbar.update(1)
                continue

            datetime_columns = df.select_dtypes(include=['datetime64']).columns
            X = df.drop(columns=[col for col in [date_column, target_column] + list(datetime_columns) if col in df.columns])
            
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype == 'bool':
                    X[col] = pd.to_numeric(X[col], errors='coerce')
            
            missing_features = set(model_features) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0.0
                    
            X = X.reindex(columns=model_features, fill_value=0.0)
            
            X = X.astype(float, errors='ignore')
            
            X = X.replace([np.inf, -np.inf], 0.0)
            X = X.fillna(0.0)
            
            try:
                X_scaled = scaler.transform(X)
                
                if not np.isfinite(X_scaled).all():
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
            except Exception as scale_error:
                logging.error(f"Scaling error for {file}: {str(scale_error)}")
                pbar.update(1)
                continue
            
            try:
                y_pred_proba = clf.predict_proba(X_scaled)
                y_pred_proba = to_cpu_array(y_pred_proba)
            except Exception as pred_error:
                logging.error(f"Prediction error for {file}: {str(pred_error)}")
                pbar.update(1)
                continue
            
            df['UpProbability'] = y_pred_proba[:, 1]
            df['DownProbability'] = y_pred_proba[:, 0]
            
            df['PositiveThreshold'] = threshold_pos
            df['NegativeThreshold'] = None
            
            epsilon = 1e-3
            df['UpProbability'] = df['UpProbability'].clip(epsilon, 1-epsilon)
            df['DownProbability'] = df['DownProbability'].clip(epsilon, 1-epsilon)

            df['UpPrediction'] = -1
            df.loc[df['UpProbability'] >= threshold_pos, 'UpPrediction'] = 1
                    
            total_predictions += len(df)
            up_predictions += (df['UpPrediction'] == 1).sum()
            
            required_columns = [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'UpProbability', 'DownProbability', 
                'PositiveThreshold', 'NegativeThreshold', 'UpPrediction', 'VIX_Close'
            ]
            
            corr_related_cols = [
                col for col in df.columns
                if any(key in col.lower() for key in ['corr_', 'corr', 'beta_', 'beta', 'alpha_', 'alpha'])
            ]
            required_columns += corr_related_cols
            
            optional_columns = ['Distance to Resistance (%)', 'Distance to Support (%)', 'volatility']
            for col in optional_columns:
                if col in df.columns:
                    required_columns.append(col)
            
            available_columns = [col for col in required_columns if col in df.columns]
            for col in set(required_columns) - set(available_columns):
                df[col] = np.nan
            
            output_df = df[[col for col in required_columns if col in df.columns]]
            
            output_file_path = os.path.join(output_directory, file)
            output_df.to_parquet(output_file_path, index=False)
            
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
        
        pbar.update(1)
    
    pbar.close()
    
    if total_predictions > 0:
        prediction_rate = (up_predictions / total_predictions) * 100
        logging.info(f"UP prediction coverage: {up_predictions:,} out of {total_predictions:,} ({prediction_rate:.2f}%)")
    
    logging.info(f"XGBoost predictions saved to {output_directory}")

def main():
    os.makedirs(config['model_output_directory'], exist_ok=True)
    os.makedirs(config['data_output_directory'], exist_ok=True)
    os.makedirs(config['calibration_output_directory'], exist_ok=True)
    os.makedirs(config['prediction_output_directory'], exist_ok=True)
    os.makedirs(os.path.dirname(config['feature_importance_output']), exist_ok=True)
    
    model_filename = "xgboost_model.joblib"
    
    if not args.predict:
        training_data, calibration_data = prepare_data_splits(
            input_directory=config['input_directory'],
            train_output_directory=config['data_output_directory'],
            calib_output_directory=config['calibration_output_directory'],
            file_selection_percentage=config['file_selection_percentage'],
            calibration_percentage=config['calibration_percentage'],
            target_column=config['target_column'],
            reuse=args.reuse,
            date_column='Date'
        )
        logging.info("Data preparation complete.")
        print("Data preparation complete, starting XGBoost model training...")
        
        train_model(training_data, calibration_data, config)

    else:
        predict_and_save(
            input_directory=config['input_directory'],
            model_path=os.path.join(config['model_output_directory'], model_filename),
            output_directory=config['prediction_output_directory'],
            target_column=config['target_column'],
            date_column='Date'
        )





if __name__ == "__main__":
    main()