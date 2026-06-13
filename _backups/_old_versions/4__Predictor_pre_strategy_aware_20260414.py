##predictor script with calibration
import os
import random
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from tqdm import tqdm
from joblib import parallel_backend
from contextlib import redirect_stdout, redirect_stderr
import io
import matplotlib.pyplot as plt
from Util import get_logger
logger = get_logger(script_name="4__Predictor")

argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=65, help="Percentage of files to process.")
argparser.add_argument("--calibpercent", type=int, default=15, help="Percentage of remaining files to use for calibration.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--nocalib", action='store_true', help="Flag to disable probability calibration.")
argparser.add_argument("--tune", action='store_true', help="Run Optuna hyperparameter tuning before training.")
argparser.add_argument("--tune_trials", type=int, default=50, help="Number of Optuna trials.")
args = argparser.parse_args()


class SmoothIsotonicCalibrator:
    """Isotonic regression smoothed with monotone cubic interpolation (PCHIP).

    Standard isotonic calibration produces a piecewise-constant step function,
    which collapses many raw XGBoost scores to the same calibrated probability.
    This wrapper fits isotonic regression then interpolates smoothly through
    the breakpoints using PCHIP, which preserves monotonicity by construction.
    """

    def __init__(self):
        self.spline = None
        self.x_min = None
        self.x_max = None

    def fit(self, raw_probs, y_true):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_probs, y_true)
        self.spline = PchipInterpolator(iso.X_thresholds_, iso.y_thresholds_)
        self.x_min = iso.X_thresholds_.min()
        self.x_max = iso.X_thresholds_.max()
        n_breakpoints = len(iso.X_thresholds_)
        n_unique_y = len(np.unique(iso.y_thresholds_))
        logging.info(f"SmoothIsotonicCalibrator: {n_breakpoints} breakpoints, "
                     f"{n_unique_y} unique isotonic levels, "
                     f"raw range [{self.x_min:.4f}, {self.x_max:.4f}]")
        return self

    def predict(self, raw_probs):
        clipped = np.clip(raw_probs, self.x_min, self.x_max)
        return np.clip(self.spline(clipped), 0.0, 1.0)


class HybridCalibrator:
    """Isotonic+PCHIP for the bulk of the distribution, Platt scaling for the tail.

    Standard isotonic calibration has sparse breakpoints above ~0.65-0.70, which
    causes PCHIP to create a near-flat plateau that destroys ranking information.
    This hybrid uses isotonic+PCHIP where breakpoints are dense (good calibration)
    and switches to a 2-parameter logistic (Platt scaling) in the sparse tail,
    preserving the ranking of raw scores where it matters most for bet sizing.
    """

    def __init__(self, min_tail_breakpoints=10, blend_width=0.03):
        self.spline = None
        self.x_min = None
        self.x_max = None
        self.crossover = None
        self.tail_model = None
        self.blend_width = blend_width
        self.min_tail_breakpoints = min_tail_breakpoints

    def fit(self, raw_probs, y_true):
        raw_probs = np.asarray(raw_probs, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        # Step 1: fit isotonic on all data (same as SmoothIsotonicCalibrator)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_probs, y_true)
        self.spline = PchipInterpolator(iso.X_thresholds_, iso.y_thresholds_)
        self.x_min = iso.X_thresholds_.min()
        self.x_max = iso.X_thresholds_.max()

        n_breakpoints = len(iso.X_thresholds_)
        n_unique_y = len(np.unique(iso.y_thresholds_))

        # Step 2: find the crossover point where isotonic becomes sparse
        # Walk from highest breakpoint downward, find where we have fewer than
        # min_tail_breakpoints remaining above that point
        thresholds_sorted = np.sort(iso.X_thresholds_)
        self.crossover = None
        for candidate in np.arange(0.50, 0.85, 0.01):
            n_above = (thresholds_sorted >= candidate).sum()
            if n_above < self.min_tail_breakpoints:
                self.crossover = candidate
                break

        if self.crossover is None:
            # Isotonic has plenty of breakpoints everywhere, no tail model needed
            logging.info(f"HybridCalibrator: {n_breakpoints} breakpoints, "
                         f"{n_unique_y} unique isotonic levels, "
                         f"raw range [{self.x_min:.4f}, {self.x_max:.4f}], "
                         f"no tail crossover needed (dense breakpoints throughout)")
            return self

        # Step 3: fit Platt scaling (logistic regression) on tail data
        tail_mask = raw_probs >= (self.crossover - 0.05)  # include some overlap for fit stability
        n_tail = tail_mask.sum()

        if n_tail < 50:
            logging.warning(f"HybridCalibrator: only {n_tail} tail samples above "
                            f"crossover {self.crossover:.2f}, skipping tail model")
            self.crossover = None
            return self

        X_tail = raw_probs[tail_mask].reshape(-1, 1)
        y_tail = y_true[tail_mask]

        self.tail_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        self.tail_model.fit(X_tail, y_tail)

        # Verify monotonicity of the tail model in the relevant range
        test_pts = np.linspace(self.crossover, self.x_max, 20).reshape(-1, 1)
        tail_preds = self.tail_model.predict_proba(test_pts)[:, 1]
        if not np.all(np.diff(tail_preds) >= -1e-6):
            logging.warning("HybridCalibrator: tail logistic model is non-monotonic, "
                            "falling back to PCHIP-only")
            self.crossover = None
            self.tail_model = None
            return self

        n_tail_bp_above = (thresholds_sorted >= self.crossover).sum()
        logging.info(f"HybridCalibrator: {n_breakpoints} breakpoints, "
                     f"{n_unique_y} unique isotonic levels, "
                     f"raw range [{self.x_min:.4f}, {self.x_max:.4f}], "
                     f"tail crossover at {self.crossover:.4f} "
                     f"({n_tail_bp_above} breakpoints above, {n_tail} tail samples for Platt fit)")
        return self

    def predict(self, raw_probs):
        raw_probs = np.asarray(raw_probs, dtype=np.float64)
        clipped = np.clip(raw_probs, self.x_min, self.x_max)

        # PCHIP prediction for everything (baseline)
        result = np.clip(self.spline(clipped), 0.0, 1.0)

        if self.crossover is None or self.tail_model is None:
            return result

        # Tail logistic prediction
        tail_pred = self.tail_model.predict_proba(clipped.reshape(-1, 1))[:, 1]

        # Blend zone: [crossover - blend_width, crossover + blend_width]
        lo = self.crossover - self.blend_width
        hi = self.crossover + self.blend_width

        # Below lo: pure PCHIP (already in result)
        # Above hi: pure tail logistic
        above_hi = clipped > hi
        result[above_hi] = tail_pred[above_hi]

        # In blend zone: linear interpolation
        in_blend = (clipped >= lo) & (clipped <= hi)
        if in_blend.any():
            w = (clipped[in_blend] - lo) / (hi - lo)  # 0 at lo, 1 at hi
            result[in_blend] = (1 - w) * result[in_blend] + w * tail_pred[in_blend]

        return np.clip(result, 0.0, 1.0)


def rank_refine_tail(calibrated_probs, raw_margins, tail_cutoff=0.65):
    """Re-spread calibrated probabilities in the tail using raw margin ranking.

    XGBoost's sigmoid compresses differences in the tails: two stocks at calibrated
    0.72 and 0.78 may have raw margins of 0.94 and 1.25 — a huge difference in
    log-odds space that is invisible after sigmoid + isotonic calibration.

    This function preserves the calibrated range [min, max] in the tail but re-orders
    and re-spaces the values according to raw margin ranks, restoring the discrimination
    that calibration destroyed.
    """
    calibrated_probs = np.asarray(calibrated_probs, dtype=np.float64)
    raw_margins = np.asarray(raw_margins, dtype=np.float64)
    mask = calibrated_probs >= tail_cutoff

    n_tail = mask.sum()
    if n_tail < 10:
        return calibrated_probs

    tail_calib = calibrated_probs[mask]
    tail_margins = raw_margins[mask]

    # Rank by raw margin (preserves XGBoost's true ordering in log-odds space)
    margin_ranks = tail_margins.argsort().argsort().astype(np.float64)
    margin_quantiles = margin_ranks / max(n_tail - 1, 1)  # 0..1

    # Map ranks to the calibrated range, preserving the overall calibration level
    tail_min = tail_calib.min()
    tail_max = tail_calib.max()

    # If the range is tiny (flat plateau), expand slightly using margin info.
    # Ceiling is capped at 0.99 (not 0.999) to prevent the threshold search from
    # landing at extreme values like 0.9944 due to artificially pushed-up probabilities.
    if tail_max - tail_min < 0.02:
        center = tail_calib.mean()
        tail_min = max(center - 0.05, tail_cutoff)
        tail_max = min(center + 0.01, 0.99)

    refined = tail_min + margin_quantiles * (tail_max - tail_min)

    result = calibrated_probs.copy()
    result[mask] = refined
    return result



config = {
    "input_directory": "Data/ProcessedData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "calibration_output_directory": "Data/ModelData/CalibrationData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "calibration_plot_output": "Data/ModelData/calibration_plot.png",
    "log_file": "data/logging/4__XGBoostPredictor.log",
    "file_selection_percentage": args.runpercent,
    "calibration_percentage": args.calibpercent,
    "target_column": "percent_change_Close",
    "apply_calibration": not args.nocalib,

    # XGBoost parameters
    "xgb_params": {
        "num_parallel_tree": 1,
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "gamma": 0.3,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": 1.6,  # overridden dynamically from data
        "random_state": 3301,
        "verbosity": 1,
        "nthread": 32,
        "tree_method": "hist",
        "early_stopping_rounds": 30,
    },
    
    # Common parameters
    "early_stopping_rounds": 10,
    "random_state": 3301,

    # Magnitude model parameters
    "magnitude_params": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "gamma": 0.3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "objective": "reg:squarederror",
        "random_state": 3301,
        "nthread": 32,
        "tree_method": "hist",
        "early_stopping_rounds": 20,
    },
    "magnitude_alpha": 0.3,  # how much magnitude adjusts direction probability
}


def drop_string_columns(df, date_column, target_column):
    columns_to_drop = [col for col in df.columns 
                       if col not in [date_column, target_column] and df[col].dtype == 'object']
    if columns_to_drop:
        logging.info(f"Dropping columns due to string data: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    return df


##=======================================[Prepare Training and Calibration Data]=======================================##

def prepare_data_splits(input_directory, train_output_directory, calib_output_directory, 
                       file_selection_percentage, calibration_percentage, target_column, reuse, date_column):
    """
    Prepare training and calibration data splits. The calibration data is taken from files
    not used in training to ensure proper evaluation.
    """
    train_output_file = os.path.join(train_output_directory, 'training_data.parquet')
    calib_output_file = os.path.join(calib_output_directory, 'calibration_data.parquet')
    file_allocation_record = os.path.join(train_output_directory, 'file_allocation.json')
    
    # Create calibration output directory if it doesn't exist
    os.makedirs(calib_output_directory, exist_ok=True)
    
    # If reusing existing data and both files exist, load and return them
    if reuse and os.path.exists(train_output_file) and os.path.exists(calib_output_file):
        logging.info("Reusing existing training and calibration data.")
        print("Reusing existing training and calibration data.")
        return pd.read_parquet(train_output_file), pd.read_parquet(calib_output_file)
    
    logging.info("Preparing new training and calibration data with anti-leakage measures.")
    
    # Define cutoff date for data filtering
    cutoff_date = pd.to_datetime('2021-01-10')
    logging.info(f"Filtering out data before {cutoff_date.strftime('%Y-%m-%d')}")
    
    # Get all valid files
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    all_files = sorted(all_files)  # Alphabetical sort often maintains chronology
    
    # Determine how many files to use for training
    total_files = len(all_files)
    train_files_count = int(total_files * file_selection_percentage / 100)
    
    # Shuffle files with fixed seed for reproducibility, then select training files
    random.seed(config['random_state'])
    random.shuffle(all_files)
    
    train_files = all_files[:train_files_count]
    remaining_files = all_files[train_files_count:]
    
    # Determine how many of the remaining files to use for calibration
    calib_files_count = int(len(remaining_files) * calibration_percentage / 100)
    calib_files = remaining_files[:calib_files_count]
    
    # Save the file allocation for future reference
    file_allocation = {
        'training_files': train_files,
        'calibration_files': calib_files,
        'unused_files': remaining_files[calib_files_count:]
    }
    
    # Save file allocation as JSON for reference
    import json
    with open(file_allocation_record, 'w') as f:
        json.dump(file_allocation, f, indent=2)
    
    logging.info(f"Training files: {len(train_files)}, Calibration files: {len(calib_files)}")
    
    # Process training files
    if os.path.exists(train_output_file):
        os.remove(train_output_file)
    
    train_data = process_files(train_files, input_directory, cutoff_date, target_column, date_column)
    if len(train_data) == 0:
        logging.error("No valid training data found after processing files and date filtering.")
        raise ValueError("No valid training data found after processing files and date filtering.")
    
    # Process calibration files
    if os.path.exists(calib_output_file):
        os.remove(calib_output_file)
    
    calib_data = process_files(calib_files, input_directory, cutoff_date, target_column, date_column)
    if len(calib_data) == 0:
        logging.warning("No valid calibration data found. Using a portion of training data for calibration.")
        # Fall back to using a small portion of training data for calibration
        train_df = pd.concat(train_data)
        train_df = train_df.sort_values(by=date_column)
        split_idx = int(len(train_df) * 0.8)
        calib_df = train_df.iloc[split_idx:]
        train_df = train_df.iloc[:split_idx]
    else:
        train_df = pd.concat(train_data)
        calib_df = pd.concat(calib_data)
    
    # Final processing
    train_df = finalize_dataset(train_df, date_column)
    calib_df = finalize_dataset(calib_df, date_column)
    
    # Save datasets
    train_df.to_parquet(train_output_file, index=False)
    calib_df.to_parquet(calib_output_file, index=False)
    
    logging.info(f"Final training data shape: {train_df.shape}")
    logging.info(f"Final calibration data shape: {calib_df.shape}")
    
    return train_df, calib_df


def process_files(file_list, input_directory, cutoff_date, target_column, date_column):
    """Process a list of files and return a list of valid dataframes."""
    pbar = tqdm(total=len(file_list), desc="Processing files")
    all_data = []
    rows_filtered_by_date = 0
    
    for file in file_list:
        try:
            df = pd.read_parquet(os.path.join(input_directory, file))
            
            # Basic validation
            if df.shape[0] <= 50 or target_column not in df.columns or date_column not in df.columns:
                pbar.update(1)
                continue
                
            # Process the dataframe
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)  # Ensure chronological order
            
            # Filter out data before cutoff date
            rows_before = len(df)
            df = df[df[date_column] >= cutoff_date]
            rows_filtered_by_date += (rows_before - len(df))
            
            # Skip if no data remains after date filtering
            if len(df) < 30:
                pbar.update(1)
                continue
            
            # Drop string columns (except date and target)
            columns_to_drop = [col for col in df.columns 
                              if col not in [date_column, target_column] and df[col].dtype == 'object']
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            # Critical fix: proper target shifting to prevent look-ahead bias
            df[target_column] = df[target_column].shift(-1)
            df = df.iloc[:-1]  # Remove last row with NaN target
            df = df.iloc[1:]   # Remove first row with nan features too

            # Binarize boolean data
            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)

            # Basic cleaning
            df = df.dropna(subset=[target_column])
            df = df[(df[target_column] <= 1000) & (df[target_column] >= -1000)]
            
            if len(df) >= 30:
                all_data.append(df)
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
            
        pbar.update(1)
    
    pbar.close()
    return all_data


def finalize_dataset(df, date_column):
    """Final processing steps for a dataset."""
    # Group by date if needed
    grouped = df.groupby(date_column)
    
    # Maintain chronological order
    ordered_groups = [group.sort_values(date_column).reset_index(drop=True) for _, group in grouped]
    
    if len(ordered_groups) == 0:
        logging.error("No groups available after grouping by date.")
        raise ValueError("No groups available after grouping by date.")
    
    final_df = pd.concat(ordered_groups).reset_index(drop=True)
    
    # Log date range of final dataset
    min_date = final_df[date_column].min().strftime('%Y-%m-%d')
    max_date = final_df[date_column].max().strftime('%Y-%m-%d')
    logging.info(f"Dataset date range: {min_date} to {max_date}")
    
    return final_df


def select_features(X_train, y_train, X_val, y_val, min_importance=0.001, max_features=60):
    """
    Two-phase feature selection:
    1. Train a quick model, keep features above importance threshold (capped at max_features)
    2. Remove highly correlated features (>0.95), keeping the higher-importance one
    """
    quick_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        early_stopping_rounds=10,
        random_state=3301,
        verbosity=0,
        tree_method='hist',
    )
    quick_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    importances = pd.Series(quick_model.feature_importances_, index=X_train.columns)
    selected = importances[importances >= min_importance].sort_values(ascending=False)
    if len(selected) > max_features:
        selected = selected.head(max_features)

    selected_features = selected.index.tolist()
    logging.info(f"Feature selection phase 1: {len(X_train.columns)} -> {len(selected_features)} features")
    logging.info(f"Top 10 features: {list(selected.head(10).items())}")

    # Phase 2: remove highly correlated features
    corr_matrix = X_train[selected_features].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = set()
    for col in upper_tri.columns:
        correlated = upper_tri.index[upper_tri[col] > 0.95].tolist()
        for corr_col in correlated:
            if importances[col] >= importances[corr_col]:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    final_features = [f for f in selected_features if f not in to_drop]
    logging.info(f"Feature selection phase 2: removed {len(to_drop)} correlated -> {len(final_features)} final features")

    return final_features


def select_features_regression(X_train, y_train, X_val, y_val, min_importance=0.001, max_features=60):
    """Feature selection for magnitude regression model using XGBRegressor."""
    quick_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        early_stopping_rounds=10,
        random_state=3301,
        verbosity=0,
        tree_method='hist',
    )
    quick_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    importances = pd.Series(quick_model.feature_importances_, index=X_train.columns)
    selected = importances[importances >= min_importance].sort_values(ascending=False)
    if len(selected) > max_features:
        selected = selected.head(max_features)

    selected_features = selected.index.tolist()
    logging.info(f"Magnitude feature selection: {len(X_train.columns)} -> {len(selected_features)} features")

    # Remove highly correlated features
    corr_matrix = X_train[selected_features].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = set()
    for col in upper_tri.columns:
        correlated = upper_tri.index[upper_tri[col] > 0.95].tolist()
        for corr_col in correlated:
            if importances[col] >= importances[corr_col]:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    final_features = [f for f in selected_features if f not in to_drop]
    logging.info(f"Magnitude feature selection: removed {len(to_drop)} correlated -> {len(final_features)} final features")
    return final_features


def purged_walk_forward_cv(dates, n_splits=3, embargo_days=5):
    """
    Walk-forward cross-validation with embargo gaps.
    Each fold trains on earlier data, validates on later data.
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    min_train_end = int(n_dates * 0.4)
    remaining_dates = n_dates - min_train_end
    fold_size = remaining_dates // n_splits

    splits = []
    for i in range(n_splits):
        train_end_idx = min_train_end + (i * fold_size)
        val_start_idx = train_end_idx + embargo_days
        val_end_idx = min(train_end_idx + fold_size, n_dates)

        if val_start_idx >= val_end_idx:
            continue

        train_end_date = unique_dates[train_end_idx]
        val_start_date = unique_dates[val_start_idx]
        val_end_date = unique_dates[val_end_idx - 1]

        train_mask = dates <= train_end_date
        val_mask = (dates >= val_start_date) & (dates <= val_end_date)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def optuna_tune(X_train, y_train, dates, n_trials=50, n_cv_splits=3):
    """Tune XGBoost hyperparameters using Optuna with temporal CV."""
    import optuna

    cv_splits = purged_walk_forward_cv(dates, n_splits=n_cv_splits, embargo_days=5)

    def objective(trial):
        params = {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 3.0),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'early_stopping_rounds': 30,
            'random_state': 3301,
            'verbosity': 0,
            'nthread': 32,
            'tree_method': 'hist',
        }

        scores = []
        for train_idx, val_idx in cv_splits:
            model = XGBClassifier(**params)
            model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                verbose=False
            )
            proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(y_train.iloc[val_idx], proba))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', study_name='xgboost_aucpr')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logging.info(f"Optuna best AUC-PR: {study.best_value:.4f}")
    logging.info(f"Optuna best params: {study.best_params}")

    return study.best_params


def plot_calibration_curves(clf, X_calib, y_calib, X_test=None, y_test=None, output_path=None):
    """
    Plot calibration curves for the original and calibrated classifier.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line representing perfect calibration
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    
    # Get original model probabilities
    y_prob = clf.predict_proba(X_calib)[:, 1]
    
    # Plot original model calibration curve
    prob_true, prob_pred = calibration_curve(y_calib, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, 's-', label='Original model (training data)')
    
    # If test data is provided, plot its calibration curve too
    if X_test is not None and y_test is not None:
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        prob_true_test, prob_pred_test = calibration_curve(y_test, y_prob_test, n_bins=10)
        plt.plot(prob_pred_test, prob_true_test, 's-', label='Original model (test data)')
    
    # Try to create calibrated versions using different methods
    methods = ['sigmoid', 'isotonic']
    
    for method in methods:
        try:
            # Create and fit calibrated classifier
            calibrated_clf = CalibratedClassifierCV(clf, method=method, cv='prefit')
            calibrated_clf.fit(X_calib, y_calib)
            
            # Get calibrated probabilities
            calibrated_prob = calibrated_clf.predict_proba(X_calib)[:, 1]
            
            # Plot calibration curve
            calib_prob_true, calib_prob_pred = calibration_curve(y_calib, calibrated_prob, n_bins=10)
            plt.plot(calib_prob_pred, calib_prob_true, 's-', 
                     label=f'Calibrated model ({method}) - training')
            
            # Plot test calibration if available
            if X_test is not None and y_test is not None:
                calib_prob_test = calibrated_clf.predict_proba(X_test)[:, 1]
                calib_true_test, calib_pred_test = calibration_curve(y_test, calib_prob_test, n_bins=10)
                plt.plot(calib_pred_test, calib_true_test, 's-', 
                         label=f'Calibrated model ({method}) - test')
                
        except Exception as e:
            logging.warning(f"Could not create {method} calibration plot: {str(e)}")
    
    # Finalize plot
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot if output path is provided
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logging.info(f"Calibration plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving calibration plot: {str(e)}")
    
    plt.close()


def train_model(training_data, calibration_data, config, target_precision=0.75):
    apply_calibration = config['apply_calibration']
    
    logging.info(f"Training XGBoost model with calibration={apply_calibration}.")
    
    model_filename = "xgb_model.joblib"
    model_output_path = os.path.join(config['model_output_directory'], model_filename)
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    # Sort data chronologically
    training_data = training_data.sort_values('Date')
    calibration_data = calibration_data.sort_values('Date')
    
    # Prepare training data
    X_train = training_data.drop(columns=[config['target_column']])
    y_train_raw = training_data[config['target_column']]  # raw continuous returns
    y_train = y_train_raw.apply(lambda x: 0 if x < 0 else 1)

    # Prepare calibration data
    X_calib = calibration_data.drop(columns=[config['target_column']])
    y_calib_raw = calibration_data[config['target_column']]  # raw continuous returns
    y_calib = y_calib_raw.apply(lambda x: 0 if x < 0 else 1)

    # Compute magnitude target: abs(return) / atr_percentage (volatility-normalized)
    # This measures "how big is the move relative to typical volatility"
    mag_vol_col = 'atr_percentage'
    if mag_vol_col in X_train.columns:
        y_train_magnitude = y_train_raw.abs() / X_train[mag_vol_col].clip(lower=0.001)
        y_calib_magnitude = y_calib_raw.abs() / X_calib[mag_vol_col].clip(lower=0.001)
        # Cap extreme outliers at 99th percentile
        mag_cap = y_train_magnitude.quantile(0.99)
        y_train_magnitude = y_train_magnitude.clip(upper=mag_cap)
        y_calib_magnitude = y_calib_magnitude.clip(upper=mag_cap)
        logging.info(f"Magnitude target stats: mean={y_train_magnitude.mean():.3f}, "
                     f"median={y_train_magnitude.median():.3f}, cap={mag_cap:.3f}")
        has_magnitude = True
    else:
        logging.warning(f"Column '{mag_vol_col}' not found, skipping magnitude model.")
        has_magnitude = False
    
    # Remove datetime columns
    datetime_columns_train = X_train.select_dtypes(include=['datetime64']).columns
    datetime_columns_calib = X_calib.select_dtypes(include=['datetime64']).columns
    
    X_train = X_train.drop(columns=datetime_columns_train)
    X_calib = X_calib.drop(columns=datetime_columns_calib)
    
    # Temporal split for validation (used for early stopping)
    # Data is already sorted by Date. Split at 80% mark with 5-day embargo gap.
    embargo_rows = 5
    split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train.iloc[:split_idx]
    y_train_final = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx + embargo_rows:]
    y_val = y_train.iloc[split_idx + embargo_rows:]
    logging.info(f"Temporal split: train={len(X_train_final)}, embargo={embargo_rows}, val={len(X_val)}")

    # Feature selection: prune noise and correlated features
    selected_features = select_features(X_train_final, y_train_final, X_val, y_val)
    X_train_final = X_train_final[selected_features]
    X_val = X_val[selected_features]
    X_calib = X_calib[selected_features]

    # Compute scale_pos_weight from actual class distribution
    n_neg = (y_train_final == 0).sum()
    n_pos = (y_train_final == 1).sum()
    computed_spw = n_neg / n_pos if n_pos > 0 else 1.0
    logging.info(f"Class distribution: neg={n_neg}, pos={n_pos}, scale_pos_weight={computed_spw:.4f}")

    # Optional Optuna hyperparameter tuning
    xgb_params = config['xgb_params'].copy()
    xgb_params["scale_pos_weight"] = computed_spw
    if args.tune:
        logging.info(f"Starting Optuna tuning with {args.tune_trials} trials...")
        train_dates = training_data['Date'].iloc[:split_idx]
        best_params = optuna_tune(X_train_final, y_train_final, train_dates, n_trials=args.tune_trials)
        xgb_params.update(best_params)
        # Restore fixed params that Optuna shouldn't override
        xgb_params['n_estimators'] = 500
        xgb_params['early_stopping_rounds'] = 30
        xgb_params['objective'] = 'binary:logistic'
        xgb_params['eval_metric'] = 'aucpr'
        xgb_params['random_state'] = 3301
        xgb_params['nthread'] = 32
        xgb_params['tree_method'] = 'hist'
        xgb_params['verbosity'] = 1
        logging.info("Using Optuna-tuned parameters for final training")

    # Train XGBoost model
    clf = XGBClassifier(**xgb_params)
    
    try:
        clf.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    except Exception as e:
        logging.error(f"Error during XGBoost fitting: {str(e)}")
        # Try without eval_set
        clf.fit(X_train_final, y_train_final)

    # AUC-PR diagnostics on validation set
    try:
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_aucpr = average_precision_score(y_val, val_proba)
        val_auroc = roc_auc_score(y_val, val_proba)
        logging.info(f"Validation AUC-PR: {val_aucpr:.4f}")
        logging.info(f"Validation AUC-ROC: {val_auroc:.4f}")

        pos_probs = val_proba[y_val == 1]
        neg_probs = val_proba[y_val == 0]
        separation_gap = pos_probs.mean() - neg_probs.mean()
        logging.info(f"Positive class prob: mean={pos_probs.mean():.4f}, median={np.median(pos_probs):.4f}")
        logging.info(f"Negative class prob: mean={neg_probs.mean():.4f}, median={np.median(neg_probs):.4f}")
        logging.info(f"Separation gap (pos - neg mean): {separation_gap:.4f}")

        if hasattr(clf, 'best_iteration'):
            logging.info(f"Best iteration (early stopping): {clf.best_iteration}")
    except Exception as e:
        logging.error(f"Error computing AUC-PR diagnostics: {str(e)}")
        val_aucpr = None
        val_auroc = None
        separation_gap = None

    # Split calibration data: 60% for isotonic fitting, 40% for threshold search
    # Calibration data is already sorted by Date, so this is a temporal split
    calib_split_idx = int(len(X_calib) * 0.6)
    X_calib_fit = X_calib.iloc[:calib_split_idx]
    y_calib_fit = y_calib.iloc[:calib_split_idx]
    X_calib_threshold = X_calib.iloc[calib_split_idx:]
    y_calib_threshold = y_calib.iloc[calib_split_idx:]
    logging.info(f"Calibration split: fit={len(X_calib_fit)}, threshold={len(X_calib_threshold)}")

    # Calibrate probabilities if requested
    smooth_calibrator = None
    if apply_calibration:
        logging.info("Calibrating model probabilities with hybrid isotonic+Platt...")
        try:
            raw_probs_fit = clf.predict_proba(X_calib_fit)[:, 1]
            smooth_calibrator = HybridCalibrator()
            smooth_calibrator.fit(raw_probs_fit, y_calib_fit.values)
            logging.info("Model calibrated using hybrid isotonic+Platt.")
        except Exception as e:
            logging.error(f"Error during hybrid calibration: {str(e)}")
            logging.warning("Proceeding with uncalibrated model.")
            smooth_calibrator = None

        # Create calibration plot using the threshold set (unseen by calibrator)
        try:
            plot_calibration_curves(
                clf, X_calib_threshold, y_calib_threshold,
                output_path=config['calibration_plot_output']
            )
        except Exception as e:
            logging.error(f"Failed to create calibration plot: {str(e)}")

    # ========================= Magnitude Model =========================
    magnitude_model = None
    magnitude_features = None
    magnitude_median = None
    if has_magnitude:
        logging.info("Training magnitude regression model (volatility-normalized)...")
        try:
            # Use the FULL X_train (before direction feature selection) for magnitude
            # because magnitude may depend on different features than direction
            X_train_all = X_train.iloc[:split_idx]
            X_val_all = X_train.iloc[split_idx + embargo_rows:]
            y_mag_train = y_train_magnitude.iloc[:split_idx]
            y_mag_val = y_train_magnitude.iloc[split_idx + embargo_rows:]

            # Feature selection for magnitude (independent of direction features)
            magnitude_features = select_features_regression(X_train_all, y_mag_train, X_val_all, y_mag_val)
            X_mag_train = X_train_all[magnitude_features]
            X_mag_val = X_val_all[magnitude_features]

            # Store median magnitude from training data for normalization at prediction time
            magnitude_median = float(y_mag_train.median())
            logging.info(f"Magnitude median (training): {magnitude_median:.4f}")

            # Train XGBRegressor
            mag_params = config['magnitude_params'].copy()
            magnitude_model = XGBRegressor(**mag_params)
            magnitude_model.fit(
                X_mag_train, y_mag_train,
                eval_set=[(X_mag_val, y_mag_val)],
                verbose=False
            )

            # Diagnostics
            mag_pred_val = magnitude_model.predict(X_mag_val)
            rho, _ = spearmanr(y_mag_val, mag_pred_val)
            mae = np.mean(np.abs(y_mag_val - mag_pred_val))
            logging.info(f"Magnitude model: Spearman rho={rho:.4f}, MAE={mae:.4f}")
            if hasattr(magnitude_model, 'best_iteration'):
                logging.info(f"Magnitude best iteration: {magnitude_model.best_iteration}")
        except Exception as e:
            logging.error(f"Error training magnitude model: {str(e)}")
            magnitude_model = None
            magnitude_features = None

    # Get predicted probabilities on held-out threshold set (not seen during calibration)
    raw_probs_threshold = clf.predict_proba(X_calib_threshold)[:, 1]
    if smooth_calibrator is not None:
        calibrated_probs_threshold = smooth_calibrator.predict(raw_probs_threshold)
    else:
        calibrated_probs_threshold = raw_probs_threshold

    # Raw margin rank injection: re-spread tail probabilities using log-odds ranking
    try:
        raw_margins_threshold = clf.predict(X_calib_threshold, output_margin=True)
        calibrated_probs_threshold = rank_refine_tail(calibrated_probs_threshold, raw_margins_threshold)
        logging.info(f"Rank refinement applied to {(calibrated_probs_threshold >= 0.65).sum()} tail predictions")
    except Exception as e:
        logging.error(f"Error during rank refinement: {str(e)}")

    # Apply magnitude weighting if available
    # Formula: adjusted = 0.5 + (dir_prob - 0.5) * (1 + alpha * (mag_norm - 1))
    # where mag_norm = predicted_magnitude / median_magnitude
    # When mag_norm = 1 (average move), no change. When mag_norm > 1, push away from 0.5.
    if magnitude_model is not None and magnitude_features is not None:
        try:
            # Get magnitude predictions on threshold set (need full feature set)
            X_calib_threshold_full = X_train.iloc[0:0]  # empty df with all columns
            X_calib_threshold_full = calibration_data.drop(columns=[config['target_column']])
            datetime_cols_ct = X_calib_threshold_full.select_dtypes(include=['datetime64']).columns
            X_calib_threshold_full = X_calib_threshold_full.drop(columns=datetime_cols_ct)
            X_calib_threshold_full = X_calib_threshold_full.iloc[calib_split_idx:]

            # Ensure magnitude features exist
            missing_mag = set(magnitude_features) - set(X_calib_threshold_full.columns)
            for f in missing_mag:
                X_calib_threshold_full[f] = 0
            X_mag_calib = X_calib_threshold_full[magnitude_features]

            mag_pred = magnitude_model.predict(X_mag_calib)
            mag_pred = np.clip(mag_pred, 0.01, None)  # floor at small positive
            mag_norm = mag_pred / magnitude_median  # ratio vs typical move

            alpha = config['magnitude_alpha']
            direction_signal = calibrated_probs_threshold - 0.5
            magnitude_weight = 1.0 + alpha * (mag_norm - 1.0)
            magnitude_weight = np.clip(magnitude_weight, 0.5, 2.0)  # safety bounds
            calibrated_probs_threshold = 0.5 + direction_signal * magnitude_weight
            calibrated_probs_threshold = np.clip(calibrated_probs_threshold, 0.001, 0.999)

            logging.info(f"Magnitude weighting applied: alpha={alpha}, "
                         f"mag_norm range=[{mag_norm.min():.3f}, {mag_norm.max():.3f}], "
                         f"adjusted prob range=[{calibrated_probs_threshold.min():.4f}, {calibrated_probs_threshold.max():.4f}]")
        except Exception as e:
            logging.error(f"Error applying magnitude weighting: {str(e)}")

    # Pack into 2-column format for compatibility with threshold search code
    y_pred_proba = np.column_stack([1 - calibrated_probs_threshold, calibrated_probs_threshold])

    # Find optimal thresholds
    logging.info(f"Calibrated probability stats: min={y_pred_proba[:, 1].min():.4f}, max={y_pred_proba[:, 1].max():.4f}, mean={y_pred_proba[:, 1].mean():.4f}")
    logging.info(f"Unique calibrated values: {len(np.unique(y_pred_proba[:, 1]))}")

    # Tail discrimination diagnostics
    tail_mask = y_pred_proba[:, 1] >= 0.65
    n_tail = tail_mask.sum()
    if n_tail >= 20:
        tail_probs = y_pred_proba[tail_mask, 1]
        tail_actuals = y_calib_threshold.values[tail_mask] if hasattr(y_calib_threshold, 'values') else y_calib_threshold[tail_mask]
        tail_winners = tail_probs[tail_actuals == 1]
        tail_losers = tail_probs[tail_actuals == 0]
        n_unique_tail = len(np.unique(tail_probs))
        logging.info(f"=== TAIL DISCRIMINATION (prob >= 0.65) ===")
        logging.info(f"  Tail samples: {n_tail}, unique values: {n_unique_tail}")
        if len(tail_winners) > 0 and len(tail_losers) > 0:
            gap = np.median(tail_winners) - np.median(tail_losers)
            logging.info(f"  Winner median: {np.median(tail_winners):.4f}, "
                         f"Loser median: {np.median(tail_losers):.4f}, "
                         f"Gap: {gap:.4f}")
            rho_tail, p_tail = spearmanr(tail_probs, tail_actuals)
            logging.info(f"  Spearman rho (prob vs outcome): {rho_tail:.4f} (p={p_tail:.2e})")
        logging.info(f"  Tail precision: {tail_actuals.mean():.4f}")
    else:
        logging.info(f"Tail discrimination: only {n_tail} samples >= 0.65, skipping diagnostics")

    # Set target values for precision and coverage
    target_min_precision_pos = 0.72  # Raised from 0.65 to push toward higher-quality thresholds
    min_predictions_percent_pos = 0.005  # Lowered from 0.01 to allow the balanced score to work
    
    target_min_precision_neg = 0.65  # For downward moves (class 0)
    min_predictions_percent_neg = 0.05

    # Find optimal thresholds for positive class
    precisions_pos, recalls_pos, thresholds_pos = precision_recall_curve(
        y_calib_threshold, y_pred_proba[:, 1], pos_label=1
    )

    # Ensure we don't have index mismatches
    if len(precisions_pos) > len(thresholds_pos):
        precisions_pos = precisions_pos[:-1]
        recalls_pos = recalls_pos[:-1]

    # Calculate prediction coverage
    prediction_coverage = []
    for threshold in thresholds_pos:
        coverage = (y_pred_proba[:, 1] >= threshold).mean()
        prediction_coverage.append(coverage)

    # Find threshold for positive class using balanced score: precision * sqrt(coverage).
    # Pure precision maximization drove the threshold to extreme values (0.9944) as an
    # artifact of rank_refine_tail compressing the tail distribution. The balanced score
    # rewards precision but penalizes near-zero coverage.
    valid_indices = (precisions_pos >= target_min_precision_pos) & (np.array(prediction_coverage) >= min_predictions_percent_pos)
    if np.any(valid_indices):
        coverage_arr = np.array(prediction_coverage)[valid_indices]
        balanced_score = precisions_pos[valid_indices] * np.sqrt(coverage_arr)
        best_idx = np.argmax(balanced_score)
        valid_idx_positions = np.where(valid_indices)[0]
        optimal_threshold_pos = thresholds_pos[valid_idx_positions[best_idx]]
        pos_precision = precisions_pos[valid_idx_positions[best_idx]]
        pos_recall = recalls_pos[valid_idx_positions[best_idx]]
        pos_coverage = prediction_coverage[valid_idx_positions[best_idx]]
    else:
        logging.warning("No threshold meets both precision and coverage requirements for UP moves.")
        valid_thresholds = [i for i, cov in enumerate(prediction_coverage) if cov >= 0.001]
        if valid_thresholds:
            best_precision_idx = np.argmax(precisions_pos[valid_thresholds])
            idx_to_use = valid_thresholds[best_precision_idx]
            optimal_threshold_pos = thresholds_pos[idx_to_use]
            pos_precision = precisions_pos[idx_to_use]
            pos_recall = recalls_pos[idx_to_use]
            pos_coverage = prediction_coverage[idx_to_use]
        else:
            optimal_threshold_pos = 0.90
            predicted_pos = y_pred_proba[:, 1] >= optimal_threshold_pos
            if predicted_pos.sum() > 0:
                pos_precision = (y_calib_threshold[predicted_pos] == 1).mean()
            else:
                pos_precision = 0
            pos_recall = (predicted_pos & (y_calib_threshold == 1)).sum() / (y_calib_threshold == 1).sum() if (y_calib_threshold == 1).sum() > 0 else 0
            pos_coverage = predicted_pos.mean()

    # Find optimal thresholds for negative class
    precisions_neg, recalls_neg, thresholds_neg = precision_recall_curve(
        1 - y_calib_threshold, y_pred_proba[:, 0], pos_label=1
    )

    if len(precisions_neg) > len(thresholds_neg):
        precisions_neg = precisions_neg[:-1]
        recalls_neg = recalls_neg[:-1]

    neg_prediction_coverage = []
    for threshold in thresholds_neg:
        coverage = (y_pred_proba[:, 0] >= threshold).mean()
        neg_prediction_coverage.append(coverage)

    valid_indices = (precisions_neg >= target_min_precision_neg) & (np.array(neg_prediction_coverage) >= min_predictions_percent_neg)
    if np.any(valid_indices):
        best_idx = np.argmax(recalls_neg[valid_indices])
        valid_idx_positions = np.where(valid_indices)[0]
        optimal_threshold_neg = thresholds_neg[valid_idx_positions[best_idx]]
        neg_precision = precisions_neg[valid_idx_positions[best_idx]]
        neg_recall = recalls_neg[valid_idx_positions[best_idx]]
        neg_coverage = neg_prediction_coverage[valid_idx_positions[best_idx]]
    else:
        reduced_precision = max(0.60, target_min_precision_neg - 0.05)
        valid_indices = (precisions_neg >= reduced_precision) & (np.array(neg_prediction_coverage) >= min_predictions_percent_neg)

        if np.any(valid_indices):
            best_idx = np.argmax(precisions_neg[valid_indices])
            valid_idx_positions = np.where(valid_indices)[0]
            optimal_threshold_neg = thresholds_neg[valid_idx_positions[best_idx]]
            neg_precision = precisions_neg[valid_idx_positions[best_idx]]
            neg_recall = recalls_neg[valid_idx_positions[best_idx]]
            neg_coverage = neg_prediction_coverage[valid_idx_positions[best_idx]]
        else:
            mean_prob = y_pred_proba[:, 0].mean()
            std_prob = y_pred_proba[:, 0].std()
            optimal_threshold_neg = min(0.75, mean_prob + std_prob)

            predicted_neg = y_pred_proba[:, 0] >= optimal_threshold_neg
            if predicted_neg.sum() > 0:
                neg_precision = ((1 - y_calib_threshold)[predicted_neg] == 1).mean()
            else:
                neg_precision = 0
            neg_recall = (predicted_neg & ((1 - y_calib_threshold) == 1)).sum() / ((1 - y_calib_threshold) == 1).sum() if ((1 - y_calib_threshold) == 1).sum() > 0 else 0
            neg_coverage = predicted_neg.mean()

    # Calculate expected profit factor
    if pos_precision > 0:
        expected_profit_factor = (pos_precision / (1 - pos_precision))
        logging.info(f"Expected profit factor for UP predictions: {expected_profit_factor:.4f}")
        logging.info(f"This means for every $1 lost, you can expect to make ${expected_profit_factor:.2f}")

    # Log threshold information
    logging.info(f"Optimal threshold for class 1 (UP): {optimal_threshold_pos:.4f} with precision {pos_precision:.4f}, recall {pos_recall:.4f}, coverage {pos_coverage:.4f}")
    logging.info(f"Optimal threshold for class 0 (DOWN): {optimal_threshold_neg:.4f} with precision {neg_precision:.4f}, recall {neg_recall:.4f}, coverage {neg_coverage:.4f}")

    # Apply thresholds to get predictions
    y_pred = np.full(len(y_calib_threshold), -1)  # Default to "no prediction" (-1)
    y_pred[y_pred_proba[:, 1] >= optimal_threshold_pos] = 1
    mask_unassigned = (y_pred == -1)
    y_pred[mask_unassigned & (y_pred_proba[:, 0] >= optimal_threshold_neg)] = 0

    # Calculate prediction coverage
    prediction_coverage = (y_pred != -1).mean() * 100
    logging.info(f"Percentage of data receiving predictions: {prediction_coverage:.2f}%")

    # Plot probability histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[:, 1], bins=50, alpha=0.7)
        plt.axvline(x=optimal_threshold_pos, color='r', linestyle='--', label=f'UP Threshold: {optimal_threshold_pos:.4f}')
        plt.title('Histogram of Predicted Probabilities for Upward Moves')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(config['model_output_directory'], 'probability_histogram.png'))
        logging.info(f"Probability histogram saved to {os.path.join(config['model_output_directory'], 'probability_histogram.png')}")
    except Exception as e:
        logging.error(f"Error creating probability histogram: {str(e)}")

    # Evaluate predictions
    mask_definitive = y_pred != -1
    y_calib_filtered = y_calib_threshold[mask_definitive]
    y_pred_filtered = y_pred[mask_definitive]
    
    if len(y_calib_filtered) > 0:
        accuracy = accuracy_score(y_calib_filtered, y_pred_filtered)
        f1 = f1_score(y_calib_filtered, y_pred_filtered, average='weighted')
        precision = precision_score(y_calib_filtered, y_pred_filtered, average='weighted')
        recall = recall_score(y_calib_filtered, y_pred_filtered, average='weighted')
        
        logging.info(f"Definitive predictions: {len(y_pred_filtered)} out of {len(y_pred)} ({len(y_pred_filtered)/len(y_pred)*100:.2f}%)")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        
        # Print classification report
        print(classification_report(y_calib_filtered, y_pred_filtered, zero_division=0))
        
        # Report class-specific metrics
        if (y_pred_filtered == 1).sum() > 0:
            up_precision = precision_score(y_calib_filtered, y_pred_filtered, pos_label=1, average='binary')
            up_recall = recall_score(y_calib_filtered, y_pred_filtered, pos_label=1, average='binary')
            logging.info(f"UP predictions precision: {up_precision:.4f}, recall: {up_recall:.4f}")
            logging.info(f"Total UP predictions: {(y_pred_filtered == 1).sum()} out of {len(y_pred_filtered)} ({(y_pred_filtered == 1).sum()/len(y_pred_filtered)*100:.2f}%)")
    else:
        logging.warning("No definitive predictions after applying thresholds.")
    
    # Save model and thresholds
    model_data = {
        'base_model': clf,
        'smooth_calibrator': smooth_calibrator,
        'calibrated_model': None,
        'is_calibrated': smooth_calibrator is not None,
        'magnitude_model': magnitude_model,
        'magnitude_features': magnitude_features,
        'magnitude_median': magnitude_median,
        'magnitude_alpha': config['magnitude_alpha'],
        'threshold_pos': optimal_threshold_pos,
        'threshold_neg': optimal_threshold_neg,
        'precision_pos': pos_precision,
        'recall_pos': pos_recall,
        'precision_neg': neg_precision,
        'recall_neg': neg_recall,
        'val_aucpr': val_aucpr,
        'val_auroc': val_auroc,
        'separation_gap': separation_gap,
        'n_estimators_used': clf.best_iteration if hasattr(clf, 'best_iteration') else config['xgb_params']['n_estimators']
    }
    
    dump(model_data, model_output_path)
    logging.info(f"Model and thresholds saved to {model_output_path}")
    
    # Handle feature importances
    try:
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        feature_importances['importance'] = feature_importances['importance'].round(5)
        feature_importances.to_parquet(config['feature_importance_output'], index=False)
        logging.info(f"Feature importances saved to {config['feature_importance_output']}")
    except Exception as e:
        logging.error(f"Error saving feature importances: {str(e)}")
    
    return model_data


def predict_and_save(input_directory, model_path, output_directory, target_column, date_column):
    logging.info("Loading the trained model with calibration for prediction.")
    
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    # Load model and thresholds
    model_data = load(model_path)
    
    smooth_calibrator = None
    if isinstance(model_data, dict):
        # Load base model (always need this for raw predictions)
        if 'base_model' in model_data:
            clf = model_data['base_model']
        elif 'model' in model_data:
            clf = model_data['model']
        else:
            clf = model_data
            logging.warning("Using model from legacy format.")

        # Load smooth calibrator if available
        if 'smooth_calibrator' in model_data and model_data['smooth_calibrator'] is not None:
            smooth_calibrator = model_data['smooth_calibrator']
            logging.info("Using smooth isotonic calibrator for predictions.")
        elif 'calibrated_model' in model_data and model_data['calibrated_model'] is not None:
            # Backward compatibility: use old CalibratedClassifierCV
            clf = model_data['calibrated_model']
            logging.info("Using legacy CalibratedClassifierCV for predictions.")
        else:
            logging.info("Using uncalibrated base model for predictions.")

        # Load magnitude model if available
        magnitude_model = model_data.get('magnitude_model', None)
        magnitude_features = model_data.get('magnitude_features', None)
        magnitude_median = model_data.get('magnitude_median', None)
        magnitude_alpha = model_data.get('magnitude_alpha', 0.3)
        if magnitude_model is not None:
            logging.info(f"Loaded magnitude model ({len(magnitude_features)} features, "
                         f"median={magnitude_median:.4f}, alpha={magnitude_alpha})")
        else:
            logging.info("No magnitude model found, using direction-only probabilities.")

        # Get thresholds
        if 'threshold_pos' in model_data:
            threshold_pos = model_data['threshold_pos']
            threshold_neg = model_data['threshold_neg']
            logging.info(f"Using optimized thresholds - Positive: {threshold_pos:.4f}, Negative: {threshold_neg:.4f}")
        else:
            threshold_pos = 0.7
            threshold_neg = 0.7
            logging.warning("Using default thresholds as no optimized thresholds found")
    else:
        clf = model_data
        threshold_pos = 0.7
        threshold_neg = 0.7
        logging.warning("Using default thresholds with legacy model format")
    
    # Determine model features
    if hasattr(clf, 'feature_names_in_'):
        model_features = clf.feature_names_in_
    elif hasattr(clf, 'get_booster') and hasattr(clf.get_booster(), 'feature_names'):
        model_features = clf.get_booster().feature_names
    elif hasattr(clf, 'feature_names_'):
        model_features = clf.feature_names_
    else:
        logging.warning("Could not determine feature names from model, using all features.")
        sample_file = os.path.join(input_directory, os.listdir(input_directory)[0])
        sample_df = pd.read_parquet(sample_file)
        datetime_columns = sample_df.select_dtypes(include=['datetime64']).columns
        model_features = [col for col in sample_df.columns if col not in [date_column, target_column] + list(datetime_columns)]
    
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)
    
    null_io = io.StringIO()
    
    # Track prediction statistics
    total_predictions = 0
    definitive_predictions = 0
    
    for file in all_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        df[date_column] = pd.to_datetime(df[date_column])
        
        if df.shape[0] < 252:
            pbar.update(1)
            continue

        # Prepare features for prediction
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        X = df.drop(columns=[col for col in [date_column, target_column] + list(datetime_columns) if col in df.columns])
        
        # Ensure X contains only the features the model was trained on
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0  # Add missing features with default values
                
        X = X.reindex(columns=model_features, fill_value=0)
        
        # Make predictions
        with parallel_backend('threading', n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                try:
                    y_pred_proba = clf.predict_proba(X)
                except Exception as e:
                    logging.error(f"Error making predictions: {str(e)}")
                    logging.error(f"Model type: {type(clf).__name__}")
                    pbar.update(1)
                    continue

        # Apply smooth calibration if available, otherwise use raw model output
        if smooth_calibrator is not None:
            raw_up = y_pred_proba[:, 1]
            calibrated_up = smooth_calibrator.predict(raw_up)
        else:
            calibrated_up = y_pred_proba[:, 1]

        # Raw margin rank injection: re-spread tail probabilities using log-odds ranking
        try:
            raw_margins = clf.predict(X, output_margin=True)
            calibrated_up = rank_refine_tail(calibrated_up, raw_margins)
        except Exception as e:
            logging.error(f"Error during rank refinement for {file}: {str(e)}")

        # Apply magnitude weighting if model available
        if magnitude_model is not None and magnitude_features is not None:
            try:
                X_full = df.drop(columns=[col for col in [date_column, target_column] + list(datetime_columns) if col in df.columns])
                missing_mag = set(magnitude_features) - set(X_full.columns)
                for f in missing_mag:
                    X_full[f] = 0
                X_mag = X_full[magnitude_features]
                mag_pred = magnitude_model.predict(X_mag)
                mag_pred = np.clip(mag_pred, 0.01, None)
                mag_norm = mag_pred / magnitude_median

                direction_signal = calibrated_up - 0.5
                mag_weight = 1.0 + magnitude_alpha * (mag_norm - 1.0)
                mag_weight = np.clip(mag_weight, 0.5, 2.0)
                calibrated_up = 0.5 + direction_signal * mag_weight
                calibrated_up = np.clip(calibrated_up, 0.001, 0.999)
            except Exception as e:
                logging.error(f"Error applying magnitude in prediction for {file}: {str(e)}")

        df['UpProbability'] = calibrated_up
        df['DownProbability'] = 1 - calibrated_up

        # Add threshold values for reference
        df['PositiveThreshold'] = threshold_pos
        df['NegativeThreshold'] = threshold_neg

        epsilon = 1e-3  # Small value to prevent zeros
        df['UpProbability'] = df['UpProbability'].clip(epsilon, 1-epsilon)
        df['DownProbability'] = df['DownProbability'].clip(epsilon, 1-epsilon)

        # Apply thresholds
        df['UpPrediction'] = -1  # Default to no prediction
        df.loc[df['UpProbability'] >= threshold_pos, 'UpPrediction'] = 1
        mask_undecided = df['UpPrediction'] == -1
        df.loc[mask_undecided & (df['DownProbability'] >= threshold_neg), 'UpPrediction'] = 0
                
        # Update prediction stats
        total_predictions += len(df)
        definitive_predictions += (df['UpPrediction'] != -1).sum()
        
        # Keep necessary columns for output
        try:
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                             'UpProbability', 'DownProbability', 
                             'PositiveThreshold', 'NegativeThreshold', 'UpPrediction', 'VIX_Close']
            
            # Include optional columns if they exist
            optional_columns = ['Distance to Resistance (%)', 'Distance to Support (%)', 'volatility']
            for col in optional_columns:
                if col in df.columns:
                    required_columns.append(col)
                    
            # Add missing required columns with NaN values
            available_columns = [col for col in required_columns if col in df.columns]
            for col in set(required_columns) - set(available_columns):
                df[col] = np.nan
                
            output_df = df[required_columns]
            
            output_file_path = os.path.join(output_directory, file)
            output_df.to_parquet(output_file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving prediction for {file}: {str(e)}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Report prediction coverage
    if total_predictions > 0:
        prediction_rate = (definitive_predictions / total_predictions) * 100
        logging.info(f"Prediction coverage: {definitive_predictions} out of {total_predictions} ({prediction_rate:.2f}%)")
    
    logging.info(f"Predictions using calibrated model and optimized thresholds saved to {output_directory}")


def main():
    # Create necessary directories
    os.makedirs(config['model_output_directory'], exist_ok=True)
    os.makedirs(config['data_output_directory'], exist_ok=True)
    os.makedirs(config['calibration_output_directory'], exist_ok=True)
    os.makedirs(config['prediction_output_directory'], exist_ok=True)
    os.makedirs(os.path.dirname(config['feature_importance_output']), exist_ok=True)
    
    model_filename = "xgb_model.joblib"
    
    if not args.predict:
        # Prepare data splits: training and calibration
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
        print("Data preparation complete starting model prep")
        
        # Train and calibrate model
        train_model(training_data, calibration_data, config)

    else:
        # Predict using trained (and calibrated) model
        predict_and_save(
            input_directory=config['input_directory'],
            model_path=os.path.join(config['model_output_directory'], model_filename),
            output_directory=config['prediction_output_directory'],
            target_column=config['target_column'],
            date_column='Date'
        )


if __name__ == "__main__":
    main()