##predictor script with calibration
import os
import random
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import PchipInterpolator
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from joblib import parallel_backend
from contextlib import redirect_stdout, redirect_stderr
import io
import matplotlib.pyplot as plt
from Util import get_logger
logger = get_logger(script_name="4__Predictor")


class SmoothIsotonicCalibrator:
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
        return self

    def predict(self, raw_probs):
        clipped = np.clip(raw_probs, self.x_min, self.x_max)
        return np.clip(self.spline(clipped), 0.0, 1.0)


class HybridCalibrator:
    """Isotonic+PCHIP for the bulk of the distribution, Platt scaling for the sparse tail."""

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

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_probs, y_true)
        self.spline = PchipInterpolator(iso.X_thresholds_, iso.y_thresholds_)
        self.x_min = iso.X_thresholds_.min()
        self.x_max = iso.X_thresholds_.max()

        thresholds_sorted = np.sort(iso.X_thresholds_)
        self.crossover = None
        for candidate in np.arange(0.50, 0.85, 0.01):
            if (thresholds_sorted >= candidate).sum() < self.min_tail_breakpoints:
                self.crossover = candidate
                break

        if self.crossover is None:
            return self

        tail_mask = raw_probs >= (self.crossover - 0.05)
        if tail_mask.sum() < 50:
            self.crossover = None
            return self

        X_tail = raw_probs[tail_mask].reshape(-1, 1)
        y_tail = y_true[tail_mask]
        self.tail_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        self.tail_model.fit(X_tail, y_tail)

        test_pts = np.linspace(self.crossover, self.x_max, 20).reshape(-1, 1)
        if not np.all(np.diff(self.tail_model.predict_proba(test_pts)[:, 1]) >= -1e-6):
            self.crossover = None
            self.tail_model = None

        return self

    def predict(self, raw_probs):
        raw_probs = np.asarray(raw_probs, dtype=np.float64)

        # Spline domain — clip ONLY for spline input (out-of-domain interpolation
        # is undefined for PCHIP). But we do NOT use this clipped value as the
        # final output above x_max — that would tie all top values together and
        # destroy ranking at tight quantiles.
        spline_input = np.clip(raw_probs, self.x_min, self.x_max)
        result = np.clip(self.spline(spline_input), 0.0, 1.0)

        # Linear extrapolation above x_max so above-max raw probs keep their
        # ordering instead of collapsing to a single tied calibrated value.
        above_max = raw_probs > self.x_max
        if above_max.any():
            delta = max((self.x_max - self.x_min) * 0.001, 1e-6)
            slope = float(self.spline(self.x_max) - self.spline(self.x_max - delta)) / delta
            slope = max(slope, 1e-4)  # ensure strictly increasing
            y_at_max = float(self.spline(self.x_max))
            result[above_max] = np.clip(
                y_at_max + slope * (raw_probs[above_max] - self.x_max),
                0.0, 1.0
            )

        if self.crossover is None or self.tail_model is None:
            return result

        # Platt tail uses RAW probs (not clipped) so it extrapolates naturally
        # beyond x_max — a logistic regression handles any input value cleanly.
        tail_pred = self.tail_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        lo = self.crossover - self.blend_width
        hi = self.crossover + self.blend_width

        above_hi = raw_probs > hi
        result[above_hi] = tail_pred[above_hi]

        in_blend = (raw_probs >= lo) & (raw_probs <= hi)
        if in_blend.any():
            w = (raw_probs[in_blend] - lo) / (hi - lo)
            result[in_blend] = (1 - w) * result[in_blend] + w * tail_pred[in_blend]

        # Enforce strict monotonicity: calibrated probability must be non-decreasing
        # in raw probability. Small numerical hiccups at the spline/Platt boundary
        # can produce reversals that destroy top-quantile precision.
        result = np.clip(result, 0.0, 1.0)
        order = np.argsort(raw_probs)
        sorted_result = np.maximum.accumulate(result[order])
        result[order] = sorted_result

        # Tiebreaker: maximum.accumulate creates ties at flat segments, which
        # collapses quantile resolution at the top of the distribution. Add a
        # microscopic offset (1e-6 max) proportional to raw_probs to restore
        # fine-grained ranking without meaningfully changing absolute values.
        # This is what lets us trade more recall for higher precision in the tail.
        tiebreaker = raw_probs * 1e-6
        result = np.clip(result + tiebreaker, 0.0, 1.0)
        return result


class _HybridWrap:
    """Module-level wrapper: exposes predict_proba for a base XGB + HybridCalibrator pair."""
    def __init__(self, base, calibrator):
        self._base = base
        self._cal = calibrator
        if hasattr(base, 'feature_names_in_'):
            self.feature_names_in_ = base.feature_names_in_

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1]
        cal = self._cal.predict(raw)
        return np.column_stack([1 - cal, cal])


argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=50, help="Percentage of files to process.")
argparser.add_argument("--calibpercent", type=int, default=20, help="Percentage of remaining files to use for calibration.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--nocalib", action='store_true', help="Flag to disable probability calibration.")
argparser.add_argument("--tune", action='store_true', help="Run Optuna hyperparameter search before final training.")
argparser.add_argument("--trials", type=int, default=80, help="Number of Optuna trials when --tune is set.")
argparser.add_argument("--tune-sample", type=int, default=200000, help="Number of training rows to sample per Optuna trial (smaller = faster).")
argparser.add_argument("--precision-target", type=float, default=0.80, help="Target precision for class 1 (UP). Threshold search aims for this.")
argparser.add_argument("--min-predictions", type=int, default=30, help="Minimum number of predicted rows for a quantile to be considered.")
args = argparser.parse_args()

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

    # XGBoost parameters — REVERSED last round's capacity bump. The train-val
    # AUC-PR gap of +0.156 showed clear overfitting. The features have signal
    # (train AUC-PR 0.69) but the model wasn't generalising. Tightening hard
    # on every regularization knob to force the model to learn cross-regime
    # patterns instead of memorising training period.
    "xgb_params": {
        "num_parallel_tree": 3,
        "n_estimators": 400,
        "max_depth": 3,                # was 5 — shallow trees generalise better across regimes
        "learning_rate": 0.04,         # was 0.025 — back up; gain from low LR doesn't help if overfitting
        "gamma": 0.4,                  # was 0.2 — stricter split criterion
        "min_child_weight": 20,        # was 3 — much bigger leaves, no sparse-leaf memorization
        "subsample": 0.6,              # was 0.75 — more stochastic
        "colsample_bytree": 0.6,       # was 0.75 — each tree sees fewer features
        "colsample_bylevel": 0.6,      # was 0.8 — each split sees fewer features
        "reg_alpha": 0.15,             # was 0.05 — stronger L1 sparsity
        "reg_lambda": 2.0,             # was 0.8 — stronger L2 smoothing of leaf weights
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": 0.9,       # overridden at train time from actual class balance
        "early_stopping_rounds": 30,   # noisier validation → more patience
        "random_state": 3301,
        "verbosity": 2,
        "nthread": 32,
    },

    # XGBoost parameters for GPU - keeping this as in your original file
    "xgb_params__STRANGE__GPU__": {
        "tree_method": "hist",
        "num_boost_round": 1,
        "device": "cuda",
        "num_parallel_tree": 256,
        "n_estimators": 8,
        "max_depth": 8,
        "learning_rate": 0.1,
        "gamma": 0.1,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        'enable_categorical': True,
        "scale_pos_weight": 2.0,
        "random_state": 3301,
        "verbosity": 2,
        "early_stopping_rounds": 15,
        "max_delta_step": 1,
        "nthread": 32,
        'use_label_encoder': True
    },
    
    # Common parameters
    "early_stopping_rounds": 10,
    "random_state": 3301
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
    
    # Lower-bound cutoff (in case ancient garbage data is present)
    cutoff_date = pd.to_datetime('2021-01-10')
    logging.info(f"Hard lower-bound cutoff: {cutoff_date.strftime('%Y-%m-%d')}")

    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    all_files = sorted(all_files)

    if file_selection_percentage < 100:
        random.seed(config['random_state'])
        random.shuffle(all_files)
        all_files = all_files[:int(len(all_files) * file_selection_percentage / 100)]

    # Scan a sample of files for the ACTUAL data range — using cutoff_date as
    # min_date was wrong: real data only starts ~2024-06, so the previous code
    # placed calib_start_date in mid-2025 and gave calibration ~13 months of data
    # (most of which got filtered out per-ticker), producing a tiny eval set.
    sample_min_dates = []
    sample_max_dates = []
    for f in all_files[:min(200, len(all_files))]:
        try:
            tmp = pd.read_parquet(os.path.join(input_directory, f), columns=[date_column])
            tmp[date_column] = pd.to_datetime(tmp[date_column])
            if len(tmp) > 0:
                sample_min_dates.append(tmp[date_column].min())
                sample_max_dates.append(tmp[date_column].max())
        except Exception:
            pass

    if sample_min_dates and sample_max_dates:
        # Use the 5th percentile of min dates to avoid being dragged early by a
        # single anomalous ticker that has a 20-year history.
        data_min_date = max(pd.Series(sample_min_dates).quantile(0.05), cutoff_date)
        data_max_date = max(sample_max_dates)
    else:
        data_min_date = cutoff_date
        data_max_date = pd.Timestamp.now()

    total_days = (data_max_date - data_min_date).days
    calib_start_date = data_min_date + pd.Timedelta(days=int(total_days * (1 - calibration_percentage / 100)))
    logging.info(f"Data range: {data_min_date.date()} → {data_max_date.date()} ({total_days} days)")
    logging.info(f"Temporal split: training {data_min_date.date()} → {calib_start_date.date()} | calibration {calib_start_date.date()} → {data_max_date.date()}")
    print(f"[split] train: {data_min_date.date()} → {calib_start_date.date()}  |  calib: {calib_start_date.date()} → {data_max_date.date()}")

    train_files = all_files
    calib_files = all_files

    import json
    file_allocation = {
        'all_files': all_files,
        'split_strategy': 'temporal',
        'train_date_range': f"{data_min_date.date()} to {calib_start_date.date()}",
        'calib_date_range': f"{calib_start_date.date()} to {data_max_date.date()}",
    }
    with open(file_allocation_record, 'w') as f:
        json.dump(file_allocation, f, indent=2, default=str)

    if os.path.exists(train_output_file):
        os.remove(train_output_file)
    train_data = process_files(train_files, input_directory, data_min_date, target_column, date_column,
                               end_date=calib_start_date)
    if len(train_data) == 0:
        logging.error("No valid training data found after processing files and date filtering.")
        raise ValueError("No valid training data found after processing files and date filtering.")

    if os.path.exists(calib_output_file):
        os.remove(calib_output_file)
    calib_data = process_files(calib_files, input_directory, calib_start_date, target_column, date_column)

    train_df = pd.concat(train_data)
    calib_df = pd.concat(calib_data) if len(calib_data) > 0 else pd.DataFrame()

    # Sanity check: calibration must be large enough for threshold search.
    # If too small (e.g. because of aggressive per-ticker filtering), pull the
    # last 20% off training to act as calibration instead.
    MIN_CALIB_ROWS = 20000
    if len(calib_df) < MIN_CALIB_ROWS:
        logging.warning(
            f"Calibration set too small ({len(calib_df)} rows < {MIN_CALIB_ROWS}). "
            f"Falling back to last 20% of training data as calibration."
        )
        print(f"[fallback] calib={len(calib_df)} rows too small; using tail of training instead")
        combined = pd.concat([train_df, calib_df]).sort_values(by=date_column).reset_index(drop=True)
        split_idx = int(len(combined) * 0.80)
        train_df = combined.iloc[:split_idx]
        calib_df = combined.iloc[split_idx:]

    train_df = finalize_dataset(train_df, date_column)
    calib_df = finalize_dataset(calib_df, date_column)

    train_df.to_parquet(train_output_file, index=False)
    calib_df.to_parquet(calib_output_file, index=False)

    # Class balance diagnostics
    t_pos = (train_df[target_column] >= 0).sum()
    t_neg = (train_df[target_column] < 0).sum()
    c_pos = (calib_df[target_column] >= 0).sum()
    c_neg = (calib_df[target_column] < 0).sum()
    logging.info(f"Training shape: {train_df.shape} | class balance: {t_neg} neg / {t_pos} pos ({t_pos/(t_pos+t_neg):.1%} positive)")
    logging.info(f"Calibration shape: {calib_df.shape} | class balance: {c_neg} neg / {c_pos} pos ({c_pos/(c_pos+c_neg):.1%} positive)")
    print(f"[balance] train: {t_pos/(t_pos+t_neg):.1%} pos | calib: {c_pos/(c_pos+c_neg):.1%} pos")

    return train_df, calib_df


def process_files(file_list, input_directory, cutoff_date, target_column, date_column, end_date=None):
    """Process a list of files and return a list of valid dataframes.

    cutoff_date: inclusive lower bound on dates (start_date >= cutoff_date)
    end_date: exclusive upper bound on dates (date < end_date); None means no upper bound
    """
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
            df = df.sort_values(by=date_column)

            # Apply date window — lower bound inclusive, upper bound exclusive
            rows_before = len(df)
            df = df[df[date_column] >= cutoff_date]
            if end_date is not None:
                df = df[df[date_column] < end_date]
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
            
            # Target shift: today's features predict tomorrow's direction.
            # shift(-1) moves tomorrow's return into the current row.
            # Drop the last row (NaN target) only — do NOT drop the first row;
            # rolling features use min_periods so the first row has valid (if sparse) values.
            df[target_column] = df[target_column].shift(-1)
            df = df.iloc[:-1]

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


def tune_hyperparameters(X_tr, y_tr, sw_tr, X_v, y_v, natural_spw, n_trials, sample_rows):
    """
    Optuna search over the regularization-heavy parameters that actually matter
    for cross-regime generalization. Uses a fixed validation set for the objective
    (val AUC-PR) and subsamples training rows to keep each trial fast.

    Returns: dict of best params (without n_estimators / early_stopping_rounds,
    which the caller controls).
    """
    import optuna
    from sklearn.metrics import average_precision_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Subsample training data for fast trials. The validation set is FULL so the
    # objective signal is stable across trials.
    n_tr_total = len(X_tr)
    rng = np.random.default_rng(3301)
    if n_tr_total > sample_rows:
        idx = rng.choice(n_tr_total, size=sample_rows, replace=False)
        X_tr_s = X_tr.iloc[idx]
        y_tr_s = y_tr.iloc[idx]
        sw_tr_s = sw_tr[idx]
    else:
        X_tr_s, y_tr_s, sw_tr_s = X_tr, y_tr, sw_tr

    logging.info(f"Optuna tuning: {n_trials} trials, {len(X_tr_s)} train rows/trial, {len(X_v)} val rows")
    print(f"[tune] {n_trials} trials on {len(X_tr_s):,} train rows, {len(X_v):,} val rows")

    def _objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 3301,
            'nthread': 16,
            'verbosity': 0,
            'tree_method': 'hist',
            'n_estimators': 250,
            'early_stopping_rounds': 15,
            # Constrained search — last run's Optuna picked aggressive params
            # (max_depth=5, reg_lambda=0.16) that won the val period but collapsed
            # on eval. We've established that tight regularization generalises.
            # Keep Optuna within that proven region.
            'max_depth':         trial.suggest_int('max_depth', 2, 4),
            'learning_rate':     trial.suggest_float('learning_rate', 0.02, 0.06, log=True),
            'min_child_weight':  trial.suggest_int('min_child_weight', 15, 60),
            'subsample':         trial.suggest_float('subsample', 0.50, 0.80),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.50, 0.80),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.50, 0.85),
            'gamma':             trial.suggest_float('gamma', 0.2, 1.2),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.05, 0.5, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1.0, 5.0, log=True),  # never below 1.0
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 2, 4),
            'scale_pos_weight':  natural_spw * trial.suggest_float('spw_factor', 0.75, 1.05),
        }
        try:
            clf = XGBClassifier(**params)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                clf.fit(X_tr_s, y_tr_s, sample_weight=sw_tr_s,
                        eval_set=[(X_v, y_v)], verbose=False)
            val_pred = clf.predict_proba(X_v)[:, 1]
            val_auc = float(average_precision_score(y_v, val_pred))

            # Gap-penalised objective: reject overfit configurations.
            # Measure training AUC-PR on a decimated subset to keep trials fast.
            train_pred = clf.predict_proba(X_tr_s.iloc[::10])[:, 1]
            train_auc = float(average_precision_score(y_tr_s.iloc[::10], train_pred))
            gap = train_auc - val_auc

            # Penalty: any gap above 0.05 costs 2x its excess from the objective.
            # Two trials with the same val AUC but different gaps will be
            # ordered: the one that generalises wins.
            gap_penalty = max(0.0, gap - 0.05) * 2.0
            score = val_auc - gap_penalty

            trial.set_user_attr('val_auc', val_auc)
            trial.set_user_attr('train_auc', train_auc)
            trial.set_user_attr('gap', gap)
            return score
        except Exception as e:
            logging.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=3301, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    progress_bar = tqdm(total=n_trials, desc="Optuna trials", ncols=100)
    def _cb(study, trial):
        progress_bar.update(1)
        # show val AUC alongside the gap-penalised score so user can see both
        v = trial.user_attrs.get('val_auc', 0.0)
        g = trial.user_attrs.get('gap', 0.0)
        progress_bar.set_postfix(score=f"{study.best_value:.4f}", val=f"{v:.4f}", gap=f"{g:+.3f}")

    study.optimize(_objective, n_trials=n_trials, callbacks=[_cb], gc_after_trial=True)
    progress_bar.close()

    best = study.best_params
    best_trial = study.best_trial
    best_val = best_trial.user_attrs.get('val_auc', study.best_value)
    best_train = best_trial.user_attrs.get('train_auc', 0.0)
    best_gap = best_trial.user_attrs.get('gap', 0.0)
    logging.info(f"Optuna best score: {study.best_value:.4f} (val AUC-PR={best_val:.4f}, train={best_train:.4f}, gap={best_gap:+.4f})")
    logging.info(f"Optuna best params: {best}")
    print(f"[tune] best gap-penalised score: {study.best_value:.4f}")
    print(f"[tune] best val AUC-PR: {best_val:.4f} | train AUC-PR: {best_train:.4f} | gap: {best_gap:+.4f}")
    print(f"[tune] best params: {best}")

    # Drop the spw_factor key — caller will recompute scale_pos_weight from it
    spw_factor = best.pop('spw_factor', 1.0)
    best['scale_pos_weight'] = natural_spw * spw_factor
    return best


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
    y_train = training_data[config['target_column']]
    y_train = y_train.apply(lambda x: 0 if x < 0 else 1)
    
    # Prepare calibration data
    X_calib = calibration_data.drop(columns=[config['target_column']])
    y_calib = calibration_data[config['target_column']]
    y_calib = y_calib.apply(lambda x: 0 if x < 0 else 1)
    
    # Remove datetime columns
    datetime_columns_train = X_train.select_dtypes(include=['datetime64']).columns
    datetime_columns_calib = X_calib.select_dtypes(include=['datetime64']).columns
    
    X_train = X_train.drop(columns=datetime_columns_train)
    X_calib = X_calib.drop(columns=datetime_columns_calib)
    
    # Chronological validation split — last 20% of dates, no random leakage.
    # A 5-row embargo is dropped between the two halves so that rolling features
    # (e.g. 20d MA) in the val set don't contain rows that were targets in training.
    embargo_rows = 5
    split_idx = int(len(X_train) * 0.80)
    X_train_final = X_train.iloc[:split_idx]
    X_val         = X_train.iloc[split_idx + embargo_rows:]   # skip the embargo gap
    y_train_final = y_train.iloc[:split_idx]
    y_val         = y_train.iloc[split_idx + embargo_rows:]

    # Recency sample weights — exponential decay with half-life 720 trading days
    half_life_days = 720
    decay = np.log(2) / half_life_days
    train_dates = pd.to_datetime(training_data['Date'].iloc[:split_idx].values)
    max_date = train_dates.max()
    age_days = (max_date - train_dates).days
    sample_weights = np.exp(-decay * age_days).astype(np.float32)

    # Data-driven scale_pos_weight: use the actual training class balance,
    # nudged by a precision-favouring factor (<1 → penalises false positives).
    n_pos = int((y_train_final == 1).sum())
    n_neg = int((y_train_final == 0).sum())
    natural_spw = n_neg / max(n_pos, 1)
    precision_bias = 0.9
    data_driven_spw = natural_spw * precision_bias
    xgb_params = config['xgb_params'].copy()
    xgb_params['scale_pos_weight'] = data_driven_spw
    logging.info(
        f"Training class balance: {n_neg} neg / {n_pos} pos "
        f"(natural spw={natural_spw:.3f}, applied spw={data_driven_spw:.3f})"
    )
    print(f"[train] {n_neg} neg / {n_pos} pos | spw={data_driven_spw:.3f}")

    # Optional Optuna hyperparameter search before the final fit.
    if args.tune:
        try:
            tuned = tune_hyperparameters(
                X_train_final, y_train_final, sample_weights,
                X_val, y_val, natural_spw,
                n_trials=args.trials,
                sample_rows=args.tune_sample,
            )
            # Merge tuned params over defaults, but keep our control knobs
            preserved = {
                'objective': xgb_params['objective'],
                'eval_metric': xgb_params['eval_metric'],
                'random_state': xgb_params['random_state'],
                'nthread': xgb_params['nthread'],
                'verbosity': xgb_params['verbosity'],
                'n_estimators': xgb_params.get('n_estimators', 400),
                'early_stopping_rounds': xgb_params.get('early_stopping_rounds', 30),
                'tree_method': 'hist',
            }
            xgb_params = {**tuned, **preserved}
            logging.info(f"Final xgb_params after tuning: {xgb_params}")
        except Exception as e:
            logging.error(f"Optuna tuning failed: {e}. Falling back to default params.")

    clf = XGBClassifier(**xgb_params)

    try:
        clf.fit(
            X_train_final, y_train_final,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    except Exception as e:
        logging.error(f"Error during XGBoost fitting: {str(e)}")
        clf.fit(X_train_final, y_train_final, sample_weight=sample_weights)

    # Under/over-fitting diagnostic: compare train vs. val AUC-PR.
    # If train >> val, the model is overfitting; if both are low, underfitting.
    try:
        from sklearn.metrics import average_precision_score
        train_pred_sample = clf.predict_proba(X_train_final.iloc[::10])[:, 1]  # decimated
        train_auc_pr = average_precision_score(y_train_final.iloc[::10], train_pred_sample)
        val_pred = clf.predict_proba(X_val)[:, 1]
        val_auc_pr = average_precision_score(y_val, val_pred)
        gap = train_auc_pr - val_auc_pr
        logging.info(f"AUC-PR: train={train_auc_pr:.4f} | val={val_auc_pr:.4f} | gap={gap:+.4f}")
        print(f"[fit] train AUC-PR={train_auc_pr:.4f} | val AUC-PR={val_auc_pr:.4f} | gap={gap:+.4f}")
        if gap > 0.10:
            logging.warning(f"Large train-val AUC-PR gap ({gap:+.4f}) → overfitting; consider tighter regularization")
        elif train_auc_pr < 0.60:
            logging.warning(f"Low train AUC-PR ({train_auc_pr:.4f}) → underfitting; consider deeper trees / less regularization")
    except Exception as e:
        logging.warning(f"Could not compute train/val AUC-PR diagnostic: {str(e)}")
    
    # Split calibration_data temporally: first 60% fits the calibrator,
    # last 40% is held out purely for threshold search.
    # This prevents the threshold search from being in-sample for the calibrator,
    # which would inflate the reported precision.
    calib_split = int(len(X_calib) * 0.60)
    X_calib_fit   = X_calib.iloc[:calib_split]
    y_calib_fit   = y_calib.iloc[:calib_split]
    X_calib_eval  = X_calib.iloc[calib_split:]
    y_calib_eval  = y_calib.iloc[calib_split:]
    logging.info(f"Calibrator fit set: {len(X_calib_fit)} rows | Threshold eval set: {len(X_calib_eval)} rows")

    # Calibrate probabilities using HybridCalibrator (PCHIP+Platt tail blend)
    hybrid_calibrator = None
    if apply_calibration:
        logging.info("Calibrating model probabilities with HybridCalibrator...")
        try:
            raw_fit_probs = clf.predict_proba(X_calib_fit)[:, 1]
            hybrid_calibrator = HybridCalibrator(min_tail_breakpoints=10, blend_width=0.03)
            hybrid_calibrator.fit(raw_fit_probs, y_calib_fit.values)
            test_out = hybrid_calibrator.predict(raw_fit_probs)
            assert np.all(np.isfinite(test_out)), "HybridCalibrator produced non-finite values"
            logging.info("HybridCalibrator fitted successfully.")
        except Exception as e:
            logging.error(f"HybridCalibrator failed, falling back to isotonic: {str(e)}")
            hybrid_calibrator = None
            try:
                fallback_cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
                fallback_cal.fit(X_calib_fit, y_calib_fit)
                hybrid_calibrator = fallback_cal
                logging.info("Fallback: isotonic calibration applied.")
            except Exception as e2:
                logging.error(f"Isotonic fallback also failed: {str(e2)}")

        try:
            plot_calibration_curves(clf, X_calib_fit, y_calib_fit, output_path=config['calibration_plot_output'])
        except Exception as e:
            logging.error(f"Failed to create calibration plot: {str(e)}")

    if hybrid_calibrator is not None and isinstance(hybrid_calibrator, HybridCalibrator):
        clf_for_prediction = _HybridWrap(clf, hybrid_calibrator)
        logging.info("Using HybridCalibrator wrapper for predictions.")
    elif hybrid_calibrator is not None:
        clf_for_prediction = hybrid_calibrator   # sklearn fallback already has predict_proba
        logging.info("Using sklearn isotonic calibration fallback.")
    else:
        clf_for_prediction = clf
        logging.info("Using uncalibrated model for predictions.")
    
    # Get predicted probabilities on the HELD-OUT eval set — calibrator has never seen this data
    y_pred_proba = clf_for_prediction.predict_proba(X_calib_eval)
    y_calib = y_calib_eval  # alias for the threshold/evaluation block below

    # Diagnostic: calibrated probability distribution. If the model collapsed,
    # this will show all probabilities packed in a narrow band.
    probs_up = y_pred_proba[:, 1]
    base_rate_eval = (y_calib == 1).mean()

    # AUTO-FALLBACK: if the calibrator squashed the tail (regime shift between
    # cal_fit and cal_eval makes isotonic flatten the mapping), the underlying
    # XGB model may still have signal. Try raw probabilities — they preserve
    # the model's actual discrimination even when the calibrator can't.
    used_raw_fallback = False
    if probs_up.max() < 0.65 and hybrid_calibrator is not None:
        raw_pred_proba = clf.predict_proba(X_calib_eval)
        raw_max = raw_pred_proba[:, 1].max()
        logging.warning(
            f"Calibrator collapsed: max calibrated prob = {probs_up.max():.3f}. "
            f"Raw XGB max = {raw_max:.3f}. Falling back to raw probabilities for "
            f"threshold search — calibration will be skipped at prediction time."
        )
        print(f"[fallback] calibrator collapsed (max={probs_up.max():.3f}); switching to RAW probs (max={raw_max:.3f})")
        y_pred_proba = raw_pred_proba
        probs_up = y_pred_proba[:, 1]
        # Switch the prediction model to the raw XGB — calibrator is broken,
        # so we save the model uncalibrated and predict_and_save will see raw probs.
        clf_for_prediction = clf
        hybrid_calibrator = None
        used_raw_fallback = True
    logging.info(f"Threshold search on {len(y_calib_eval)}-row held-out eval set (out-of-sample for calibrator)")
    logging.info(
        f"Calibrated prob distribution: "
        f"q10={np.quantile(probs_up, 0.10):.3f} | "
        f"q50={np.quantile(probs_up, 0.50):.3f} | "
        f"q90={np.quantile(probs_up, 0.90):.3f} | "
        f"q95={np.quantile(probs_up, 0.95):.3f} | "
        f"q99={np.quantile(probs_up, 0.99):.3f} | "
        f"max={probs_up.max():.3f}"
    )
    logging.info(f"Eval base rate (class 1): {base_rate_eval:.3f}")
    print(
        f"[probs] q50={np.quantile(probs_up, 0.50):.3f} "
        f"q90={np.quantile(probs_up, 0.90):.3f} "
        f"q99={np.quantile(probs_up, 0.99):.3f} "
        f"max={probs_up.max():.3f} | base rate {base_rate_eval:.1%}"
    )

    # Guardrail: if the calibrated probability max is suspiciously low, the model
    # has collapsed — either training failed to discriminate or calibrator squashed
    # the tail. Warn loudly so the user doesn't waste compute on a doomed run.
    if probs_up.max() < 0.65:
        logging.warning(
            f"COLLAPSED MODEL: max calibrated prob is {probs_up.max():.3f} — "
            f"the calibrator thinks even the most confident predictions are barely above base rate. "
            f"This usually means the hyperparameters don't generalise to the eval period. "
            f"Try running without --tune (use defaults) or with --trials 100+ to give Optuna more search budget."
        )
        print(f"[ALERT] max calibrated prob {probs_up.max():.3f} < 0.65 — model has effectively collapsed")

    # Target precision (CLI-configurable). Bumping --precision-target 0.85 makes
    # the search reject anything below that and rely on the quantile fallback.
    target_min_precision_pos = args.precision_target
    min_predictions_percent_pos = 0.005

    target_min_precision_neg = 0.65
    min_predictions_percent_neg = 0.05

    # Minimum sample size for any candidate quantile to be considered.
    # Lower = allow tighter quantiles (more precision, fewer signals, noisier estimate).
    MIN_N_PRED = args.min_predictions

    # Find optimal thresholds for positive class via the full PR curve
    precisions_pos, recalls_pos, thresholds_pos = precision_recall_curve(
        y_calib, y_pred_proba[:, 1], pos_label=1
    )

    if len(precisions_pos) > len(thresholds_pos):
        precisions_pos = precisions_pos[:-1]
        recalls_pos = recalls_pos[:-1]

    prediction_coverage = np.array([(y_pred_proba[:, 1] >= t).mean() for t in thresholds_pos])

    # Also do a fine-grained sweep in [0.50, 0.97] with 0.005 step for precision maximisation
    fine_thresholds = np.arange(0.50, 0.97, 0.005)
    fine_precisions, fine_recalls, fine_coverage = [], [], []
    for ft in fine_thresholds:
        mask = y_pred_proba[:, 1] >= ft
        if mask.sum() > 0:
            fine_precisions.append((y_calib[mask] == 1).mean())
            fine_recalls.append(mask[y_calib == 1].mean())
        else:
            fine_precisions.append(0.0)
            fine_recalls.append(0.0)
        fine_coverage.append(mask.mean())
    fine_precisions = np.array(fine_precisions)
    fine_recalls    = np.array(fine_recalls)
    fine_coverage   = np.array(fine_coverage)

    # Merge PR-curve and fine-sweep arrays for a richer search space
    all_thresholds  = np.concatenate([thresholds_pos, fine_thresholds])
    all_precisions  = np.concatenate([precisions_pos, fine_precisions])
    all_recalls     = np.concatenate([recalls_pos, fine_recalls])
    all_coverage    = np.concatenate([prediction_coverage, fine_coverage])

    # Helper: measure precision/recall/coverage at a given threshold
    def _metrics_at(t):
        m = y_pred_proba[:, 1] >= t
        if m.sum() < 5:
            return 0.0, 0.0, m.mean(), int(m.sum())
        # Use positional indexing — y_pred_proba is numpy positional, y_calib is pandas
        p = float((y_calib.values[m] == 1).mean())
        r = float(m[y_calib.values == 1].mean()) if (y_calib == 1).sum() > 0 else 0.0
        return p, r, float(m.mean()), int(m.sum())

    valid_indices = (all_precisions >= target_min_precision_pos) & (all_coverage >= min_predictions_percent_pos)
    if np.any(valid_indices):
        # Best case: a threshold genuinely hits target precision with adequate coverage.
        valid_mask = np.where(valid_indices)[0]
        best_sub = np.lexsort((all_recalls[valid_mask], all_precisions[valid_mask]))[-1]
        best_idx = valid_mask[best_sub]
        optimal_threshold_pos = float(all_thresholds[best_idx])
        pos_precision, pos_recall, pos_coverage, _ = _metrics_at(optimal_threshold_pos)
        logging.info(f"Found threshold meeting precision target: {optimal_threshold_pos:.4f}")
    else:
        # Fallback: pick the threshold at the (1 - coverage_target) quantile of predicted
        # probabilities. This guarantees a fixed coverage (e.g. top 1%) regardless of how
        # the calibrated distribution is shaped. Try progressively tighter quantiles to
        # maximise precision while keeping enough rows for a meaningful metric.
        logging.warning(
            f"No threshold meets precision target {target_min_precision_pos:.2f} "
            f"with coverage >= {min_predictions_percent_pos:.3%}. "
            f"Using quantile-based fallback."
        )
        # Tighter quantile sweep — the precision/recall curve has a long flat
        # middle and a sharp uptick at the very top. We need to find that uptick.
        candidate_coverages = [
            0.02, 0.01, 0.0075, 0.005, 0.0035, 0.0025,
            0.0015, 0.001, 0.00075, 0.0005, 0.00035, 0.00025, 0.00015, 0.0001
        ]
        fallback_results = []     # (precision, threshold, recall, coverage, n_pred, edge)
        for cov_target in candidate_coverages:
            thr = float(np.quantile(probs_up, 1.0 - cov_target))
            p, r, cov, n_pred = _metrics_at(thr)
            if n_pred < MIN_N_PRED:
                logging.info(f"  quantile cov={cov_target:.4%}: n_pred={n_pred} < {MIN_N_PRED} (skipped)")
                continue
            edge = p - base_rate_eval
            logging.info(
                f"  quantile cov={cov_target:.4%}: thr={thr:.4f} "
                f"prec={p:.4f} recall={r:.4f} n={n_pred} edge={edge:+.4f}"
            )
            print(f"  fallback cov={cov_target:.4%} → prec={p:.3f} n={n_pred} edge={edge:+.3f}")
            fallback_results.append((p, thr, r, cov, n_pred, edge))

        if fallback_results:
            # Prefer raw PRECISION (not edge) — that's what the user actually wants.
            # Require precision > base_rate + 0.05 to ensure real signal, otherwise reject.
            min_acceptable_edge = 0.05
            qualified = [r for r in fallback_results if r[5] > min_acceptable_edge]
            if qualified:
                # Among qualified candidates, pick the one with highest raw precision.
                # Break ties by larger n_pred (more reliable estimate).
                qualified.sort(key=lambda x: (x[0], x[4]), reverse=True)
                pos_precision, optimal_threshold_pos, pos_recall, pos_coverage, n_pred_sel, edge_sel = qualified[0]
                logging.info(
                    f"Selected quantile fallback: thr={optimal_threshold_pos:.4f} "
                    f"prec={pos_precision:.4f} recall={pos_recall:.4f} "
                    f"coverage={pos_coverage:.4%} n={n_pred_sel} edge={edge_sel:+.4f}"
                )
                print(f"[chosen] thr={optimal_threshold_pos:.4f} prec={pos_precision:.3f} n={n_pred_sel} cov={pos_coverage:.3%}")
            else:
                # No quantile shows >5pp edge over base rate → model genuinely has no usable
                # tail. Emit zero predictions rather than degenerate yes-to-all.
                optimal_threshold_pos = max(0.99, float(probs_up.max()) + 0.01)
                pos_precision, pos_recall, pos_coverage, _ = _metrics_at(optimal_threshold_pos)
                logging.warning(
                    f"No quantile beats base rate by >{min_acceptable_edge:.0%}. "
                    f"Setting threshold above max prob ({optimal_threshold_pos:.4f}) → zero predictions."
                )
                print(f"[chosen] threshold disabled — no usable edge in tail; signals will be empty")
        else:
            optimal_threshold_pos = max(0.99, float(probs_up.max()) + 0.01)
            pos_precision, pos_recall, pos_coverage, _ = _metrics_at(optimal_threshold_pos)
            logging.warning(f"All quantile candidates had n_pred<20 → threshold set to {optimal_threshold_pos:.4f}")

    # --- DOWN threshold via quantile fallback (mirrors UP logic) ---
    # The old code defaulted to (mean_prob + std_prob) which produces a very low
    # threshold → "predict down for everyone not predicted up" → degenerate report.
    probs_down = y_pred_proba[:, 0]
    base_rate_neg = float((y_calib == 0).mean())

    def _neg_metrics_at(t):
        m = probs_down >= t
        if m.sum() < 5:
            return 0.0, 0.0, m.mean(), int(m.sum())
        p = float((y_calib[m] == 0).mean())
        r = float(m[y_calib == 0].mean()) if (y_calib == 0).sum() > 0 else 0.0
        return p, r, float(m.mean()), int(m.sum())

    # Coarse PR curve search at the target precision
    precisions_neg, recalls_neg, thresholds_neg = precision_recall_curve(
        1 - y_calib, probs_down, pos_label=1
    )
    if len(precisions_neg) > len(thresholds_neg):
        precisions_neg = precisions_neg[:-1]
        recalls_neg = recalls_neg[:-1]
    neg_cov = np.array([(probs_down >= t).mean() for t in thresholds_neg])

    neg_valid = (precisions_neg >= target_min_precision_neg) & (neg_cov >= min_predictions_percent_neg)
    if np.any(neg_valid):
        idx_pool = np.where(neg_valid)[0]
        sel = idx_pool[np.argmax(recalls_neg[idx_pool])]
        optimal_threshold_neg = float(thresholds_neg[sel])
        neg_precision = float(precisions_neg[sel])
        neg_recall    = float(recalls_neg[sel])
        neg_coverage  = float(neg_cov[sel])
    else:
        # Quantile fallback for DOWN class
        logging.warning(f"No DOWN threshold meets precision target {target_min_precision_neg:.2f}; using quantile fallback.")
        candidate_coverages_neg = [0.05, 0.02, 0.01, 0.005, 0.0025]
        neg_results = []
        for cov_target in candidate_coverages_neg:
            thr = float(np.quantile(probs_down, 1.0 - cov_target))
            p, r, cov, n_pred = _neg_metrics_at(thr)
            if n_pred < 20:
                continue
            edge = p - base_rate_neg
            logging.info(f"  DOWN quantile cov={cov_target:.4%}: thr={thr:.4f} prec={p:.4f} recall={r:.4f} n={n_pred} edge={edge:+.4f}")
            neg_results.append((p, thr, r, cov, n_pred, edge))

        if neg_results:
            qualified_neg = [r for r in neg_results if r[5] > 0.05]
            if qualified_neg:
                qualified_neg.sort(key=lambda x: (x[0], x[4]), reverse=True)
                neg_precision, optimal_threshold_neg, neg_recall, neg_coverage, _, _ = qualified_neg[0]
                logging.info(f"Selected DOWN threshold {optimal_threshold_neg:.4f} prec={neg_precision:.4f}")
            else:
                # No usable down signal — disable downs
                optimal_threshold_neg = max(0.99, float(probs_down.max()) + 0.01)
                neg_precision, neg_recall, neg_coverage, _ = _neg_metrics_at(optimal_threshold_neg)
                logging.warning(f"No DOWN edge over base rate. Disabling down predictions (thr={optimal_threshold_neg:.4f}).")
        else:
            optimal_threshold_neg = max(0.99, float(probs_down.max()) + 0.01)
            neg_precision, neg_recall, neg_coverage, _ = _neg_metrics_at(optimal_threshold_neg)

    # Calculate expected profit factor
    if pos_precision > 0:
        expected_profit_factor = (pos_precision / (1 - pos_precision))
        logging.info(f"Expected profit factor for UP predictions: {expected_profit_factor:.4f}")
        logging.info(f"This means for every $1 lost, you can expect to make ${expected_profit_factor:.2f}")

    # Log threshold information
    logging.info(f"Optimal threshold for class 1 (UP): {optimal_threshold_pos:.4f} with precision {pos_precision:.4f}, recall {pos_recall:.4f}, coverage {pos_coverage:.4f}")
    logging.info(f"Optimal threshold for class 0 (DOWN): {optimal_threshold_neg:.4f} with precision {neg_precision:.4f}, recall {neg_recall:.4f}, coverage {neg_coverage:.4f}")

    # Apply thresholds to get predictions
    y_pred = np.full(len(y_calib), -1)  # Default to "no prediction" (-1)
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
    y_calib_filtered = y_calib[mask_definitive]
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
    
    # Save model and thresholds.
    # When raw-fallback fired, hybrid_calibrator was nulled and clf_for_prediction
    # is the base XGB. predict_and_save will use 'base_model' since calibrated_model is None.
    _is_calibrated = (hybrid_calibrator is not None) and (not used_raw_fallback)
    model_data = {
        'base_model': clf,
        'calibrated_model': clf_for_prediction if _is_calibrated else None,
        'is_calibrated': _is_calibrated,
        'used_raw_fallback': used_raw_fallback,
        'threshold_pos': optimal_threshold_pos,
        'threshold_neg': optimal_threshold_neg,
        'precision_pos': pos_precision,
        'recall_pos': pos_recall,
        'precision_neg': neg_precision,
        'recall_neg': neg_recall
    }
    if used_raw_fallback:
        logging.info("Saved model in RAW mode (no calibrator) — thresholds are in raw probability space.")
        print(f"[save] model saved in RAW mode | threshold_pos={optimal_threshold_pos:.4f} (raw space)")
    
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
    
    if isinstance(model_data, dict):
        if 'calibrated_model' in model_data and model_data['calibrated_model'] is not None:
            # Use calibrated model if available
            clf = model_data['calibrated_model']
            logging.info("Using calibrated model for predictions.")
        elif 'base_model' in model_data:
            # Use base model if calibrated model is not available
            clf = model_data['base_model']
            logging.info("Using base model for predictions.")
        elif 'model' in model_data:
            # Backward compatibility
            clf = model_data['model']
            logging.info("Using model from older format.")
        else:
            # For very old model files
            clf = model_data
            logging.warning("Using model from legacy format.")
            
        # Get thresholds
        if 'threshold_pos' in model_data:
            threshold_pos = model_data['threshold_pos']
            threshold_neg = model_data['threshold_neg']
            logging.info(f"Using optimized thresholds - Positive: {threshold_pos:.4f}, Negative: {threshold_neg:.4f}")
        else:
            # Default thresholds
            threshold_pos = 0.7
            threshold_neg = 0.7
            logging.warning("Using default thresholds as no optimized thresholds found")
    else:
        # Legacy format
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
        
        # Assign probabilities - these are CALIBRATED probabilities if a calibrated model was used
        df['UpProbability'] = y_pred_proba[:, 1]
        df['DownProbability'] = y_pred_proba[:, 0]
        
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