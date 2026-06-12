"""Clean-sheet predictor (2026-05-16 redesign).

Pipeline:
  Train:
    1. Load ProcessedData/*.parquet, per-ticker shift target by -1 (next-day return),
       add 5-day forward return, concatenate.
    2. Apply FilterRubric Step 1 universe gate (Close>=5, dollar_volume_ma_10>=5M,
       atr_percentage<=0.05, RSI not in [30,40)).
    3. Add cross-sectional rank features (per-day percentile of every numeric
       column) -> 200 raw features become ~400 (raw + xs).
    4. Build labels: y_1d (binary up), y_5d (binary up over 5d), y_topq (per-day
       top-quintile by 1d return).
    5. Date-based train/calib split with 5-day embargo.
    6. Optionally tune each base learner via Optuna with per-day top-K precision
       objective on walk-forward CV folds.
    7. Generate walk-forward OOF predictions for 4 base learners:
       - XGBClassifier on y_1d
       - XGBRanker (pairwise) on y_topq with per-day groups
       - XGBClassifier on y_5d
       - Logistic regression on rank-normalised features
    8. Train logistic meta-stacker on the OOF predictions.
    9. Refit base learners on full training data.
    10. Predict on calibration slice -> meta-stacker probability -> fit Beta
        calibrator -> compute conformal threshold for target precision.
    11. Save full pipeline.

  Predict (--predict):
    1. Load all ProcessedData/*.parquet.
    2. Apply universe filter, mark non-passing rows for UpPrediction=-1 (kept
       in output, but ranks computed only on passing universe).
    3. Cross-sectional rank features on passing universe.
    4. Predict base learners -> meta-stacker -> calibrate.
    5. Consensus AND-gate: row fires only if its meta-stacker prob >= conformal
       threshold AND all 4 base learners place it in that day's top decile.
    6. Write per-ticker parquet to Data/RFpredictions/<TICKER>.parquet with
       columns: Date, Open, High, Low, Close, Volume, UpProbability,
       DownProbability, PositiveThreshold, NegativeThreshold, UpPrediction,
       VIX_Close (+ optional Distance to Support/Resistance, volatility).

Long-only by design. DownProbability = 1 - UpProbability is emitted only to
preserve the backtester contract; the model is not trained on the down side.
"""

import os
import sys
import json
import argparse
import logging
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io

# IMPORTANT: configure the ROOT logger first.  Util.get_logger() only attaches
# handlers to a named ("4__Predictor") logger, so module-level `logging.info`
# calls (which use root) would otherwise go to the void and we'd see no
# progress output during long-running phases.  force=True clears any prior
# config.  Stream handler -> stdout so output isn't reordered against tqdm
# (which writes to stderr).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

from xgboost import XGBClassifier, XGBRanker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Util import get_logger

logger = get_logger(script_name="4__Predictor")
# Prevent the named logger from re-emitting through root (would print twice).
logger.propagate = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


##=============================================================================##
##                                  CLI                                          ##
##=============================================================================##

argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=65,
                       help="Percent of total rows to use for training (date-ordered).")
argparser.add_argument("--calibpercent", type=int, default=15,
                       help="Percent of total rows to use for calibration (after train+embargo).")
argparser.add_argument("--clear", action="store_true",
                       help="Clear model and training/calibration cache before run.")
argparser.add_argument("--predict", action="store_true",
                       help="Inference mode: load saved pipeline, write per-ticker predictions.")
argparser.add_argument("--reuse", action="store_true",
                       help="Reuse cached training/calibration parquets if present.")
argparser.add_argument("--nocalib", action="store_true",
                       help="Disable Beta calibration (use raw meta-stacker output).")
argparser.add_argument("--tune", action="store_true",
                       help="Run Optuna hyperparameter tuning for each base learner.")
argparser.add_argument("--tune_trials", type=int, default=20,
                       help="Optuna trials per base learner (default 20).")
argparser.add_argument("--target_precision", type=float, default=0.75,
                       help="Conformal target precision on calibration slice.")
argparser.add_argument("--max_coverage", type=float, default=0.05,
                       help="Upper cap on coverage when picking conformal threshold.")
argparser.add_argument("--fast", action="store_true",
                       help="Quick mode: fewer Optuna trials (10), 2 CV folds, "
                            "tighter search range, narrower n_estimators. "
                            "Sacrifices quality for ~3x faster iteration.")
argparser.add_argument("--no_clf5d", action="store_true",
                       help="Skip the 5d horizon classifier. Useful when clf5d "
                            "OOF top-K precision is near random and the stacker "
                            "is overweighting it.")
args, _ = argparser.parse_known_args()


##=============================================================================##
##                                  CONFIG                                       ##
##=============================================================================##

config = {
    "input_directory": "Data/ProcessedData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "calibration_output_directory": "Data/ModelData/CalibrationData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "calibration_plot_output": "Data/ModelData/calibration_plot.png",
    "pipeline_path": "Data/ModelData/pipeline.joblib",

    "target_column": "percent_change_Close",   # gets shifted -1 to become next-day return
    "date_column": "Date",
    "ticker_column": "Ticker",

    "file_selection_percentage": args.runpercent,
    "calibration_percentage": args.calibpercent,
    "embargo_days": 5,

    "topq_fraction": 0.20,            # per-day top-quintile labels for the ranker
    "horizon_5d": 5,
    "n_oof_folds": 2 if args.fast else 4,
    "n_tune_folds": 2 if args.fast else 3,
    "tune_topk_frac": 0.01,           # per-day top-K used in Optuna objective
    "recency_half_life_days": 720.0,  # recency-weighted training
    "min_train_rows_per_fold": 50_000,
    "fast_mode": args.fast,

    # Default XGBClassifier hyperparams (used when --tune is not passed)
    "xgb_clf_defaults": {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.04,
        "gamma": 0.3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 3301,
        "early_stopping_rounds": 40,
    },

    # Default XGBRanker hyperparams (pairwise)
    "xgb_rank_defaults": {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.04,
        "gamma": 0.3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 3301,
        "early_stopping_rounds": 40,
    },

    # Default 5d classifier (slightly shallower; longer horizon = more noise)
    "xgb_clf5d_defaults": {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.04,
        "gamma": 0.3,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 7919,
        "early_stopping_rounds": 40,
    },

    "logistic_defaults": {
        "C": 0.5,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,
        "n_jobs": -1,
    },

    "stacker_C": 1.0,
}

config["apply_calibration"] = not args.nocalib


##=============================================================================##
##                               UTILITIES                                       ##
##=============================================================================##


def downcast_numeric(df):
    """Cast float64 columns to float32 to halve memory."""
    floats = df.select_dtypes(include=["float64"]).columns
    if len(floats) > 0:
        df[floats] = df[floats].astype("float32")
    return df


def drop_non_numeric_features(df, keep_cols):
    """Drop string / object columns except those in keep_cols."""
    drop_cols = [c for c in df.columns
                 if c not in keep_cols and df[c].dtype == "object"]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def recency_weights(dates, half_life_days):
    """Exponential decay weights with given half-life, normalised to mean 1."""
    dates = pd.to_datetime(dates)
    age = (dates.max() - dates).dt.days.values.astype(np.float64)
    w = np.power(0.5, age / half_life_days)
    return w / w.mean()


# ---- phase / progress logging helpers ----------------------------------------
import time as _time

class _Phase:
    """Context manager that logs entry, exit, and elapsed wall-time for a phase."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = _time.time()
        logging.info(f">>> {self.name} ...")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = _time.time() - self.t0
        if exc_type is None:
            logging.info(f"<<< {self.name} done in {dt:.1f}s.")
        else:
            logging.error(f"!!! {self.name} FAILED after {dt:.1f}s: {exc}")
        return False


def _optuna_log_callback(label, total_trials):
    """Callback that logs every Optuna trial completion with running best."""
    state = {"start": _time.time()}
    def cb(study, trial):
        elapsed = _time.time() - state["start"]
        try:
            best = study.best_value
        except Exception:
            best = float("nan")
        val_str = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        logging.info(f"  [{label}] trial {trial.number+1}/{total_trials} "
                     f"value={val_str} best={best:.4f} elapsed={elapsed:.0f}s")
    return cb


##=============================================================================##
##                            UNIVERSE FILTER                                    ##
##=============================================================================##


def apply_quality_filter(df,
                         min_close=5.0,
                         min_dollar_volume=5_000_000.0,
                         max_atr_pct=0.05,
                         rsi_exclude_lo=30.0,
                         rsi_exclude_hi=40.0):
    """FilterRubric Step 1 hard universe gate.

    Keeps only the rows the backtester can profitably trade. Applied identically
    at training and inference so the model never sees a name it can't act on.
    """
    n0 = len(df)
    mask = pd.Series(True, index=df.index)
    if "Close" in df.columns:
        mask &= df["Close"] >= min_close
    if "dollar_volume_ma_10" in df.columns:
        mask &= df["dollar_volume_ma_10"] >= min_dollar_volume
    if "atr_percentage" in df.columns:
        mask &= df["atr_percentage"] <= max_atr_pct
    if "RSI" in df.columns:
        mask &= ~((df["RSI"] >= rsi_exclude_lo) & (df["RSI"] < rsi_exclude_hi))
    out = df.loc[mask].copy()
    logging.info(f"Universe filter: {n0:,} -> {len(out):,} rows "
                 f"({100*len(out)/max(n0,1):.1f}% retained).")
    return out


##=============================================================================##
##                       CROSS-SECTIONAL RANK FEATURES                           ##
##=============================================================================##


def select_feature_columns(df, exclude):
    """Numeric feature columns, excluding labels / id / passthrough columns."""
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def filter_cross_sectional_candidates(df, feature_cols, date_col="Date",
                                      sample_dates=200, xs_to_total_min=0.05):
    """Identify features that actually vary cross-sectionally per day.

    Market-wide features (VIX_*, sector indices) have the same value for every
    ticker on a date.  Their cross-sectional rank with method="average" is
    pinned to 0.5 (all tied) -- ranking them is degenerate noise.

    A feature is kept (varying) when:
        mean_per_date_std / overall_std > xs_to_total_min

    This ratio is ~0 for market-wide features (per-date std = 0) and 0.5-1.0
    for stock-varying features.  Setting the threshold at 0.05 catches both
    perfectly-flat features (VIX_Close) and derived features with rolling-window
    artifacts (VIX_Z_50d) that have negligible cross-sectional variation but
    pass an absolute-std cutoff.

    Returns (varying, market_wide).  We sample dates rather than scanning the
    whole panel because per-date std across 2.5M rows x 200 cols is slow.
    """
    unique_dates = df[date_col].drop_duplicates()
    if len(unique_dates) > sample_dates:
        idx = np.linspace(0, len(unique_dates) - 1, sample_dates, dtype=int)
        sampled = unique_dates.iloc[idx]
    else:
        sampled = unique_dates
    sub = df[df[date_col].isin(sampled)]
    per_date_std = sub.groupby(date_col)[feature_cols].std()
    mean_per_date_std = per_date_std.mean().fillna(0.0)
    overall_std = sub[feature_cols].std().fillna(0.0).replace(0, 1.0)
    xs_to_total = (mean_per_date_std / overall_std).fillna(0.0)
    varying = [c for c in feature_cols if xs_to_total[c] > xs_to_total_min]
    market_wide = [c for c in feature_cols if c not in varying]
    return varying, market_wide


def add_cross_sectional_ranks(df, feature_cols, date_col="Date", suffix="_xs",
                              batch_size=25):
    """Add per-day percentile rank for each feature column that VARIES across
    tickers within a day.

    Market-wide features (same value for every ticker on a date) are skipped
    here because their cross-sectional rank is degenerate -- 0.5 for everyone.
    The raw column stays in the feature set; only the redundant `_xs` column
    is omitted.

    Done in feature batches so progress is visible -- one giant `groupby.rank`
    on 200 cols x 2.5M rows can take 20+ minutes with no output.
    """
    t0 = _time.time()
    varying, market_wide = filter_cross_sectional_candidates(df, feature_cols,
                                                             date_col=date_col)
    logging.info(f"Cross-section filter: {len(feature_cols)} candidates -> "
                 f"{len(varying)} varying + {len(market_wide)} market-wide "
                 f"(skipped) in {_time.time()-t0:.1f}s")
    if market_wide:
        sample_mw = market_wide[:6] + (["..."] if len(market_wide) > 6 else [])
        logging.info(f"  market-wide skipped: {sample_mw}")

    n_feat = len(varying)
    n_dates = int(df[date_col].nunique())
    n_rows = len(df)
    if n_feat == 0:
        logging.warning("No varying features to rank cross-sectionally.")
        return df
    logging.info(f"Cross-sectional rank: {n_feat} features x {n_dates:,} dates "
                 f"x {n_rows:,} rows (batches of {batch_size}).")
    grouped = df.groupby(date_col, sort=False)
    out_chunks = []
    n_batches = (n_feat + batch_size - 1) // batch_size
    pbar = tqdm(total=n_batches, desc="xs-rank batches")
    for i in range(0, n_feat, batch_size):
        batch = varying[i:i + batch_size]
        t0 = _time.time()
        ranks = grouped[batch].rank(pct=True, method="average", na_option="keep")
        ranks.columns = [c + suffix for c in batch]
        out_chunks.append(ranks.astype("float32"))
        logging.info(f"  xs-rank batch {len(out_chunks)}/{n_batches}: "
                     f"{len(batch)} features in {_time.time()-t0:.1f}s "
                     f"({batch[0]}..{batch[-1]})")
        pbar.update(1)
    pbar.close()
    logging.info("Concatenating rank columns onto base DataFrame...")
    df = pd.concat([df] + out_chunks, axis=1)
    return df


##=============================================================================##
##                          LABELS / GROUPS                                      ##
##=============================================================================##


def make_per_day_topq_labels(returns, dates, top_frac=0.20):
    """Per-day top-quintile binary labels (1 if return is top-q% that day)."""
    s = pd.Series(returns).reset_index(drop=True)
    d = pd.Series(dates).reset_index(drop=True)
    cutoff = s.groupby(d).transform(lambda x: x.quantile(1 - top_frac))
    return (s >= cutoff).astype(int).values


def compute_group_sizes(dates):
    """Per-day group sizes for XGBRanker, in chronological order.

    NOTE: caller must pass *sorted-by-date* dates; group sizes are emitted in the
    order dates first appear.
    """
    s = pd.Series(dates).reset_index(drop=True)
    return s.groupby(s, sort=False).size().values.astype(int)


##=============================================================================##
##                              DATA LOADING                                     ##
##=============================================================================##


def process_files(file_list, input_directory, target_column, date_column,
                  horizon_5d=5):
    """Per-ticker: shift target -1 (next-day), add ret_5d, drop NaN tails."""
    pbar = tqdm(total=len(file_list), desc="Loading tickers")
    out = []
    for fn in file_list:
        try:
            path = os.path.join(input_directory, fn)
            df = pd.read_parquet(path)
            if df.shape[0] <= 60 or target_column not in df.columns or date_column not in df.columns:
                pbar.update(1)
                continue
            df = df.sort_values(date_column).reset_index(drop=True)
            # Per-ticker next-day return (label, not feature)
            df[target_column] = df[target_column].shift(-1)
            # 5-day forward return (label, not feature)
            if "Close" in df.columns:
                df["ret_5d"] = df["Close"].shift(-horizon_5d) / df["Close"] - 1.0
            else:
                df["ret_5d"] = np.nan
            df = df.iloc[:-max(horizon_5d, 1)]  # drop trailing rows with NaN labels
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=[target_column])
            df = df[(df[target_column] <= 5) & (df[target_column] >= -5)]
            if df.shape[0] > 0:
                # Downcast per-ticker so concat doesn't materialise float64
                df = downcast_numeric(df)
                out.append(df)
        except Exception as e:
            logging.warning(f"Skipping {fn}: {e}")
        pbar.update(1)
    pbar.close()
    return out


def finalize_dataset(df, date_column):
    """Final cleanup pre-training."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    df = downcast_numeric(df)
    return df


def prepare_data_splits(config, args):
    """Load all tickers, filter universe, build labels and ranks, split train/calib."""
    input_directory = config["input_directory"]
    train_dir = config["data_output_directory"]
    calib_dir = config["calibration_output_directory"]
    target_column = config["target_column"]
    date_column = config["date_column"]
    file_selection_percentage = config["file_selection_percentage"]
    calibration_percentage = config["calibration_percentage"]
    embargo_days = config["embargo_days"]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    train_path = os.path.join(train_dir, "training_data.parquet")
    calib_path = os.path.join(calib_dir, "calibration_data.parquet")
    alloc_path = os.path.join(train_dir, "file_allocation.json")

    if args.reuse and os.path.exists(train_path) and os.path.exists(calib_path):
        logging.info("Reusing cached training and calibration parquets.")
        return pd.read_parquet(train_path), pd.read_parquet(calib_path)

    if args.clear:
        for p in [train_path, calib_path]:
            if os.path.exists(p):
                os.remove(p)

    all_files = sorted(f for f in os.listdir(input_directory) if f.endswith(".parquet"))
    logging.info(f"Loading {len(all_files):,} ticker files...")
    with _Phase("Load & per-ticker label shift"):
        parts = process_files(all_files, input_directory, target_column, date_column,
                              horizon_5d=config["horizon_5d"])
        if not parts:
            raise ValueError("No valid ticker data after processing.")
        t0 = _time.time()
        logging.info(f"  concat {len(parts):,} ticker frames...")
        combined = pd.concat(parts, ignore_index=True)
        logging.info(f"  concat done in {_time.time()-t0:.1f}s, "
                     f"shape={combined.shape}, "
                     f"mem={combined.memory_usage(deep=False).sum()/1e9:.2f}GB")
        del parts  # free per-ticker frames
        t0 = _time.time()
        logging.info("  downcasting float64 -> float32...")
        combined = downcast_numeric(combined)
        logging.info(f"  downcast done in {_time.time()-t0:.1f}s, "
                     f"mem={combined.memory_usage(deep=False).sum()/1e9:.2f}GB")
        t0 = _time.time()
        logging.info("  to_datetime + sort_values(Date)...")
        combined[date_column] = pd.to_datetime(combined[date_column])
        combined = combined.sort_values(date_column).reset_index(drop=True)
        logging.info(f"  sort done in {_time.time()-t0:.1f}s.")

    with _Phase("Universe filter (FilterRubric Step 1)"):
        combined = apply_quality_filter(combined).reset_index(drop=True)
        if combined.empty:
            raise ValueError("Universe filter dropped everything.")
        logging.info(f"  post-filter shape={combined.shape}, "
                     f"mem={combined.memory_usage(deep=False).sum()/1e9:.2f}GB")

    with _Phase("Cross-sectional rank features"):
        exclude = {date_column, target_column, "ret_5d",
                   "Open", "High", "Low", "Close", "Volume",
                   config["ticker_column"]}
        feature_cols = select_feature_columns(combined, exclude)
        combined = add_cross_sectional_ranks(combined, feature_cols, date_column)
        logging.info(f"  post-rank shape={combined.shape}, "
                     f"mem={combined.memory_usage(deep=False).sum()/1e9:.2f}GB")

    with _Phase("Drop strings + per-day top-quintile labels"):
        t0 = _time.time()
        combined = drop_non_numeric_features(combined, keep_cols={date_column})
        logging.info(f"  drop strings: {_time.time()-t0:.1f}s, "
                     f"shape={combined.shape}")
        t0 = _time.time()
        combined["y_topq"] = make_per_day_topq_labels(
            combined[target_column].values, combined[date_column].values,
            top_frac=config["topq_fraction"])
        logging.info(f"  topq labels: {_time.time()-t0:.1f}s "
                     f"(rate={combined['y_topq'].mean():.3f})")

    with _Phase("Row-percentile train/calib split"):
        n = len(combined)
        train_end_row = max(int(n * file_selection_percentage / 100) - 1, 0)
        train_end_date = pd.Timestamp(combined[date_column].iloc[train_end_row])
        calib_start_date = train_end_date + pd.Timedelta(days=embargo_days)
        calib_pool = combined[combined[date_column] >= calib_start_date]
        if calib_pool.empty:
            raise ValueError("No rows available for calibration after embargo.")
        calib_target_rows = int(n * calibration_percentage / 100)
        calib_end_row = min(calib_target_rows, len(calib_pool)) - 1
        calib_end_date = pd.Timestamp(calib_pool[date_column].iloc[calib_end_row])

        t0 = _time.time()
        train_df = combined[combined[date_column] <= train_end_date].copy()
        calib_df = combined[(combined[date_column] >= calib_start_date) &
                            (combined[date_column] <= calib_end_date)].copy()
        del combined
        logging.info(f"  split done in {_time.time()-t0:.1f}s: "
                     f"train={len(train_df):,} rows ({train_end_date.date()}), "
                     f"calib={len(calib_df):,} rows "
                     f"({calib_start_date.date()} - {calib_end_date.date()})")

        train_df = finalize_dataset(train_df, date_column)
        calib_df = finalize_dataset(calib_df, date_column)
        logging.info(f"  train mem={train_df.memory_usage(deep=False).sum()/1e9:.2f}GB, "
                     f"calib mem={calib_df.memory_usage(deep=False).sum()/1e9:.2f}GB")

    split_info = {
        "train_end": str(train_end_date.date()),
        "calib_start": str(calib_start_date.date()),
        "calib_end": str(calib_end_date.date()),
        "train_rows": int(len(train_df)),
        "calib_rows": int(len(calib_df)),
        "n_features_raw": int(len(feature_cols)),
        "n_features_with_xs": int(2 * len(feature_cols)),
        "topq_base_rate": float(train_df["y_topq"].mean()),
        "up_base_rate": float((train_df[target_column] > 0).mean()),
        "up5d_base_rate": float((train_df["ret_5d"] > 0).mean()),
    }
    with open(alloc_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logging.info(f"Split: train {split_info['train_rows']:,} rows "
                 f"({split_info['train_end']}) | "
                 f"calib {split_info['calib_rows']:,} rows "
                 f"({split_info['calib_start']} - {split_info['calib_end']}).")
    logging.info(f"Base rates: 1d_up={split_info['up_base_rate']:.3f}, "
                 f"5d_up={split_info['up5d_base_rate']:.3f}, "
                 f"topq={split_info['topq_base_rate']:.3f}")

    with _Phase("Write training/calibration parquets"):
        t0 = _time.time()
        train_df.to_parquet(train_path, index=False)
        logging.info(f"  train parquet ({len(train_df):,} rows x "
                     f"{len(train_df.columns)} cols) written in {_time.time()-t0:.1f}s.")
        t0 = _time.time()
        calib_df.to_parquet(calib_path, index=False)
        logging.info(f"  calib parquet ({len(calib_df):,} rows x "
                     f"{len(calib_df.columns)} cols) written in {_time.time()-t0:.1f}s.")
    return train_df, calib_df


##=============================================================================##
##                        WALK-FORWARD CV / OOF                                  ##
##=============================================================================##


def walk_forward_splits(dates, n_folds=4, embargo_days=5, min_train_rows=50_000):
    """Expanding-window time-series CV: train [0, train_end_i], val [val_start_i, val_end_i].

    Returns list of (train_idx, val_idx) tuples. OOF preds available for the
    union of validation slices.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    n = len(dates)
    # Each fold has val window of width 1/(n_folds+1); train window grows.
    width = n // (n_folds + 1)
    splits = []
    for i in range(n_folds):
        val_start_row = (i + 1) * width
        val_end_row = (i + 2) * width
        if val_end_row >= n:
            val_end_row = n - 1
        val_start_date = dates.iloc[val_start_row]
        val_end_date = dates.iloc[val_end_row]
        train_end_date = val_start_date - pd.Timedelta(days=embargo_days)
        train_idx = np.where(dates <= train_end_date)[0]
        val_idx = np.where((dates >= val_start_date) & (dates <= val_end_date))[0]
        if len(train_idx) < min_train_rows or len(val_idx) < 1000:
            continue
        splits.append((train_idx, val_idx))
    return splits


##=============================================================================##
##                              BASE LEARNERS                                    ##
##=============================================================================##


def _silent_fit(model, *fit_args, **fit_kwargs):
    """Suppress XGBoost stdout chatter during fit."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        model.fit(*fit_args, **fit_kwargs)
    return model


def fit_xgb_classifier(X_tr, y_tr, X_val, y_val, params, sample_weight=None):
    base_rate = float(np.mean(y_tr))
    pos_weight = (1 - base_rate) / max(base_rate, 1e-6)
    p = dict(params)
    p.setdefault("scale_pos_weight", min(max(pos_weight, 0.3), 3.0))
    model = XGBClassifier(**p)
    fit_kw = dict(eval_set=[(X_val, y_val)], verbose=False)
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    _silent_fit(model, X_tr, y_tr, **fit_kw)
    return model


def fit_xgb_ranker(X_tr, y_tr, group_tr, X_val, y_val, group_val, params):
    """y must be integer relevance (0 or 1 is fine for pairwise). group is array
    of per-day row counts in chronological order. X must be sorted by date."""
    model = XGBRanker(**params)
    _silent_fit(model, X_tr, y_tr, group=group_tr,
                eval_set=[(X_val, y_val)], eval_group=[group_val], verbose=False)
    return model


def fit_logistic_rank(X_tr_ranks, y_tr, params):
    """Logistic regression on rank-normalised features only.

    Ranks live in [0,1] but can be NaN where the underlying raw feature was NaN.
    Logistic regression rejects NaN, so we impute to 0.5 (rank-space neutral).
    """
    X_filled = np.asarray(X_tr_ranks, dtype=np.float32)
    if np.isnan(X_filled).any():
        X_filled = np.where(np.isnan(X_filled), 0.5, X_filled)
    scaler = StandardScaler().fit(X_filled)
    Xs = scaler.transform(X_filled)
    model = LogisticRegression(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xs, y_tr)
    return model, scaler


def predict_xgb_classifier(model, X):
    return model.predict_proba(X)[:, 1]


def predict_xgb_ranker(model, X):
    return model.predict(X)


def predict_logistic(model, scaler, X):
    X_arr = np.asarray(X, dtype=np.float32)
    if np.isnan(X_arr).any():
        X_arr = np.where(np.isnan(X_arr), 0.5, X_arr)
    return model.predict_proba(scaler.transform(X_arr))[:, 1]


##=============================================================================##
##                                OPTUNA                                         ##
##=============================================================================##


def _per_day_topk_precision(scores, labels, dates, k_frac=0.01):
    """Average per-day precision in the top k_frac of model scores.

    Closer match to "trade <10 per day" than AUC-PR.
    """
    df = pd.DataFrame({"s": scores, "y": labels, "d": pd.Series(dates).values})
    precs = []
    for _, g in df.groupby("d", sort=False):
        if len(g) < 20:
            continue
        n_top = max(int(len(g) * k_frac), 1)
        top = g.nlargest(n_top, "s")
        precs.append(top["y"].mean())
    return float(np.mean(precs)) if precs else 0.0


def tune_xgb_classifier(X, y, dates, defaults, n_trials, n_folds, topk_frac, label_name=""):
    """Optuna search on XGBClassifier with per-day top-K precision objective.

    Returns (best_params_dict, study_info_dict).  study_info contains
    'best_value' and 'best_per_fold' for the end-of-training summary.
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError:
        logging.warning("Optuna not installed; using defaults.")
        return defaults, {"best_value": None, "best_per_fold": None}

    splits = walk_forward_splits(dates, n_folds=n_folds)
    if not splits:
        logging.warning(f"[{label_name}] No valid CV folds; using defaults.")
        return defaults, {"best_value": None, "best_per_fold": None}
    logging.info(f"[{label_name}] tuning: {n_trials} trials x {len(splits)} folds "
                 f"(fold sizes: {[len(tr)+len(va) for tr, va in splits]})")

    def objective(trial):
        params = dict(defaults)
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        })
        scores = []
        for fold_i, (tr, va) in enumerate(splits):
            t0 = _time.time()
            logging.info(f"  [{label_name}] trial {trial.number+1}/{n_trials} "
                         f"fold {fold_i+1}/{len(splits)} fit "
                         f"(train={len(tr):,} val={len(va):,}, "
                         f"depth={params['max_depth']}, "
                         f"n_est={params['n_estimators']}, "
                         f"lr={params['learning_rate']:.4f})...")
            model = fit_xgb_classifier(
                X.iloc[tr], y[tr], X.iloc[va], y[va], params)
            p = predict_xgb_classifier(model, X.iloc[va])
            fold_score = _per_day_topk_precision(p, y[va], dates[va], topk_frac)
            scores.append(fold_score)
            logging.info(f"  [{label_name}] trial {trial.number+1} "
                         f"fold {fold_i+1} done: top-{topk_frac:.0%} "
                         f"prec={fold_score:.4f} ({_time.time()-t0:.1f}s)")
            trial.report(np.mean(scores), step=fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("per_fold", [round(s, 4) for s in scores])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                pruner=MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_optuna_log_callback(label_name, n_trials)])
    best = dict(defaults)
    best.update(study.best_params)
    info = {
        "best_value": float(study.best_value),
        "best_per_fold": study.best_trial.user_attrs.get("per_fold"),
    }
    logging.info(f"[{label_name}] Optuna best top-{topk_frac:.0%} precision: "
                 f"{study.best_value:.4f}")
    logging.info(f"[{label_name}] best params: {study.best_params}")
    return best, info


def tune_xgb_ranker(X, y_topq, dates, defaults, n_trials, n_folds, topk_frac):
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError:
        logging.warning("Optuna not installed; using defaults.")
        return defaults, {"best_value": None, "best_per_fold": None}

    splits = walk_forward_splits(dates, n_folds=n_folds)
    if not splits:
        return defaults, {"best_value": None, "best_per_fold": None}
    logging.info(f"[ranker] tuning: {n_trials} trials x {len(splits)} folds")

    def objective(trial):
        params = dict(defaults)
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        })
        scores = []
        for fold_i, (tr, va) in enumerate(splits):
            t0 = _time.time()
            logging.info(f"  [ranker] trial {trial.number+1}/{n_trials} "
                         f"fold {fold_i+1}/{len(splits)} fit "
                         f"(train={len(tr):,} val={len(va):,}, "
                         f"depth={params['max_depth']}, "
                         f"n_est={params['n_estimators']}, "
                         f"lr={params['learning_rate']:.4f})...")
            X_tr = X.iloc[tr]
            X_va = X.iloc[va]
            d_tr = pd.Series(dates[tr]).reset_index(drop=True)
            d_va = pd.Series(dates[va]).reset_index(drop=True)
            order_tr = np.argsort(d_tr.values, kind="stable")
            order_va = np.argsort(d_va.values, kind="stable")
            g_tr = compute_group_sizes(d_tr.values[order_tr])
            g_va = compute_group_sizes(d_va.values[order_va])
            model = fit_xgb_ranker(
                X_tr.iloc[order_tr], y_topq[tr][order_tr], g_tr,
                X_va.iloc[order_va], y_topq[va][order_va], g_va, params)
            p = predict_xgb_ranker(model, X_va.iloc[order_va])
            fold_score = _per_day_topk_precision(
                p, y_topq[va][order_va], d_va.values[order_va], topk_frac)
            scores.append(fold_score)
            logging.info(f"  [ranker] trial {trial.number+1} "
                         f"fold {fold_i+1} done: top-{topk_frac:.0%} "
                         f"prec={fold_score:.4f} ({_time.time()-t0:.1f}s)")
            trial.report(np.mean(scores), step=fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("per_fold", [round(s, 4) for s in scores])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                pruner=MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_optuna_log_callback("ranker", n_trials)])
    best = dict(defaults)
    best.update(study.best_params)
    info = {
        "best_value": float(study.best_value),
        "best_per_fold": study.best_trial.user_attrs.get("per_fold"),
    }
    logging.info(f"[ranker] Optuna best top-{topk_frac:.0%} precision: "
                 f"{study.best_value:.4f}")
    return best, info


##=============================================================================##
##                                   OOF                                         ##
##=============================================================================##


def generate_oof_predictions(train_df, feature_cols, rank_cols,
                             y_1d, y_5d, y_topq, dates,
                             params_clf1d, params_rank, params_clf5d, params_logit,
                             n_folds, weights=None, use_clf5d=True):
    """Walk-forward OOF predictions for all 4 base learners.

    Returns:
        oof: DataFrame indexed by row position in train_df, with columns
             ['p_clf1d', 'p_rank', 'p_clf5d', 'p_logit', 'oof_valid'].
             Rows in folds we didn't validate on get NaN + oof_valid=False.
    """
    n = len(train_df)
    oof = pd.DataFrame({
        "p_clf1d": np.full(n, np.nan, dtype=np.float32),
        "p_rank": np.full(n, np.nan, dtype=np.float32),
        "p_clf5d": np.full(n, np.nan, dtype=np.float32),
        "p_logit": np.full(n, np.nan, dtype=np.float32),
        "oof_valid": np.zeros(n, dtype=bool),
    })

    splits = walk_forward_splits(dates, n_folds=n_folds,
                                 min_train_rows=config["min_train_rows_per_fold"])
    if not splits:
        raise RuntimeError("walk_forward_splits returned no folds; reduce min_train_rows.")

    X_full = train_df[feature_cols]
    X_full_ranks = train_df[rank_cols]

    for fold_i, (tr, va) in enumerate(splits):
        fold_t0 = _time.time()
        logging.info(f"  OOF fold {fold_i+1}/{len(splits)}: "
                     f"train={len(tr):,} val={len(va):,}")

        # ----- XGBClassifier 1d -----
        t0 = _time.time()
        m1 = fit_xgb_classifier(
            X_full.iloc[tr], y_1d[tr], X_full.iloc[va], y_1d[va],
            params_clf1d,
            sample_weight=(weights[tr] if weights is not None else None))
        oof.loc[va, "p_clf1d"] = predict_xgb_classifier(m1, X_full.iloc[va]).astype(np.float32)
        logging.info(f"    clf1d fit+predict: {_time.time()-t0:.1f}s")

        # ----- XGBClassifier 5d (optional) -----
        if use_clf5d:
            t0 = _time.time()
            m5 = fit_xgb_classifier(
                X_full.iloc[tr], y_5d[tr], X_full.iloc[va], y_5d[va],
                params_clf5d,
                sample_weight=(weights[tr] if weights is not None else None))
            oof.loc[va, "p_clf5d"] = predict_xgb_classifier(m5, X_full.iloc[va]).astype(np.float32)
            logging.info(f"    clf5d fit+predict: {_time.time()-t0:.1f}s")
        else:
            logging.info("    clf5d skipped (--no_clf5d).")

        # ----- XGBRanker (must be date-sorted within fold) -----
        t0 = _time.time()
        d_tr = pd.Series(dates[tr]).reset_index(drop=True)
        d_va = pd.Series(dates[va]).reset_index(drop=True)
        ord_tr = np.argsort(d_tr.values, kind="stable")
        ord_va = np.argsort(d_va.values, kind="stable")
        g_tr = compute_group_sizes(d_tr.values[ord_tr])
        g_va = compute_group_sizes(d_va.values[ord_va])
        mR = fit_xgb_ranker(
            X_full.iloc[tr].iloc[ord_tr], y_topq[tr][ord_tr], g_tr,
            X_full.iloc[va].iloc[ord_va], y_topq[va][ord_va], g_va,
            params_rank)
        r_scores = predict_xgb_ranker(mR, X_full.iloc[va].iloc[ord_va])
        r_lo, r_hi = float(np.min(r_scores)), float(np.max(r_scores))
        rng = max(r_hi - r_lo, 1e-9)
        r_scaled = ((r_scores - r_lo) / rng).astype(np.float32)
        va_idx_sorted = np.array(va)[ord_va]
        oof.loc[va_idx_sorted, "p_rank"] = r_scaled
        logging.info(f"    ranker fit+predict: {_time.time()-t0:.1f}s")

        # ----- Logistic on ranks -----
        t0 = _time.time()
        mL, scL = fit_logistic_rank(X_full_ranks.iloc[tr], y_1d[tr],
                                    params_logit)
        oof.loc[va, "p_logit"] = predict_logistic(mL, scL, X_full_ranks.iloc[va]).astype(np.float32)
        logging.info(f"    logit fit+predict: {_time.time()-t0:.1f}s")

        oof.loc[va, "oof_valid"] = True
        logging.info(f"  fold {fold_i+1} total: {_time.time()-fold_t0:.1f}s")

    valid = oof["oof_valid"].sum()
    logging.info(f"OOF: {valid:,} valid rows ({100*valid/n:.1f}% of training).")
    return oof


##=============================================================================##
##                              META-STACKER                                     ##
##=============================================================================##


def fit_meta_stacker(oof, y, inputs, C=1.0):
    """Logistic regression stacker on OOF base-learner predictions.

    `inputs` is the list of OOF column names to use (e.g. without "p_clf5d"
    when --no_clf5d is set).
    """
    mask = oof["oof_valid"].values
    X = oof.loc[mask, inputs].values
    yv = y[mask]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    stk = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
    stk.fit(Xs, yv)
    coef = dict(zip(inputs, stk.coef_.ravel().tolist()))
    logging.info(f"Stacker fit on {mask.sum():,} OOF rows; inputs={inputs}; "
                 f"coefs={ {k: f'{v:+.3f}' for k,v in coef.items()} }")
    return stk, sc


def predict_meta(stacker, scaler, base_pred_df, inputs):
    X = base_pred_df[inputs].values
    return stacker.predict_proba(scaler.transform(X))[:, 1]


##=============================================================================##
##                             BETA CALIBRATION                                  ##
##=============================================================================##


class BetaCalibrator:
    """Three-parameter Beta calibration (Kull et al. 2017).

    Calibrated p = sigmoid(a*logit(p) + b*log((1-p)/p) + c).  Cannot collapse to
    a flat output the way isotonic can when the calibration set has degenerate
    structure.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 0.0
        self._fitted = False

    @staticmethod
    def _safe(p):
        return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)

    def _transform(self, p, a, b, c):
        p = self._safe(p)
        z = a * np.log(p / (1 - p)) + b * np.log((1 - p) / p) + c
        return expit(z)

    def fit(self, p, y):
        p = self._safe(p)
        y = np.asarray(y).astype(np.float64)

        def nll(theta):
            a, b, c = theta
            q = self._transform(p, a, b, c)
            q = np.clip(q, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize(nll, x0=[1.0, 0.0, 0.0], method="L-BFGS-B")
        if res.success:
            self.a, self.b, self.c = res.x.tolist()
        self._fitted = True
        return self

    def predict(self, p):
        if not self._fitted:
            return np.asarray(p, dtype=np.float64)
        return self._transform(p, self.a, self.b, self.c)


def fit_calibrator(raw_probs, y, min_range=0.05):
    """Fit Beta calibrator; fall back to identity if calibrated range is too narrow."""
    cal = BetaCalibrator().fit(raw_probs, y)
    q = cal.predict(raw_probs)
    rng = float(np.max(q) - np.min(q))
    if rng < min_range:
        logging.warning(f"Calibrator output range {rng:.3f} < {min_range}; "
                        "falling back to raw probabilities.")
        cal._fitted = False
    else:
        logging.info(f"Beta calibrator OK (range {rng:.3f}).")
    return cal


##=============================================================================##
##                           CONFORMAL THRESHOLD                                 ##
##=============================================================================##


def conformal_threshold(scores, labels, target_precision=0.75,
                        min_coverage=0.001, max_coverage=0.05):
    """Pick the lowest score cutoff where precision >= target on calibration set,
    constrained to coverage in [min_coverage, max_coverage].

    Returns (threshold, precision_at_threshold, coverage_at_threshold).
    If no threshold meets target, returns the precision-maximising threshold in
    the coverage band.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(s)
    if n == 0:
        return 0.5, 0.0, 0.0

    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    ranks = np.arange(1, n + 1)
    precs = cum_pos / ranks
    covs = ranks / n

    band = (covs >= min_coverage) & (covs <= max_coverage)
    if not band.any():
        # Coverage band collapsed; widen to top 5%
        band = covs <= max(max_coverage, 0.05)

    target_hits = band & (precs >= target_precision)
    if target_hits.any():
        # Pick the LARGEST coverage that still meets target (lowest threshold, max recall)
        idx_candidates = np.where(target_hits)[0]
        chosen_idx = idx_candidates.max()
    else:
        # Fall back: pick the highest-precision point in the coverage band
        band_idx = np.where(band)[0]
        chosen_idx = band_idx[np.argmax(precs[band_idx])]
        logging.warning(f"Target precision {target_precision} not reachable in "
                        f"[{min_coverage:.4f}, {max_coverage:.4f}]; using "
                        f"precision-max point.")
    tau = float(s_sorted[chosen_idx])
    p = float(precs[chosen_idx])
    c = float(covs[chosen_idx])
    logging.info(f"Conformal threshold: tau={tau:.4f}, precision={p:.3f}, "
                 f"coverage={c:.4f} ({int(c*n):,} fires of {n:,}).")
    return tau, p, c


##=============================================================================##
##                             CONSENSUS GATE                                    ##
##=============================================================================##


def per_day_top_decile_mask(scores, dates, decile_frac=0.10):
    """Per-day, True for rows in the top decile by score."""
    df = pd.DataFrame({"s": scores, "d": pd.Series(dates).values})
    cutoff = df.groupby("d", sort=False)["s"].transform(lambda x: x.quantile(1 - decile_frac))
    return (df["s"] >= cutoff).values


def consensus_mask(base_scores_df, dates, inputs, decile_frac=0.10):
    """Row fires only if ALL listed base learners place it in that day's top decile."""
    m = np.ones(len(dates), dtype=bool)
    for col in inputs:
        m &= per_day_top_decile_mask(base_scores_df[col].values, dates, decile_frac)
    return m


##=============================================================================##
##                              PIPELINE                                         ##
##=============================================================================##


def train_pipeline(train_df, calib_df, config, args):
    target = config["target_column"]
    date_col = config["date_column"]
    horizon_5d = config["horizon_5d"]

    with _Phase("Build feature lists + labels + weights"):
        # ------ features / labels ------
        label_cols = {target, "ret_5d", "y_topq"}
        pass_cols = {"Open", "High", "Low", "Close", "Volume", date_col}
        feature_cols = [c for c in train_df.columns
                        if c not in label_cols and c not in pass_cols
                        and pd.api.types.is_numeric_dtype(train_df[c])
                        and not c.endswith("_xs")]
        rank_cols = [c for c in train_df.columns if c.endswith("_xs")]
        all_feature_cols = feature_cols + rank_cols
        logging.info(f"  feature counts: raw={len(feature_cols)}, "
                     f"rank={len(rank_cols)}, total={len(all_feature_cols)}.")

        t0 = _time.time()
        n0_tr, n0_ca = len(train_df), len(calib_df)
        train_df = train_df.dropna(subset=[target, "ret_5d", "y_topq"]).reset_index(drop=True)
        calib_df = calib_df.dropna(subset=[target]).reset_index(drop=True)
        if "ret_5d" not in calib_df.columns:
            calib_df["ret_5d"] = np.nan
        logging.info(f"  dropna: train {n0_tr:,}->{len(train_df):,}, "
                     f"calib {n0_ca:,}->{len(calib_df):,} "
                     f"({_time.time()-t0:.1f}s)")

        t0 = _time.time()
        y_1d = (train_df[target] > 0).astype(int).values
        y_5d = (train_df["ret_5d"] > 0).astype(int).values
        y_topq = train_df["y_topq"].astype(int).values
        dates = train_df[date_col].values
        weights = recency_weights(train_df[date_col], config["recency_half_life_days"])
        logging.info(f"  labels+weights: {_time.time()-t0:.1f}s | "
                     f"y_1d rate={y_1d.mean():.3f}, "
                     f"y_5d rate={y_5d.mean():.3f}, "
                     f"y_topq rate={y_topq.mean():.3f}, "
                     f"weight range=[{weights.min():.3f}, {weights.max():.3f}]")

    # ------ Optuna tuning (per learner) ------
    p_clf1d = dict(config["xgb_clf_defaults"])
    p_rank = dict(config["xgb_rank_defaults"])
    p_clf5d = dict(config["xgb_clf5d_defaults"])
    p_logit = dict(config["logistic_defaults"])
    diagnostics = {"optuna_best": {}, "optuna_per_fold": {}, "next_steps": []}

    use_clf5d = not args.no_clf5d
    if not use_clf5d:
        logging.info("--no_clf5d: skipping clf5d everywhere (tuning, OOF, "
                     "stacker, refit, consensus, inference).")

    if args.tune:
        with _Phase("Optuna: XGBClassifier (1d up)"):
            p_clf1d, info_clf1d = tune_xgb_classifier(
                train_df[all_feature_cols], y_1d, dates,
                config["xgb_clf_defaults"], args.tune_trials,
                config["n_tune_folds"], config["tune_topk_frac"], "clf1d")
            diagnostics["optuna_best"]["clf1d"] = info_clf1d["best_value"]
            diagnostics["optuna_per_fold"]["clf1d"] = info_clf1d["best_per_fold"]
        if use_clf5d:
            with _Phase("Optuna: XGBClassifier (5d up)"):
                p_clf5d, info_clf5d = tune_xgb_classifier(
                    train_df[all_feature_cols], y_5d, dates,
                    config["xgb_clf5d_defaults"], args.tune_trials,
                    config["n_tune_folds"], config["tune_topk_frac"], "clf5d")
                diagnostics["optuna_best"]["clf5d"] = info_clf5d["best_value"]
                diagnostics["optuna_per_fold"]["clf5d"] = info_clf5d["best_per_fold"]
        with _Phase("Optuna: XGBRanker (top-q per day)"):
            p_rank, info_rank = tune_xgb_ranker(
                train_df[all_feature_cols], y_topq, dates,
                config["xgb_rank_defaults"], args.tune_trials,
                config["n_tune_folds"], config["tune_topk_frac"])
            diagnostics["optuna_best"]["rank"] = info_rank["best_value"]
            diagnostics["optuna_per_fold"]["rank"] = info_rank["best_per_fold"]
        # Skip logistic tuning -- defaults are robust and Optuna budget is precious

    # Which base learners feed the stacker / consensus / inference
    stacker_inputs = ["p_clf1d", "p_rank", "p_logit"]
    if use_clf5d:
        stacker_inputs.insert(2, "p_clf5d")
    base_eval_inputs = ["p_clf1d", "p_rank", "p_clf5d", "p_logit"] if use_clf5d else stacker_inputs

    # ------ Walk-forward OOF for stacker ------
    with _Phase("Walk-forward OOF predictions"):
        oof = generate_oof_predictions(
            train_df, all_feature_cols, rank_cols,
            y_1d, y_5d, y_topq, dates,
            p_clf1d, p_rank, p_clf5d, p_logit,
            n_folds=config["n_oof_folds"], weights=weights,
            use_clf5d=use_clf5d)

        # OOF metrics per base learner (genuine holdout)
        from sklearn.metrics import roc_auc_score, average_precision_score
        oof_valid_mask = oof["oof_valid"].values
        y_oof = y_1d[oof_valid_mask]
        d_oof = dates[oof_valid_mask]
        diagnostics["oof_metrics"] = {}
        for name in base_eval_inputs:
            scores = oof.loc[oof_valid_mask, name].values
            try:
                auc = float(roc_auc_score(y_oof, scores))
                aucpr = float(average_precision_score(y_oof, scores))
            except Exception:
                auc, aucpr = float("nan"), float("nan")
            top1 = _per_day_topk_precision(scores, y_oof, d_oof, k_frac=0.01)
            diagnostics["oof_metrics"][name] = {
                "auc": auc, "aucpr": aucpr, "top1_prec": float(top1)}
            logging.info(f"  OOF {name}: AUC={auc:.4f} AUC-PR={aucpr:.4f} "
                         f"top-1%-prec={top1:.4f}")

    # ------ Meta-stacker ------
    with _Phase("Meta-stacker fit"):
        stacker, stack_scaler = fit_meta_stacker(oof, y_1d,
                                                 inputs=stacker_inputs,
                                                 C=config["stacker_C"])
        coefs = stacker.coef_.ravel() / stack_scaler.scale_
        diagnostics["stacker_coefs"] = dict(zip(
            stacker_inputs, [float(c) for c in coefs]))

        # Stacked meta probability on OOF rows, for an honest pre-calib metric
        meta_oof = predict_meta(stacker, stack_scaler,
                                oof.loc[oof_valid_mask, stacker_inputs],
                                inputs=stacker_inputs)
        try:
            auc_m = float(roc_auc_score(y_oof, meta_oof))
            aucpr_m = float(average_precision_score(y_oof, meta_oof))
        except Exception:
            auc_m, aucpr_m = float("nan"), float("nan")
        top1_m = _per_day_topk_precision(meta_oof, y_oof, d_oof, k_frac=0.01)
        diagnostics["oof_metrics_meta"] = {
            "auc": auc_m, "aucpr": aucpr_m, "top1_prec": float(top1_m)}
        logging.info(f"  OOF META: AUC={auc_m:.4f} AUC-PR={aucpr_m:.4f} "
                     f"top-1%-prec={top1_m:.4f}")

    # ------ Refit base learners on full training data ------
    with _Phase("Refit base learners on full training data"):
        n_tr = len(train_df)
        val_start = int(n_tr * 0.80)
        X_full = train_df[all_feature_cols]
        X_ranks_full = train_df[rank_cols]
        t0 = _time.time()
        final_clf1d = fit_xgb_classifier(
            X_full.iloc[:val_start], y_1d[:val_start],
            X_full.iloc[val_start:], y_1d[val_start:], p_clf1d,
            sample_weight=weights[:val_start])
        logging.info(f"  final clf1d: {_time.time()-t0:.1f}s")
        if use_clf5d:
            t0 = _time.time()
            final_clf5d = fit_xgb_classifier(
                X_full.iloc[:val_start], y_5d[:val_start],
                X_full.iloc[val_start:], y_5d[val_start:], p_clf5d,
                sample_weight=weights[:val_start])
            logging.info(f"  final clf5d: {_time.time()-t0:.1f}s")
        else:
            final_clf5d = None
            logging.info("  final clf5d skipped (--no_clf5d).")
        t0 = _time.time()
        d_tr = pd.Series(dates[:val_start]).reset_index(drop=True)
        d_va = pd.Series(dates[val_start:]).reset_index(drop=True)
        ord_tr = np.argsort(d_tr.values, kind="stable")
        ord_va = np.argsort(d_va.values, kind="stable")
        g_tr = compute_group_sizes(d_tr.values[ord_tr])
        g_va = compute_group_sizes(d_va.values[ord_va])
        final_rank = fit_xgb_ranker(
            X_full.iloc[:val_start].iloc[ord_tr], y_topq[:val_start][ord_tr], g_tr,
            X_full.iloc[val_start:].iloc[ord_va], y_topq[val_start:][ord_va], g_va,
            p_rank)
        logging.info(f"  final ranker: {_time.time()-t0:.1f}s")
        t0 = _time.time()
        final_logit, logit_scaler = fit_logistic_rank(X_ranks_full, y_1d, p_logit)
        logging.info(f"  final logit: {_time.time()-t0:.1f}s")

        rk_val_scores = predict_xgb_ranker(final_rank, X_full.iloc[val_start:].iloc[ord_va])
        rk_lo = float(np.min(rk_val_scores))
        rk_hi = float(np.max(rk_val_scores))

    # ------ Score calibration slice ------
    with _Phase("Score calibration slice through meta-stacker"):
        X_cal = calib_df[all_feature_cols]
        X_cal_ranks = calib_df[rank_cols]
        d_cal = calib_df[date_col].values
        y_cal = (calib_df[target] > 0).astype(int).values

        p_clf1d_cal = predict_xgb_classifier(final_clf1d, X_cal)
        p_clf5d_cal = (predict_xgb_classifier(final_clf5d, X_cal)
                       if final_clf5d is not None else np.full(len(X_cal), np.nan))
        p_logit_cal = predict_logistic(final_logit, logit_scaler, X_cal_ranks)
        d_cal_s = pd.Series(d_cal).reset_index(drop=True)
        ord_cal = np.argsort(d_cal_s.values, kind="stable")
        rk_scores_cal_sorted = predict_xgb_ranker(final_rank, X_cal.iloc[ord_cal])
        rk_scores_cal = np.empty_like(rk_scores_cal_sorted)
        rk_scores_cal[ord_cal] = rk_scores_cal_sorted
        rng = max(rk_hi - rk_lo, 1e-9)
        p_rank_cal = np.clip((rk_scores_cal - rk_lo) / rng, 0, 1)

        base_cal = pd.DataFrame({
            "p_clf1d": p_clf1d_cal,
            "p_rank":  p_rank_cal,
            "p_clf5d": p_clf5d_cal,
            "p_logit": p_logit_cal,
        })
        meta_cal = predict_meta(stacker, stack_scaler, base_cal,
                                inputs=stacker_inputs)

    # ------ Calibrator + conformal threshold ------
    with _Phase("Calibrator fit + conformal threshold"):
        if config["apply_calibration"]:
            cal = fit_calibrator(meta_cal, y_cal)
            diagnostics["calibrator_status"] = (
                "Beta (fitted)" if cal._fitted else "raw passthrough (Beta range too narrow)")
        else:
            cal = BetaCalibrator()  # identity (not fitted)
            diagnostics["calibrator_status"] = "disabled (--nocalib)"
        p_calibrated = cal.predict(meta_cal)

        consensus_decile_frac = 0.10
        cons_cal = consensus_mask(base_cal, d_cal, inputs=stacker_inputs,
                                  decile_frac=consensus_decile_frac)
        if cons_cal.sum() < 100:
            logging.warning(f"Only {cons_cal.sum()} consensus fires on calib slice; "
                            "relaxing decile gate to 20% for threshold calibration.")
            consensus_decile_frac = 0.20
            cons_cal = consensus_mask(base_cal, d_cal, inputs=stacker_inputs,
                                      decile_frac=consensus_decile_frac)
            diagnostics["next_steps"].append(
                "Consensus AND-gate too strict at 10% per-learner -- relaxed to "
                "20%. If precision is poor downstream, consider reducing to 3 "
                "base learners or dropping the consensus filter entirely.")

        p_gated = np.where(cons_cal, p_calibrated, -np.inf)
        tau, prec_at, cov_at = conformal_threshold(
            p_gated, y_cal, target_precision=args.target_precision,
            min_coverage=0.001, max_coverage=args.max_coverage)

        # Per-day fire stats on calibration slice (what production will actually emit)
        fire_mask = (p_calibrated >= tau) & cons_cal
        df_fires = pd.DataFrame({"d": pd.Series(d_cal).values,
                                 "fire": fire_mask.astype(int)})
        per_day = df_fires.groupby("d", sort=False)["fire"].sum()
        diagnostics["fires_per_day"] = {
            "mean": float(per_day.mean()),
            "median": float(per_day.median()),
            "max": float(per_day.max()),
            "days_with_fire": int((per_day > 0).sum()),
            "total_days": int(len(per_day)),
        }
        if prec_at < args.target_precision:
            diagnostics["next_steps"].append(
                f"Target precision {args.target_precision} not reachable on calib slice. "
                "Either lower --target_precision, raise --max_coverage, or train on "
                "more data (--runpercent).")
        if diagnostics["fires_per_day"]["mean"] < 0.5:
            diagnostics["next_steps"].append(
                "Average fewer than 0.5 fires/day on calib slice -- strategy may "
                "be too restrictive. Loosen tau (lower --target_precision) or relax consensus.")

    # ------ Diagnostic plot ------
    try:
        os.makedirs(os.path.dirname(config["calibration_plot_output"]), exist_ok=True)
        plot_calibration_diagnostic(meta_cal, p_calibrated, y_cal,
                                    config["calibration_plot_output"])
    except Exception as e:
        logging.warning(f"Calibration plot failed: {e}")

    # ------ Feature importance from clf1d ------
    try:
        os.makedirs(os.path.dirname(config["feature_importance_output"]), exist_ok=True)
        imp = pd.DataFrame({
            "feature": all_feature_cols,
            "importance_clf1d": final_clf1d.feature_importances_,
        }).sort_values("importance_clf1d", ascending=False)
        imp.to_parquet(config["feature_importance_output"], index=False)
        logging.info(f"Top 10 features (clf1d):\n{imp.head(10).to_string(index=False)}")
    except Exception as e:
        logging.warning(f"Could not save feature importance: {e}")

    pipeline = {
        "version": "2026-05-16-clean-sheet",
        "feature_cols_raw": feature_cols,
        "feature_cols_rank": rank_cols,
        "all_feature_cols": all_feature_cols,
        "params": {"clf1d": p_clf1d, "rank": p_rank, "clf5d": p_clf5d,
                   "logit": p_logit},
        "model_clf1d": final_clf1d,
        "model_clf5d": final_clf5d,
        "model_rank": final_rank,
        "ranker_score_range": (rk_lo, rk_hi),
        "model_logit": final_logit,
        "logit_scaler": logit_scaler,
        "stacker": stacker,
        "stacker_scaler": stack_scaler,
        "calibrator": cal,
        "threshold": tau,
        "threshold_precision_calib": prec_at,
        "threshold_coverage_calib": cov_at,
        "target_column": target,
        "date_column": date_col,
        "horizon_5d": horizon_5d,
        "consensus_decile_frac": consensus_decile_frac,
        "stacker_inputs": stacker_inputs,
        "use_clf5d": use_clf5d,
        "diagnostics": diagnostics,
    }
    return pipeline


def save_pipeline(pipeline, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(pipeline, path)
    logging.info(f"Pipeline saved to {path}.")


def load_pipeline(path):
    return load(path)


def plot_calibration_diagnostic(raw, calibrated, y, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p, title in zip(axes, [raw, calibrated],
                            ["Raw meta-stacker", "Beta calibrated"]):
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        idx = np.clip(idx, 0, 9)
        bin_centers, observed = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 30:
                bin_centers.append(p[m].mean())
                observed.append(y[m].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.plot(bin_centers, observed, "o-", label=title)
        ax.set_xlabel("predicted prob")
        ax.set_ylabel("observed positive rate")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


##=============================================================================##
##                              INFERENCE                                        ##
##=============================================================================##


def predict_and_save(input_directory, pipeline_path, output_directory):
    """Score every ticker file and write per-ticker parquets compatible with the
    backtester contract.
    """
    pipeline = load_pipeline(pipeline_path)
    date_col = pipeline["date_column"]
    ticker_col = config["ticker_column"]
    feature_cols_raw = pipeline["feature_cols_raw"]
    feature_cols_rank = pipeline["feature_cols_rank"]
    all_feature_cols = pipeline["all_feature_cols"]
    rk_lo, rk_hi = pipeline["ranker_score_range"]
    tau = pipeline["threshold"]
    decile_frac = pipeline["consensus_decile_frac"]
    stacker_inputs = pipeline.get("stacker_inputs",
                                  ["p_clf1d", "p_rank", "p_clf5d", "p_logit"])
    use_clf5d = pipeline.get("use_clf5d", True)
    logging.info(f"Inference: stacker inputs = {stacker_inputs} "
                 f"(use_clf5d={use_clf5d})")

    os.makedirs(output_directory, exist_ok=True)

    files = sorted(f for f in os.listdir(input_directory) if f.endswith(".parquet"))
    logging.info(f"Inference: loading {len(files):,} tickers...")

    # ---- Load all tickers, keep Ticker + Date + features + passthrough ----
    parts = []
    pbar = tqdm(total=len(files), desc="Loading tickers")
    for fn in files:
        try:
            path = os.path.join(input_directory, fn)
            df = pd.read_parquet(path)
            if date_col not in df.columns or df.empty:
                pbar.update(1); continue
            if ticker_col not in df.columns:
                df[ticker_col] = os.path.splitext(fn)[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            parts.append(df)
        except Exception as e:
            logging.warning(f"Skipping {fn}: {e}")
        pbar.update(1)
    pbar.close()
    if not parts:
        logging.error("No inference data loaded.")
        return

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values([date_col]).reset_index(drop=True)
    logging.info(f"Inference combined shape: {combined.shape}")

    # ---- Universe filter (mark, don't drop -- we still want output per row) ----
    quality_mask = pd.Series(True, index=combined.index)
    if "Close" in combined.columns:
        quality_mask &= combined["Close"] >= 5.0
    if "dollar_volume_ma_10" in combined.columns:
        quality_mask &= combined["dollar_volume_ma_10"] >= 5_000_000
    if "atr_percentage" in combined.columns:
        quality_mask &= combined["atr_percentage"] <= 0.05
    if "RSI" in combined.columns:
        quality_mask &= ~((combined["RSI"] >= 30) & (combined["RSI"] < 40))
    n_pass = int(quality_mask.sum())
    logging.info(f"Universe filter at inference: {n_pass:,}/{len(combined):,} "
                 f"rows pass ({100*n_pass/len(combined):.1f}%).")

    # ---- Cross-sectional ranks computed ONLY on passing universe per day ----
    filtered = combined.loc[quality_mask].copy()
    missing_raw = [c for c in feature_cols_raw if c not in filtered.columns]
    if missing_raw:
        logging.warning(f"{len(missing_raw)} training features missing at "
                        f"inference; will fill with 0. Example: {missing_raw[:3]}")
        for c in missing_raw:
            filtered[c] = 0.0
            combined[c] = 0.0

    logging.info("Computing cross-sectional ranks on filtered universe...")
    ranks = (filtered.groupby(date_col)[feature_cols_raw]
                     .rank(pct=True, method="average", na_option="keep"))
    ranks.columns = [c + "_xs" for c in feature_cols_raw]
    filtered = pd.concat([filtered, ranks.astype("float32")], axis=1)

    # For non-passing rows: fill rank features with neutral 0.5 so the model can
    # still emit a probability (which the row's UpPrediction=-1 will gate out).
    for c in feature_cols_rank:
        if c not in combined.columns:
            combined[c] = 0.5
    combined.loc[quality_mask, feature_cols_rank] = filtered[feature_cols_rank].values

    # ---- Base learner predictions ----
    logging.info("Running base learner predictions...")
    X = combined[all_feature_cols].fillna(0.0)
    X_ranks = combined[feature_cols_rank].fillna(0.5)

    p_clf1d = predict_xgb_classifier(pipeline["model_clf1d"], X)
    if use_clf5d and pipeline.get("model_clf5d") is not None:
        p_clf5d = predict_xgb_classifier(pipeline["model_clf5d"], X)
    else:
        p_clf5d = np.full(len(X), np.nan)
    p_logit = predict_logistic(pipeline["model_logit"], pipeline["logit_scaler"], X_ranks)

    # Ranker: sort by date for prediction, then unscramble
    order = np.argsort(combined[date_col].values, kind="stable")
    rk_sorted = predict_xgb_ranker(pipeline["model_rank"], X.iloc[order])
    rk = np.empty_like(rk_sorted)
    rk[order] = rk_sorted
    rng = max(rk_hi - rk_lo, 1e-9)
    p_rank = np.clip((rk - rk_lo) / rng, 0, 1)

    base_pred_df = pd.DataFrame({
        "p_clf1d": p_clf1d, "p_rank": p_rank,
        "p_clf5d": p_clf5d, "p_logit": p_logit,
    })

    # ---- Meta-stacker + calibration ----
    meta_p = predict_meta(pipeline["stacker"], pipeline["stacker_scaler"],
                          base_pred_df, inputs=stacker_inputs)
    up_prob = pipeline["calibrator"].predict(meta_p)

    # ---- Consensus AND-gate computed within the FILTERED universe per day ----
    cons = np.zeros(len(combined), dtype=bool)
    if quality_mask.any():
        filt_idx = np.where(quality_mask.values)[0]
        d_filt = combined[date_col].values[filt_idx]
        cons_filt = np.ones(len(filt_idx), dtype=bool)
        for col in stacker_inputs:
            sc = base_pred_df[col].values[filt_idx]
            cons_filt &= per_day_top_decile_mask(sc, d_filt, decile_frac)
        cons[filt_idx] = cons_filt

    # ---- Fire decision ----
    fire = (up_prob >= tau) & cons & quality_mask.values
    up_pred = np.where(quality_mask.values, fire.astype(int), -1)
    logging.info(f"Fires: {int(fire.sum()):,} of {len(combined):,} rows "
                 f"({100*fire.sum()/max(len(combined),1):.3f}%). "
                 f"Universe-passing: {int(fire.sum()):,}/{n_pass:,} "
                 f"({100*fire.sum()/max(n_pass,1):.3f}%).")

    # ---- Assemble output ----
    combined["UpProbability"] = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
    combined["DownProbability"] = (1.0 - combined["UpProbability"]).astype(np.float32)
    combined["PositiveThreshold"] = np.float32(tau)
    combined["NegativeThreshold"] = np.float32(np.nan)
    combined["UpPrediction"] = up_pred.astype(np.int8)

    out_cols = [date_col, "Open", "High", "Low", "Close", "Volume",
                "UpProbability", "DownProbability", "PositiveThreshold",
                "NegativeThreshold", "UpPrediction"]
    if "VIX_Close" in combined.columns:
        out_cols.append("VIX_Close")
    for opt in ["Distance to Resistance (%)", "Distance to Support (%)", "volatility"]:
        if opt in combined.columns:
            out_cols.append(opt)
    out_cols = [c for c in out_cols if c in combined.columns]

    # ---- Write per-ticker parquet ----
    pbar = tqdm(total=combined[ticker_col].nunique(), desc="Writing predictions")
    for tk, grp in combined.groupby(ticker_col, sort=False):
        out_path = os.path.join(output_directory, f"{tk}.parquet")
        try:
            grp[out_cols].to_parquet(out_path, index=False)
        except Exception as e:
            logging.warning(f"Failed to write {tk}: {e}")
        pbar.update(1)
    pbar.close()
    logging.info(f"Wrote predictions to {output_directory}.")


##=============================================================================##
##                                  MAIN                                         ##
##=============================================================================##


def main():
    os.makedirs(config["model_output_directory"], exist_ok=True)
    os.makedirs(config["data_output_directory"], exist_ok=True)
    os.makedirs(config["calibration_output_directory"], exist_ok=True)
    os.makedirs(config["prediction_output_directory"], exist_ok=True)

    # --fast: clamp trials and surface what changed
    if args.fast and args.tune_trials > 10:
        logging.info(f"--fast: capping tune_trials {args.tune_trials} -> 10, "
                     f"n_tune_folds=2, n_oof_folds=2.")
        args.tune_trials = 10

    if args.clear:
        for p in [config["pipeline_path"]]:
            if os.path.exists(p):
                os.remove(p)
                logging.info(f"Cleared {p}.")

    if args.predict:
        if not os.path.exists(config["pipeline_path"]):
            raise FileNotFoundError(
                f"No pipeline at {config['pipeline_path']}. Train first "
                "(run without --predict).")
        predict_and_save(
            input_directory=config["input_directory"],
            pipeline_path=config["pipeline_path"],
            output_directory=config["prediction_output_directory"])
        return

    # Train mode
    run_t0 = _time.time()
    train_df, calib_df = prepare_data_splits(config, args)
    logging.info("Data preparation complete; training pipeline.")
    pipeline = train_pipeline(train_df, calib_df, config, args)
    save_pipeline(pipeline, config["pipeline_path"])
    print_training_summary(pipeline, run_t0)


def print_training_summary(pipeline, run_t0):
    """End-of-training human-readable diagnostic block."""
    diag = pipeline.get("diagnostics", {})
    lines = []
    lines.append("=" * 78)
    lines.append("TRAINING COMPLETE")
    lines.append("=" * 78)
    lines.append(f"Total wall-clock time: {(_time.time()-run_t0)/60:.1f} min")
    lines.append(f"Pipeline saved to: {config['pipeline_path']}")
    lines.append("")
    lines.append("--- Optuna best per-day top-1% precision (CV folds) ---")
    for k in ("clf1d", "clf5d", "rank"):
        v = diag.get("optuna_best", {}).get(k)
        per_fold = diag.get("optuna_per_fold", {}).get(k)
        if v is not None:
            extra = f"  folds={per_fold}" if per_fold else ""
            lines.append(f"  {k:6s}: {v:.4f}{extra}")
    if "stacker_coefs" in diag:
        lines.append("")
        lines.append("--- Stacker weights (logistic on OOF base preds) ---")
        for name, coef in diag["stacker_coefs"].items():
            bar = "#" * max(1, int(abs(coef) * 20))
            sign = "-" if coef < 0 else "+"
            lines.append(f"  {name:8s} {sign}{abs(coef):.3f}  {bar}")
    if "oof_metrics" in diag:
        lines.append("")
        lines.append("--- OOF metrics per base learner (genuine holdout) ---")
        lines.append(f"  {'learner':10s} {'AUC':>7s} {'AUC-PR':>8s} {'top1%prec':>10s}")
        for name, m in diag["oof_metrics"].items():
            lines.append(f"  {name:10s} {m['auc']:>7.4f} {m['aucpr']:>8.4f} "
                         f"{m['top1_prec']:>10.4f}")
        m = diag.get("oof_metrics_meta")
        if m:
            lines.append(f"  {'META':10s} {m['auc']:>7.4f} {m['aucpr']:>8.4f} "
                         f"{m['top1_prec']:>10.4f}  (after stacking)")
    lines.append("")
    lines.append("--- Calibration slice (out-of-sample after embargo) ---")
    lines.append(f"  calibrator: {diag.get('calibrator_status','?')}")
    lines.append(f"  conformal threshold tau: {pipeline['threshold']:.4f}")
    lines.append(f"  precision @ tau:         "
                 f"{pipeline['threshold_precision_calib']:.4f} "
                 f"(target was {args.target_precision})")
    lines.append(f"  coverage @ tau:          "
                 f"{pipeline['threshold_coverage_calib']:.4f} "
                 f"of calib slice")
    if "fires_per_day" in diag:
        f = diag["fires_per_day"]
        lines.append(f"  per-day fires:           "
                     f"mean={f['mean']:.2f}  median={f['median']:.0f}  "
                     f"max={f['max']:.0f}  days_with_fire={f['days_with_fire']}/"
                     f"{f['total_days']} ({100*f['days_with_fire']/max(f['total_days'],1):.0f}%)")
    if "next_steps" in diag and diag["next_steps"]:
        lines.append("")
        lines.append("--- Notes ---")
        for n in diag["next_steps"]:
            lines.append(f"  * {n}")
    lines.append("=" * 78)
    msg = "\n" + "\n".join(lines)
    logging.info(msg)


if __name__ == "__main__":
    main()
