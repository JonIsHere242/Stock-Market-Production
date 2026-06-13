"""Self-contained XGBoost pipeline — data loading, labeling, training, prediction.

Merges prepare_data.py + simple_xgb.py + predict_to_rf.py into one file.
Originally written as xgb_pipeline.py (2026-05-19), renamed to 4__Predictor.py
(2026-05-20) when the old ensemble predictor was archived to _old_versions/.

RESULT THIS CONFIG WAS BUILT TO REPLICATE
==========================================
Date:           2026-05-19
Ann Return:     172.31%
Sharpe:         11.44
Win Rate:       47.44%  (after fees)
Max Drawdown:   11.49%
PSR:            98.58%
Total Return:   177.78%

Monthly performance (OOS window Dec 2025 – Apr 2026):
  2025-12:  -1.93% excess
  2026-01: +25.81% excess  [Unicorn]
  2026-02: +22.14% excess  [Unicorn]
  2026-03:  +9.28% excess  [Excellent]
  2026-04:  -7.70% excess

EXACT CONFIG THAT PRODUCED THESE RESULTS
==========================================
  runpercent         = 75   (train through 2025-11-20, 459k rows)
  embargo_days       = 5
  label_mode         = topq (per-day top-20% next-day return)
  add_xs_features    = True (cross-sectional percentile ranks)
  drop_vol_features  = False  ← CRITICAL: vol features are top-ranked
  tune               = True
  tune_objective     = top1_meanret
  tune_subsample     = 0.35
  n_trials           = 100
  recency_half_life  = 720 days
  top_frac_per_day   = 0.01 (top 1% per day fires)

2026-06-12: Phase 12 UpProbability mapping changed from per-day cross-sectional
rank (xs_rank) to per-stock trailing percentile (ts_pctile, now the default) so
the output is the "unusually high vs own history" quantity the strategy's
can_buy percentile gate actually consumes (gate reads PAST bars since the
2026-06-10 lookahead repair). The result table above was produced under
xs_rank; pass --upprob_mode xs_rank to reproduce it. Training is untouched --
regenerate RFpredictions with --predict_only, no retrain needed.

Pipeline steps
==============
  Phase 1:  Load per-ticker parquets from Data/ProcessedData/
  Phase 2:  Label engineering (shift target -1, add ret_5d)
  Phase 3:  Universe filter — FilterRubric Step 1
  Phase 4:  Shuffle within each date (kills row-order leakage)
  Phase 5:  Train/calib split with N-day embargo
  Phase 6:  Build feature matrix — xs rank features if enabled
  Phase 7:  Build labels (topq / risk_adj_topq / binary_up)
  Phase 8:  Recency weights
  Phase 9:  Optuna hyperparameter tuning
  Phase 10: Train final XGBClassifier
  Phase 11: Evaluate on calibration slice + save model/reports
  Phase 12: Inference — score all ProcessedData tickers, write RFpredictions
            UpProbability = per-stock trailing percentile of the model score
            (ts_pctile default, matches can_buy's own-history percentile gate;
            --upprob_mode xs_rank = legacy per-day cross-sectional mapping)

Run:
    python 4__Predictor.py                         # full train + predict
    python 4__Predictor.py --predict_only          # inference only (model saved)
    python 4__Predictor.py --inspect               # print cached report
    python 4__Predictor.py --n_trials 10           # quick diagnostic run
    python 4__Predictor.py --no_tune               # skip Optuna (match pre-Optuna baseline)
    python 4__Predictor.py --upprob_mode xs_rank   # legacy cross-sectional UpProb mapping

BEST PRACTICE — STANDARD RETRAIN
==================================
The bare `python 4__Predictor.py` invocation is the canonical retrain. Every
default below is the winning-config value (see "EXACT CONFIG ..." above) and
should NOT be overridden unless you are running a diagnostic ablation:
    runpercent=75, embargo_days=5, label_mode=topq, add_xs_features=True,
    drop_vol_features=False, tune=True, tune_objective=top1_meanret,
    n_trials=100, tune_subsample=0.35, recency_half_life_days=720,
    top_frac_per_day=0.01
Do NOT raise --runpercent beyond 75 for a standard retrain — the OOS validation
slice (Dec 2025+) is load-bearing for the backtester's reported metrics, and
shrinking it makes the backtest mostly in-sample. After this script completes,
run the backtester: `python 5__NightlyBackTester.py --force [--sample N]`.
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from joblib import dump, load as joblib_load

import warnings

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score,
)
from scipy.special import expit
from scipy.optimize import minimize


# -------------------------------------------------------------------------- #
# Logging                                                                    #
# -------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


# -------------------------------------------------------------------------- #
# CLI — defaults match the 172% winning config                               #
# -------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Self-contained XGBoost pipeline (data -> train -> predict)."
)

# I/O
parser.add_argument("--input_dir", default="Data/ProcessedData")
parser.add_argument("--model_dir", default="Data/XGBPipeline", help="Where model, reports, and calib scores are saved.")
parser.add_argument("--output_dir", default="Data/RFpredictions", help="Where per-ticker RFpredictions parquets are written.")
parser.add_argument("--max_files", type=int, default=None, help="Cap on tickers loaded (for quick tests).")

# Data split
parser.add_argument("--runpercent", type=int, default=75, help="Percent of rows (sorted by date) used for training. 75 = train through 2025-11-20 with current data.")
parser.add_argument("--train_end_date", default=None, help="Override --runpercent: train through this exact date (YYYY-MM-DD). For refit-cadence experiments — hold params fixed, vary only the cutoff.")
parser.add_argument("--calibpercent", type=int, default=15, help="Percent of rows used for calibration (after embargo).")
parser.add_argument("--embargo_days", type=int, default=5, help="Calendar-day gap between train end and calib start.")
parser.add_argument("--horizon_5d", type=int, default=5, help="Horizon for the secondary ret_5d label.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for within-date shuffling.")

# Columns
parser.add_argument("--target_column", default="percent_change_Close")
parser.add_argument("--date_column", default="Date")
parser.add_argument("--ticker_column", default="Ticker")

# Label
parser.add_argument("--label_mode", choices=["binary_up", "topq", "risk_adj_topq"], default="topq", help="topq = per-day top-20%% next-day return (winning config). risk_adj_topq = top-20%% of return/vol. binary_up = simple next-day return > 0.")
parser.add_argument("--topq_frac", type=float, default=0.20)
parser.add_argument("--vol_col", default="Realized_Vol_21d")
parser.add_argument("--vol_floor", type=float, default=1e-3)

# Features
parser.add_argument("--add_xs_features", action="store_true", default=True, help="Add per-day percentile-rank column for each numeric feature (doubles feature count). ON by default — these were top-ranked in the winning model.")
parser.add_argument("--no_xs_features", action="store_true", help="Disable --add_xs_features.")
parser.add_argument("--drop_vol_features", action="store_true", help="Drop volatility-family features. OFF by default — vol features were the top-ranked features in the winning model. Only set this for diagnostic ablations.")
parser.add_argument("--drop_feature_patterns", type=str, default=None, help="Comma-separated substrings to drop from feature list.")
parser.add_argument("--drop_features_exact", type=str, default=None, help="Comma-separated EXACT column names to drop (useful when a betrayal feature name is a substring of a feature you want to keep, e.g. 'composite' vs 'market_regime_composite').")

# Training
parser.add_argument("--n_estimators", type=int, default=500)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--min_child_weight", type=int, default=5)
parser.add_argument("--reg_alpha", type=float, default=0.5)
parser.add_argument("--reg_lambda", type=float, default=2.0)
parser.add_argument("--subsample", type=float, default=0.8)
parser.add_argument("--colsample_bytree", type=float, default=0.6)
parser.add_argument("--early_stopping_rounds", type=int, default=30)
parser.add_argument("--scale_pos_weight", type=float, default=None)
parser.add_argument("--recency_half_life_days", type=float, default=720, help="Exponential-decay recency weight half-life in days. 720 = samples 2 years old have half the weight of newest.")

# Optuna
parser.add_argument("--no_tune", action="store_true", help="Skip Optuna tuning — use default XGB params. Matches the original May-16 run (no Optuna) that produced 84%%+ before config drift.")
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--tune_subsample", type=float, default=0.35, help="Row fraction for each Optuna trial (speeds up tuning). Final fit uses full training data.")
parser.add_argument("--tune_objective", choices=["top1_prec", "top1_meanret", "aucpr"], default="top1_meanret", help="Optuna objective. top1_meanret = mean return of top-1%% picks per day (the objective used in the winning run).")

# Inference
parser.add_argument("--upprob_mode", choices=["ts_pctile", "xs_rank"], default="ts_pctile", help="UpProbability mapping. ts_pctile (default) = per-stock trailing percentile of the model score mapped to [0.30, 0.70] -- the 'unusually high vs own history' quantity can_buy's percentile gate expects. xs_rank = legacy per-day cross-sectional rank mapping (rollback).")
parser.add_argument("--ts_window", type=int, default=252, help="ts_pctile: trailing window in rows per ticker.")
parser.add_argument("--ts_min_periods", type=int, default=60, help="ts_pctile: min history rows before a percentile is emitted; earlier rows get UpProbability=0.30.")
parser.add_argument("--top_frac_per_day", type=float, default=0.01, help="xs_rank mode only: per-day fraction mapped to UpProb in [0.45, 0.70]. 0.01 = top 1%% per day (~10-15 names with universe ~1400).")

# Calibration + conformal threshold
parser.add_argument("--nocalib", action="store_true", help="Disable Beta probability calibration on calib slice. By default, a 3-param Beta calibrator is fit and saved for diagnostics (does not alter UpPrediction logic).")
parser.add_argument("--target_precision", type=float, default=0.75, help="Conformal precision target: find lowest score cutoff where calib-slice precision >= this value. Logged as diagnostic; does not gate UpPrediction.")
parser.add_argument("--max_coverage", type=float, default=0.05, help="Upper cap on coverage when searching conformal threshold.")

# Convenience
parser.add_argument("--reuse", action="store_true", help="Reuse cached PreparedData splits from a previous run (skips Phases 1-5, goes straight to feature/label build).")
parser.add_argument("--fast", action="store_true", help="Quick diagnostic mode: cap Optuna to 10 trials and load only 500 tickers. Useful for smoke-testing changes.")

# Modes
parser.add_argument("--predict_only", action="store_true", help="Skip training. Load saved model and run inference only.")
parser.add_argument("--inspect", action="store_true", help="Print cached report + summary without running anything.")
parser.add_argument("--oos_only", action="store_true", help="Skip training/inference. Load the saved model and measure true out-of-sample decay: score the held-out tail (rows after the calib window — data the model never saw in train OR calib) and print a rank-band x split table of per-day mean return + precision. Headline = top-1%% vs shoulder-band (0.90-0.95) decay. Writes oos_report.txt + oos_scores.parquet.")
parser.add_argument("--oos_max_tickers", type=int, default=None, help="Cap tickers loaded for --oos_only (quick smoke test). None = all.")
parser.add_argument("--walkforward", action="store_true", help="Walk-forward OOS engine: load+feature-build ONCE, then for a suite of training configs x monthly anchors, retrain and measure true-OOS band edge (net of market beta) on the following month. Judges configs across many independent OOS windows. Writes walkforward_report.txt + walkforward_results.parquet.")
parser.add_argument("--wf_configs", default="baseline,recent_252,recent_126,hl_180,hl_90,decontam", help="Comma list of walk-forward configs to compare. Available: baseline, recent_252, recent_126, hl_180, hl_90, decontam.")
parser.add_argument("--wf_min_anchor", default="2025-06", help="Earliest anchor year-month (YYYY-MM). Anchor = last train date of that month; OOS = the following month.")
parser.add_argument("--wf_trees", type=int, default=200, help="Fixed n_estimators per walk-forward fit (no Optuna/early-stop — isolates the data/feature/weight effect across configs).")
parser.add_argument("--wf_max_tickers", type=int, default=None, help="Cap tickers loaded for --walkforward (smoke test). None = all.")
parser.add_argument("--wf_sweep", action="store_true", help="PARAM-SWEEP mode: parallel, multi-seed walk-forward over XGB hyperparameter configs (depth/reg/trees) on identical baseline data, to isolate the overfitting lever and beat the nondeterminism noise floor. Reports seed-averaged top1_net per config with anchor win%%. Writes walkforward_sweep_report.txt + walkforward_sweep_results.parquet.")
parser.add_argument("--wf_sweep_configs", default="prod_optuna,d3,d4,d5,d6,d8_reg,d4_strongreg,d5_slow", help="Comma list of param-sweep configs (see SWEEP_CONFIGS registry).")
parser.add_argument("--wf_seeds", type=int, default=4, help="Seeds per (config,anchor) — averaged to beat the ~0.02 XGB-hist nondeterminism noise floor.")
parser.add_argument("--wf_workers", type=int, default=4, help="Concurrent fits (threads; XGB releases the GIL during fit). Total cores ~= wf_workers * wf_threads.")
parser.add_argument("--wf_threads", type=int, default=8, help="n_jobs per fit. Default 8; with wf_workers=4 that's 32 cores.")
parser.add_argument("--wf_device", default="cpu", help="XGB device for walk-forward fits: cpu or cuda (or cuda:0/cuda:1).")

args = parser.parse_args()

# Resolve the xs features flag (default=True but --no_xs_features disables it)
USE_XS    = args.add_xs_features and not args.no_xs_features
TUNE      = not args.no_tune
USE_CALIB = not args.nocalib

# --fast: quick smoke-test mode
if args.fast:
    if args.n_trials > 10:
        args.n_trials = 10
    if args.max_files is None:
        args.max_files = 500
    logging.info("--fast: n_trials=10, max_files=500")

# -------------------------------------------------------------------------- #
# Utilities                                                                  #
# -------------------------------------------------------------------------- #
class Phase:
    """Context manager that logs timing for each named pipeline step."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        logging.info(f"\n>>> {self.name} ...")
        return self
    def __exit__(self, *exc):
        logging.info(f"<<< {self.name} done in {time.time()-self.t0:.1f}s.")


class BetaCalibrator:
    """3-parameter Beta calibrator (Kull et al. 2017).

    Maps raw probabilities p through sigmoid(a*logit(p) + b*log((1-p)/p) + c).
    Fitted by minimising NLL on a held-out calibration slice.
    When a=1, b=0, c=0 the transform is identity — safe default when fitting fails.
    """
    def __init__(self):
        self.a = 1.0; self.b = 0.0; self.c = 0.0
        self._fitted = False

    @staticmethod
    def _safe(p):
        return np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)

    def _transform(self, p, a, b, c):
        p = self._safe(p)
        z = a * np.log(p / (1 - p)) + b * np.log((1 - p) / p) + c
        return expit(z)

    def fit(self, p, y):
        p = self._safe(p)
        y = np.asarray(y, dtype=np.float64)

        def nll(theta):
            a, b, c = theta
            q = np.clip(self._transform(p, a, b, c), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(nll, x0=[1.0, 0.0, 0.0], method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-10})
        if res.success:
            self.a, self.b, self.c = float(res.x[0]), float(res.x[1]), float(res.x[2])
        self._fitted = True
        return self

    def predict(self, p):
        if not self._fitted:
            return np.asarray(p, dtype=np.float64)
        return self._transform(p, self.a, self.b, self.c)

    def to_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c, "fitted": self._fitted}


def conformal_threshold(scores, labels, target_precision=0.75,
                        min_coverage=0.001, max_coverage=0.05):
    """Find the lowest score cutoff where calib-slice precision >= target.

    Searches candidate thresholds from high to low; keeps the last one
    (lowest threshold = highest coverage) that still achieves target precision
    within coverage bounds.

    Returns (threshold, precision_at_threshold, coverage_at_threshold).
    Returns (NaN, NaN, 0.0) if no threshold meets the criteria.
    """
    if len(scores) == 0:
        return float("nan"), float("nan"), 0.0
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    unique = np.sort(np.unique(scores))[::-1]  # descending
    best = (float("nan"), float("nan"), 0.0)
    for thr in unique:
        fired = scores >= thr
        cov = float(fired.mean())
        if cov < min_coverage or cov > max_coverage:
            continue
        prec = float(labels[fired].mean())
        if prec >= target_precision:
            best = (float(thr), prec, cov)
    return best


def downcast(df):
    floats = df.select_dtypes(include=["float64"]).columns
    if len(floats):
        df[floats] = df[floats].astype("float32")
    return df


def mem_gb(df):
    return df.memory_usage(deep=False).sum() / 1e9


# -------------------------------------------------------------------------- #
# Phase 1+2: Load tickers + label engineering                                #
# -------------------------------------------------------------------------- #
def load_and_label_tickers(input_dir, target_col, date_col, horizon_5d,
                            max_files=None):
    """Load every parquet in input_dir, shift target -1 (next-day return),
    add ret_5d (5-day forward return), drop last rows that have no label."""
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
    if max_files:
        files = files[:max_files]
        logging.info(f"  --max_files {max_files}: using {len(files)} tickers")

    parts = []
    n_skip_short = n_drop_nan = n_drop_outlier = 0

    def _process(fn, arrow_table):
        nonlocal n_skip_short, n_drop_nan, n_drop_outlier
        df = arrow_table.to_pandas()
        if target_col not in df.columns or date_col not in df.columns:
            return None
        if df.shape[0] <= 60:
            n_skip_short += df.shape[0]
            return None
        df = df.sort_values(date_col).reset_index(drop=True)
        df[target_col] = df[target_col].shift(-1)
        df["ret_5d"] = (df["Close"].shift(-horizon_5d) / df["Close"] - 1.0
                        if "Close" in df.columns else np.nan)
        df = df.iloc[:-max(horizon_5d, 1)]
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna(subset=[target_col])
        n_drop_nan += before - len(df)
        before = len(df)
        df = df[(df[target_col] <= 5) & (df[target_col] >= -5)]
        n_drop_outlier += before - len(df)
        return downcast(df) if not df.empty else None

    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_fn = {
            ex.submit(pq.read_table, os.path.join(input_dir, fn)): fn
            for fn in files
        }
        for future in tqdm(as_completed(future_to_fn), total=len(files),
                           desc="Loading tickers"):
            fn = future_to_fn[future]
            try:
                result = _process(fn, future.result())
                if result is not None:
                    parts.append(result)
            except Exception as e:
                logging.warning(f"  skipping {fn}: {e}")

    logging.info(f"  accounting: skipped_short={n_skip_short:,}  "
                 f"nan_dropped={n_drop_nan:,}  outliers_dropped={n_drop_outlier:,}")
    return parts


# -------------------------------------------------------------------------- #
# Phase 3: Universe filter — FilterRubric Step 1                             #
# -------------------------------------------------------------------------- #
def apply_quality_filter(df,
                          min_close=5.0,
                          min_dollar_volume=5_000_000.0,
                          max_atr_pct=0.05,
                          rsi_exclude_lo=30.0,
                          rsi_exclude_hi=40.0):
    """Apply the FilterRubric Step 1 universe gate.

    Returns (filtered_df, boolean_mask_aligned_to_input_index).
    The mask is useful at inference time (mark, don't drop).
    """
    n0 = len(df)
    mask = pd.Series(True, index=df.index)

    if "Close" in df.columns:
        gate = df["Close"] >= min_close
        logging.info(f"  gate Close>={min_close}: removes {int((~gate & mask).sum()):,}")
        mask &= gate
    if "dollar_volume_ma_10" in df.columns:
        gate = df["dollar_volume_ma_10"] >= min_dollar_volume
        logging.info(f"  gate dollar_vol>={min_dollar_volume:,.0f}: removes {int((~gate & mask).sum()):,}")
        mask &= gate
    if "atr_percentage" in df.columns:
        gate = df["atr_percentage"] <= max_atr_pct
        logging.info(f"  gate atr_pct<={max_atr_pct}: removes {int((~gate & mask).sum()):,}")
        mask &= gate
    if "RSI" in df.columns:
        gate = ~((df["RSI"] >= rsi_exclude_lo) & (df["RSI"] < rsi_exclude_hi))
        logging.info(f"  gate RSI not in [{rsi_exclude_lo},{rsi_exclude_hi}): "
                     f"removes {int((~gate & mask).sum()):,}")
        mask &= gate

    out = df.loc[mask].copy()
    logging.info(f"  universe: {n0:,} -> {len(out):,} rows "
                 f"({100*len(out)/max(n0,1):.1f}% retained)")
    return out, mask


# -------------------------------------------------------------------------- #
# Phase 4: Shuffle within each date                                          #
# -------------------------------------------------------------------------- #
def shuffle_within_date(df, date_col, seed):
    """Randomise row order within each date.

    Kills any signal a model could learn from intra-day row order
    (alphabetic ticker order, file load order, etc.).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_shuf"] = rng.random(len(df), dtype=np.float32)
    df = df.sort_values([date_col, "_shuf"], kind="stable").reset_index(drop=True)
    return df.drop(columns="_shuf")


# -------------------------------------------------------------------------- #
# Phase 5: Train/calib split with embargo                                    #
# -------------------------------------------------------------------------- #
def time_split_with_embargo(df, train_pct, calib_pct, embargo_days, date_col,
                            train_end_date=None):
    n = len(df)
    if train_end_date is not None:
        train_end_date = pd.Timestamp(train_end_date)
    else:
        train_end_row  = max(int(n * train_pct / 100) - 1, 0)
        train_end_date = pd.Timestamp(df[date_col].iloc[train_end_row])
    calib_start    = train_end_date + pd.Timedelta(days=embargo_days)

    pool = df[df[date_col] >= calib_start]
    if pool.empty:
        raise ValueError("No rows available after embargo for calibration.")

    calib_target   = int(n * calib_pct / 100)
    calib_end_row  = min(calib_target, len(pool)) - 1
    calib_end_date = pd.Timestamp(pool[date_col].iloc[calib_end_row])

    train_df = df[df[date_col] <= train_end_date].copy()
    calib_df = df[(df[date_col] >= calib_start) &
                  (df[date_col] <= calib_end_date)].copy()

    meta = {
        "train_end_date":   str(train_end_date.date()),
        "calib_start_date": str(calib_start.date()),
        "calib_end_date":   str(calib_end_date.date()),
        "embargo_days":     embargo_days,
        "train_rows":       len(train_df),
        "calib_rows":       len(calib_df),
        "train_dates":      train_df[date_col].nunique(),
        "calib_dates":      calib_df[date_col].nunique(),
    }
    logging.info(f"  train: {meta['train_end_date']}  "
                 f"({meta['train_rows']:,} rows, {meta['train_dates']} dates)")
    logging.info(f"  calib: {meta['calib_start_date']} -> "
                 f"{meta['calib_end_date']}  "
                 f"({meta['calib_rows']:,} rows, {meta['calib_dates']} dates)")
    return train_df, calib_df, meta


# -------------------------------------------------------------------------- #
# Phase 6: Feature matrix                                                    #
# -------------------------------------------------------------------------- #
NON_FEATURES = {
    "percent_change_Close", "ret_5d",
    "Date", "Ticker",
    "Open", "High", "Low", "Close", "Volume",
}


def is_vol_feature(name):
    """Heuristic: True for volatility-family feature names."""
    n = name.lower()
    if n.endswith("_xs"):
        n = n[:-3]
    patterns = [
        r"^atr(_|$|\d|%)", r"realized_vol", r"^cv_\d",
        r"hc_ratio", r"high_close_ratio", r"low_close_ratio",
        r"intraday_range", r"percent_range",
        r"^volatility", r"_volatility",
        r"vol_v\d", r"vix_vs_realized", r"vix_adjusted_atr",
        r"^pct_change_std",
    ]
    return any(re.search(p, n) for p in patterns)


def select_base_features(df, extra_exclude=None):
    """Return numeric columns that are valid input features."""
    exclude = set(NON_FEATURES)
    if extra_exclude:
        exclude |= set(extra_exclude)
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def add_xs_rank_features(df, feature_cols, date_col, batch_size=40):
    """Add a per-day percentile-rank column for each base feature.

    Doubles the feature count. Market-wide constants get all-tied ranks at 0.5
    (XGB ignores them). Done in batches for visibility.
    """
    n_feat = len(feature_cols)
    n_batches = (n_feat + batch_size - 1) // batch_size
    logging.info(f"  xs-rank for {n_feat} features over "
                 f"{df[date_col].nunique():,} dates ({n_batches} batches)...")
    grouped = df.groupby(date_col, sort=False)
    chunks = []
    for i in range(0, n_feat, batch_size):
        batch = feature_cols[i:i + batch_size]
        t0 = time.time()
        ranks = grouped[batch].rank(pct=True, method="average", na_option="keep")
        ranks.columns = [c + "_xs" for c in batch]
        chunks.append(ranks.astype(np.float32))
        logging.info(f"    batch {len(chunks)}/{n_batches}: "
                     f"{len(batch)} features in {time.time()-t0:.1f}s")
    return pd.concat([df] + chunks, axis=1)


# -------------------------------------------------------------------------- #
# Phase 7: Label construction                                                #
# -------------------------------------------------------------------------- #
def topq_label(returns, dates, top_frac=0.20):
    """Label = 1 if row is in the per-day top `top_frac` of next-day return."""
    s = pd.Series(returns).reset_index(drop=True)
    d = pd.Series(dates).reset_index(drop=True)
    cutoff = s.groupby(d).transform(lambda x: x.quantile(1 - top_frac))
    return (s >= cutoff).astype(int).values


def risk_adj_topq_label(returns, vol, dates, top_frac=0.20, vol_floor=1e-3):
    """Label = 1 if row is in the per-day top `top_frac` of return / vol."""
    r = pd.Series(returns).reset_index(drop=True)
    v = pd.Series(vol).reset_index(drop=True).clip(lower=vol_floor)
    d = pd.Series(dates).reset_index(drop=True)
    rar = r / v
    cutoff = rar.groupby(d).transform(lambda x: x.quantile(1 - top_frac))
    return (rar >= cutoff).astype(int).values


def build_labels(train_df, calib_df, label_mode, target_col, date_col,
                  topq_frac, vol_col, vol_floor):
    if label_mode == "binary_up":
        y_train = (train_df[target_col] > 0).astype(int).values
        y_calib = (calib_df[target_col] > 0).astype(int).values
        logging.info(f"  label = next-day return > 0")
    elif label_mode == "topq":
        y_train = topq_label(train_df[target_col].values,
                             train_df[date_col].values, top_frac=topq_frac)
        y_calib = topq_label(calib_df[target_col].values,
                             calib_df[date_col].values, top_frac=topq_frac)
        logging.info(f"  label = per-day top {topq_frac*100:.0f}% next-day return")
    elif label_mode == "risk_adj_topq":
        if vol_col not in train_df.columns:
            raise ValueError(f"vol_col '{vol_col}' missing from data")
        y_train = risk_adj_topq_label(
            train_df[target_col].values, train_df[vol_col].values,
            train_df[date_col].values, top_frac=topq_frac, vol_floor=vol_floor)
        y_calib = risk_adj_topq_label(
            calib_df[target_col].values, calib_df[vol_col].values,
            calib_df[date_col].values, top_frac=topq_frac, vol_floor=vol_floor)
        logging.info(f"  label = per-day top {topq_frac*100:.0f}% of return/{vol_col}")
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")

    logging.info(f"  P(y=1) train={y_train.mean():.4f}  calib={y_calib.mean():.4f}")
    return y_train, y_calib


# -------------------------------------------------------------------------- #
# Phase 8: Recency weights                                                   #
# -------------------------------------------------------------------------- #
def recency_weights(dates, half_life_days):
    """Exponential-decay weights so recent samples count more.

    Most-recent sample weight = 1. A sample `half_life_days` older → 0.5.
    Mean-normalised so loss scale stays comparable to unweighted training.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    age = (dates.max() - dates).dt.days.values.astype(np.float64)
    w = np.power(0.5, age / max(half_life_days, 1.0))
    return (w / w.mean()).astype(np.float32)


# -------------------------------------------------------------------------- #
# Phase 9: Optuna tuning                                                     #
# -------------------------------------------------------------------------- #
def per_day_top_k_precision(scores, labels, dates, k_frac=0.01):
    s = pd.DataFrame({"s": scores, "y": labels, "d": pd.Series(dates).values})
    precs = []
    for _, g in s.groupby("d", sort=False):
        if len(g) < 20:
            continue
        top = g.nlargest(max(int(len(g) * k_frac), 1), "s")
        precs.append(top["y"].mean())
    return float(np.mean(precs)) if precs else float("nan")


def per_day_top_k_mean_return(scores, returns, dates, k_frac=0.01):
    s = pd.DataFrame({"s": scores, "r": returns, "d": pd.Series(dates).values})
    rets = []
    for _, g in s.groupby("d", sort=False):
        if len(g) < 20:
            continue
        top = g.nlargest(max(int(len(g) * k_frac), 1), "s")
        rets.append(top["r"].mean())
    return float(np.mean(rets)) if rets else float("nan")


def run_optuna_tuning(X_train, y_train, ret_train, dates_train, sw_train,
                       n_trials, objective_name, tune_subsample=1.0):
    """Optuna search on an 80/20 inner walk-forward split of train.

    Returns (best_params, best_value, best_iter).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(X_train)
    cut = int(n * 0.80)
    X_it = X_train.iloc[:cut]
    X_iv = X_train.iloc[cut:]
    y_it = y_train[:cut]
    y_iv = y_train[cut:]
    ret_iv   = ret_train[cut:] if ret_train is not None else None
    d_iv     = dates_train[cut:]
    sw_it = sw_train[:cut] if sw_train is not None else None
    sw_iv = sw_train[cut:] if sw_train is not None else None

    if tune_subsample < 1.0:
        keep  = int(len(X_it) * tune_subsample)
        start = len(X_it) - keep
        X_it  = X_it.iloc[start:]
        y_it  = y_it[start:]
        if sw_it is not None:
            sw_it = sw_it[start:]
        logging.info(f"  tune_subsample={tune_subsample}: inner-train -> "
                     f"{len(X_it):,} rows (most recent)")

    logging.info(f"  inner split: tr={len(X_it):,}  val={len(X_iv):,}")
    logging.info(f"  optuna objective: {objective_name}")

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 300, 1200),
            max_depth         = trial.suggest_int("max_depth", 4, 9),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            gamma             = trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        )
        m = XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42,
            early_stopping_rounds=30, verbosity=0, **params,
        )
        fk = dict(eval_set=[(X_iv, y_iv)], verbose=False)
        if sw_it is not None:
            fk["sample_weight"] = sw_it
            fk["sample_weight_eval_set"] = [sw_iv]
        m.fit(X_it, y_it, **fk)
        p = m.predict_proba(X_iv)[:, 1]

        if objective_name == "top1_prec":
            score = per_day_top_k_precision(p, y_iv, d_iv)
        elif objective_name == "top1_meanret":
            if ret_iv is None:
                raise ValueError("top1_meanret requires return column")
            score = per_day_top_k_mean_return(p, ret_iv, d_iv)
        else:
            score = float(average_precision_score(y_iv, p))

        trial.set_user_attr("best_iter", int(m.best_iteration))
        return score if not np.isnan(score) else -1.0

    def _log(study, trial):
        bi = trial.user_attrs.get("best_iter", -1)
        logging.info(f"  trial {trial.number+1}/{n_trials}  "
                     f"value={trial.value:.4f}  best_iter={bi}  "
                     f"best_so_far={study.best_value:.4f}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[_log],
                   show_progress_bar=False)

    logging.info(f"  best trial #{study.best_trial.number}  "
                 f"value={study.best_value:.4f}")
    logging.info(f"  best params: {study.best_params}")
    return (study.best_params,
            study.best_value,
            study.best_trial.user_attrs.get("best_iter"))


# -------------------------------------------------------------------------- #
# Phase 10: Train final model                                                #
# -------------------------------------------------------------------------- #
def train_model(X_train, y_train, tuned_params, sw_train, scale_pos_weight,
                n_estimators, max_depth, learning_rate, min_child_weight,
                reg_alpha, reg_lambda, subsample, colsample_bytree,
                early_stopping_rounds):
    """Fit final XGBClassifier on full training data.

    Uses last 20% of train as internal val for early stopping.
    """
    val_start = int(len(X_train) * 0.80)
    sw_fit = sw_val = None
    if sw_train is not None:
        sw_fit = sw_train[:val_start]
        sw_val = sw_train[val_start:]

    xgb_params = dict(tuned_params) if tuned_params else dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, min_child_weight=min_child_weight,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
    )
    logging.info(f"  XGB params: {xgb_params}")

    spw = scale_pos_weight if scale_pos_weight is not None else 1.0
    clf = XGBClassifier(
        scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=42,
        early_stopping_rounds=early_stopping_rounds, verbosity=1,
        **xgb_params,
    )
    fk = dict(eval_set=[(X_train.iloc[val_start:], y_train[val_start:])],
              verbose=False)
    if sw_fit is not None:
        fk["sample_weight"] = sw_fit
        fk["sample_weight_eval_set"] = [sw_val]
    clf.fit(X_train.iloc[:val_start], y_train[:val_start], **fk)
    logging.info(f"  trained {clf.best_iteration+1} trees "
                 f"(early stopped from {n_estimators})")
    return clf


# -------------------------------------------------------------------------- #
# Phase 11: Evaluate + save                                                  #
# -------------------------------------------------------------------------- #
def evaluate_and_save(clf, X_calib, y_calib, calib_df, feature_cols,
                       train_df, model_dir, tuned_params, label_mode,
                       topq_frac, vol_col, target_col, date_col, ticker_col,
                       use_calib=True, target_precision=0.75, max_coverage=0.05):
    os.makedirs(model_dir, exist_ok=True)
    model_path   = os.path.join(model_dir, "xgb.joblib")
    scores_path  = os.path.join(model_dir, "calib_scores.parquet")
    summary_path = os.path.join(model_dir, "summary.json")
    report_path  = os.path.join(model_dir, "report.txt")
    calib_path   = os.path.join(model_dir, "calibrator.joblib")

    p_calib = clf.predict_proba(X_calib)[:, 1]
    d_calib = calib_df[date_col].values

    auc   = float(roc_auc_score(y_calib, p_calib))
    aucpr = float(average_precision_score(y_calib, p_calib))
    top1_prec = per_day_top_k_precision(p_calib, y_calib, d_calib, k_frac=0.01)
    top5_prec = per_day_top_k_precision(p_calib, y_calib, d_calib, k_frac=0.05)

    # Fixed-threshold metrics
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    thresh_metrics = {}
    n_days_calib = calib_df[date_col].nunique()
    for t in thresholds:
        fire = p_calib >= t
        if fire.sum() < 10:
            thresh_metrics[t] = {"fires": int(fire.sum()), "precision": float("nan"),
                                 "fires_per_day": float(fire.sum() / max(n_days_calib, 1))}
        else:
            thresh_metrics[t] = {
                "fires": int(fire.sum()),
                "precision": float(precision_score(y_calib, fire, zero_division=0)),
                "fires_per_day": float(fire.sum() / max(n_days_calib, 1)),
            }

    # Save per-row scores
    out_cols = [date_col] + ([ticker_col] if ticker_col in calib_df.columns else [])
    if "Close" in calib_df.columns:
        out_cols.append("Close")
    scores_df = calib_df[out_cols].copy()
    scores_df["score"]         = p_calib.astype(np.float32)
    scores_df["label_up"]      = y_calib.astype(np.int8)
    scores_df["actual_return"] = calib_df[target_col].astype(np.float32).values
    scores_df.to_parquet(scores_path, index=False)

    # Save model
    dump(clf, model_path)

    # Beta calibration (diagnostic — does not alter inference UpPrediction logic)
    calibrator_info = {"fitted": False}
    p_calib_cal = p_calib.copy()   # calibrated scores (same as raw if disabled)
    conf_thr = conf_prec = conf_cov = float("nan")
    if use_calib:
        try:
            cal = BetaCalibrator()
            cal.fit(p_calib, y_calib)
            p_calib_cal = cal.predict(p_calib).astype(np.float64)
            dump(cal, calib_path)
            calibrator_info = cal.to_dict()
            calibrator_info["path"] = calib_path
            logging.info(f"  BetaCalibrator fitted: a={cal.a:.4f} b={cal.b:.4f} "
                         f"c={cal.c:.4f}  -> saved to {calib_path}")
            conf_thr, conf_prec, conf_cov = conformal_threshold(
                p_calib_cal, y_calib,
                target_precision=target_precision,
                max_coverage=max_coverage,
            )
            logging.info(f"  Conformal threshold (target_prec={target_precision}): "
                         f"thr={conf_thr:.4f}  prec={conf_prec:.4f}  "
                         f"cov={conf_cov:.4f}")
        except Exception as e:
            logging.warning(f"  BetaCalibrator fit failed ({e}); skipping calibration")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    summary = {
        "label_mode": label_mode,
        "tuned_params": tuned_params,
        "n_features": len(feature_cols),
        "train_rows": len(train_df),
        "calib_rows": len(calib_df),
        "best_iteration": int(clf.best_iteration),
        "metrics": {
            "auc": auc, "aucpr": aucpr,
            "top_1pct_precision": top1_prec,
            "top_5pct_precision": top5_prec,
        },
        "fixed_threshold_metrics": {str(k): v for k, v in thresh_metrics.items()},
        "p_calib_distribution": {
            "min": float(np.min(p_calib)), "p25": float(np.percentile(p_calib, 25)),
            "p50": float(np.median(p_calib)), "p75": float(np.percentile(p_calib, 75)),
            "p99": float(np.percentile(p_calib, 99)), "max": float(np.max(p_calib)),
        },
        "beta_calibrator": calibrator_info,
        "conformal_threshold": {
            "threshold": conf_thr, "precision": conf_prec, "coverage": conf_cov,
            "target_precision": target_precision, "max_coverage": max_coverage,
        },
        "top_features": imp.head(20).to_dict(orient="records"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Human-readable report
    baseline = y_calib.mean()
    lines = [
        "=" * 78,
        "XGB PIPELINE REPORT",
        "=" * 78,
        f"Label:          {label_mode}  (topq_frac={topq_frac})",
        f"Features:       {len(feature_cols)}",
        f"Train rows:     {len(train_df):,}",
        f"Calib rows:     {len(calib_df):,}  ({n_days_calib} days)",
        f"Trees:          {clf.best_iteration+1}",
        f"Base rate P(y=1) on calib: {baseline:.4f}",
        "",
        "--- Calibration-slice metrics ---",
        f"  AUC:              {auc:.4f}",
        f"  AUC-PR:           {aucpr:.4f}",
        f"  Top-1%-prec/day:  {top1_prec:.4f}",
        f"  Top-5%-prec/day:  {top5_prec:.4f}",
        "",
        "--- Fixed-threshold precision ---",
        f"  {'thr':>5} {'fires':>8} {'fires/day':>10} {'precision':>10} {'vs_base':>8}",
    ]
    for t, m in thresh_metrics.items():
        edge = f"{m['precision']-baseline:+.3f}" if not np.isnan(m["precision"]) else ""
        lines.append(f"  {t:>5.2f} {m['fires']:>8,} {m['fires_per_day']:>10.2f} "
                     f"{m['precision'] if not np.isnan(m['precision']) else 'n/a':>10} "
                     f"{edge:>8}")
    d = summary["p_calib_distribution"]
    lines += [
        "",
        "--- p_calib distribution ---",
        f"  min={d['min']:.3f} p25={d['p25']:.3f} p50={d['p50']:.3f} "
        f"p75={d['p75']:.3f} p99={d['p99']:.3f} max={d['max']:.3f}",
    ]
    if use_calib and calibrator_info.get("fitted"):
        lines += [
            "",
            "--- Beta calibrator (diagnostic) ---",
            f"  a={calibrator_info['a']:.4f}  b={calibrator_info['b']:.4f}  "
            f"c={calibrator_info['c']:.4f}",
            f"  Conformal threshold @ prec>={target_precision}: "
            f"thr={conf_thr:.4f}  prec={conf_prec:.4f}  cov={conf_cov:.4f}",
            f"  (diagnostic only — does not gate UpPrediction)",
        ]
    lines += [
        "",
        "--- Top 15 features ---",
    ]
    for _, row in imp.head(15).iterrows():
        lines.append(f"  {row['importance']:>8.5f}  {row['feature']}")
    lines.append("=" * 78)

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    logging.info("\n" + report)

    return model_path, scores_path, summary_path, calib_path


# -------------------------------------------------------------------------- #
# Phase 12: Inference — score all tickers, write RFpredictions               #
# -------------------------------------------------------------------------- #
def map_pct_rank_to_upprob(pct_rank, top_frac):
    """Map per-day percentile rank of model score to UpProbability.

    Top `top_frac` of each day -> [0.45, 0.70]  (passes backtester gate)
    Rest                       -> [0.30, 0.44]
    """
    threshold = 1.0 - top_frac
    above = pct_rank >= threshold
    up = np.empty_like(pct_rank, dtype=np.float64)
    up[above]  = 0.45 + 0.25 * (pct_rank[above] - threshold) / max(top_frac, 1e-9)
    up[~above] = 0.30 + 0.14 * pct_rank[~above] / max(threshold, 1e-9)
    return np.clip(up, 0.30, 0.70)


def per_stock_trailing_percentile(scores, tickers, dates, window, min_periods):
    """Percentile of each row's score within its own ticker's trailing
    `window` rows (current row included). NaN until min_periods rows exist.

    This is the "unusually high vs own history" quantity that can_buy's
    per-stock percentile gate expects UpProbability to be.
    """
    t_codes, _ = pd.factorize(np.asarray(tickers))
    d64 = np.asarray(dates, dtype="datetime64[ns]").view("i8")
    order = np.lexsort((d64, t_codes))
    tmp = pd.DataFrame({
        "t": t_codes[order],
        "s": np.asarray(scores, dtype=np.float64)[order],
    })
    p = (tmp.groupby("t", sort=False)["s"]
            .rolling(window, min_periods=min_periods)
            .rank(pct=True)
            .reset_index(level=0, drop=True)
            .sort_index())
    out = np.full(len(tmp), np.nan, dtype=np.float32)
    out[order] = p.to_numpy(np.float32)
    return out


def run_inference(model_path, input_dir, output_dir, date_col, ticker_col,
                   top_frac_per_day, max_files=None, calib_path=None,
                   upprob_mode="ts_pctile", ts_window=252, ts_min_periods=60):
    """Load saved model, score all tickers, write RFpredictions parquets."""
    os.makedirs(output_dir, exist_ok=True)

    with Phase("Load saved model"):
        clf = joblib_load(model_path)
        if not hasattr(clf, "feature_names_in_"):
            raise RuntimeError("Model has no feature_names_in_")
        feat_names      = list(clf.feature_names_in_)
        xs_features     = [c for c in feat_names if c.endswith("_xs")]
        raw_features    = [c for c in feat_names if not c.endswith("_xs")]
        xs_src          = [c[:-3] for c in xs_features]
        logging.info(f"  model expects {len(feat_names)} features: "
                     f"{len(raw_features)} raw + {len(xs_features)} xs")

        calibrator = None
        if calib_path and os.path.exists(calib_path):
            try:
                calibrator = joblib_load(calib_path)
                logging.info(f"  BetaCalibrator loaded from {calib_path} "
                             f"(a={calibrator.a:.4f} b={calibrator.b:.4f} "
                             f"c={calibrator.c:.4f})")
            except Exception as e:
                logging.warning(f"  failed to load calibrator ({e}); using raw scores")

    with Phase("Load all ticker files for inference"):
        import pyarrow as pa
        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
        if max_files:
            files = files[:max_files]

        def _load_arrow_infer(fn):
            tbl = pq.read_table(os.path.join(input_dir, fn))
            if date_col not in tbl.schema.names or len(tbl) == 0:
                return None
            if ticker_col not in tbl.schema.names:
                tbl = tbl.append_column(
                    ticker_col,
                    pa.array([os.path.splitext(fn)[0]] * len(tbl), type=pa.string()),
                )
            return tbl

        arrow_tables = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_to_fn = {ex.submit(_load_arrow_infer, fn): fn for fn in files}
            for future in tqdm(as_completed(future_to_fn), total=len(files), desc="Loading"):
                fn = future_to_fn[future]
                try:
                    tbl = future.result()
                    if tbl is not None:
                        arrow_tables.append(tbl)
                except Exception as e:
                    logging.warning(f"  skipping {fn}: {e}")

        if not arrow_tables:
            raise RuntimeError("No ticker data loaded for inference.")

        # Single concat + single to_pandas() — avoids 4265 individual GIL hits
        try:
            combined = pa.concat_tables(arrow_tables, promote_options="default").to_pandas()
        except Exception:
            combined = pd.concat([t.to_pandas() for t in arrow_tables], ignore_index=True)
        del arrow_tables

        combined[date_col] = pd.to_datetime(combined[date_col])
        combined = downcast(combined)
        combined = combined.sort_values(date_col).reset_index(drop=True)
        logging.info(f"  combined: {combined.shape}  "
                     f"{combined[ticker_col].nunique()} tickers  "
                     f"{combined[date_col].nunique()} dates")

    with Phase("Universe filter (mark, don't drop)"):
        _, quality_mask = apply_quality_filter(combined)
        n_pass = int(quality_mask.sum())
        logging.info(f"  universe-passing: {n_pass:,} / {len(combined):,} "
                     f"({100*n_pass/len(combined):.1f}%)")

    with Phase("Fill missing model features"):
        for c in raw_features:
            if c not in combined.columns:
                combined[c] = 0.0
        for c in xs_src:
            if c not in combined.columns:
                combined[c] = 0.0

    with Phase("Compute cross-sectional ranks (universe rows only)"):
        if xs_features:
            available_xs_src = [c for c in xs_src if c in combined.columns]
            xs_present = [c + "_xs" for c in available_xs_src]
            filtered = combined.loc[quality_mask, available_xs_src]
            grouped  = filtered.groupby(combined.loc[quality_mask, date_col], sort=False)
            ranks    = grouped.rank(pct=True, method="average", na_option="keep")
            ranks.columns = xs_present
            xs_arr = np.full((len(combined), len(xs_present)), np.float32(0.5), dtype=np.float32)
            xs_arr[quality_mask.values] = ranks[xs_present].values.astype(np.float32)
            xs_df = pd.DataFrame(xs_arr, columns=xs_present, index=combined.index)
            combined = pd.concat([combined, xs_df], axis=1)
            del xs_arr, xs_df
            logging.info(f"  computed xs ranks for {len(xs_present)} features")

    with Phase("Predict"):
        X = combined[feat_names].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        scores = clf.predict_proba(X)[:, 1].astype(np.float32)
        logging.info(f"  raw scores: min={scores.min():.4f} p50={np.median(scores):.4f} "
                     f"p95={np.percentile(scores, 95):.4f} max={scores.max():.4f}")
        if calibrator is not None:
            scores_cal = calibrator.predict(scores).astype(np.float32)
            logging.info(f"  calibrated: min={scores_cal.min():.4f} "
                         f"p50={np.median(scores_cal):.4f} "
                         f"p95={np.percentile(scores_cal, 95):.4f} "
                         f"max={scores_cal.max():.4f}")
        else:
            scores_cal = scores

    if upprob_mode == "ts_pctile":
        with Phase("Per-stock trailing percentile -> UpProbability"):
            # UpProbability = where today's score sits in THIS stock's own
            # trailing-window score history, mapped linearly to [0.30, 0.70].
            # can_buy's percentile-of-own-history gate then receives the
            # quantity it was designed for (high = unusually high for this name).
            ts_p = per_stock_trailing_percentile(
                scores_cal, combined[ticker_col].values,
                combined[date_col].values, ts_window, ts_min_periods)
            warm = np.isnan(ts_p)
            pct = np.nan_to_num(ts_p, nan=0.0).astype(np.float64)

            up_prob = 0.30 + 0.40 * pct
            up_prob[warm] = 0.30
            up_prob[~quality_mask.values] = 0.30   # suppress out-of-universe names

            # Mirror can_buy's sufficient-data gate (own-history [65th, 98th)
            # percentile). BT_PLOW/BT_PHIGH env overrides match the backtester.
            gate_lo = float(os.environ.get("BT_PLOW", 65.0)) / 100.0
            gate_hi = float(os.environ.get("BT_PHIGH", 98.0)) / 100.0
            in_gate = (~warm) & quality_mask.values & (pct >= gate_lo) & (pct < gate_hi)
            up_pred = np.where(quality_mask.values,
                               np.where(in_gate, 1, 0), -1).astype(np.int8)
            pos_threshold = np.float32(0.30 + 0.40 * gate_lo)

            n_fire = int((up_pred == 1).sum())
            n_days = combined[date_col].nunique()
            logging.info(f"  window={ts_window}  min_periods={ts_min_periods}  "
                         f"warmup rows: {int(warm.sum()):,} ({100*warm.mean():.1f}%)")
            logging.info(f"  in-gate rows (own-history pctile in "
                         f"[{gate_lo:.2f},{gate_hi:.2f})): {n_fire:,}  "
                         f"avg {n_fire/max(n_days,1):.1f}/day over {n_days} dates")
    else:
        with Phase("Rescale scores to backtester UpProbability (xs_rank legacy)"):
            pct_rank = np.zeros(len(combined), dtype=np.float32)
            filt_idx = np.where(quality_mask.values)[0]
            if len(filt_idx):
                df_filt = pd.DataFrame({
                    "s": scores_cal[filt_idx],   # rank on calibrated scores
                    "d": combined[date_col].values[filt_idx],
                })
                pr = df_filt.groupby("d", sort=False)["s"].rank(
                    pct=True, method="average").values
                pct_rank[filt_idx] = pr.astype(np.float32)

            up_prob = map_pct_rank_to_upprob(pct_rank, top_frac_per_day)
            up_prob[~quality_mask.values] = 0.30

            is_top  = pct_rank >= (1.0 - top_frac_per_day)
            up_pred = np.where(quality_mask.values & is_top, 1,
                               np.where(quality_mask.values, 0, -1)).astype(np.int8)
            pos_threshold = np.float32(1.0 - top_frac_per_day)

            n_fire = int((up_pred == 1).sum())
            n_days = combined[date_col].nunique()
            logging.info(f"  fires: {n_fire:,}  avg {n_fire/max(n_days,1):.1f}/day "
                         f"over {n_days} dates")

    with Phase("Write per-ticker parquets to RFpredictions"):
        combined["UpProbability"]     = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
        combined["DownProbability"]   = (1.0 - combined["UpProbability"]).astype(np.float32)
        combined["PositiveThreshold"] = pos_threshold
        combined["NegativeThreshold"] = np.float32(np.nan)
        combined["UpPrediction"]      = up_pred
        combined["raw_score"]         = scores
        if calibrator is not None:
            combined["cal_score"] = scores_cal

        base_out = [date_col, "Open", "High", "Low", "Close", "Volume",
                    "UpProbability", "DownProbability",
                    "PositiveThreshold", "NegativeThreshold",
                    "UpPrediction", "raw_score"]
        optional = ["VIX_Close", "Distance to Resistance (%)",
                    "Distance to Support (%)", "volatility"]
        out_cols = [c for c in base_out + optional if c in combined.columns]

        n_written = 0
        pbar = tqdm(total=combined[ticker_col].nunique(), desc="Writing")
        for tk, grp in combined.groupby(ticker_col, sort=False):
            try:
                grp[out_cols].to_parquet(
                    os.path.join(output_dir, f"{tk}.parquet"), index=False)
                n_written += 1
            except Exception as e:
                logging.warning(f"  failed {tk}: {e}")
            pbar.update(1)
        pbar.close()
        logging.info(f"  wrote {n_written:,} ticker parquets to {output_dir}")


# -------------------------------------------------------------------------- #
# OOS decay harness — measure true out-of-sample generalization              #
# -------------------------------------------------------------------------- #
# Rank bands are defined on the per-day percentile rank of the model score
# (0 = worst score that day, 1 = best). The strategy trades the top ~1%, so the
# headline is the top-1% band; the [0.90,0.95) "shoulder" band is tracked
# because prior decay logs found it the only zone whose edge survives OOS.
OOS_BANDS = [
    ("top 1%   [.99,1.0]", 0.99, 1.0001),
    ("1-2%     [.98,.99)", 0.98, 0.99),
    ("2-5%     [.95,.98)", 0.95, 0.98),
    ("5-7%     [.93,.95)", 0.93, 0.95),
    ("7-10%    [.90,.93)", 0.90, 0.93),
    ("10-20%   [.80,.90)", 0.80, 0.90),
    ("20-30%   [.70,.80)", 0.70, 0.80),
    ("30-50%   [.50,.70)", 0.50, 0.70),
    ("bot 50%  [.00,.50)", 0.00, 0.50),
]
# Highlighted bands for the headline decay comparison.
OOS_HEADLINE = [
    ("top1pct",  0.99, 1.0001),
    ("shoulder", 0.90, 0.95),
]


def oos_band_stats(split_df, lo, hi, ret_col, label_col, rank_col, date_col):
    """Per-band stats for one split. Returns a dict of metrics.

    per-day mean return = average across days of (that day's mean return for the
    band) — this is the strategy-relevant number (each day you hold the band).
    """
    m = (split_df[rank_col] >= lo) & (split_df[rank_col] < hi)
    sub = split_df.loc[m]
    n_days = split_df[date_col].nunique()
    if len(sub) == 0 or n_days == 0:
        return {"n": 0, "picks_per_day": 0.0, "pooled_ret": float("nan"),
                "perday_ret": float("nan"), "perday_sharpe": float("nan"),
                "precision": float("nan"), "hit": float("nan")}
    daily = sub.groupby(date_col)[ret_col].mean()
    pdm = float(daily.mean())
    pds = float(daily.std())
    return {
        "n": int(len(sub)),
        "picks_per_day": len(sub) / n_days,
        "pooled_ret": float(sub[ret_col].mean()) * 100.0,
        "perday_ret": pdm * 100.0,
        "perday_sharpe": (pdm / pds * np.sqrt(252)) if pds > 0 else float("nan"),
        "precision": float(sub[label_col].mean()),
        "hit": float((sub[ret_col] > 0).mean()),
    }


def oos_band_table_lines(split_df, name, ret_col, label_col, rank_col, date_col):
    n_days = split_df[date_col].nunique()
    d0, d1 = split_df[date_col].min(), split_df[date_col].max()
    lines = [
        f"--- {name}: {str(pd.Timestamp(d0).date())} -> {str(pd.Timestamp(d1).date())}"
        f"  ({len(split_df):,} rows, {n_days} days) ---",
        f"  {'band':<19} {'picks/d':>8} {'perday_ret%':>12} {'sharpe':>8} "
        f"{'pooled_ret%':>12} {'precision':>10} {'hit':>7}",
    ]
    for label, lo, hi in OOS_BANDS:
        s = oos_band_stats(split_df, lo, hi, ret_col, label_col, rank_col, date_col)
        lines.append(
            f"  {label:<19} {s['picks_per_day']:>8.1f} {s['perday_ret']:>12.4f} "
            f"{s['perday_sharpe']:>8.2f} {s['pooled_ret']:>12.4f} "
            f"{s['precision']:>10.4f} {s['hit']:>7.4f}")
    return lines


def run_oos_evaluation():
    """Load the saved model and quantify true out-of-sample decay.

    Re-derives the exact train/calib/OOS date boundaries (same row-percentile
    logic as time_split_with_embargo) so the OOS slice is precisely the rows the
    model never saw in train OR calib. Validates the reconstruction against the
    saved model's train_rows before trusting any number.
    """
    model_path   = os.path.join(args.model_dir, "xgb.joblib")
    summary_path = os.path.join(args.model_dir, "summary.json")
    out_report   = os.path.join(args.model_dir, "oos_report.txt")
    out_scores   = os.path.join(args.model_dir, "oos_scores.parquet")
    date_col, ticker_col, target_col = args.date_column, args.ticker_column, args.target_column

    if not os.path.exists(model_path):
        logging.error(f"No model at {model_path}. Train first.")
        return

    with Phase("OOS: load saved model"):
        clf = joblib_load(model_path)
        if not hasattr(clf, "feature_names_in_"):
            raise RuntimeError("Model has no feature_names_in_")
        feat_names   = list(clf.feature_names_in_)
        xs_features  = [c for c in feat_names if c.endswith("_xs")]
        raw_features = [c for c in feat_names if not c.endswith("_xs")]
        xs_src       = [c[:-3] for c in xs_features]
        logging.info(f"  model expects {len(feat_names)} features "
                     f"({len(raw_features)} raw + {len(xs_features)} xs)")

    expected_train_rows = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                expected_train_rows = json.load(f).get("train_rows")
        except Exception:
            pass

    with Phase("OOS: load + label all tickers"):
        parts = load_and_label_tickers(
            args.input_dir, target_col, date_col, args.horizon_5d,
            max_files=args.oos_max_tickers)
        if not parts:
            logging.error("No data loaded — check --input_dir")
            return
        df = pd.concat(parts, ignore_index=True)
        del parts
        df[date_col] = pd.to_datetime(df[date_col])
        logging.info(f"  loaded {df.shape[0]:,} rows")

    with Phase("OOS: universe filter (FilterRubric Step 1)"):
        df, _ = apply_quality_filter(df)
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

    # Reconstruct the exact train/calib boundaries used at training time.
    with Phase("OOS: reconstruct split boundaries"):
        n = len(df)
        train_end_row  = max(int(n * args.runpercent / 100) - 1, 0)
        train_end_date = pd.Timestamp(df[date_col].iloc[train_end_row])
        calib_start    = train_end_date + pd.Timedelta(days=args.embargo_days)
        pool           = df[df[date_col] >= calib_start]
        calib_target   = int(n * args.calibpercent / 100)
        calib_end_row  = min(calib_target, len(pool)) - 1
        calib_end_date = pd.Timestamp(pool[date_col].iloc[calib_end_row])
        recon_train_rows = int((df[date_col] <= train_end_date).sum())
        logging.info(f"  universe rows:    {n:,}")
        logging.info(f"  train end:        {train_end_date.date()}  "
                     f"(reconstructed train_rows={recon_train_rows:,})")
        logging.info(f"  calib window:     {calib_start.date()} -> {calib_end_date.date()}")
        logging.info(f"  OOS (true holdout): {calib_end_date.date()} (exclusive) onward")
        if expected_train_rows is not None:
            match = "EXACT" if recon_train_rows == expected_train_rows else "MISMATCH"
            logging.info(f"  boundary check vs summary.json train_rows="
                         f"{expected_train_rows:,}: {match}"
                         + ("" if match == "EXACT" else
                            "  (data snapshot changed since training — boundaries approximate)"))

    with Phase("OOS: build model feature matrix"):
        for c in raw_features:
            if c not in df.columns:
                df[c] = 0.0
        for c in xs_src:
            if c not in df.columns:
                df[c] = 0.0
        if xs_features:
            grouped = df.groupby(date_col, sort=False)[xs_src]
            ranks = grouped.rank(pct=True, method="average", na_option="keep")
            ranks.columns = [c + "_xs" for c in xs_src]
            df = pd.concat([df, ranks.astype(np.float32)], axis=1)
        X = df[feat_names].astype(np.float32).replace([np.inf, -np.inf], np.nan)

    with Phase("OOS: score + per-day rank + label"):
        df["_score"] = clf.predict_proba(X)[:, 1].astype(np.float32)
        df["_rank"]  = df.groupby(date_col)["_score"].rank(pct=True, method="average").astype(np.float32)
        df["_ret"]   = df[target_col].astype(np.float32)
        df["_label"] = topq_label(df[target_col].values, df[date_col].values,
                                  top_frac=args.topq_frac).astype(np.int8)

    # Slice into the three regimes.
    is_df    = df[df[date_col] <= train_end_date]
    calib_df = df[(df[date_col] >= calib_start) & (df[date_col] <= calib_end_date)]
    oos_df   = df[df[date_col] > calib_end_date]

    if len(oos_df) == 0:
        logging.error("  OOS slice is empty — no data after the calib window. "
                      "Lower --runpercent/--calibpercent or add fresher data.")
        return

    # Save per-row scores for downstream EDA without re-scoring.
    keep = [date_col, ticker_col, "_score", "_rank", "_ret", "_label"]
    keep = [c for c in keep if c in df.columns]
    scores_out = df[keep].copy()
    scores_out["split"] = np.where(
        df[date_col] <= train_end_date, "train",
        np.where(df[date_col] > calib_end_date, "oos",
                 np.where(df[date_col] >= calib_start, "calib", "embargo")))
    scores_out.to_parquet(out_scores, index=False)

    # Build the report.
    lines = ["=" * 92, "OOS DECAY REPORT — true out-of-sample generalization", "=" * 92,
             f"Model:    {model_path}",
             f"Features: {len(feat_names)}   Label: topq (top_frac={args.topq_frac})   "
             f"rank = per-day percentile of model score",
             ""]
    lines += oos_band_table_lines(is_df, "IS / TRAIN (in-sample — fit-optimistic)",
                                  "_ret", "_label", "_rank", date_col)
    lines.append("")
    lines += oos_band_table_lines(calib_df, "CALIB (near-OOS — used for threshold/calib only)",
                                  "_ret", "_label", "_rank", date_col)
    lines.append("")
    lines += oos_band_table_lines(oos_df, "OOS (TRUE HOLDOUT — never seen in train or calib)",
                                  "_ret", "_label", "_rank", date_col)

    # Headline decay: calib -> oos and train -> oos for the key bands.
    lines += ["", "=" * 92, "HEADLINE DECAY (per-day mean return %)", "=" * 92,
              f"  {'band':<10} {'IS/train':>10} {'calib':>10} {'OOS':>10} "
              f"{'calib->OOS':>12} {'decay%':>9}"]
    for name, lo, hi in OOS_HEADLINE:
        si = oos_band_stats(is_df,    lo, hi, "_ret", "_label", "_rank", date_col)
        sc = oos_band_stats(calib_df, lo, hi, "_ret", "_label", "_rank", date_col)
        so = oos_band_stats(oos_df,   lo, hi, "_ret", "_label", "_rank", date_col)
        drop_abs = so["perday_ret"] - sc["perday_ret"]
        drop_pct = (drop_abs / abs(sc["perday_ret"]) * 100.0
                    if sc["perday_ret"] not in (0.0,) and not np.isnan(sc["perday_ret"]) else float("nan"))
        lines.append(
            f"  {name:<10} {si['perday_ret']:>10.4f} {sc['perday_ret']:>10.4f} "
            f"{so['perday_ret']:>10.4f} {drop_abs:>12.4f} {drop_pct:>8.1f}%")
    lines += ["",
              "Read: if the top-1% OOS per-day return collapses toward 0 while the shoulder",
              "band holds, the model's tradeable edge is overfit to the top tail. The strategy",
              f"fires the top {args.top_frac_per_day*100:.0f}% (rank>={1-args.top_frac_per_day:.2f}).",
              "=" * 92]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  OOS report  -> {out_report}")
    logging.info(f"  OOS scores  -> {out_scores}  (per-row: score, rank, ret, label, split)")


# -------------------------------------------------------------------------- #
# Walk-forward OOS engine — judge configs across many independent windows     #
# -------------------------------------------------------------------------- #
def is_regime_feature(name):
    """True for features that may encode the prevailing market regime (and so
    risk memorizing the OLD regime rather than generalizing)."""
    n = name.lower()
    if n.endswith("_xs"):
        n = n[:-3]
    pats = ["regime", "beta_iwm", "beta_spy", "beta_qqq", "beta_dia",
            "hc_predict", "market_state", "vix_regime"]
    return any(p in n for p in pats)


def _wf_perday_ret(sub, date_col, ret_col="_ret"):
    """Mean across days of each day's mean return, in %."""
    if len(sub) == 0:
        return float("nan")
    return float(sub.groupby(date_col)[ret_col].mean().mean()) * 100.0


def wf_config_spec(name, df, date_col, sorted_dates, feat_names, anchor):
    """Return (train_mask, feature_subset, half_life_days) for a named config.

    train_mask selects rows used to fit; feature_subset the columns; half_life
    the recency-weight decay. Only the data/feature/weight treatment varies —
    XGB params are held fixed so the comparison isolates the regime-adaptation
    lever.
    """
    upto = df[date_col] <= anchor
    if name == "baseline":
        return upto, feat_names, 720.0
    if name in ("recent_252", "recent_126"):
        k = 252 if name == "recent_252" else 126
        prior = sorted_dates[sorted_dates <= anchor]
        floor = prior[-k] if len(prior) >= k else prior[0]
        return upto & (df[date_col] >= floor), feat_names, 720.0
    if name == "hl_180":
        return upto, feat_names, 180.0
    if name == "hl_90":
        return upto, feat_names, 90.0
    if name == "decontam":
        feats = [f for f in feat_names if not is_regime_feature(f)]
        return upto, feats, 720.0
    if name.startswith("stale_"):
        # Train through (anchor - N months) but still score the SAME OOS month
        # (anchor+1). Isolates the STALENESS penalty: identical features/params/
        # OOS window, only the training cutoff moves back. baseline = stale_0mo.
        n_mo = int(name.split("_")[1].replace("mo", ""))
        cutoff = anchor - pd.DateOffset(months=n_mo)
        return (df[date_col] <= cutoff), feat_names, 720.0
    raise ValueError(f"unknown wf config: {name}")


def run_walkforward_oos():
    """Load+feature-build once, then evaluate a suite of training configs across
    monthly walk-forward anchors. For each (config, anchor): fit on the config's
    training slice, score the FOLLOWING month (true OOS), and record the top-1%
    and shoulder-band per-day return NET of the bottom-50% market-drift baseline.
    """
    date_col, ticker_col, target_col = args.date_column, args.ticker_column, args.target_column
    out_report  = os.path.join(args.model_dir, "walkforward_report.txt")
    out_parquet = os.path.join(args.model_dir, "walkforward_results.parquet")
    configs = [c.strip() for c in args.wf_configs.split(",") if c.strip()]
    fixed_params = dict(max_depth=6, learning_rate=0.05, min_child_weight=5,
                        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5,
                        reg_lambda=2.0, tree_method="hist",
                        objective="binary:logistic", eval_metric="aucpr",
                        n_jobs=-1, random_state=42)

    # ---- load + filter + feature matrix + label (ONCE) ----
    with Phase("WF: load + label all tickers"):
        parts = load_and_label_tickers(args.input_dir, target_col, date_col,
                                       args.horizon_5d, max_files=args.wf_max_tickers)
        if not parts:
            logging.error("No data loaded — check --input_dir")
            return
        df = pd.concat(parts, ignore_index=True)
        del parts
        df[date_col] = pd.to_datetime(df[date_col])

    with Phase("WF: universe filter"):
        df, _ = apply_quality_filter(df)
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

    with Phase("WF: build feature matrix (base + xs ranks)"):
        base_cols = select_base_features(df)
        if USE_XS:
            df = add_xs_rank_features(df, base_cols, date_col)
            feat_names = base_cols + [c + "_xs" for c in base_cols]
        else:
            feat_names = base_cols
        logging.info(f"  features: {len(feat_names)}")

    with Phase("WF: topq label"):
        df["_label"] = topq_label(df[target_col].values, df[date_col].values,
                                  top_frac=args.topq_frac).astype(np.int8)
        df["_ret"] = df[target_col].astype(np.float32)

    # ---- anchors: each month m -> train_end = last date in m, OOS = month m+1 ----
    sorted_dates = np.sort(df[date_col].unique())
    df["_ym"] = df[date_col].dt.to_period("M")
    months = sorted(df["_ym"].unique())
    min_anchor = pd.Period(args.wf_min_anchor, freq="M")
    anchors = []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        if m < min_anchor:
            continue
        oos_rows = df[df["_ym"] == nxt]
        if oos_rows[date_col].nunique() < 5:
            continue
        anchor_date = df.loc[df["_ym"] == m, date_col].max()
        anchors.append((str(m), anchor_date, nxt))
    logging.info(f"  anchors: {len(anchors)}  "
                 f"({anchors[0][0]}->{anchors[-1][0]})  configs: {configs}")

    # Precompute per-config feature matrices as float32 numpy once? Keep as df;
    # XGB accepts the DataFrame slice directly. Fit cost dominates anyway.
    rows = []
    total = len(configs) * len(anchors)
    done = 0
    for cfg in configs:
        for ym, anchor_date, oos_period in anchors:
            done += 1
            train_mask, feats, half_life = wf_config_spec(
                cfg, df, date_col, sorted_dates, feat_names, anchor_date)
            tr = df.loc[train_mask]
            if len(tr) < 5000:
                logging.info(f"  [{done}/{total}] {cfg} @ {ym}: train too small "
                             f"({len(tr)}), skip")
                continue
            sw = recency_weights(tr[date_col], half_life)
            clf = XGBClassifier(n_estimators=args.wf_trees, **fixed_params)
            clf.fit(tr[feats].astype(np.float32).replace([np.inf, -np.inf], np.nan),
                    tr["_label"].values, sample_weight=sw)

            oos = df[df["_ym"] == oos_period].copy()
            oos["_score"] = clf.predict_proba(
                oos[feats].astype(np.float32).replace([np.inf, -np.inf], np.nan))[:, 1]
            oos["_rank"] = oos.groupby(date_col)["_score"].rank(pct=True, method="average")

            base = _wf_perday_ret(oos[oos["_rank"] < 0.50], date_col)
            top1 = _wf_perday_ret(oos[oos["_rank"] >= 0.99], date_col)
            shou = _wf_perday_ret(oos[(oos["_rank"] >= 0.90) & (oos["_rank"] < 0.95)], date_col)
            rows.append({
                "config": cfg, "anchor_month": ym,
                "oos_month": str(oos_period),
                "train_rows": int(len(tr)), "n_feats": len(feats),
                "oos_days": int(oos[date_col].nunique()),
                "base": base, "top1": top1, "shoulder": shou,
                "top1_net": top1 - base, "shoulder_net": shou - base,
            })
            logging.info(f"  [{done}/{total}] {cfg:<11} train_end {str(anchor_date.date())} "
                         f"-> OOS {str(oos_period)}: "
                         f"top1_net={top1-base:+.4f}  shoulder_net={shou-base:+.4f}")

    res = pd.DataFrame(rows)
    res.to_parquet(out_parquet, index=False)

    # ---- report ----
    lines = ["=" * 100,
             "WALK-FORWARD OOS — per-day mean return NET of bottom-50% market-drift baseline",
             "=" * 100,
             f"Configs: {configs}   anchors: {len(anchors)}   trees: {args.wf_trees} (fixed, no tune)",
             "Each row = train on config slice up to anchor month-end, OOS = the FOLLOWING month.",
             ""]
    # Per-config detail
    for cfg in configs:
        c = res[res["config"] == cfg].sort_values("oos_month")
        if len(c) == 0:
            continue
        lines.append(f"### {cfg} ###")
        lines.append(f"  {'OOS month':<10} {'train_rows':>11} {'days':>5} "
                     f"{'top1_net%':>10} {'shoulder_net%':>14}")
        for _, r in c.iterrows():
            lines.append(f"  {r['oos_month']:<10} {r['train_rows']:>11,} {r['oos_days']:>5} "
                         f"{r['top1_net']:>10.4f} {r['shoulder_net']:>14.4f}")
        lines.append("")
    # Summary comparison (pooled across anchors)
    lines += ["=" * 100, "CONFIG COMPARISON (pooled across OOS windows)", "=" * 100,
              f"  {'config':<12} {'n_win':>6} {'top1_net mean':>14} {'median':>9} "
              f"{'win%':>6} {'shoulder_net mean':>18} {'median':>9} {'win%':>6}"]
    summ = []
    for cfg in configs:
        c = res[res["config"] == cfg]
        if len(c) == 0:
            continue
        t = c["top1_net"]; s = c["shoulder_net"]
        summ.append((cfg, len(c), t.mean(), t.median(), (t > 0).mean(),
                     s.mean(), s.median(), (s > 0).mean()))
    # sort by pooled top1_net mean (strategy-relevant) descending
    for cfg, n, tm, tmd, tw, sm, smd, sw_ in sorted(summ, key=lambda x: -x[2]):
        lines.append(f"  {cfg:<12} {n:>6} {tm:>14.4f} {tmd:>9.4f} {tw*100:>5.0f}% "
                     f"{sm:>18.4f} {smd:>9.4f} {sw_*100:>5.0f}%")
    lines += ["",
              "top1_net = strategy-relevant (fires top ~1%). A config wins if it raises pooled",
              "top1_net mean AND win% vs baseline across independent OOS windows — not one month.",
              "=" * 100]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  WF report  -> {out_report}")
    logging.info(f"  WF results -> {out_parquet}")


# -------------------------------------------------------------------------- #
# Walk-forward PARAM SWEEP — parallel, multi-seed, isolates the overfit lever  #
# -------------------------------------------------------------------------- #
# All configs train on identical baseline data (all history up to anchor, all
# features, 720d half-life). Only XGB hyperparameters vary, so any difference is
# the params. prod_optuna = the live production Optuna config (reference to beat).
_SWEEP_BASE = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                   reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5, gamma=0.0)
SWEEP_CONFIGS = {
    "prod_optuna":  dict(n_estimators=304, max_depth=8, learning_rate=0.0509,
                         min_child_weight=15, subsample=0.886, colsample_bytree=0.444,
                         reg_alpha=0.0048, reg_lambda=0.0223, gamma=0.283),
    "d3":           dict(_SWEEP_BASE, n_estimators=300, max_depth=3),
    "d4":           dict(_SWEEP_BASE, n_estimators=300, max_depth=4),
    "d5":           dict(_SWEEP_BASE, n_estimators=300, max_depth=5),
    "d6":           dict(_SWEEP_BASE, n_estimators=300, max_depth=6),
    "d8_reg":       dict(_SWEEP_BASE, n_estimators=300, max_depth=8),   # deep but properly regularized (depth vs reg)
    "d4_strongreg": dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                         min_child_weight=30, subsample=0.7, colsample_bytree=0.5,
                         reg_alpha=2.0, reg_lambda=5.0, gamma=1.0),
    "d5_slow":      dict(n_estimators=600, max_depth=5, learning_rate=0.02,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.6,
                         reg_alpha=1.0, reg_lambda=3.0, gamma=0.0),
}


def _sweep_band_nets(scores, lo, hi, dates_all, ret_all):
    """Return (top1_net, shoulder_net) per-day-mean returns net of bot-50% baseline.
    Operates on the contiguous OOS row-range [lo:hi] (views, no copy)."""
    odf = pd.DataFrame({"d": dates_all[lo:hi], "r": ret_all[lo:hi], "s": scores})
    odf["rk"] = odf.groupby("d")["s"].rank(pct=True, method="average")
    def pdr(mask):
        sub = odf.loc[mask]
        return float(sub.groupby("d")["r"].mean().mean()) * 100.0 if len(sub) else float("nan")
    base = pdr(odf["rk"] < 0.50)
    top1 = pdr(odf["rk"] >= 0.99)
    shou = pdr((odf["rk"] >= 0.90) & (odf["rk"] < 0.95))
    return top1 - base, shou - base


def run_walkforward_sweep():
    """Parallel, multi-seed walk-forward sweep over XGB hyperparameter configs.

    Isolates the overfitting lever: every config trains on the SAME baseline data
    (all history up to each anchor, all features, 720d recency) and is scored on
    the following month; only the hyperparameters differ. Each (config, anchor) is
    fit with `wf_seeds` seeds and averaged to beat the ~0.02 hist nondeterminism.
    Fits run concurrently in threads (XGB releases the GIL).
    """
    date_col, target_col = args.date_column, args.target_column
    out_report  = os.path.join(args.model_dir, "walkforward_sweep_report.txt")
    out_parquet = os.path.join(args.model_dir, "walkforward_sweep_results.parquet")
    configs = [c.strip() for c in args.wf_sweep_configs.split(",") if c.strip()]
    for c in configs:
        if c not in SWEEP_CONFIGS:
            raise ValueError(f"unknown sweep config '{c}'. Known: {list(SWEEP_CONFIGS)}")

    with Phase("SWEEP: load + label all tickers"):
        parts = load_and_label_tickers(args.input_dir, target_col, date_col,
                                       args.horizon_5d, max_files=args.wf_max_tickers)
        if not parts:
            logging.error("No data loaded.")
            return
        df = pd.concat(parts, ignore_index=True)
        del parts
        df[date_col] = pd.to_datetime(df[date_col])

    with Phase("SWEEP: universe filter"):
        df, _ = apply_quality_filter(df)
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

    with Phase("SWEEP: build feature matrix (base + xs ranks)"):
        base_cols = select_base_features(df)
        if USE_XS:
            df = add_xs_rank_features(df, base_cols, date_col)
            feat_names = base_cols + [c + "_xs" for c in base_cols]
        else:
            feat_names = base_cols
        logging.info(f"  features: {len(feat_names)}")

    with Phase("SWEEP: arrays (one-time float32 cast)"):
        Xall = df[feat_names].to_numpy(np.float32)
        Xall[~np.isfinite(Xall)] = np.nan
        y_all = topq_label(df[target_col].values, df[date_col].values,
                           top_frac=args.topq_frac).astype(np.int8)
        ret_all   = df[target_col].to_numpy(np.float32)
        dates_all = df[date_col].values.astype("datetime64[ns]")
        ym_all    = df[date_col].dt.to_period("M").astype(str).to_numpy()
        logging.info(f"  Xall {Xall.shape} ~{Xall.nbytes/1e9:.1f}GB")

    # anchors: month m -> train<=last date of m, OOS = month m+1
    months = sorted(pd.unique(ym_all))
    min_anchor = str(pd.Period(args.wf_min_anchor, freq="M"))
    anchors = []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        if m < min_anchor:
            continue
        if (ym_all == nxt).sum() == 0:
            continue
        oos_days = len(np.unique(dates_all[ym_all == nxt]))
        if oos_days < 5:
            continue
        anchor_date = dates_all[ym_all == m].max()
        anchors.append((m, anchor_date, nxt))
    logging.info(f"  anchors: {len(anchors)} ({anchors[0][0]}->{anchors[-1][0]})")
    logging.info(f"  configs: {configs}  seeds: {args.wf_seeds}  "
                 f"workers: {args.wf_workers}x{args.wf_threads}thr  device: {args.wf_device}")

    # Precompute per-anchor: train prefix length k (data is date-sorted, so
    # date<=anchor is a contiguous prefix -> Xall[:k] is a VIEW, no copy), the
    # recency weights (depend only on anchor, not config/seed), and the contiguous
    # OOS row-range. This removes the GIL-held 1.15GB fancy-index copy that was
    # serializing the threaded fits.
    anchor_info = {}   # m -> (k, sw, oos_lo, oos_hi, oos_period)
    for (m, anchor_date, nxt) in anchors:
        k = int(np.searchsorted(dates_all, anchor_date, side="right"))
        sw = recency_weights(pd.Series(dates_all[:k]), 720.0)
        oi = np.where(ym_all == nxt)[0]
        anchor_info[m] = (k, sw, int(oi[0]), int(oi[-1]) + 1, nxt)

    jobs = [(cfg, m, seed)
            for cfg in configs
            for (m, _ad, _nxt) in anchors
            for seed in range(args.wf_seeds)]
    logging.info(f"  total fits: {len(jobs)}")

    def _one(job):
        cfg, m, seed = job
        k, sw, olo, ohi, oos_period = anchor_info[m]
        params = dict(SWEEP_CONFIGS[cfg])
        n_est = params.pop("n_estimators", 300)
        clf = XGBClassifier(n_estimators=n_est, tree_method="hist",
                            objective="binary:logistic", eval_metric="aucpr",
                            device=args.wf_device, n_jobs=args.wf_threads,
                            random_state=1000 + seed, **params)
        clf.fit(Xall[:k], y_all[:k], sample_weight=sw)      # views, zero-copy
        sc = clf.predict_proba(Xall[olo:ohi])[:, 1]
        t1n, shn = _sweep_band_nets(sc, olo, ohi, dates_all, ret_all)
        return {"config": cfg, "oos_month": oos_period, "seed": seed,
                "train_rows": int(k), "top1_net": t1n, "shoulder_net": shn}

    rows = []
    done = 0
    with Phase(f"SWEEP: {len(jobs)} fits ({args.wf_workers} concurrent)"):
        with ThreadPoolExecutor(max_workers=args.wf_workers) as ex:
            futs = {ex.submit(_one, j): j for j in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    r = fut.result()
                    rows.append(r)
                    if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                        logging.info(f"  [{done}/{len(jobs)}] {r['config']:<12} "
                                     f"{r['oos_month']} s{r['seed']}: top1_net={r['top1_net']:+.4f}")
                except Exception as e:
                    j = futs[fut]
                    logging.warning(f"  job {j[0]}@{j[1]} s{j[2]} failed: {repr(e)[:120]}")

    res = pd.DataFrame(rows)
    res.to_parquet(out_parquet, index=False)

    # Seed-average per (config, anchor), then pool across anchors.
    per_anchor = (res.groupby(["config", "oos_month"])
                     .agg(top1_net=("top1_net", "mean"),
                          top1_std=("top1_net", "std"),
                          shoulder_net=("shoulder_net", "mean"))
                     .reset_index())

    lines = ["=" * 104,
             "WALK-FORWARD PARAM SWEEP — seed-averaged top1_net (per-day mean ret NET of bot-50% baseline)",
             "=" * 104,
             f"configs={configs}  anchors={len(anchors)}  seeds={args.wf_seeds}  "
             f"(identical baseline data; only XGB params vary)",
             ""]
    # per-config per-anchor (seed-averaged) detail
    for cfg in configs:
        c = per_anchor[per_anchor["config"] == cfg].sort_values("oos_month")
        lines.append(f"### {cfg} ###   (params: {SWEEP_CONFIGS[cfg]})")
        lines.append(f"  {'OOS month':<10} {'top1_net%':>10} {'seed_std':>9} {'shoulder%':>10}")
        for _, r in c.iterrows():
            lines.append(f"  {r['oos_month']:<10} {r['top1_net']:>10.4f} "
                         f"{r['top1_std']:>9.4f} {r['shoulder_net']:>10.4f}")
        lines.append("")

    # pooled comparison
    lines += ["=" * 104, "CONFIG COMPARISON (pooled across anchors; seed-averaged per anchor)", "=" * 104,
              f"  {'config':<13} {'anchors':>7} {'top1_net mean':>14} {'median':>9} {'win%':>6} "
              f"{'anchor_std':>11} {'avg_seed_std':>13} {'shoulder mean':>14}"]
    summ = []
    for cfg in configs:
        c = per_anchor[per_anchor["config"] == cfg]
        if len(c) == 0:
            continue
        t = c["top1_net"]
        summ.append((cfg, len(c), t.mean(), t.median(), (t > 0).mean(),
                     t.std(), c["top1_std"].mean(), c["shoulder_net"].mean()))
    for cfg, n, tm, tmd, tw, tstd, sds, sm in sorted(summ, key=lambda x: -x[2]):
        lines.append(f"  {cfg:<13} {n:>7} {tm:>14.4f} {tmd:>9.4f} {tw*100:>5.0f}% "
                     f"{tstd:>11.4f} {sds:>13.4f} {sm:>14.4f}")
    lines += ["",
              "top1_net = strategy-relevant. avg_seed_std = the nondeterminism noise floor for a",
              "single fit; anchor_std = month-to-month dispersion. A config genuinely beats another",
              "only if the gap exceeds ~avg_seed_std/sqrt(seeds). prod_optuna = live config (to beat).",
              "=" * 104]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  SWEEP report  -> {out_report}")
    logging.info(f"  SWEEP results -> {out_parquet}")


# -------------------------------------------------------------------------- #
# Main                                                                       #
# -------------------------------------------------------------------------- #
def main():
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "xgb.joblib")

    # --inspect: just print cached report
    if args.inspect:
        summary_path = os.path.join(args.model_dir, "summary.json")
        report_path  = os.path.join(args.model_dir, "report.txt")
        if not os.path.exists(summary_path):
            logging.error(f"No cached summary at {summary_path}. Train first.")
            return
        with open(summary_path) as f:
            logging.info("Cached summary:\n" + json.dumps(json.load(f), indent=2))
        if os.path.exists(report_path):
            with open(report_path) as f:
                logging.info("\nCached report:\n" + f.read())
        return

    # --oos_only: measure true out-of-sample decay on the saved model, no retrain
    if args.oos_only:
        run_oos_evaluation()
        return

    # --walkforward: compare training configs across many independent OOS windows
    if args.walkforward:
        run_walkforward_oos()
        return

    # --wf_sweep: parallel multi-seed hyperparameter sweep (isolate overfit lever)
    if args.wf_sweep:
        run_walkforward_sweep()
        return

    run_t0 = time.time()

    # ------------------------------------------------------------------ #
    # TRAINING PHASES                                                     #
    # ------------------------------------------------------------------ #
    if not args.predict_only:

        prepared_dir   = os.path.join(args.model_dir, "PreparedData")
        train_cache    = os.path.join(prepared_dir, "train.parquet")
        calib_cache    = os.path.join(prepared_dir, "calib.parquet")
        reuse_ok       = (args.reuse and os.path.exists(train_cache)
                          and os.path.exists(calib_cache))

        if reuse_ok:
            with Phase("Phase 1-5: Load cached PreparedData splits (--reuse)"):
                train_df = pd.read_parquet(train_cache)
                calib_df = pd.read_parquet(calib_cache)
                train_df[args.date_column] = pd.to_datetime(train_df[args.date_column])
                calib_df[args.date_column] = pd.to_datetime(calib_df[args.date_column])
                logging.info(f"  train: {len(train_df):,} rows from {train_cache}")
                logging.info(f"  calib: {len(calib_df):,} rows from {calib_cache}")
        else:
            # Phase 1+2: Load tickers + label engineering
            with Phase("Phase 1+2: Load tickers + label engineering"):
                parts = load_and_label_tickers(
                    args.input_dir, args.target_column, args.date_column,
                    args.horizon_5d, max_files=args.max_files)
                logging.info(f"  loaded {len(parts)} tickers, "
                             f"{sum(len(p) for p in parts):,} rows total")

            with Phase("Concatenate"):
                if not parts:
                    logging.error("No data loaded — check --input_dir")
                    return
                df = pd.concat(parts, ignore_index=True)
                del parts
                df[args.date_column] = pd.to_datetime(df[args.date_column])
                logging.info(f"  shape={df.shape}, mem={mem_gb(df):.2f} GB")

            # Phase 3: Universe filter
            with Phase("Phase 3: Universe filter (FilterRubric Step 1)"):
                df, _ = apply_quality_filter(df)

            # Phase 4: Shuffle within each date
            with Phase("Phase 4: Shuffle within each date"):
                df = df.sort_values(args.date_column, kind="stable").reset_index(drop=True)
                df = shuffle_within_date(df, args.date_column, args.seed)

            # Phase 5: Train/calib split
            with Phase("Phase 5: Train/calib split with embargo"):
                train_df, calib_df, split_meta = time_split_with_embargo(
                    df, args.runpercent, args.calibpercent,
                    args.embargo_days, args.date_column,
                    train_end_date=args.train_end_date)
                del df

            # Save splits for --reuse on next run
            os.makedirs(prepared_dir, exist_ok=True)
            train_df.to_parquet(train_cache, index=False)
            calib_df.to_parquet(calib_cache, index=False)
            logging.info(f"  PreparedData cached -> {prepared_dir}/")

        # Phase 6: Feature matrix
        with Phase("Phase 6: Build feature matrix"):
            base_cols = select_base_features(train_df)
            logging.info(f"  base numeric features: {len(base_cols)}")

            if args.drop_vol_features:
                dropped = [c for c in base_cols if is_vol_feature(c)]
                base_cols = [c for c in base_cols if not is_vol_feature(c)]
                logging.info(f"  --drop_vol_features removed {len(dropped)} cols "
                             f"(WARNING: vol features are top-ranked in winning config)")

            if args.drop_feature_patterns:
                pats = [p.strip() for p in args.drop_feature_patterns.split(",") if p.strip()]
                dropped = [c for c in base_cols if any(p in c for p in pats)]
                base_cols = [c for c in base_cols if not any(p in c for p in pats)]
                logging.info(f"  drop_feature_patterns removed {len(dropped)} cols")

            if args.drop_features_exact:
                exact = {n.strip() for n in args.drop_features_exact.split(",") if n.strip()}
                dropped = [c for c in base_cols if c in exact]
                base_cols = [c for c in base_cols if c not in exact]
                logging.info(f"  drop_features_exact removed {len(dropped)} cols: {dropped}")

            if USE_XS:
                logging.info("  --add_xs_features: appending per-day rank columns")
                train_df = add_xs_rank_features(train_df, base_cols, args.date_column)
                calib_df = add_xs_rank_features(calib_df, base_cols, args.date_column)
                feature_cols = base_cols + [c + "_xs" for c in base_cols]
            else:
                feature_cols = base_cols

            logging.info(f"  total features: {len(feature_cols)}")
            X_train = train_df[feature_cols]
            X_calib = calib_df[feature_cols]

        # Phase 7: Labels
        with Phase("Phase 7: Build labels"):
            y_train, y_calib = build_labels(
                train_df, calib_df,
                label_mode=args.label_mode,
                target_col=args.target_column,
                date_col=args.date_column,
                topq_frac=args.topq_frac,
                vol_col=args.vol_col,
                vol_floor=args.vol_floor,
            )

        # Phase 8: Recency weights
        sw_train_full = None
        if args.recency_half_life_days:
            with Phase("Phase 8: Recency weights"):
                sw_train_full = recency_weights(
                    train_df[args.date_column], args.recency_half_life_days)
                logging.info(f"  half_life={args.recency_half_life_days}d  "
                             f"newest:oldest ratio = "
                             f"{sw_train_full.max()/max(sw_train_full.min(),1e-9):.1f}x")

        # Phase 9: Optuna tuning
        tuned_params = None
        if TUNE:
            with Phase(f"Phase 9: Optuna tuning "
                       f"({args.n_trials} trials, {args.tune_objective})"):
                ret_full = train_df[args.target_column].astype(np.float32).values
                d_full   = train_df[args.date_column].values
                tuned_params, best_val, _ = run_optuna_tuning(
                    X_train=X_train, y_train=y_train,
                    ret_train=ret_full, dates_train=d_full,
                    sw_train=sw_train_full,
                    n_trials=args.n_trials,
                    objective_name=args.tune_objective,
                    tune_subsample=args.tune_subsample,
                )
                logging.info(f"  inner-val {args.tune_objective} = {best_val:.4f}")
        else:
            logging.info("  --no_tune: skipping Optuna, using default XGB params")

        # Phase 10: Train final model
        with Phase("Phase 10: Train final XGBClassifier"):
            clf = train_model(
                X_train, y_train, tuned_params, sw_train_full,
                scale_pos_weight=args.scale_pos_weight,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                min_child_weight=args.min_child_weight,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                early_stopping_rounds=args.early_stopping_rounds,
            )

        # Phase 11: Evaluate + save
        with Phase("Phase 11: Evaluate on calib slice + save model"):
            model_path, scores_path, summary_path, calib_path = evaluate_and_save(
                clf, X_calib, y_calib, calib_df, feature_cols,
                train_df, args.model_dir, tuned_params,
                label_mode=args.label_mode,
                topq_frac=args.topq_frac,
                vol_col=args.vol_col,
                target_col=args.target_column,
                date_col=args.date_column,
                ticker_col=args.ticker_column,
                use_calib=USE_CALIB,
                target_precision=args.target_precision,
                max_coverage=args.max_coverage,
            )
            logging.info(f"  model saved -> {model_path}")
            logging.info(f"  calib scores -> {scores_path}")
            logging.info(f"  summary -> {summary_path}")
            if USE_CALIB:
                logging.info(f"  calibrator -> {calib_path}")

    # ------------------------------------------------------------------ #
    # Phase 12: Inference                                                 #
    # ------------------------------------------------------------------ #
    # When --predict_only is set, calib_path was never assigned above.
    if args.predict_only:
        calib_path = os.path.join(args.model_dir, "calibrator.joblib")

    with Phase("Phase 12: Inference — score all tickers -> RFpredictions"):
        run_inference(
            model_path=model_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            date_col=args.date_column,
            ticker_col=args.ticker_column,
            top_frac_per_day=args.top_frac_per_day,
            max_files=args.max_files,
            calib_path=calib_path if USE_CALIB else None,
            upprob_mode=args.upprob_mode,
            ts_window=args.ts_window,
            ts_min_periods=args.ts_min_periods,
        )

    total_min = (time.time() - run_t0) / 60
    logging.info(f"\n{'='*70}")
    logging.info(f"Pipeline done in {total_min:.1f}m")
    logging.info(f"  Model:          {model_path}")
    logging.info(f"  RFpredictions:  {args.output_dir}/")
    logging.info(f"  Next step:      python 5__NightlyBackTester.py --force")
    logging.info(f"{'='*70}")


if __name__ == "__main__":
    main()