"""Very simple XGBoost predictor.

Reads the prepared train/calib parquets produced by prepare_data.py and
trains ONE XGBClassifier on (y = next-day return > 0).  No stacking, no
ranker, no cross-sectional ranks, no consensus AND-gate, no conformal.

The point of this script is to establish a clean baseline.  If a simple
XGB beats the elaborate pipeline we built earlier, the elaborate pipeline
has problems.  If it doesn't beat it either, the issue is in the data or
the feature set, not the modeling architecture.

Inputs (from prepare_data.py):
    Data/PreparedData/train.parquet
    Data/PreparedData/calib.parquet

Outputs:
    Data/SimpleModel/xgb.joblib       — trained model
    Data/SimpleModel/calib_scores.parquet — Date, Ticker, score, label
    Data/SimpleModel/summary.json     — metrics + run config
    Data/SimpleModel/report.txt       — human-readable diagnostic

Run:
    python simple_xgb.py

Inspect:
    python simple_xgb.py --inspect
"""

import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import pandas as pd
from joblib import dump, load

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="Data/PreparedData")
parser.add_argument("--output_dir", default="Data/SimpleModel")
parser.add_argument("--target_column", default="percent_change_Close")
parser.add_argument("--date_column", default="Date")
parser.add_argument("--ticker_column", default="Ticker")
parser.add_argument("--n_estimators", type=int, default=500)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--min_child_weight", type=int, default=5,
                    help="Minimum sum of instance weight in a child node. "
                         "Higher = more conservative splits.")
parser.add_argument("--reg_alpha", type=float, default=0.5,
                    help="L1 regularisation on leaf weights.")
parser.add_argument("--reg_lambda", type=float, default=2.0,
                    help="L2 regularisation on leaf weights.")
parser.add_argument("--subsample", type=float, default=0.8)
parser.add_argument("--colsample_bytree", type=float, default=0.6)
parser.add_argument("--early_stopping_rounds", type=int, default=30)
parser.add_argument("--recency_half_life_days", type=float, default=None,
                    help="If set, weight each training sample by 0.5^(age/half_life). "
                         "Recent samples count more in gradient.  Try 180.")
parser.add_argument("--recent_only_days", type=int, default=None,
                    help="If set, hard-cut train data to the most recent N calendar days "
                         "before training.  E.g. 120 = roughly 5 months.  Aggressive "
                         "recency bias; use instead of or alongside --recency_half_life_days.")
parser.add_argument("--label_mode",
                    choices=["binary_up", "topq", "risk_adj_topq"],
                    default="risk_adj_topq",
                    help="binary_up: y = (next-day return > 0). "
                         "topq: y = (in per-day top quintile of next-day return). "
                         "risk_adj_topq: y = (in per-day top quintile of "
                         "next-day return / Realized_Vol_21d).")
parser.add_argument("--topq_frac", type=float, default=0.20,
                    help="Fraction defining the per-day top quantile.")
parser.add_argument("--vol_col", default="Realized_Vol_21d",
                    help="Column used as the volatility denominator for "
                         "risk_adj_topq.")
parser.add_argument("--vol_floor", type=float, default=1e-3,
                    help="Lower bound on volatility to prevent divide-by-zero.")
parser.add_argument("--add_xs_features", action="store_true",
                    help="Add a per-day percentile-rank column for each numeric "
                         "feature (doubles feature count).")
parser.add_argument("--drop_vol_features", action="store_true",
                    help="Drop volatility-family features (ATR, Realized_Vol, "
                         "cv_*, HC_Ratio, *_close_ratio, percent_range, etc.) "
                         "before training.  Diagnostic: tests whether the "
                         "model has any non-volatility signal.")
parser.add_argument("--scale_pos_weight", type=float, default=None,
                    help="Override XGB scale_pos_weight (default=1).")
parser.add_argument("--tune", action="store_true",
                    help="Run Optuna hyperparameter search (n_trials) before "
                         "final training.  Objective = per-day top-1% precision "
                         "on an inner walk-forward validation slice of train.")
parser.add_argument("--n_trials", type=int, default=100,
                    help="Number of Optuna trials when --tune is set.")
parser.add_argument("--tune_subsample", type=float, default=0.35,
                    help="Row fraction sampled FOR EACH TRIAL during tuning "
                         "(time-ordered tail kept for inner-val).  Lowers "
                         "wall-clock per trial.  Final fit still uses full "
                         "train.  Set to 1.0 to disable.")
parser.add_argument("--tune_objective",
                    choices=["top1_prec", "top1_meanret", "aucpr"],
                    default="top1_meanret",
                    help="Optuna objective: top-1%% per-day precision, top-1%% "
                         "per-day mean realised return, or overall AUC-PR.")
parser.add_argument("--inspect", action="store_true")
args = parser.parse_args()


# -------------------------------------------------------------------------- #
class Phase:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time()
        logging.info(f">>> {self.name} ...")
        return self
    def __exit__(self, *exc):
        logging.info(f"<<< {self.name} done in {time.time()-self.t0:.1f}s.")


def select_features(df, exclude):
    """Return numeric columns excluding date / ticker / labels / OHLCV passthrough."""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def is_vol_feature(name):
    """Heuristic: True for volatility-family feature names.

    Strips a trailing '_xs' so the same patterns match raw and xs versions.
    """
    import re
    n = name.lower()
    if n.endswith("_xs"):
        n = n[:-3]
    patterns = [
        r"^atr(_|$|\d|%)",          # atr_14, atr_percentage, ATR%
        r"realized_vol",
        r"^cv_\d",                   # cv_10d, cv_20d, cv_50d (+ variants)
        r"hc_ratio",
        r"high_close_ratio",
        r"low_close_ratio",
        r"intraday_range",
        r"percent_range",
        r"^volatility",
        r"_volatility",
        r"vol_v\d",                  # stress_vol_v7
        r"vix_vs_realized",
        r"vix_adjusted_atr",
        r"^pct_change_std",
    ]
    return any(re.search(p, n) for p in patterns)


def per_day_top_k_precision(scores, labels, dates, k_frac=0.01):
    """Average per-day precision in the top k_frac of model scores."""
    s = pd.DataFrame({"s": scores, "y": labels, "d": pd.Series(dates).values})
    precs, daily_n = [], []
    for _, g in s.groupby("d", sort=False):
        if len(g) < 20:
            continue
        n_top = max(int(len(g) * k_frac), 1)
        top = g.nlargest(n_top, "s")
        precs.append(top["y"].mean())
        daily_n.append(n_top)
    if not precs:
        return float("nan"), 0
    return float(np.mean(precs)), int(np.mean(daily_n))


def recency_weights(dates, half_life_days):
    """Exponential-decay weights as a function of sample age.

    Most-recent sample weight = 1.  A sample `half_life_days` older has weight
    0.5.  Two half-lives older = 0.25.  Output is mean-normalised so the
    average weight is ~1 (keeps loss scale comparable to unweighted training).
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    age = (dates.max() - dates).dt.days.values.astype(np.float64)
    w = np.power(0.5, age / max(half_life_days, 1.0))
    return (w / w.mean()).astype(np.float32)


def topq_label(returns, dates, top_frac=0.20):
    """Per-day top-quantile binary label.

    For each Date, label = 1 if the row's return is in the top `top_frac` of
    that day's cross-section.  Base rate is exactly `top_frac` by construction.
    """
    s = pd.Series(returns).reset_index(drop=True)
    d = pd.Series(dates).reset_index(drop=True)
    cutoff = s.groupby(d).transform(lambda x: x.quantile(1 - top_frac))
    return (s >= cutoff).astype(int).values


def add_xs_rank_features(df, feature_cols, date_col, batch_size=40):
    """Add a per-day percentile-rank column for each numeric feature.

    For every Date, every feature is also represented by its [0, 1] rank in
    that day's cross-section.  Market-wide features (same value for every
    ticker on a date) end up with all-tied ranks at 0.5, which XGB ignores
    on its own -- no explicit filter needed.

    Done in feature batches for visibility (one big groupby.rank on 300+
    columns is opaque and slow).
    """
    n_feat = len(feature_cols)
    logging.info(f"  computing xs rank for {n_feat} features over "
                 f"{df[date_col].nunique():,} dates (batches of {batch_size})...")
    grouped = df.groupby(date_col, sort=False)
    out_chunks = []
    n_batches = (n_feat + batch_size - 1) // batch_size
    for i in range(0, n_feat, batch_size):
        batch = feature_cols[i:i + batch_size]
        t0 = time.time()
        ranks = grouped[batch].rank(pct=True, method="average", na_option="keep")
        ranks.columns = [c + "_xs" for c in batch]
        out_chunks.append(ranks.astype(np.float32))
        logging.info(f"    batch {len(out_chunks)}/{n_batches}: "
                     f"{len(batch)} features in {time.time()-t0:.1f}s")
    return pd.concat([df] + out_chunks, axis=1)


def risk_adj_topq_label(returns, vol, dates, top_frac=0.20, vol_floor=1e-3):
    """Per-day top-quantile of risk-adjusted next-day return.

    risk_adj = return / max(vol, vol_floor).  Label = 1 if a row's risk_adj is
    in the top `top_frac` of that day's cross-section.  This penalises high-
    volatility stocks: they need an outsized move to qualify, while stable
    stocks with modest gains can.
    """
    r = pd.Series(returns).reset_index(drop=True)
    v = pd.Series(vol).reset_index(drop=True).clip(lower=vol_floor)
    d = pd.Series(dates).reset_index(drop=True)
    rar = r / v
    cutoff = rar.groupby(d).transform(lambda x: x.quantile(1 - top_frac))
    return (rar >= cutoff).astype(int).values


def per_day_top_k_mean_return(scores, returns, dates, k_frac=0.01):
    """Mean realised return on the top k_frac of model scores, per day, averaged."""
    s = pd.DataFrame({"s": scores, "r": returns, "d": pd.Series(dates).values})
    rets = []
    for _, g in s.groupby("d", sort=False):
        if len(g) < 20:
            continue
        n_top = max(int(len(g) * k_frac), 1)
        top = g.nlargest(n_top, "s")
        rets.append(top["r"].mean())
    if not rets:
        return float("nan")
    return float(np.mean(rets))


def run_optuna_tuning(X_train, y_train, ret_train, dates_train, sw_train,
                      n_trials, objective_name, tune_subsample=1.0):
    """Optuna search using inner walk-forward split of train.

    Inner split: first 80% as train, last 20% as validation (time-ordered, so
    this preserves a temporal gap).  Objective evaluated on the inner-val slice.

    `tune_subsample` < 1.0 keeps only the most-recent fraction of the inner-
    train rows (the val slice is kept whole so the objective stays comparable).
    Final fit uses full train.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = len(X_train)
    cut = int(n * 0.80)
    X_inner_tr = X_train.iloc[:cut]
    X_inner_val = X_train.iloc[cut:]
    y_inner_tr = y_train[:cut]
    y_inner_val = y_train[cut:]
    ret_inner_val = ret_train[cut:] if ret_train is not None else None
    d_inner_val = dates_train[cut:]
    sw_inner_tr = sw_train[:cut] if sw_train is not None else None
    sw_inner_val = sw_train[cut:] if sw_train is not None else None

    if tune_subsample < 1.0:
        keep = int(len(X_inner_tr) * tune_subsample)
        start = len(X_inner_tr) - keep
        X_inner_tr = X_inner_tr.iloc[start:]
        y_inner_tr = y_inner_tr[start:]
        if sw_inner_tr is not None:
            sw_inner_tr = sw_inner_tr[start:]
        logging.info(f"  tune_subsample={tune_subsample}: "
                     f"inner-train trimmed to most-recent {keep:,} rows")

    logging.info(f"  inner-tune split: tr={len(X_inner_tr):,}  val={len(X_inner_val):,}")
    logging.info(f"  optuna objective: {objective_name}")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        )
        m = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=30,
            verbosity=0,
            **params,
        )
        fit_kwargs = dict(
            eval_set=[(X_inner_val, y_inner_val)],
            verbose=False,
        )
        if sw_inner_tr is not None:
            fit_kwargs["sample_weight"] = sw_inner_tr
            fit_kwargs["sample_weight_eval_set"] = [sw_inner_val]
        m.fit(X_inner_tr, y_inner_tr, **fit_kwargs)
        p = m.predict_proba(X_inner_val)[:, 1]

        if objective_name == "top1_prec":
            score, _ = per_day_top_k_precision(p, y_inner_val, d_inner_val, k_frac=0.01)
        elif objective_name == "top1_meanret":
            if ret_inner_val is None:
                raise ValueError("top1_meanret objective requires return column")
            score = per_day_top_k_mean_return(p, ret_inner_val, d_inner_val, k_frac=0.01)
        else:  # aucpr
            score = float(average_precision_score(y_inner_val, p))

        trial.set_user_attr("best_iter", int(m.best_iteration))
        return score if not np.isnan(score) else -1.0

    def _cb(study, trial):
        bi = trial.user_attrs.get("best_iter", -1)
        logging.info(f"  trial {trial.number+1}/{n_trials} done: "
                     f"value={trial.value:.4f}  best_iter={bi}  "
                     f"best_so_far={study.best_value:.4f}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

    logging.info(f"  best trial: #{study.best_trial.number} "
                 f"value={study.best_value:.4f}")
    logging.info(f"  best params: {study.best_params}")
    return study.best_params, study.best_value, study.best_trial.user_attrs.get("best_iter", None)


# -------------------------------------------------------------------------- #
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "xgb.joblib")
    scores_path = os.path.join(args.output_dir, "calib_scores.parquet")
    summary_path = os.path.join(args.output_dir, "summary.json")
    report_path = os.path.join(args.output_dir, "report.txt")

    if args.inspect:
        if not os.path.exists(summary_path):
            logging.error(f"No cached summary at {summary_path}. Train first.")
            return
        with open(summary_path) as f:
            logging.info("Cached summary:\n" + json.dumps(json.load(f), indent=2))
        if os.path.exists(report_path):
            with open(report_path) as f:
                logging.info(f"\nCached report:\n{f.read()}")
        return

    run_t0 = time.time()

    with Phase("Load train + calib parquets"):
        train_path = os.path.join(args.input_dir, "train.parquet")
        calib_path = os.path.join(args.input_dir, "calib.parquet")
        if not (os.path.exists(train_path) and os.path.exists(calib_path)):
            logging.error(f"Missing prepared data at {args.input_dir}.")
            logging.error("Run `python prepare_data.py` first.")
            return
        train_df = pd.read_parquet(train_path)
        calib_df = pd.read_parquet(calib_path)
        logging.info(f"  train: {train_df.shape}, calib: {calib_df.shape}")

    with Phase("Build feature matrix + labels"):
        if args.recent_only_days is not None:
            dates_train = pd.to_datetime(train_df[args.date_column])
            cutoff = dates_train.max() - pd.Timedelta(days=args.recent_only_days)
            before = len(train_df)
            train_df = train_df[dates_train >= cutoff].reset_index(drop=True)
            logging.info(f"  --recent_only_days={args.recent_only_days}: "
                         f"kept {len(train_df):,} rows (dropped {before-len(train_df):,}) "
                         f"from {cutoff.date()} onward")

        exclude = {args.target_column, "ret_5d",
                   args.date_column, args.ticker_column,
                   "Open", "High", "Low", "Close", "Volume"}
        base_feature_cols = select_features(train_df, exclude)
        logging.info(f"  base numeric features: {len(base_feature_cols)}")

        if args.drop_vol_features:
            dropped = [c for c in base_feature_cols if is_vol_feature(c)]
            base_feature_cols = [c for c in base_feature_cols if not is_vol_feature(c)]
            logging.info(f"  --drop_vol_features: removed {len(dropped)} cols "
                         f"(examples: {dropped[:8]})")
            logging.info(f"  remaining base features: {len(base_feature_cols)}")

        if args.add_xs_features:
            logging.info("  --add_xs_features: appending per-day rank columns")
            t0 = time.time()
            train_df = add_xs_rank_features(train_df, base_feature_cols, args.date_column)
            calib_df = add_xs_rank_features(calib_df, base_feature_cols, args.date_column)
            logging.info(f"  xs-rank features added in {time.time()-t0:.1f}s "
                         f"(train shape now {train_df.shape})")
            feature_cols = base_feature_cols + [c + "_xs" for c in base_feature_cols]
        else:
            feature_cols = base_feature_cols

        logging.info(f"  total feature count: {len(feature_cols)}")
        X_train = train_df[feature_cols]
        X_calib = calib_df[feature_cols]

        if args.label_mode == "binary_up":
            y_train = (train_df[args.target_column] > 0).astype(int).values
            y_calib = (calib_df[args.target_column] > 0).astype(int).values
            logging.info(f"  label = (next-day return > 0)")
        elif args.label_mode == "topq":
            y_train = topq_label(train_df[args.target_column].values,
                                 train_df[args.date_column].values,
                                 top_frac=args.topq_frac)
            y_calib = topq_label(calib_df[args.target_column].values,
                                 calib_df[args.date_column].values,
                                 top_frac=args.topq_frac)
            logging.info(f"  label = (next-day return in per-day top "
                         f"{args.topq_frac*100:.0f}%)")
        elif args.label_mode == "risk_adj_topq":
            if args.vol_col not in train_df.columns or args.vol_col not in calib_df.columns:
                raise ValueError(f"vol column '{args.vol_col}' missing -- "
                                 "can't compute risk-adjusted label")
            y_train = risk_adj_topq_label(
                train_df[args.target_column].values,
                train_df[args.vol_col].values,
                train_df[args.date_column].values,
                top_frac=args.topq_frac, vol_floor=args.vol_floor)
            y_calib = risk_adj_topq_label(
                calib_df[args.target_column].values,
                calib_df[args.vol_col].values,
                calib_df[args.date_column].values,
                top_frac=args.topq_frac, vol_floor=args.vol_floor)
            logging.info(f"  label = (next-day return / {args.vol_col} in "
                         f"per-day top {args.topq_frac*100:.0f}%)")
        else:
            raise ValueError(f"unknown label_mode: {args.label_mode}")

        logging.info(f"  P(y=1) in train: {y_train.mean():.4f}")
        logging.info(f"  P(y=1) in calib: {y_calib.mean():.4f}")

    with Phase("Train XGBClassifier (single model, no tuning)"):
        # Use the last 20% of train as internal val for early stopping
        n_tr = len(X_train)
        val_start = int(n_tr * 0.80)
        spw = args.scale_pos_weight if args.scale_pos_weight is not None else 1.0

        # Recency-weighted training: recent samples count more in the gradient.
        # Computed AFTER sorting by date in prepare_data, so position == time order.
        sw_train = sw_val = None
        if args.recency_half_life_days:
            w_all = recency_weights(train_df[args.date_column],
                                    args.recency_half_life_days)
            sw_train = w_all[:val_start]
            sw_val = w_all[val_start:]
            logging.info(f"  recency weights: half_life={args.recency_half_life_days}d, "
                         f"min={w_all.min():.4f} max={w_all.max():.4f} "
                         f"(ratio newest:oldest = {w_all.max()/max(w_all.min(),1e-9):.1f}x)")

        tuned_params = None
        if args.tune:
            with Phase(f"Optuna tuning ({args.n_trials} trials, "
                       f"objective={args.tune_objective})"):
                ret_train_full = train_df[args.target_column].astype(np.float32).values
                d_train_full = train_df[args.date_column].values
                tuned_params, best_val, best_iter = run_optuna_tuning(
                    X_train=X_train,
                    y_train=y_train,
                    ret_train=ret_train_full,
                    dates_train=d_train_full,
                    sw_train=(recency_weights(train_df[args.date_column],
                                              args.recency_half_life_days)
                              if args.recency_half_life_days else None),
                    n_trials=args.n_trials,
                    objective_name=args.tune_objective,
                    tune_subsample=args.tune_subsample,
                )
                logging.info(f"  using tuned params (inner-val "
                             f"{args.tune_objective}={best_val:.4f})")

        if tuned_params is not None:
            xgb_params = dict(tuned_params)
        else:
            xgb_params = dict(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                min_child_weight=args.min_child_weight,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
            )

        logging.info(f"  XGB params: {xgb_params}")
        clf = XGBClassifier(
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=args.early_stopping_rounds,
            verbosity=1,
            **xgb_params,
        )
        fit_kwargs = dict(
            eval_set=[(X_train.iloc[val_start:], y_train[val_start:])],
            verbose=False,
        )
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
            fit_kwargs["sample_weight_eval_set"] = [sw_val]
        clf.fit(X_train.iloc[:val_start], y_train[:val_start], **fit_kwargs)
        logging.info(f"  trained {clf.best_iteration+1} trees "
                     f"(early stopped from {args.n_estimators})")

    with Phase("Score calibration slice + compute metrics"):
        p_calib = clf.predict_proba(X_calib)[:, 1]
        d_calib = calib_df[args.date_column].values

        auc = float(roc_auc_score(y_calib, p_calib))
        aucpr = float(average_precision_score(y_calib, p_calib))
        top1_prec, top1_n = per_day_top_k_precision(p_calib, y_calib, d_calib, k_frac=0.01)
        top5_prec, top5_n = per_day_top_k_precision(p_calib, y_calib, d_calib, k_frac=0.05)

        # Fixed-threshold metrics for reference
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        thresh_metrics = {}
        for t in thresholds:
            fire = p_calib >= t
            if fire.sum() < 10:
                thresh_metrics[t] = {"fires": int(fire.sum()),
                                     "precision": float("nan"),
                                     "fires_per_day": float(fire.sum() / max(calib_df[args.date_column].nunique(), 1))}
                continue
            prec = float(precision_score(y_calib, fire, zero_division=0))
            thresh_metrics[t] = {
                "fires": int(fire.sum()),
                "precision": prec,
                "fires_per_day": float(fire.sum() / max(calib_df[args.date_column].nunique(), 1)),
            }

        # Save per-row scores so user can audit / feed elsewhere
        out_cols = [args.date_column]
        if args.ticker_column in calib_df.columns:
            out_cols.append(args.ticker_column)
        out_cols += ["Close"] if "Close" in calib_df.columns else []
        scores_df = calib_df[out_cols].copy()
        scores_df["score"] = p_calib.astype(np.float32)
        scores_df["label_up"] = y_calib.astype(np.int8)
        scores_df["actual_return"] = calib_df[args.target_column].astype(np.float32).values
        scores_df.to_parquet(scores_path, index=False)

    with Phase("Save model + summary + report"):
        dump(clf, model_path)

        # Feature importance
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)

        summary = {
            "args": vars(args),
            "tuned_params": tuned_params,
            "n_features": len(feature_cols),
            "train_rows": len(train_df),
            "calib_rows": len(calib_df),
            "best_iteration": int(clf.best_iteration),
            "metrics": {
                "auc": auc,
                "aucpr": aucpr,
                "top_1pct_precision": top1_prec,
                "top_1pct_picks_per_day": top1_n,
                "top_5pct_precision": top5_prec,
                "top_5pct_picks_per_day": top5_n,
            },
            "fixed_threshold_metrics": {str(k): v for k, v in thresh_metrics.items()},
            "p_calib_distribution": {
                "min": float(np.min(p_calib)),
                "p25": float(np.percentile(p_calib, 25)),
                "p50": float(np.median(p_calib)),
                "p75": float(np.percentile(p_calib, 75)),
                "p99": float(np.percentile(p_calib, 99)),
                "max": float(np.max(p_calib)),
            },
            "top_features": imp.head(20).to_dict(orient="records"),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Human-readable report
        n_days_calib = calib_df[args.date_column].nunique()
        baseline = y_calib.mean()
        if args.label_mode == "binary_up":
            label_desc = "next-day return > 0"
        elif args.label_mode == "topq":
            label_desc = f"per-day top {args.topq_frac*100:.0f}% next-day return"
        else:
            label_desc = (f"per-day top {args.topq_frac*100:.0f}% of "
                          f"(return / {args.vol_col})")
        lines = []
        lines.append("=" * 78)
        lines.append("SIMPLE XGB REPORT")
        lines.append("=" * 78)
        lines.append(f"Label:         {label_desc}")
        lines.append(f"Features used: {len(feature_cols)}")
        lines.append(f"Train rows:    {len(train_df):,}")
        lines.append(f"Calib rows:    {len(calib_df):,}  ({n_days_calib} days)")
        lines.append(f"Trees trained: {clf.best_iteration+1} (early stop @ "
                     f"{args.early_stopping_rounds} rounds)")
        lines.append(f"Base rate P(y=1) on calib: {baseline:.4f}")
        lines.append("")
        lines.append("--- Calibration-slice metrics ---")
        lines.append(f"  AUC:                {auc:.4f}")
        lines.append(f"  AUC-PR:             {aucpr:.4f}")
        lines.append(f"  Top-1%-prec/day:    {top1_prec:.4f}  ({top1_n} picks/day)")
        lines.append(f"  Top-5%-prec/day:    {top5_prec:.4f}  ({top5_n} picks/day)")
        lines.append("")
        lines.append("--- Fixed-threshold precision ---")
        lines.append(f"  {'thr':>5} {'fires':>8} {'fires/day':>10} {'precision':>10} "
                     f"{'vs_base':>8}")
        for t, m in thresh_metrics.items():
            edge = ""
            if not np.isnan(m["precision"]):
                edge = f"{m['precision'] - baseline:+.3f}"
            lines.append(f"  {t:>5.2f} {m['fires']:>8,} {m['fires_per_day']:>10.2f} "
                         f"{m['precision'] if not np.isnan(m['precision']) else 'n/a':>10} "
                         f"{edge:>8}")
        lines.append("")
        lines.append("--- p_calib distribution ---")
        d = summary["p_calib_distribution"]
        lines.append(f"  min={d['min']:.3f} p25={d['p25']:.3f} p50={d['p50']:.3f} "
                     f"p75={d['p75']:.3f} p99={d['p99']:.3f} max={d['max']:.3f}")
        lines.append("")
        lines.append("--- Top 15 features ---")
        for _, row in imp.head(15).iterrows():
            lines.append(f"  {row['importance']:>8.5f}  {row['feature']}")
        lines.append("=" * 78)
        report = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report)
        logging.info("\n" + report)

    logging.info(f"\nTotal time: {(time.time()-run_t0)/60:.1f} min")
    logging.info(f"Outputs:")
    logging.info(f"  - {model_path}")
    logging.info(f"  - {scores_path}")
    logging.info(f"  - {summary_path}")
    logging.info(f"  - {report_path}")
    logging.info(f"\nRe-print the report with: python simple_xgb.py --inspect")


if __name__ == "__main__":
    main()
