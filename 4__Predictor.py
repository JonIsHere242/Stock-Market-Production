"""Self-contained XGBoost pipeline - data loading, labeling, training, prediction.

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

Monthly performance (OOS window Dec 2025 - Apr 2026):
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

Pipeline steps
==============
  Phase 1:  Load per-ticker parquets from Data/ProcessedData/
  Phase 2:  Label engineering (shift target -1, add ret_5d)
  Phase 3:  Universe filter - FilterRubric Step 1
  Phase 4:  Shuffle within each date (kills row-order leakage)
  Phase 5:  Train/calib split with N-day embargo
  Phase 6:  Build feature matrix - xs rank features if enabled
  Phase 7:  Build labels (topq / risk_adj_topq / binary_up)
  Phase 8:  Recency weights
  Phase 9:  Optuna hyperparameter tuning
  Phase 10: Train final XGBClassifier
  Phase 11: Evaluate on calibration slice + save model/reports
  Phase 12: Inference - score all ProcessedData tickers, write RFpredictions
  Phase 13: Pred-space factor neutralization (--neutralize; was 4.5__NeutralizePreds.py)
  Phase 14: Conviction-momentum tilt (ON by default; was 4.6__ConvictionMomentum.py)

Run:
    python 4__Predictor.py                         # full train + predict
    python 4__Predictor.py --predict_only          # inference only (model saved)
    python 4__Predictor.py --predict_only --neutralize             # ship config (Phase 14 is default-on)
    python 4__Predictor.py --predict_only --neutralize --no_conviction_momentum  # un-tilted ablation
    python 4__Predictor.py --cm_retilt Data/RFpredictions --cm_k 1.0 --cm_retilt_out Data/_cm_k1
                                                   # re-tilt existing preds for an A/B, no inference
    python 4__Predictor.py --inspect               # print cached report
    python 4__Predictor.py --n_trials 10           # quick diagnostic run
    python 4__Predictor.py --no_tune               # skip Optuna (match pre-Optuna baseline)
    python 4__Predictor.py --tune_device cuda       # run the Optuna search on GPU (~2 hr -> ~20 min; see FAST TUNING below)

BEST PRACTICE - STANDARD RETRAIN
==================================
The bare `python 4__Predictor.py` invocation is the canonical retrain. Every
default below is the winning-config value (see "EXACT CONFIG ..." above) and
should NOT be overridden unless you are running a diagnostic ablation:
    runpercent=75, embargo_days=5, label_mode=topq, add_xs_features=True,
    drop_vol_features=False, tune=True, tune_objective=top1_meanret,
    n_trials=100, tune_subsample=0.35, recency_half_life_days=720,
    top_frac_per_day=0.01
Do NOT raise --runpercent beyond 75 for a standard retrain - the OOS validation
slice (Dec 2025+) is load-bearing for the backtester's reported metrics, and
shrinking it makes the backtest mostly in-sample. After this script completes,
run the backtester: `python 5__NightlyBackTester.py --force [--sample N]`.

FAST TUNING - RUN THE OPTUNA SEARCH ON GPU (~2 hr -> ~20 min)
==============================================================
The Optuna search (Phase 9) is the long pole of a retrain. Each trial fits an
XGB `hist` model on the inner-train slice (~232k rows x ~1,416 features after the
xs-rank expansion); the cost is dominated by per-round histogram construction
scaled by the trial's depth/colsample/subsample, repeated every trial. Measured
CPU cost is ~80-130s per trial, so the canonical `n_trials=100` search is ~2.5-3
hours. Tree count barely moves it - it is the per-round work on the wide feature
matrix that hurts.

The lever is the GPU. Add `--tune_device cuda`:

    python 4__Predictor.py ... --tune_device cuda

  - Scope: ONLY the Optuna search runs on the GPU. The final model (Phase 10)
    still trains on CPU, so the saved model and all inference (Phase 12) are
    completely unchanged - no GPU is needed at predict time.
  - Speed: GPU `hist` on this feature width is roughly an order of magnitude
    faster on the search, taking the 100-trial run from ~2-3 hr toward ~20 min.
    (GPU is SLOWER on tiny data due to kernel-launch overhead - the win only
    shows up at this row x feature scale.)
  - Reproducibility: GPU `hist` chooses slightly different split points than CPU
    `hist`, so the tuned hyperparameters will not be bit-identical to a CPU
    search. They are equally good - tuning is a search, not a fixed computation - 
    but if you need to reproduce a specific saved model's params exactly, keep
    the default `--tune_device cpu`.
  - GPU memory: the inner-train + eval QuantileDMatrices are ~1.5-2 GB plus
    per-node histograms (fine on an 8 GB+ card). If you OOM, lower
    `--tune_subsample` (default 0.35) to shrink the inner-train matrix.

The default is `--tune_device cpu`, so the canonical retrain above is unchanged
unless you opt in.

PHASE 13 - PRED-SPACE FACTOR NEUTRALIZATION (the old 4.5 stage, now in here)
============================================================================
`4.5__NeutralizePreds.py` was folded into this file on 2026-07-27. It used to
re-read the whole RFpredictions dir + the whole feature panel, rewrite every
parquet into a tmp dir, and rename RFpredictions -> RFpredictions_raw. All of
that data is already in memory at the end of Phase 12, so it now runs inline:

    python 4__Predictor.py --predict_only --neutralize \
        --neut_mode spy200v2 --neut_dose_above 0.15 --neut_dose_below 0.30

  - What it does: per day, project the cross-section of raw_score onto
    [beta_spy, atr_percentage, log dollar_volume_ma_10], subtract
    dose*projection, then quantile-map back onto that day's ORIGINAL
    UpProbability values. The per-day marginal is preserved exactly - only the
    within-day ordering changes, so the backtester's threshold/percentile
    machinery sees the same distribution. Zero fitted parameters.
  - Dose: `--neut_mode spy200v2` uses 0.15 above the SPY 200d EMA and 0.30
    below (validated 2026-07-12: live 80.2% vs 52.2% for fixed 0.30, 2 seeds).
    `--neut_mode fixed --neut_dose 0.3` is the older validated overlay.
  - Rollback: the un-neutralized value is written per row as `raw_up_prob`
    (this replaces the old RFpredictions_raw directory).
  - `UpPrediction` is deliberately NOT recomputed from the neutralized ordering
    - it stays the raw top-1% flag, exactly as the standalone stage left it.
  - FAIL-SAFE: on any gate failure (too few tickers, signal day un-neutralized
    because the factor panel is stale) the RAW preds are written and the script
    exits 1 - the nightly runner alerts but the pipeline continues on the plain
    ship signal.
  - Default is OFF so research/ablation runs keep raw preds; the live runners
    (trading_system.ps1, rerun_for_live.ps1) pass the flags explicitly.
"""

import os
import sys
import json
import time
import argparse
import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from auxiliary._quiet_progress import tqdm
from joblib import dump, load as joblib_load

import warnings

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score,
)
from scipy.special import expit
from scipy.optimize import minimize

# Diagnostics sidecars (diagnostics/hooks.py). Every call is a no-op unless the
# DIAG_OUT env var names a directory; the stand-in keeps this script independent
# of the diagnostics folder. Output is byte-identical with DIAG_OUT unset.
try:
    from diagnostics import hooks as _diag
except Exception:
    class _diag:
        enabled = staticmethod(lambda: False)
        dump_json = dump_parquet = append_jsonl = stamp = staticmethod(lambda *a, **k: None)


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
# CLI - defaults match the 172% winning config                               #
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
parser.add_argument("--train_end_date", default=None, help="Override --runpercent: train through this exact date (YYYY-MM-DD). For refit-cadence experiments - hold params fixed, vary only the cutoff.")
parser.add_argument("--calibpercent", type=int, default=15, help="Percent of rows used for calibration (after embargo).")
parser.add_argument("--embargo_days", type=int, default=5, help="Calendar-day gap between train end and calib start.")
parser.add_argument("--horizon_5d", type=int, default=5, help="Horizon for the secondary ret_5d label.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for within-date shuffling.")

# Columns
parser.add_argument("--target_column", default="percent_change_Close")
parser.add_argument("--date_column", default="Date")
parser.add_argument("--ticker_column", default="Ticker")

# Label
parser.add_argument("--label_mode", choices=["binary_up", "topq", "risk_adj_topq", "topq_5d", "thresh_5d", "thresh_5d_xs", "thresh_mfe_5d", "thresh_5d_vol", "book_5d", "book_5d_all", "book_5d_stop", "book_ratchet", "book_ratchet_all"], default="topq", help="topq = per-day top-20%% next-day return (winning config). risk_adj_topq = top-20%% of return/vol. binary_up = simple next-day return > 0.")
parser.add_argument("--topq_frac", type=float, default=0.20)
parser.add_argument("--thresh_5d", type=float, default=0.08, help="thresh_5d label: y=1 when the 5-day forward close-to-close return (ret_5d) >= this. Payoff-targeted label (predictor lab 2026-09-02): the book's profit is the frequency of +10%%-class winners inside the ~5-day hold, which the 1-day topq label does not target. topq_5d = per-day top --topq_frac of ret_5d instead.")
parser.add_argument("--vol_col", default="Realized_Vol_21d")
parser.add_argument("--vol_floor", type=float, default=1e-3)

# Features
parser.add_argument("--add_xs_features", action="store_true", default=True, help="Add per-day percentile-rank column for each numeric feature (doubles feature count). ON by default - these were top-ranked in the winning model.")
parser.add_argument("--no_xs_features", action="store_true", help="Disable --add_xs_features.")
parser.add_argument("--drop_vol_features", action="store_true", help="Drop volatility-family features. OFF by default - vol features were the top-ranked features in the winning model. Only set this for diagnostic ablations.")
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
parser.add_argument("--load_start_date", default=None, help="Phase 1: drop panel rows before this date at read (deep-history panels; default None = load everything, prod path unchanged).")
parser.add_argument("--x_float32", action="store_true", help="Phase 6: cast the feature matrix to uniform float32 and drop feature columns from the frames (same model, about half the fit-phase RAM). Default off.")
parser.add_argument("--prefilter_quality", action="store_true", help="Phase 1: apply the universe quality gate per ticker at read (identical rows downstream, ~4x less RAM). Default off.")
parser.add_argument("--load_end_date", default=None, help="Phase 1: drop rows after this date (after labels are built). Use cutoff + calib window.")
parser.add_argument("--early_stopping_rounds", type=int, default=30, help="0 = no holdout and no early stop: fit a fixed --n_estimators on every training row (predictor lab 2026-09-05).")
parser.add_argument("--refit_full", action="store_true", help="After the early-stopped fit picks the tree count, refit on ALL training rows (the holdout is the newest 20%% of the window). Default off: prod path unchanged.")
parser.add_argument("--scale_pos_weight", type=float, default=None)
parser.add_argument("--train_row_filter", default=None, metavar="QUERY",
                    help="REGIME SPECIALIST (predictor lab 2026-09-03): pandas query applied to the TRAIN split only, "
                         "after the cache load, e.g. \"vix_close >= 20\" or \"vix_close < 20\". The calib split, the "
                         "calibrator and inference are untouched, so the specialist scores every name every day and "
                         "only its training population changes. Default None: prod path unchanged.")
parser.add_argument("--payoff_weight_cap", type=float, default=None,
                    help="Phase 8: multiply sample weights by (0.005 + min(|ret_5d|, cap)); big movers dominate the fit. Default None: off.")
parser.add_argument("--add_weekday", action="store_true", help="Phase 6/12: add day-of-week (0-4) as a feature named dow.")
parser.add_argument("--pool_top_n", type=int, default=0,
                    help="POOL CAP (Order 66): after Phase 14, keep only the top N names per day by final UpProbability; every other "
                         "name at or above 0.40 is floored to 0.30 so the book sees an N-name pool. 0 = off (prod path unchanged).")
parser.add_argument("--limit_features", action="store_true",
                    help="Phase 6/12: features measured at the trigger LIMIT price (Close x (1 - k x daily vol)) instead of the close: "
                         "lim_depth, lim_to_support, lim_to_resistance, lim_gap_room.")
parser.add_argument("--day_features", action="store_true",
                    help="Phase 6/12: cross-sectional market-day context (same-day, no lookahead): day_dip_breadth, day_ret_mean, "
                         "day_ret_disp, day_vol_mean, day_n.")
parser.add_argument("--book_touch_k", type=float, default=1.5, help="book_5d label: limit = Close x (1 - k x daily vol).")
parser.add_argument("--book_stop", type=float, default=0.03, help="book_5d label: stop below the fill.")
parser.add_argument("--book_target", type=float, default=0.08, help="book_5d label: target above the fill within horizon_5d sessions.")
parser.add_argument("--book_beta", action="store_true", help="book labels: depth = k x beta_spy x daily vol (the broker rule) instead of k x vol.")
parser.add_argument("--book_min_depth", type=float, default=0.0, help="book labels: clamp the limit depth below (backtester floor 0.015).")
parser.add_argument("--book_max_depth", type=float, default=1.0, help="book labels: clamp the limit depth above (broker 0.15).")
parser.add_argument("--book_ratchet", type=float, default=0.03, help="book_ratchet label: exit when Close <= running max Close x (1 - this) after the fill.")
parser.add_argument("--book_ratchet_win", type=float, default=0.05, help="book_ratchet label: win if the ratchet-exit return >= this.")
parser.add_argument("--payoff_weight_col", default="ret_5d", help="column used by --payoff_weight_cap (ret_5d, book_ret, book_ret_ratchet).")
parser.add_argument("--read_float32", action="store_true",
                    help="Phase 1: cast feature columns to float32 at parquet read (labels, prices, keys and Phase-13 factors stay float64). "
                         "Cuts a fresh-cache run from ~48 GB to ~36 GB but is NOT bit-identical (rank ties shift). Default OFF: prod path unchanged.")
parser.add_argument("--bag_mode", choices=["prob", "rank"], default="prob",
                    help="How --bag_seeds members combine: prob = mean probability (compresses the tail), rank = mean percentile rank.")
parser.add_argument("--bag_seeds", type=int, default=1, help="Predictor lab (2026-09-02): fit N XGB models with different random seeds and average their probabilities (BaggedXGB wrapper). 1 = single model, unchanged. Targets Sharpe via lower seed variance; --wf_bag measured the same effect offline.")
parser.add_argument("--recency_half_life_days", type=float, default=720, help="Exponential-decay recency weight half-life in days. 720 = samples 2 years old have half the weight of newest.")

# Optuna
parser.add_argument("--no_tune", action="store_true", help="Skip Optuna tuning - use default XGB params. Matches the original May-16 run (no Optuna) that produced 84%%+ before config drift.")
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument("--tune_subsample", type=float, default=0.35, help="Row fraction for each Optuna trial (speeds up tuning). Final fit uses full training data.")
parser.add_argument("--tune_objective", choices=["top1_prec", "top1_meanret", "aucpr"], default="top1_meanret", help="Optuna objective. top1_meanret = mean return of top-1%% picks per day (the objective used in the winning run).")
parser.add_argument("--tune_device", default="cpu", help="XGB device for the Optuna SEARCH fits only: cpu or cuda (or cuda:0). Default cpu = identical to the canonical run. 'cuda' runs the search on GPU (5-20x faster on wide feature sets). The final model is still trained on CPU, so the saved model + inference are unchanged. NOTE: GPU hist picks slightly different split points, so the tuned params won't be bit-identical to a CPU search (but are equally good).")
parser.add_argument("--tune_gpus", type=int, default=1, help="Spread the Optuna SEARCH across N GPUs (round-robins trials over cuda:0..cuda:N-1 with n_jobs=N). Only active with --tune_device cuda. WITHIN-search parallelism: one CPU copy of the data feeds both GPUs (RAM-safe), unlike launching N processes (~38GB each). 2x 3070 -> ~1.6x search throughput.")

# Inference
parser.add_argument("--top_frac_per_day", type=float, default=0.01, help="Per-day fraction mapped to UpProb in [0.45, 0.70]. 0.01 = top 1%% per day (~10-15 names with universe ~1400).")

# Calibration + conformal threshold
parser.add_argument("--nocalib", action="store_true", help="Disable Beta probability calibration on calib slice. By default, a 3-param Beta calibrator is fit and saved for diagnostics (does not alter UpPrediction logic).")
parser.add_argument("--target_precision", type=float, default=0.75, help="Conformal precision target: find lowest score cutoff where calib-slice precision >= this value. Logged as diagnostic; does not gate UpPrediction.")
parser.add_argument("--max_coverage", type=float, default=0.05, help="Upper cap on coverage when searching conformal threshold.")

# Convenience
parser.add_argument("--reuse", action="store_true", help="Reuse cached PreparedData splits from a previous run (skips Phases 1-5, goes straight to feature/label build).")
parser.add_argument("--fast", action="store_true", help="Quick diagnostic mode: cap Optuna to 10 trials and load only 500 tickers. Useful for smoke-testing changes.")

# Modes
parser.add_argument("--predict_only", action="store_true", help="Skip training. Load saved model and run inference only.")
# Inference peak-RAM control. X is len(combined) x len(feat_names) float32: at 3.07M rows
# x 1414 features that is 16.2 GiB allocated ON TOP of `combined`, and it OOM'd the
# nightly on 2026-08-13 and again on 2026-08-27 as the panel grew. The signal only needs
# recent history: 5__NightlyBackTester feeds ~400 calendar days (~275 bars), so rows older
# than that are computed and thrown away. This keeps the LAST N rows per ticker.
# Default None = OFF, byte-identical to every run before this flag existed.
# Cross-sectional (_xs) features rank WITHIN a day across tickers, so trimming each
# ticker's tail equally leaves recent days' cross-sections complete. Keep N comfortably
# above the conviction-momentum EMA spans so the last bar's tilt is fully warmed up.
parser.add_argument("--infer_tail", type=int, default=None, metavar="N",
                    help="Inference only: keep just the last N rows per ticker. Caps peak "
                         "RAM (X is rows x features x 4 bytes). None = all rows (default, "
                         "unchanged behaviour). 500 halves a 3.07M-row panel and leaves "
                         "ample EMA warm-up over the ~275 bars the backtester actually reads.")
parser.add_argument("--inspect", action="store_true", help="Print cached report + summary without running anything.")
parser.add_argument("--oos_only", action="store_true", help="Skip training/inference. Load the saved model and measure true out-of-sample decay: score the held-out tail (rows after the calib window - data the model never saw in train OR calib) and print a rank-band x split table of per-day mean return + precision. Headline = top-1%% vs shoulder-band (0.90-0.95) decay. Writes oos_report.txt + oos_scores.parquet.")
parser.add_argument("--oos_max_tickers", type=int, default=None, help="Cap tickers loaded for --oos_only (quick smoke test). None = all.")
parser.add_argument("--walkforward", action="store_true", help="Walk-forward OOS engine: load+feature-build ONCE, then for a suite of training configs x monthly anchors, retrain and measure true-OOS band edge (net of market beta) on the following month. Judges configs across many independent OOS windows. Writes walkforward_report.txt + walkforward_results.parquet.")
parser.add_argument("--wf_configs", default="baseline,recent_252,recent_126,hl_180,hl_90,decontam", help="Comma list of walk-forward configs to compare. Available: baseline, recent_252, recent_126, hl_180, hl_90, decontam.")
parser.add_argument("--wf_min_anchor", default="2025-06", help="Earliest anchor year-month (YYYY-MM). Anchor = last train date of that month; OOS = the following month.")
parser.add_argument("--wf_max_anchor", default=None, help="Latest anchor year-month (YYYY-MM), inclusive. Default None = run to the end of the data. Use with --wf_min_anchor to pin a walk-forward to one regime (e.g. pre- vs post-break), so a config's edge can be tested for regime-conditionality instead of being averaged across the break.")
parser.add_argument("--wf_trees", type=int, default=200, help="Fixed n_estimators per walk-forward fit (no Optuna/early-stop - isolates the data/feature/weight effect across configs).")
parser.add_argument("--wf_max_tickers", type=int, default=None, help="Cap tickers loaded for --walkforward (smoke test). None = all.")
parser.add_argument("--wf_sweep", action="store_true", help="PARAM-SWEEP mode: parallel, multi-seed walk-forward over XGB hyperparameter configs (depth/reg/trees) on identical baseline data, to isolate the overfitting lever and beat the nondeterminism noise floor. Reports seed-averaged top1_net per config with anchor win%%. Writes walkforward_sweep_report.txt + walkforward_sweep_results.parquet.")
parser.add_argument("--wf_sweep_configs", default="prod_optuna,d3,d4,d5,d6,d8_reg,d4_strongreg,d5_slow", help="Comma list of param-sweep configs (see SWEEP_CONFIGS registry).")
parser.add_argument("--wf_sweep_windows", default="full720", help="Comma list of TRAINING-WINDOW specs to CROSS with --wf_sweep_configs, so complexity and window length are varied JOINTLY (they are coupled: a longer window buys samples but imports more non-stationarity). Grammar: 'full' or 'full<HL>' = all history up to the anchor with recency half-life HL days; 'w<K>' or 'w<K>hl<HL>' = only the last K trading days. Default 'full720' reproduces the old fixed-window behaviour exactly.")
parser.add_argument("--wf_seeds", type=int, default=4, help="Seeds per (config,anchor) - averaged to beat the ~0.02 XGB-hist nondeterminism noise floor.")
parser.add_argument("--wf_workers", type=int, default=4, help="Concurrent fits (threads; XGB releases the GIL during fit). Total cores ~= wf_workers * wf_threads.")
parser.add_argument("--wf_threads", type=int, default=8, help="n_jobs per fit. Default 8; with wf_workers=4 that's 32 cores.")
parser.add_argument("--wf_device", default="cpu", help="XGB device for walk-forward fits: cpu or cuda (or cuda:0/cuda:1).")
# --- BAG-EVAL mode (additive, flag-gated; measures bagging, does NOT ship it) ---
parser.add_argument("--wf_bag", action="store_true", help="BAG-EVAL mode: for each (config,anchor) fit S seeds ONCE, then measure whether BAGGING the predictions (averaging probs OR per-day ranks across K seeds) preserves top1_net while shrinking the per-seed noise floor. Sweeps bag size K. Writes walkforward_bag_report.txt + walkforward_bag_results.parquet. Touches NO live model/predictions.")
parser.add_argument("--wf_bag_configs", default="prod_optuna", help="Comma list of SWEEP_CONFIGS to bag-eval (default the live production config).")
parser.add_argument("--wf_bag_seeds", type=int, default=12, help="S = seeds fit per (config,anchor). Larger S = tighter variance estimate (and more fits). Bags are drawn from these S.")
parser.add_argument("--wf_bag_ks", default="1,3,5,8", help="Comma list of bag sizes K to evaluate (K=1 = single-seed noise floor baseline).")
parser.add_argument("--wf_bag_bags", type=int, default=10, help="B = random K-seed bags sampled (without replacement within a bag) to estimate across-bag std. K=1 uses all S singles instead.")
parser.add_argument("--wf_bag_mode", default="both", choices=["prob", "rank", "both"], help="Bag-combine method: 'prob' (avg probabilities), 'rank' (avg per-day percentile ranks), or 'both'.")
parser.add_argument("--wf_feature_file", default=None, help="Optional path to a newline-separated list of BASE feature names. When set, walk-forward sweep restricts select_base_features() to this list before xs-rank expansion, so a frozen model's exact feature set can be retrained against the current full panel. None = use every feature the panel offers.")

# --- Phase 13: pred-space factor neutralization (was the separate 4.5__NeutralizePreds.py) --
parser.add_argument("--neutralize", action="store_true", help="Phase 13: partially neutralize the predictions against [beta_spy, atr_percentage, log dollar_volume_ma_10] per day, then quantile-map back onto that day's original UpProbability values (per-day marginal preserved, only within-day ordering changes). PRODUCTION runs this ON with --neut_mode spy200v2. OFF by default so research/ablation runs keep raw ship preds.")
parser.add_argument("--neut_mode", choices=["fixed", "spy200v2"], default="spy200v2", help="fixed: constant --neut_dose. spy200v2: per-day dose = --neut_dose_above when SPY close >= its 200d EMA that day, else --neut_dose_below (validated 2026-07-12: live 80.2%% vs 52.2%% for fixed 0.30, 2 seeds).")
parser.add_argument("--neut_dose", type=float, default=0.3, help="Dose for --neut_mode fixed (validated 2026-07-01/07-03).")
parser.add_argument("--neut_dose_above", type=float, default=0.15, help="spy200v2 dose on days SPY closed at/above its 200d EMA.")
parser.add_argument("--neut_dose_below", type=float, default=0.30, help="spy200v2 dose on days SPY closed below its 200d EMA (= the incumbent fixed dose).")
parser.add_argument("--neut_flag_max_stale_days", type=int, default=14, help="Max calendar-day ffill of the SPY/EMA flag; staler days fall back to --neut_dose_below (safe degrade).")
parser.add_argument("--neut_min_tickers", type=int, default=1000, help="Gate: skip neutralization (ship RAW preds, exit non-zero) if inference covered fewer tickers than this.")
parser.add_argument("--neut_min_changed", type=float, default=0.05, help="Gate: the signal (last) day must actually be neutralized -- abort to RAW preds if fewer than this fraction of that day's names changed (stale factor panel).")

# --- Phase 14: conviction-momentum tilt (was the separate 4.6__ConvictionMomentum.py) -------
# ON BY DEFAULT since 2026-07-29. It is the ship config and every path that produces a live
# book (trading_system.ps1, rerun_for_live.ps1, the README invocation) passed the flag
# anyway, so opt-in bought nothing and left room to ship an un-tilted book by forgetting it.
# The residuals in the Phase 14 block below are NOT closed by this default -- read them.
parser.add_argument("--no_conviction_momentum", action="store_true", help="Disable Phase 14, the conviction-momentum tilt, which is ON by default. Use for research/ablation runs that need the un-tilted preds.")
parser.add_argument("--conviction_momentum", action="store_true", help="Phase 14: tilt the FINAL (post-Phase-13) UpProbability toward names whose model conviction is RISING -- up' = clip(up + k * mean_spans(up - EMA_span(up))). Per ticker, causal, adds no data and drops no tickers. NO-OP as of 2026-07-29: the tilt is the default, and this flag is kept only so the existing ship invocations keep parsing. Pass --no_conviction_momentum to turn it off.")
parser.add_argument("--cm_k", type=float, default=0.5, help="Tilt strength. 0.5 = max validated return; 1.0 = best validated drawdown.")
parser.add_argument("--cm_spans", default="3,6,12", help="Comma-separated EMA spans; the deviation is AVERAGED over them (multi-scale, the validated default). A single value, e.g. 6, gives the single-scale variant.")
parser.add_argument("--cm_lo", type=float, default=0.30, help="Clip floor (panel-native).")
parser.add_argument("--cm_hi", type=float, default=0.70, help="Clip ceiling (panel-native).")
parser.add_argument("--cm_min_tickers", type=int, default=1000, help="Gate: skip the tilt (ship un-tilted preds, exit non-zero) if inference covered fewer tickers than this.")
parser.add_argument("--cm_min_changed", type=float, default=0.05, help="Gate: the signal (last) day must actually move -- abort to un-tilted preds if fewer than this fraction of that day's names changed (short history => inert transform).")
# RETILT mode: tilt an EXISTING prediction dir on disk, without re-running inference. This
# was the only capability 4.6__ConvictionMomentum.py had that Phase 14 did not, and it is
# what an A/B or a k/spans sweep needs (re-scoring 1,900 tickers to change one scalar is
# absurd). Ported here so there is exactly ONE implementation of the transform.
parser.add_argument("--cm_retilt", metavar="SRC_DIR", default=None, help="RETILT MODE: skip the whole pipeline and apply the Phase 14 tilt to the prediction parquets already in SRC_DIR. Reads the per-row pre-tilt column if present, so re-running is idempotent rather than compounding. Writes to --cm_retilt_out; SRC_DIR is untouched unless --cm_retilt_apply.")
parser.add_argument("--cm_retilt_out", default=os.path.join("Data", "_cm_tmp"), help="Retilt mode: where to write the tilted preds. Backtest this dir directly with 5__NightlyBackTester.py --data_dir.")
parser.add_argument("--cm_retilt_apply", action="store_true", help="Retilt mode: swap the result into --cm_retilt (rollback copy kept at <SRC_DIR>_precm). Without this the source dir is NEVER touched.")
parser.add_argument("--cm_retilt_dry_run", action="store_true", help="Retilt mode: compute and report only, write nothing at all.")
parser.add_argument("--cm_retilt_book", type=int, default=12, help="Retilt mode: book size for the reported entry/exit delta.")
parser.add_argument("--cm_retilt_workers", type=int, default=16, help="Retilt mode: parquet read/write threads.")

args = parser.parse_args()

# Resolve the xs features flag (default=True but --no_xs_features disables it)
USE_XS    = args.add_xs_features and not args.no_xs_features
TUNE      = not args.no_tune
USE_CALIB = not args.nocalib
# Phase 14 is the ship default; --conviction_momentum survives only as an accepted no-op.
CM_ON     = not args.no_conviction_momentum

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
        try:
            import psutil as _ps
            _rss = f"  rss {_ps.Process().memory_info().rss/1e9:.1f} GB"
        except Exception:
            _rss = ""
        logging.info(f"<<< {self.name} done in {time.time()-self.t0:.1f}s.{_rss}")


class CalibratorInversionError(RuntimeError):
    """Fitted Beta calibration would REVERSE the ranking. Never accepted silently."""


class BetaCalibrator:
    """3-parameter Beta calibrator (Kull et al. 2017).

    Maps raw probabilities p through sigmoid(a*logit(p) + b*log((1-p)/p) + c).
    Fitted by minimising NLL on a held-out calibration slice.
    When a=1, b=0, c=0 the transform is identity - safe default when fitting fails.
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
        # SIGN GUARD. log(p/(1-p)) and log((1-p)/p) are negatives of each other, so
        # the whole transform collapses to sigmoid((a-b)*logit(p) + c) and (a-b) is
        # the ONLY thing that carries the ordering. A fitted (a-b) <= 0 monotonically
        # REVERSES it: the names the model likes least would score highest and the
        # traded book would be the exact inverse of the intended one, with no error
        # anywhere. This runs offline in the nightly, so failing loudly is correct;
        # do not fall back to identity, because a silent fallback is how a broken
        # calibration slice gets shipped. `_fitted` stays False so the object is
        # unusable rather than half-fitted.
        slope = self.a - self.b
        if not np.isfinite(slope) or slope <= 0:
            raise CalibratorInversionError(
                f"BetaCalibrator fitted a non-positive slope: a={self.a:.6f} "
                f"b={self.b:.6f} c={self.c:.6f} -> (a-b)={slope:.6f}. The transform "
                f"reduces to sigmoid((a-b)*logit(p)+c), so (a-b) <= 0 INVERTS the "
                f"entire traded ranking. Refusing to fit. Inspect the calibration "
                f"slice (rows={len(p)}, positives={float(np.mean(y)):.4f}) before "
                f"re-running."
            )
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
    # Column-by-column: select_dtypes + bulk assignment consolidate every float64
    # block into one contiguous array (15.8 GiB on the 3M-row inference panel,
    # OOM on 2026-07-30). Casting one column at a time caps the peak at a single
    # column's copy.
    for c in df.columns[(df.dtypes == "float64").to_numpy()]:
        df[c] = df[c].to_numpy(dtype="float32")
    return df


def mem_gb(df):
    return df.memory_usage(deep=False).sum() / 1e9


# -------------------------------------------------------------------------- #
# Phase 1+2: Load tickers + label engineering                                #
# -------------------------------------------------------------------------- #
def load_and_label_tickers(input_dir, target_col, date_col, horizon_5d,
                            max_files=None, usecols_fn=None, read_float32=False,
                            book_touch_k=1.5, book_stop=0.03, book_target=0.08, book_vol_col="Realized_Vol_21d",
                            book_beta=False, book_min_depth=0.0, book_max_depth=1.0, book_ratchet=0.03,
                            load_start_date=None, load_end_date=None, prefilter_quality=False):
    """Load every parquet in input_dir, shift target -1 (next-day return),
    add ret_5d (5-day forward return), drop last rows that have no label.

    usecols_fn: optional name->bool predicate. When given, only the columns it
    keeps are read from each parquet (intersected per-file so a missing column
    never errors). Used to skip feature families that are dropped in Phase 6
    anyway - pure I/O + memory savings, identical surviving rows/columns.
    """
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
    if max_files:
        files = files[:max_files]
        logging.info(f"  --max_files {max_files}: using {len(files)} tickers")

    # Read-time float32 (2026-09-03): the 3.06M-row training frame in float64 is ~18 GB
    # before the concat copy, and a fresh-cache run peaked at 45.7 GB (watchdog-killed).
    # Feature columns are downcast to float32 for the model anyway, so casting them here
    # is identical downstream; label sources, prices, keys and the Phase-13 factors keep
    # float64 so labels and neutralization are bit-for-bit unchanged.
    _keep64 = set(NON_FEATURES) | set(NEUT_FACTORS) | {target_col, date_col, "Ticker",
               "ret_today", "dollar_volume_ma_10", "atr_percentage", "RSI"}

    def _cast32(tbl):
        if not read_float32:
            return tbl
        import pyarrow as _pa
        fields = [(_pa.field(f.name, _pa.float32()) if (_pa.types.is_float64(f.type) and f.name not in _keep64) else f)
                  for f in tbl.schema]
        target = _pa.schema(fields)
        return tbl if target == tbl.schema else tbl.cast(target)

    def _read(fn):
        path = os.path.join(input_dir, fn)
        if usecols_fn is None:
            return _cast32(pq.read_table(path))
        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if usecols_fn(c)]
        return _cast32(pf.read(columns=cols))

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
        # LOAD WINDOW (2026-09-05, default off): a deep-history panel holds 6+ years and Phase 1
        # keeps every row in RAM (peak ~12 GB per panel-year); a run only needs its training
        # window plus the calib slice. Start is applied before labels (labels look forward only),
        # end after them so the last rows keep their horizon.
        if load_start_date is not None:
            df = df[pd.to_datetime(df[date_col]) >= pd.Timestamp(load_start_date)].reset_index(drop=True)
            if df.shape[0] <= 60:
                n_skip_short += df.shape[0]
                return None
        # Stash TODAY's return as an explicit feature BEFORE the in-place label shift.
        # LEAK GUARD (2026-07-01): the v2 panels name the target in lowercase
        # ('percent_change_close') but NON_FEATURES only excluded the capital-C name,
        # so a fresh full-feature train (no --match_model_features) put TOMORROW's
        # return (raw + _xs rank) into the feature set. The ship model dodged this only
        # because its curated 708-feature list predates the v2 lowercase naming.
        df["ret_today"] = df[target_col]
        df[target_col] = df[target_col].shift(-1)
        df["ret_5d"] = (df["Close"].shift(-horizon_5d) / df["Close"] - 1.0
                        if "Close" in df.columns else np.nan)
        # Max favorable excursion over the next horizon_5d sessions (predictor lab
        # 2026-09-02): the book's take-profit / ratchet exits fire on intraday HIGHS, so
        # a label on the forward max high matches the payoff more literally than
        # close-to-close. Only used by --label_mode thresh_mfe_5d; costs one column.
        if "High" in df.columns and "Close" in df.columns:
            _fwd_max = pd.concat([df["High"].shift(-k) for k in range(1, horizon_5d + 1)],
                                 axis=1).max(axis=1)
            df["mfe_5d"] = _fwd_max / df["Close"] - 1.0
        else:
            df["mfe_5d"] = np.nan
        # BOOK-RULE label columns (predictor lab 2026-09-03): simulate the trigger arm on each
        # row. Limit = Close x (1 - k x daily vol); touch if next session's Low reaches it; from
        # that fill, walk horizon_5d sessions: win if High >= fill x (1+target) strictly before
        # Low <= fill x (1-stop) (same-session tie counts as a stop). book_touch: 0/1;
        # book_win: 0/1 for touching rows, NaN otherwise. Only --label_mode book_5d uses them.
        if all(c in df.columns for c in ("Low", "High", "Close")) and book_vol_col in df.columns:
            _v = df[book_vol_col].astype(float)
            if np.nanmedian(_v.values) > 0.1:          # annualised -> daily
                _v = _v / np.sqrt(252.0)
            _depth = book_touch_k * _v
            if book_beta and "beta_spy" in df.columns:
                _depth = _depth * pd.to_numeric(df["beta_spy"], errors="coerce").abs().fillna(1.0)
            _depth = _depth.clip(book_min_depth, book_max_depth)
            _limit = df["Close"] * (1.0 - _depth)
            _lows = np.column_stack([df["Low"].shift(-k).values for k in range(1, horizon_5d + 1)])
            _highs = np.column_stack([df["High"].shift(-k).values for k in range(1, horizon_5d + 1)])
            _closes = np.column_stack([df["Close"].shift(-k).values for k in range(1, horizon_5d + 1)])
            _fill = _limit.values[:, None]
            _touch = _lows[:, 0] <= _limit.values
            _stop_hit = _lows <= _fill * (1.0 - book_stop)
            _tgt_hit = _highs >= _fill * (1.0 + book_target)
            _first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), horizon_5d + 1)
            _fs, _ft = _first(_stop_hit), _first(_tgt_hit)
            _win = (_ft < _fs).astype(float)
            _stopf = (_fs <= _ft) & _stop_hit.any(axis=1)
            # realised return under the fixed-target rule: +target, -stop, or the last close
            _ret = np.where(_ft < _fs, book_target, np.where(_stopf, -book_stop, _closes[:, -1] / _limit.values - 1.0))
            # ratchet exit: after the fill, exit at the first close <= running max close x (1 - ratchet),
            # else the last close; stop-first days still stop out
            _runmax = np.maximum.accumulate(np.nan_to_num(_closes, nan=-np.inf), axis=1)
            _rat_hit = _closes <= _runmax * (1.0 - book_ratchet)
            _fr = _first(_rat_hit)
            _rat_ret = np.where(_fr <= horizon_5d - 1, _closes[np.arange(len(df)), np.minimum(_fr, horizon_5d - 1)] / _limit.values - 1.0,
                                _closes[:, -1] / _limit.values - 1.0)
            _rat_ret = np.where(_stopf & (_fs < _fr), -book_stop, _rat_ret)
            df["book_touch"] = _touch.astype(np.int8)
            df["book_win"] = np.where(_touch, _win, np.nan)
            df["book_stop"] = np.where(_touch, _stopf.astype(float), np.nan)
            df["book_ret"] = np.where(_touch, np.clip(np.nan_to_num(_ret, nan=0.0), -0.5, 0.5), np.nan)
            df["book_ret_ratchet"] = np.where(_touch, np.clip(np.nan_to_num(_rat_ret, nan=0.0), -0.5, 0.5), np.nan)
        else:
            df["book_touch"] = 0
            df["book_win"] = np.nan
            df["book_stop"] = np.nan
            df["book_ret"] = np.nan
            df["book_ret_ratchet"] = np.nan
        df = df.iloc[:-max(horizon_5d, 1)]
        if load_end_date is not None:
            df = df[pd.to_datetime(df[date_col]) <= pd.Timestamp(load_end_date)]
        if prefilter_quality:
            # Same FilterRubric gate main() applies to the concatenated frame BEFORE the split and
            # before the per-day xs ranks, so applying it per ticker here (after the labels, which
            # look forward only) is identical downstream and keeps ~75% of the rows out of RAM.
            df, _ = apply_quality_filter(df, quiet=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna(subset=[target_col])
        n_drop_nan += before - len(df)
        before = len(df)
        df = df[(df[target_col] <= 5) & (df[target_col] >= -5)]
        n_drop_outlier += before - len(df)
        return downcast(df) if not df.empty else None

    _by_fn = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        future_to_fn = {ex.submit(_read, fn): fn for fn in files}
        for future in tqdm(as_completed(future_to_fn), total=len(files),
                           desc="Loading tickers"):
            fn = future_to_fn[future]
            try:
                result = _process(fn, future.result())
                if result is not None:
                    _by_fn[fn] = result
            except Exception as e:
                logging.warning(f"  skipping {fn}: {e}")
    # DETERMINISTIC ORDER (2026-09-05): the threads finish in a different order every run,
    # and shuffle_within_date assigns its random keys by row POSITION, so the concat order
    # used to change the within-date shuffle and with it the fitted model (the predictor
    # lab found two "identical" fits whose stitched books differed by 70pp). Emit the parts
    # in filename order so the same flags + seed give the same rows in the same order.
    parts = [_by_fn[fn] for fn in files if fn in _by_fn]
    try:  # hand the Arrow read buffers back to the OS (the 16 reader threads leave arenas behind)
        import pyarrow as _pa
        _held = _pa.total_allocated_bytes() / 1e9
        _pa.default_memory_pool().release_unused()
        logging.info(f"  arrow pool: {_held:.2f} GB still allocated after load ({_pa.default_memory_pool().backend_name}); release_unused() called")
    except Exception:
        pass

    logging.info(f"  accounting: skipped_short={n_skip_short:,}  "
                 f"nan_dropped={n_drop_nan:,}  outliers_dropped={n_drop_outlier:,}")
    _diag.dump_json("T_load", {"n_files": len(files), "n_parts": len(parts),
                               "rows": int(sum(len(p) for p in parts)),
                               "skipped_short_rows": n_skip_short, "nan_dropped": n_drop_nan,
                               "outliers_dropped": n_drop_outlier, "horizon_5d": horizon_5d})
    return parts


# -------------------------------------------------------------------------- #
# Phase 3: Universe filter - FilterRubric Step 1                             #
# -------------------------------------------------------------------------- #
def apply_quality_filter(df,
                          min_close=5.0,
                          min_dollar_volume=5_000_000.0,
                          max_atr_pct=0.05,
                          rsi_exclude_lo=30.0,
                          rsi_exclude_hi=40.0, quiet=False):
    """Apply the FilterRubric Step 1 universe gate.

    Returns (filtered_df, boolean_mask_aligned_to_input_index).
    The mask is useful at inference time (mark, don't drop).
    """
    n0 = len(df)
    mask = pd.Series(True, index=df.index)
    gates = {}

    if "Close" in df.columns:
        gate = df["Close"] >= min_close
        _n = int((~gate & mask).sum())
        if not quiet: logging.info(f"  gate Close>={min_close}: removes {_n:,}")
        gates["close_ge_%g" % min_close] = _n
        mask &= gate
    if "dollar_volume_ma_10" in df.columns:
        gate = df["dollar_volume_ma_10"] >= min_dollar_volume
        _n = int((~gate & mask).sum())
        if not quiet: logging.info(f"  gate dollar_vol>={min_dollar_volume:,.0f}: removes {_n:,}")
        gates["dollar_vol_ge_%g" % min_dollar_volume] = _n
        mask &= gate
    if "atr_percentage" in df.columns:
        gate = df["atr_percentage"] <= max_atr_pct
        _n = int((~gate & mask).sum())
        if not quiet: logging.info(f"  gate atr_pct<={max_atr_pct}: removes {_n:,}")
        gates["atr_pct_le_%g" % max_atr_pct] = _n
        mask &= gate
    if "RSI" in df.columns:
        gate = ~((df["RSI"] >= rsi_exclude_lo) & (df["RSI"] < rsi_exclude_hi))
        _n = int((~gate & mask).sum())
        if not quiet: logging.info(f"  gate RSI not in [{rsi_exclude_lo},{rsi_exclude_hi}): "
                     f"removes {_n:,}")
        gates["rsi_not_in_%g_%g" % (rsi_exclude_lo, rsi_exclude_hi)] = _n
        mask &= gate

    out = df.loc[mask].copy()
    if not quiet: logging.info(f"  universe: {n0:,} -> {len(out):,} rows "
                 f"({100*len(out)/max(n0,1):.1f}% retained)")
    _diag.append_jsonl("P_quality_gates", {"n0": int(n0), "n_pass": int(len(out)), "gates": gates,
                                           "columns_present": [c for c in ("Close", "dollar_volume_ma_10",
                                                                           "atr_percentage", "RSI")
                                                               if c in df.columns]})
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
    "percent_change_Close", "ret_5d", "mfe_5d", "book_touch", "book_win", "book_stop", "book_ret", "book_ret_ratchet", "book_win_ratchet",
    # LEAK GUARD (2026-07-01): v2 panels use the LOWERCASE target name; without this
    # entry the in-place-shifted label (tomorrow's return) becomes a feature in any
    # fresh full-feature train. Today's return survives as 'ret_today' instead.
    "percent_change_close",
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


def add_limit_features(df, vol_col, k=1.5):
    """Features measured at the trigger limit price rather than the close (Order 66, 2026-09-03).
    Uses columns the panel already has; missing inputs give NaN (XGB handles them)."""
    v = pd.to_numeric(df.get(vol_col), errors="coerce").astype(float)
    if np.nanmedian(v.values) > 0.1:
        v = v / np.sqrt(252.0)
    depth = (k * v).clip(0.0, 0.30)
    df["lim_depth"] = depth.astype(np.float32)
    ds = (pd.to_numeric(df.get("Distance to Support (%)"), errors="coerce") / 100.0).clip(-0.5, 0.9)
    dr = (pd.to_numeric(df.get("Distance to Resistance (%)"), errors="coerce") / 100.0).clip(-0.5, 5.0)
    df["lim_to_support"] = ((1.0 - depth) / (1.0 - ds) - 1.0).clip(-1.0, 1.0).astype(np.float32)     # limit vs support level
    df["lim_to_resistance"] = ((1.0 + dr) / (1.0 - depth) - 1.0).clip(-1.0, 3.0).astype(np.float32)  # room from limit to resistance
    df["lim_gap_room"] = (df["lim_to_resistance"] - 0.08).astype(np.float32)                        # room beyond the +8% target
    return ["lim_depth", "lim_to_support", "lim_to_resistance", "lim_gap_room"]


def add_day_features(df, date_col, vol_col, ret_col="ret_today", k=1.5):
    """Cross-sectional market-day context, same-day information only (Order 66, 2026-09-03)."""
    v = pd.to_numeric(df.get(vol_col), errors="coerce").astype(float)
    if np.nanmedian(v.values) > 0.1:
        v = v / np.sqrt(252.0)
    r = pd.to_numeric(df.get(ret_col), errors="coerce").astype(float)
    g = pd.DataFrame({"d": df[date_col].values, "r": r.values, "dip": (r.values < -k * v.values).astype(float), "v": v.values})
    agg = g.groupby("d").agg(day_dip_breadth=("dip", "mean"), day_ret_mean=("r", "mean"), day_ret_disp=("r", "std"),
                             day_vol_mean=("v", "mean"), day_n=("r", "size"))
    for c in agg.columns:
        df[c] = pd.Series(df[date_col].values).map(agg[c]).values.astype(np.float32)
    return list(agg.columns)


def build_labels(train_df, calib_df, label_mode, target_col, date_col,
                  topq_frac, vol_col, vol_floor, thresh_5d=0.08, ratchet_win=0.05):
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
    elif label_mode == "topq_5d":
        # Payoff-targeted (predictor lab 2026-09-02): per-day top-q of the 5-DAY forward
        # return, i.e. names that move big inside the book's typical hold, not tomorrow's
        # top-20%. ret_5d is built in Phase 2 and is in NON_FEATURES, so it cannot leak.
        for _df in (train_df, calib_df):
            if "ret_5d" not in _df.columns:
                raise ValueError("label_mode topq_5d needs the ret_5d column from Phase 2")
        y_train = topq_label(train_df["ret_5d"].values, train_df[date_col].values, top_frac=topq_frac)
        y_calib = topq_label(calib_df["ret_5d"].values, calib_df[date_col].values, top_frac=topq_frac)
        logging.info(f"  label = per-day top {topq_frac*100:.0f}% of 5-day forward return (ret_5d)")
    elif label_mode == "thresh_5d":
        for _df in (train_df, calib_df):
            if "ret_5d" not in _df.columns:
                raise ValueError("label_mode thresh_5d needs the ret_5d column from Phase 2")
        y_train = (train_df["ret_5d"].values >= thresh_5d).astype(int)
        y_calib = (calib_df["ret_5d"].values >= thresh_5d).astype(int)
        logging.info(f"  label = 5-day forward return >= {thresh_5d:+.2%} (thresh_5d)")
    elif label_mode == "thresh_mfe_5d":
        for _df in (train_df, calib_df):
            if "mfe_5d" not in _df.columns:
                raise ValueError("label_mode thresh_mfe_5d needs the mfe_5d column from Phase 2 (fresh cache)")
        y_train = (train_df["mfe_5d"].values >= thresh_5d).astype(int)
        y_calib = (calib_df["mfe_5d"].values >= thresh_5d).astype(int)
        logging.info(f"  label = 5-day forward MAX HIGH vs close >= {thresh_5d:+.2%} (thresh_mfe_5d)")
    elif label_mode == "thresh_5d_vol":
        # +threshold measured in units of the name's own 5-day volatility (annualised vol
        # columns are converted to daily): equalises positives across vol buckets.
        def _volscaled(df):
            v = df[vol_col].astype(float).values
            if np.nanmedian(v) > 0.1:
                v = v / np.sqrt(252.0)
            return df["ret_5d"].values / np.maximum(v * np.sqrt(5.0), vol_floor)
        y_train = (_volscaled(train_df) >= thresh_5d).astype(int)
        y_calib = (_volscaled(calib_df) >= thresh_5d).astype(int)
        logging.info(f"  label = 5-day return / (5-day vol) >= {thresh_5d:.2f} (thresh_5d_vol)")
    elif label_mode == "book_ratchet":
        # win = the ratchet-exit trade from the fill returned >= --book_ratchet_win (train on touching rows)
        for _df in (train_df, calib_df):
            if "book_ret_ratchet" not in _df.columns:
                raise ValueError("label_mode book_ratchet needs the book_ret_ratchet column from Phase 2 (fresh cache)")
        y_train = (np.nan_to_num(train_df["book_ret_ratchet"].values, nan=-1.0) >= ratchet_win).astype(int)
        y_calib = (np.nan_to_num(calib_df["book_ret_ratchet"].values, nan=-1.0) >= ratchet_win).astype(int)
        logging.info(f"  label = ratchet-exit return >= {ratchet_win:+.2%} (book_ratchet); positives {y_train.mean():.3%}")
    elif label_mode == "book_ratchet_all":
        # P(touch AND ratchet-exit return >= win) on ALL rows (non-touch = 0)
        for _df in (train_df, calib_df):
            if "book_ret_ratchet" not in _df.columns:
                raise ValueError("label_mode book_ratchet_all needs the book_ret_ratchet column from Phase 2 (fresh cache)")
        y_train = (np.nan_to_num(train_df["book_ret_ratchet"].values, nan=-1.0) >= ratchet_win).astype(int)
        y_calib = (np.nan_to_num(calib_df["book_ret_ratchet"].values, nan=-1.0) >= ratchet_win).astype(int)
        logging.info(f"  label = touch AND ratchet-exit return >= {ratchet_win:+.2%}, all rows (book_ratchet_all); positives {y_train.mean():.3%}")
    elif label_mode == "book_5d_stop":
        # WICK MODEL: P(stop first | touch). Train with --train_row_filter 'book_touch == 1'.
        for _df in (train_df, calib_df):
            if "book_stop" not in _df.columns:
                raise ValueError("label_mode book_5d_stop needs the book_stop column from Phase 2 (fresh cache)")
        y_train = np.nan_to_num(train_df["book_stop"].values, nan=0.0).astype(int)
        y_calib = np.nan_to_num(calib_df["book_stop"].values, nan=0.0).astype(int)
        logging.info(f"  label = stop-first under the book rule (book_5d_stop); positives {y_train.mean():.3%}")
    elif label_mode == "book_5d_all":
        # P(touch AND win) on ALL rows (non-touch = 0): ranking by this is ranking by expected
        # book wins per pool name, the scorecard's ev_touch. No row filter needed.
        for _df in (train_df, calib_df):
            if "book_win" not in _df.columns:
                raise ValueError("label_mode book_5d_all needs the book_win column from Phase 2 (fresh cache)")
        y_train = np.nan_to_num(train_df["book_win"].values, nan=0.0).astype(int)
        y_calib = np.nan_to_num(calib_df["book_win"].values, nan=0.0).astype(int)
        logging.info(f"  label = touch AND win under the book rule, all rows (book_5d_all); positives {y_train.mean():.3%}")
    elif label_mode == "book_5d":
        for _df in (train_df, calib_df):
            if "book_win" not in _df.columns:
                raise ValueError("label_mode book_5d needs the book_win column from Phase 2 (fresh cache)")
        y_train = np.nan_to_num(train_df["book_win"].values, nan=0.0).astype(int)
        y_calib = np.nan_to_num(calib_df["book_win"].values, nan=0.0).astype(int)
        logging.info("  label = book rule (touch, then target before stop); train touch rows "
                     f"{int(train_df['book_touch'].sum()):,} of {len(train_df):,} (use --train_row_filter 'book_touch == 1')")
    elif label_mode == "thresh_5d_xs":
        # Same big-move label, measured in EXCESS of that day's cross-sectional mean 5-day
        # return, so market-wide rallies do not mint positives (predictor lab 2026-09-02).
        for _df in (train_df, calib_df):
            if "ret_5d" not in _df.columns:
                raise ValueError("label_mode thresh_5d_xs needs the ret_5d column from Phase 2")
        def _xs_excess(df):
            r = pd.Series(df["ret_5d"].values)
            d = pd.Series(df[date_col].values)
            return (r - r.groupby(d).transform("mean")).values
        y_train = (_xs_excess(train_df) >= thresh_5d).astype(int)
        y_calib = (_xs_excess(calib_df) >= thresh_5d).astype(int)
        logging.info(f"  label = 5-day forward return minus day mean >= {thresh_5d:+.2%} (thresh_5d_xs)")
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
                       n_trials, objective_name, tune_subsample=1.0, device="cpu", n_gpus=1):
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
    logging.info(f"  optuna objective: {objective_name}  device: {device}  n_gpus: {n_gpus}")
    multi_gpu = device.startswith("cuda") and n_gpus > 1

    def objective(trial):
        dev = f"cuda:{trial.number % n_gpus}" if multi_gpu else device
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
            tree_method="hist", device=dev, n_jobs=-1, random_state=42,
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
    _n_jobs = n_gpus if (device.startswith("cuda") and n_gpus > 1) else 1
    study.optimize(objective, n_trials=n_trials, callbacks=[_log],
                   show_progress_bar=False, n_jobs=_n_jobs)
    if _diag.enabled():
        try:
            _diag.dump_parquet("T_optuna_trials", study.trials_dataframe())
        except Exception:
            pass
        _diag.dump_json("T_optuna", {"inner_train_rows": int(len(X_it)), "inner_val_rows": int(len(X_iv)),
                                     "inner_cut_row": int(cut), "tune_subsample": tune_subsample,
                                     "objective": objective_name, "n_trials": n_trials, "device": device,
                                     "best_trial": int(study.best_trial.number), "best_value": float(study.best_value),
                                     "best_params": study.best_params})

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
                early_stopping_rounds, seed=42, refit_full=False):
    """Fit final XGBClassifier on full training data.

    Uses last 20% of train as internal val for early stopping. NOTE (2026-09-05): the
    frame is date-sorted, so that 20% is the most RECENT fifth of the training window
    and the early-stopped model never trains on it. --refit_full re-fits on every row
    with the tree count the early stop chose; --early_stopping_rounds 0 skips the
    holdout entirely and fits a fixed --n_estimators on every row.
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
        tree_method="hist", n_jobs=-1, random_state=seed,
        early_stopping_rounds=early_stopping_rounds, verbosity=1,
        **xgb_params,
    )
    if early_stopping_rounds is not None and early_stopping_rounds <= 0:
        # FIXED TREES (2026-09-05, flag-gated): no holdout, no early stop, every row.
        clf = XGBClassifier(scale_pos_weight=spw, objective="binary:logistic", eval_metric="aucpr",
                            tree_method="hist", n_jobs=-1, random_state=seed, verbosity=1, **xgb_params)
        clf.fit(X_train, y_train, **({"sample_weight": sw_train} if sw_train is not None else {}))
        clf.get_booster().set_attr(best_iteration=str(int(xgb_params.get("n_estimators", n_estimators)) - 1))
        logging.info(f"  trained {clf.best_iteration+1} trees (FIXED, no early stop, all {len(X_train):,} rows)")
        return clf
    fk = dict(eval_set=[(X_train.iloc[val_start:], y_train[val_start:])],
              verbose=False)
    if sw_fit is not None:
        fk["sample_weight"] = sw_fit
        fk["sample_weight_eval_set"] = [sw_val]
    clf.fit(X_train.iloc[:val_start], y_train[:val_start], **fk)
    logging.info(f"  trained {clf.best_iteration+1} trees "
                 f"(early stopped from {n_estimators})")
    if refit_full:
        # REFIT ON EVERY ROW (2026-09-05, flag-gated): keep the early-stopped tree count,
        # drop the holdout so the newest fifth of the window is in the fit.
        n_best = int(clf.best_iteration) + 1
        p2 = dict(xgb_params); p2["n_estimators"] = n_best
        clf2 = XGBClassifier(scale_pos_weight=spw, objective="binary:logistic", eval_metric="aucpr",
                             tree_method="hist", n_jobs=-1, random_state=seed, verbosity=1, **p2)
        clf2.fit(X_train, y_train, **({"sample_weight": sw_train} if sw_train is not None else {}))
        clf2.get_booster().set_attr(best_iteration=str(n_best - 1))
        logging.info(f"  --refit_full: refit {n_best} trees on all {len(X_train):,} rows (holdout was {len(X_train)-val_start:,})")
        clf = clf2
    if _diag.enabled():
        try:
            _ev = clf.evals_result()
        except Exception:
            _ev = None
        _diag.dump_json("T_fit", {"val_start_row": int(val_start), "n_fit": int(val_start),
                                  "n_val": int(len(X_train) - val_start), "best_iteration": int(clf.best_iteration),
                                  "n_estimators": xgb_params.get("n_estimators", n_estimators),
                                  "params": xgb_params, "evals_result": _ev})
    return clf


class BaggedXGB:
    """Average of N seed-varied XGBClassifiers. Quacks like one classifier for
    everything Phases 11-12 touch: predict_proba, feature_names_in_,
    feature_importances_, best_iteration, evals_result. Pickles by reference to
    this module, so load it from a process running 4__Predictor.py (as the
    nightly does); other tools should read summary.json instead."""
    def __init__(self, members, bag_mode="prob"):
        self.members = list(members)
        self.bag_mode = bag_mode
        self.feature_names_in_ = self.members[0].feature_names_in_
        self.best_iteration = int(round(np.mean([m.best_iteration for m in self.members])))
        self.feature_importances_ = np.mean([m.feature_importances_ for m in self.members], axis=0)

    def predict_proba(self, X):
        if getattr(self, "bag_mode", "prob") == "rank":
            # Rank-average (predictor lab 2026-09-02): averaging PROBABILITIES compressed
            # the extreme-tail scores the 3-slot book lives on (bag3 lost 11pp on the
            # honest window). Averaging each member's percentile rank keeps the tail
            # shape while still voting across seeds.
            from scipy.stats import rankdata
            n = float(len(X))
            p = np.mean([rankdata(m.predict_proba(X)[:, 1]) / n for m in self.members], axis=0)
            return np.column_stack([1.0 - p, p])
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def evals_result(self):
        return self.members[0].evals_result()


def train_bagged(n_bags, seed, bag_mode="prob", **kw):
    """Fit n_bags models via train_model with seeds seed..seed+n-1 and wrap them.
    n_bags == 1 returns the plain classifier, exactly the pre-flag behaviour."""
    if n_bags <= 1:
        return train_model(seed=seed, **kw)
    members = []
    for i in range(n_bags):
        logging.info(f"  bag member {i+1}/{n_bags} (seed {seed + i})")
        members.append(train_model(seed=seed + i, **kw))
    bag = BaggedXGB(members, bag_mode=bag_mode)
    logging.info(f"  BaggedXGB: {n_bags} members ({bag_mode}-averaged), mean best_iteration {bag.best_iteration}")
    return bag


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

    # Beta calibration (diagnostic - does not alter inference UpPrediction logic)
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
        except CalibratorInversionError:
            # An inverted calibration is not a "skip calibration" case: it means the
            # fit says the ranking should be flipped. Let it kill the run.
            raise
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
            f"  (diagnostic only - does not gate UpPrediction)",
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
# Phase 12: Inference - score all tickers, write RFpredictions               #
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


# -------------------------------------------------------------------------- #
# Phase 13: pred-space PARTIAL FACTOR NEUTRALIZATION                          #
# -------------------------------------------------------------------------- #
# Was the standalone stage 4.5__NeutralizePreds.py (2026-07-01 .. 2026-07-27);
# folded in here 2026-07-27 so the predictor emits ship-ready preds in one pass.
# Merging it removed a full re-read of RFpredictions + ProcessedData_v2 (the old
# stage's dominant cost) and the RFpredictions -> _raw directory swap: the factor
# panel and the scores are already in memory at this point, and the pre-
# neutralization value is preserved per row in the `raw_up_prob` column.
#
# Per day: project the cross-section of raw_score onto [beta_spy,
# atr_percentage, log dollar_volume_ma_10] and subtract dose*projection, then
# quantile-map the neutralized scores back onto that day's ORIGINAL
# UpProbability values. The per-day marginal is preserved EXACTLY -- only the
# within-day ordering changes, so the backtester's threshold/percentile
# machinery sees the same distribution.
#
# Validated 2026-07-01 (pinned BT_AS_OF=2026-06-26, BT_SAMPLE_SEED 42-45: base
# 254.5%/Sh 3.68 vs dose-0.3 305.7-340.8%/Sh 4.05-4.37), 2026-07-03 on the live
# window, and 2026-07-12 for the spy200v2 adaptive dose (live 80.2% vs 52.2%).
NEUT_FACTORS = ["beta_spy", "atr_percentage", "dollar_volume_ma_10"]


def spy_above_200ema_flag():
    """Date-indexed bool Series: SPY close >= its 200d EMA.

    Splices the deep index lake with the live one, then tries a yfinance tail
    freshen (the live lake can lag by days). EMA is computed over the full
    spliced history. Returns None if no SPY parquet exists.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    parts = []
    for d in [os.path.join(root, "Data", "IndexesFull"),
              os.path.join(root, "Data", "Indexes")]:
        p = os.path.join(d, "SPY.parquet")
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)
        if "Date" not in s.columns:
            s = s.reset_index()
        dcol = [c for c in s.columns if str(c).lower() == "date"][0]
        s = s.rename(columns={dcol: "Date"})[["Date", "Close"]].dropna()
        s["Date"] = pd.to_datetime(s["Date"])
        parts.append(s)
    if not parts:
        return None
    spy = pd.concat(parts).drop_duplicates("Date", keep="last").sort_values("Date")
    try:
        import yfinance as yf
        tail = yf.download("SPY", start=str((spy["Date"].max()
                           - pd.Timedelta(days=10)).date()), progress=False,
                           auto_adjust=False)
        tail = tail.reset_index()
        if hasattr(tail.columns, "get_level_values"):
            tail.columns = [c[0] if isinstance(c, tuple) else c for c in tail.columns]
        tail = tail[["Date", "Close"]].dropna()
        tail["Date"] = pd.to_datetime(tail["Date"])
        spy = pd.concat([spy, tail]).drop_duplicates("Date", keep="last").sort_values("Date")
    except Exception as e:
        logging.warning(f"  yfinance SPY freshen failed ({type(e).__name__}); "
                        f"lake flag ends {spy['Date'].max().date()}")
    spy = spy.reset_index(drop=True)
    ema = spy["Close"].ewm(span=200, adjust=False).mean()
    return pd.Series((spy["Close"] >= ema).values,
                     index=spy["Date"].values).sort_index()


def neutralize_upprob(dates, tickers, up_prob, score, factors,
                      mode="spy200v2", dose=0.3, dose_above=0.15,
                      dose_below=0.30, flag_max_stale_days=14):
    """Per-day partial factor-neutralization of `score`, quantile-mapped back
    onto `up_prob`.

    dates    : datetime64 array (one row per ticker-day)
    tickers  : ticker array (only used to fix the within-day row order, so the
               result does not depend on file-read order)
    up_prob  : the UpProbability values that would ship un-neutralized
    score    : the raw model score being neutralized (raw_score)
    factors  : dict name -> float64 array for NEUT_FACTORS (missing -> all-NaN)

    Returns (new_up, meta). new_up is a per-day permutation of up_prob.
    """
    n = len(dates)
    dcodes, dvals = pd.factorize(pd.to_datetime(dates), sort=True)
    n_days = int(dcodes.max()) + 1 if n else 0
    # Date-major, ticker-minor: reproduces the old stage's sort_values(["Date","Ticker"]),
    # which is what breaks ties in the quantile map below. factorize(sort=True) gives
    # alphabetical ticker codes, so lexsort stays on ints (an object-array lexsort over
    # millions of rows is minutes of Python string compares).
    tcodes = pd.factorize(np.asarray(tickers), sort=True)[0]
    order = np.lexsort((tcodes, dcodes))
    bounds = np.searchsorted(dcodes[order], np.arange(n_days + 1))

    # per-day dose
    day_dose = np.full(n_days, float(dose))
    meta = {"mode": mode, "n_days": n_days,
            "dose_desc": ("%.2f" % dose if mode == "fixed"
                          else "spy200v2 %.2f/%.2f" % (dose_above, dose_below))}
    if mode == "spy200v2":
        flag = spy_above_200ema_flag()
        if flag is None:
            raise RuntimeError("spy200v2: no SPY parquet found in Data/Indexes[Full]")
        n_stale = 0
        for di in range(n_days):
            d = pd.Timestamp(dvals[di])
            sub = flag.loc[:d]
            if len(sub) == 0 or (d - pd.Timestamp(sub.index[-1])).days > flag_max_stale_days:
                day_dose[di] = dose_below      # safe degrade = incumbent dose
                n_stale += 1
            else:
                day_dose[di] = dose_above if bool(sub.iloc[-1]) else dose_below
        sig_d = pd.Timestamp(dvals[-1])
        sig_flag = flag.loc[:sig_d]
        sig_stale = (len(sig_flag) == 0
                     or (sig_d - pd.Timestamp(sig_flag.index[-1])).days > flag_max_stale_days)
        logging.info("  spy200v2: dose_above=%.2f on %.1f%% of %d days; %d days degraded "
                     "to dose_below (stale flag); signal-day flag date %s (above=%s)"
                     % (dose_above, 100 * float((day_dose == dose_above).mean()), n_days,
                        n_stale, sig_flag.index[-1] if len(sig_flag) else "NONE",
                        bool(sig_flag.iloc[-1]) if len(sig_flag) else "n/a"))
        if sig_stale:
            logging.warning("  signal-day SPY flag stale/missing -> trading dose_below "
                            "%.2f today" % dose_below)
        meta["signal_day_flag_stale"] = bool(sig_stale)

    # design matrix columns (log-transform dollar volume, as validated)
    XV = {
        "beta":  np.asarray(factors["beta_spy"], dtype=np.float64),
        "atr":   np.asarray(factors["atr_percentage"], dtype=np.float64),
        "logdv": np.log(np.clip(np.asarray(factors["dollar_volume_ma_10"],
                                           dtype=np.float64), 1.0, None)),
    }
    sv = np.asarray(score, dtype=np.float64)
    neut = np.full(n, np.nan)
    for di in range(n_days):
        idx = order[bounds[di]:bounds[di + 1]]
        if len(idx) < 50:                      # too thin to fit -> pass through
            neut[idx] = sv[idx]
            continue
        y = sv[idx]
        good_y = np.isfinite(y)
        cols = [np.ones(len(idx))]
        for k in XV:
            c = XV[k][idx]
            med = np.nanmedian(c)
            c = np.where(np.isfinite(c), c, 0.0 if not np.isfinite(med) else med)
            c = np.clip(c, np.nanpercentile(c, 1), np.nanpercentile(c, 99))
            sd = c.std()
            cols.append((c - c.mean()) / (sd if sd > 1e-12 else 1.0))
        X = np.column_stack(cols)
        proj = np.zeros(len(idx))
        if good_y.sum() >= 20:
            coef, *_ = np.linalg.lstsq(X[good_y], y[good_y], rcond=None)
            proj = X @ coef - (X @ coef)[good_y].mean() + y[good_y].mean()
            proj = proj - y[good_y].mean()     # centered systematic component
        out = y - day_dose[di] * proj
        out[~good_y] = y[~good_y]
        neut[idx] = out

    # quantile-map back onto each day's ORIGINAL UpProbability values
    up = np.asarray(up_prob)
    new_up = np.full(n, np.nan, dtype=up.dtype if up.dtype.kind == "f" else np.float64)
    for di in range(n_days):
        idx = order[bounds[di]:bounds[di + 1]]
        u = up[idx]
        nv = neut[idx]
        ok = np.isfinite(u) & np.isfinite(nv)
        if ok.sum() < 5:
            new_up[idx] = u
            continue
        ranks = np.empty(ok.sum(), dtype=np.int64)
        ranks[np.argsort(nv[ok], kind="stable")] = np.arange(ok.sum())
        tmp = u.copy()
        tmp[ok] = np.sort(u[ok])[ranks]
        new_up[idx] = tmp
    if _diag.enabled():
        _diag.dump_parquet("P_neut_by_day", pd.DataFrame({"Date": pd.to_datetime(dvals), "dose": day_dose,
                                                          "n_rows": np.diff(bounds)}))
    return new_up, meta


##############################################################################
# Phase 14: CONVICTION-MOMENTUM TILT   (was the standalone 4.6__ConvictionMomentum.py)
#
#     up' = clip( up + k * mean_over_spans( up - EMA_span(up) ), lo, hi )
#
# Algebraically (1+k)*up - k*EMA: a tilt toward names whose model conviction is
# RISING and away from names where it is fading. It is a first derivative of
# conviction, which the flat level does not expose.
#
# WHY IT WORKS: `can_buy` in 5__NightlyBackTester.py is a WITHIN-TICKER SPIKE
# DETECTOR -- it fires when a ticker's UpProbability clears that ticker's OWN
# rolling 90th/95th percentile. Amplifying a name's deviation from its own trend
# makes genuine rising-conviction moves clear that bar decisively. The exact
# inverse (EMA smoothing) damps those peaks and LOST 27-32pp in full sim.
#
# EVIDENCE (2026-07-27, full slot sim, BT_SAMPLE_SEED pinned), main 252d window:
#   baseline, 8 seeds  : ann 59.40-62.13 | Sh 1.48-1.54 | DD 25.34-27.12
#   k=.5 MULTI 3/6/12  : ann 98.73       | Sh 2.40      | DD 12.16
#   -> +36pp above the baseline's BEST of 8 seeds; zero overlap on any metric.
# Every arm of the family beats baseline in all 3 tested windows. Seed-INVARIANT
# (42/43/44/999): the tilt removes ties, so BT_SAMPLE_SEED goes inert. Parameter
# plateau over k .5/1.0 x span 3/6/12 -- not a knife-edge.
#
# RESIDUAL (do not let the fold-in launder this): the 3 validation windows
# OVERLAP (~2 independent periods); pre-2023 slices are unreachable because
# Data/RFpredictions only spans 2023-08-30+; and the baseline prints NEGATIVE in
# the older asof windows, contradicting the live record, still unexplained.
# Rank-IC FALLS as returns rise (0.18 -> 0.11 -> 0.06) -- IC would have rejected
# this, which is why it is judged on the full sim only.
##############################################################################

def conviction_momentum_tilt(dates, tickers, up_prob, k=0.5, spans=(3, 6, 12),
                             lo=0.30, hi=0.70):
    """Per-ticker causal conviction tilt. Returns tilted values in INPUT row order.

    `combined` is not guaranteed to be sorted by (ticker, date), and the EMA is a
    time-series recursion -- computing it on unsorted rows silently produces a
    different, wrong number rather than an error. So sort explicitly, transform,
    then restore the original row order via the preserved index.
    """
    df = pd.DataFrame({"t": np.asarray(tickers), "d": np.asarray(dates),
                       "up": np.asarray(up_prob, dtype=np.float64)})
    df = df.sort_values(["t", "d"], kind="mergesort")     # stable: ties keep input order
    g = df.groupby("t", sort=False)["up"]

    tot = None
    for sp in spans:
        dev = df["up"] - g.transform(lambda s, _sp=sp: s.ewm(span=_sp, adjust=False).mean())
        tot = dev if tot is None else tot + dev
    new = (df["up"] + k * (tot / float(len(spans)))).clip(lo, hi)
    return new.sort_index().to_numpy()                     # back to input row order


def _cm_kwargs():
    """Phase 14 parameters off the CLI. Shared by the inference path and by --cm_retilt,
    so a sweep cannot silently tilt with parameters other than the ones it printed."""
    spans = tuple(int(x) for x in str(args.cm_spans).split(",") if str(x).strip())
    if not spans:
        raise SystemExit("--cm_spans parsed to nothing")
    return dict(k=args.cm_k, spans=spans, lo=args.cm_lo, hi=args.cm_hi)


def run_cm_retilt():
    """RETILT MODE -- Phase 14 applied to an EXISTING prediction dir, no inference.

    This was the whole remaining reason 4.6__ConvictionMomentum.py existed: an A/B or a
    k/spans sweep needs to re-tilt preds that are already on disk, and re-scoring ~1,900
    tickers to change one scalar is absurd. It calls the SAME conviction_momentum_tilt()
    Phase 14 uses, so the two can no longer drift apart the way two copies could.

    IDEMPOTENCE (this is new, and it matters now that Phase 14 is the default): the base
    is `pre_cm_up_prob` when that column exists, i.e. the post-neutralization pre-tilt
    value. Pointing this at the live dir therefore RE-tilts from the same starting point
    instead of compounding a second tilt onto an already-tilted book, which is exactly
    what the standalone tool did if you ran it twice.
    """
    src = args.cm_retilt
    ck = _cm_kwargs()
    logging.info("RETILT: k=%.2f spans=%s clip=[%.2f,%.2f]"
                 % (ck["k"], list(ck["spans"]), ck["lo"], ck["hi"]))
    logging.info("  source: %s" % src)
    if not os.path.isdir(src):
        raise SystemExit("RETILT ABORT: source dir does not exist: %s" % src)
    files = sorted(f for f in os.listdir(src) if f.endswith(".parquet"))
    logging.info("  %d ticker files" % len(files))
    if len(files) < args.cm_min_tickers:
        raise SystemExit("RETILT ABORT: only %d files (< %d) -- %s left untouched."
                         % (len(files), args.cm_min_tickers, src))

    write = not args.cm_retilt_dry_run
    out = args.cm_retilt_out
    if write:
        if os.path.abspath(out) == os.path.abspath(src):
            raise SystemExit("RETILT ABORT: --cm_retilt_out equals the source dir; that "
                             "would destroy the rollback base. Use --cm_retilt_apply.")
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)

    def _one(fn):
        d = pd.read_parquet(os.path.join(src, fn))
        if "UpProbability" not in d.columns or args.date_column not in d.columns:
            return None
        d[args.date_column] = pd.to_datetime(d[args.date_column])
        d = d.sort_values(args.date_column).reset_index(drop=True)
        based_on_pre = "pre_cm_up_prob" in d.columns
        base_col = "pre_cm_up_prob" if based_on_pre else "UpProbability"
        base = pd.to_numeric(d[base_col], errors="coerce").astype(np.float32)
        new = conviction_momentum_tilt(d[args.date_column].values,
                                       np.full(len(d), fn[:-8]), base, **ck)
        new = new.astype(np.float32)
        bad_nan = int(((~np.isfinite(new)) & np.isfinite(base)).sum())
        oob = int(((new < ck["lo"] - 1e-6) | (new > ck["hi"] + 1e-6)).sum())
        moved = int((np.abs(new.astype(np.float64) - base.astype(np.float64)) > 1e-9).sum())
        if write:
            d["pre_cm_up_prob"] = base
            d["UpProbability"] = new
            d["DownProbability"] = (1.0 - new).astype(np.float32)
            d.to_parquet(os.path.join(out, fn), index=False)
        return dict(ticker=fn[:-8], moved=moved, rows=len(d), nan=bad_nan, oob=oob,
                    pre=based_on_pre, date=d[args.date_column].iloc[-1],
                    old=float(base.iloc[-1]), new=float(new[-1]))

    rows, n_nan, n_oob, n_moved, n_rows, n_pre = [], 0, 0, 0, 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.cm_retilt_workers) as ex:
        for fu in as_completed([ex.submit(_one, f) for f in files]):
            r = fu.result()
            if r is None:
                continue
            rows.append(r)
            n_nan += r["nan"]; n_oob += r["oob"]
            n_moved += r["moved"]; n_rows += r["rows"]; n_pre += int(r["pre"])
    logging.info("  processed %d tickers / %d rows in %.0fs"
                 % (len(rows), n_rows, time.time() - t0))
    logging.info("  tilt base: %d files from pre_cm_up_prob (already-tilted source, "
                 "re-tilted not double-tilted), %d from UpProbability"
                 % (n_pre, len(rows) - n_pre))

    # The standalone's hard gates: a tilt that emits a non-finite value, escapes the clip,
    # or leaves the signal day inert is a BROKEN transform, not a weak one.
    def _abort(m):
        raise SystemExit("RETILT GATE FAILED: %s\nABORT -- %s left untouched." % (m, src))

    if n_nan:
        _abort("transform introduced %d non-finite values" % n_nan)
    if n_oob:
        _abort("%d values escaped the clip range" % n_oob)
    if len(rows) < args.cm_min_tickers:
        _abort("only %d tickers processed (< %d)" % (len(rows), args.cm_min_tickers))

    S = pd.DataFrame(rows)
    sig_date = S["date"].max()
    sd = S[S["date"] == sig_date].copy()
    changed = float((np.abs(sd["new"] - sd["old"]) > 1e-9).mean()) if len(sd) else 0.0
    logging.info("  signal day %s: %d names, %.1f%% moved"
                 % (pd.Timestamp(sig_date).date(), len(sd), 100 * changed))
    if changed < args.cm_min_changed:
        _abort("signal day barely moved (%.1f%% < %.0f%%) -- inert transform, short history?"
               % (100 * changed, 100 * args.cm_min_changed))

    for nm, col in (("pre-tilt ", "old"), ("post-tilt", "new")):
        v = sd[col]
        logging.info("  %s  min %.4f  p50 %.4f  p95 %.4f  max %.4f  sd %.4f"
                     % (nm, v.min(), v.median(), v.quantile(.95), v.max(), v.std()))
    b = args.cm_retilt_book
    old_top = set(sd.nlargest(b, "old")["ticker"])
    new_top = set(sd.nlargest(b, "new")["ticker"])
    logging.info("  top-%d book delta: %d of %d unchanged | entering %s | leaving %s"
                 % (b, len(old_top & new_top), b,
                    sorted(new_top - old_top) or "-", sorted(old_top - new_top) or "-"))
    logging.info("  NOTE: top-N is only indicative. `can_buy` fires per ticker against "
                 "that TICKER'S OWN rolling percentile, so the tilt changes WHICH names "
                 "trigger over time. Judge it by the backtest, not by this table.")
    logging.info("  rows moved: %.1f%% (%d of %d)"
                 % (100 * n_moved / max(n_rows, 1), n_moved, n_rows))

    if args.cm_retilt_dry_run:
        logging.info("--cm_retilt_dry_run: nothing written.")
        return
    logging.info("wrote %d files -> %s" % (len(rows), out))
    if not args.cm_retilt_apply:
        logging.info("--cm_retilt_apply NOT set: %s is untouched (the safe default)." % src)
        logging.info("Backtest it with: python 5__NightlyBackTester.py --force --data_dir %s"
                     % out)
        return
    roll = src.rstrip("/\\") + "_precm"
    if os.path.isdir(roll):
        shutil.rmtree(roll)
    os.rename(src, roll)
    os.rename(out, src)
    logging.info("APPLIED: %s is now tilted (k=%.2f spans=%s). Rollback dir: %s"
                 % (src, ck["k"], list(ck["spans"]), roll))


def run_inference(model_path, input_dir, output_dir, date_col, ticker_col,
                   top_frac_per_day, max_files=None, calib_path=None,
                   neutralize=False, neut_kwargs=None,
                   neut_min_tickers=1000, neut_min_changed=0.05,
                   conviction_momentum=False, cm_kwargs=None,
                   cm_min_tickers=1000, cm_min_changed=0.05, infer_tail=None, pool_top_n=0):
    """Load saved model, score all tickers, write RFpredictions parquets.

    With neutralize=True the written UpProbability is the Phase-13 neutralized
    one and the un-neutralized value is kept in `raw_up_prob`. Returns a status
    dict: {"neutralized": bool, "gate_failed": bool, ...}.
    """
    os.makedirs(output_dir, exist_ok=True)
    status = {"neutralized": False, "gate_failed": False}

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

        # Read only the columns the model + universe filter + output need. The
        # dropped feature families never appear in feat_names, so projecting them
        # away at read time is identical downstream (missing model features are
        # still backfilled to 0.0 below). Big I/O + memory cut on wide v2 data.
        needed = {date_col, ticker_col, "Open", "High", "Low", "Close", "Volume",
                  "dollar_volume_ma_10", "atr_percentage", "RSI",
                  "VIX_Close", "vix_close", "Distance to Resistance (%)",
                  "Distance to Support (%)", "volatility",
                  # source for the ret_today alias below (models trained post-leak-guard)
                  "percent_change_close", "percent_change_Close"}
        needed |= set(raw_features) | set(xs_src)
        needed |= {"Realized_Vol_21d"}   # limit/day features (Order 66) derive from it at inference
        # Phase-13 factors: needed even when the model itself doesn't use them.
        needed |= set(NEUT_FACTORS)

        def _load_arrow_infer(fn):
            pf = pq.ParquetFile(os.path.join(input_dir, fn))
            names = pf.schema_arrow.names
            if date_col not in names or pf.metadata.num_rows == 0:
                return None
            tbl = pf.read(columns=[c for c in names if c in needed])
            # Peak-RAM cap: keep only this ticker's most recent rows. Files are stored
            # date-ascending, so the tail IS the recent history. No-op when unset.
            if infer_tail and tbl.num_rows > infer_tail:
                # take(), not slice(): a slice is zero-copy and keeps the WHOLE file's buffers
                # alive until the concat, so 4,209 files x every row sat in RAM (35 GB on a
                # 6-year panel at tail 300). take() materialises only the tail rows. Same rows.
                tbl = tbl.take(pa.array(range(tbl.num_rows - infer_tail, tbl.num_rows)))
            # Peak-RAM cap 2 (2026-09-02): the concatenated panel is ~2.1M rows x ~1,400
            # float64 columns (~24 GB in Arrow, doubled by to_pandas), which OOMed a
            # 63 GB box at 49.6 GB. downcast() below turns every float64 into float32
            # anyway, so casting feature columns to float32 HERE is identical for the
            # model; only the Phase-13 factor columns (fit in native float64) and the
            # key/price columns keep their types.
            _keep64 = set(NEUT_FACTORS) | {date_col, ticker_col, "Open", "High", "Low",
                                           "Close", "Volume", "percent_change_close",
                                           "percent_change_Close"}
            _fields = []
            for f in tbl.schema:
                if pa.types.is_float64(f.type) and f.name not in _keep64:
                    _fields.append(pa.field(f.name, pa.float32()))
                else:
                    _fields.append(f)
            _target = pa.schema(_fields)
            if _target != tbl.schema:
                tbl = tbl.cast(_target)
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

        # Single concat + single to_pandas() - avoids 4265 individual GIL hits
        try:
            _big = pa.concat_tables(arrow_tables, promote_options="default")
            del arrow_tables
            # self_destruct releases each Arrow buffer as its pandas block is built,
            # so the peak is ~1x the table instead of ~2x.
            combined = _big.to_pandas(split_blocks=True, self_destruct=True)
            del _big
        except Exception:
            combined = pd.concat([t.to_pandas() for t in arrow_tables], ignore_index=True)
            del arrow_tables

        combined[date_col] = pd.to_datetime(combined[date_col])
        if "dow" in raw_features and "dow" not in combined.columns:
            combined["dow"] = combined[date_col].dt.dayofweek.astype(np.float32)
        # v2 FeatureFramework emits lowercase 'vix_close'; the backtester and the
        # RFpredictions output schema require uppercase 'VIX_Close'. Alias so
        # inference carries it through (otherwise the backtester drops the universe).
        if "VIX_Close" not in combined.columns and "vix_close" in combined.columns:
            combined["VIX_Close"] = combined["vix_close"]
        # ret_today feature alias: at train time this was TODAY's return (stashed before
        # the label shift); at inference the raw panel target column IS today's return.
        # No-op for models (like the current ship) that don't have this feature.
        if "ret_today" not in combined.columns:
            for _src in ("percent_change_close", "percent_change_Close"):
                if _src in combined.columns:
                    combined["ret_today"] = combined[_src]
                    break
        if "lim_depth" in raw_features and "lim_depth" not in combined.columns:
            add_limit_features(combined, "Realized_Vol_21d", 1.5)
        if "day_n" in raw_features and "day_n" not in combined.columns:
            add_day_features(combined, date_col, "Realized_Vol_21d", k=1.5)
        # Phase 13 fits its per-day regression on the panel's native float64 factor
        # values; downcast() below rounds every float64 column to float32, so stash
        # them first (and carry the stash through the sort by the frame's own index).
        neut_f64 = None
        if neutralize:
            neut_f64 = {c: (combined[c].to_numpy(np.float64, copy=True)
                            if c in combined.columns
                            else np.full(len(combined), np.nan))
                        for c in NEUT_FACTORS}
        combined = downcast(combined)
        # Sort by taking positions, not sort_values + reset_index: the latter leaves a
        # shuffled index whose reset_index(drop=True) deep-copies and consolidates every
        # float32 column into ONE contiguous block (703 cols x 3.06M rows = 8.02 GiB), on
        # top of the frame already resident. That OOM'd the 2026-08-13 nightly run. .take()
        # with a RangeIndex reaches the same row order at one block-wise copy.
        # kind must stay "quicksort": that is what sort_values() used, and the tie order
        # within a date is part of the contract downstream (verified row-identical; the
        # stable/mergesort variants reorder ties and do NOT reproduce the old frame).
        _pos = np.argsort(combined[date_col].to_numpy(), kind="quicksort")
        combined = combined.take(_pos)
        combined.index = pd.RangeIndex(len(combined))
        if neut_f64 is not None:
            neut_f64 = {k: v[_pos] for k, v in neut_f64.items()}
        logging.info(f"  combined: {combined.shape}  "
                     f"{combined[ticker_col].nunique()} tickers  "
                     f"{combined[date_col].nunique()} dates")
        if _diag.enabled():
            _diag.stamp("4__Predictor")
            _diag.dump_json("P_load", {"n_files": len(files), "rows": int(len(combined)),
                                       "tickers": int(combined[ticker_col].nunique()),
                                       "dates": int(combined[date_col].nunique()), "infer_tail": infer_tail,
                                       "model_features": len(feat_names), "raw": len(raw_features),
                                       "xs": len(xs_features), "columns_loaded": int(len(combined.columns)),
                                       "calibrator": ({"a": calibrator.a, "b": calibrator.b, "c": calibrator.c}
                                                      if calibrator is not None else None)})
            _diag.dump_parquet("P_rows_by_day", combined.groupby(date_col).size().reset_index(name="n"))

    with Phase("Universe filter (mark, don't drop)"):
        _, quality_mask = apply_quality_filter(combined)
        n_pass = int(quality_mask.sum())
        logging.info(f"  universe-passing: {n_pass:,} / {len(combined):,} "
                     f"({100*n_pass/len(combined):.1f}%)")

    with Phase("Fill missing model features"):
        # A feature the model expects but the panel lacks becomes a constant 0.0 (and
        # its _xs rank a constant 0.5). Same effect as before; now counted and logged.
        missing_raw = [c for c in raw_features if c not in combined.columns]
        missing_xs = [c for c in xs_src if c not in combined.columns]
        for c in missing_raw:
            combined[c] = 0.0
        for c in missing_xs:
            combined[c] = 0.0
        if missing_raw or missing_xs:
            logging.warning(f"  ZERO-FILLED {len(missing_raw)} raw + {len(missing_xs)} xs-source model "
                            f"features absent from the panel: "
                            f"{(missing_raw + missing_xs)[:12]}{' ...' if len(missing_raw) + len(missing_xs) > 12 else ''}")
        _diag.dump_json("P_zero_fill", {"missing_raw": missing_raw, "missing_xs_src": missing_xs,
                                        "n_missing": len(missing_raw) + len(missing_xs),
                                        "n_model_features": len(feat_names)})

    with Phase("Build prediction matrix X (mem-efficient: batched f32 xs-rank, no concat)"):
        # Build X directly as ONE float32 array in feat_names order: raw features copy from
        # `combined`; xs features get the per-day percentile rank of their source column
        # (quality rows only, 0.5 elsewhere), ranked in COLUMN BATCHES so the float64 rank
        # intermediate is capped to one batch. Replaces the old float64 full-rank + the
        # pd.concat frame-doubling + the .astype/.replace copies that spiked RAM at inference.
        # BIT-IDENTICAL output (verified: 0 mismatched cells, predict matches) at ~60% lower peak.
        qmask = quality_mask.values
        qidx  = np.where(qmask)[0]
        X = np.empty((len(combined), len(feat_names)), dtype=np.float32)
        dvals = combined.loc[quality_mask, date_col]
        xs_idx = []
        for j, col in enumerate(feat_names):
            if col.endswith("_xs"):
                X[:, j] = np.float32(0.5)
                xs_idx.append((j, col[:-3]))
            else:
                X[:, j] = combined[col].to_numpy(np.float32, copy=False)
        if xs_idx:
            BATCH = 50
            for i in range(0, len(xs_idx), BATCH):
                chunk = xs_idx[i:i + BATCH]
                srcs  = [s for _, s in chunk]
                rv = (combined.loc[quality_mask, srcs]
                      .groupby(dvals, sort=False)
                      .rank(pct=True, method="average", na_option="keep")
                      .to_numpy(np.float32))     # cast f64->f32 immediately, drop the f64 frame
                for bi, (j, s) in enumerate(chunk):
                    X[qidx, j] = rv[:, bi]
                del rv
        np.putmask(X, ~np.isfinite(X), np.nan)    # inf -> nan (XGBoost missing)
        logging.info(f"  built X {X.shape} ({len(xs_idx)} xs cols, mem-efficient)")
        if _diag.enabled():
            # NaN share on a row stride, not the full 16 GiB matrix.
            _sub = X[::97]
            _diag.dump_json("P_matrix", {"shape": [int(v) for v in X.shape], "n_xs_cols": len(xs_idx),
                                         "n_quality_rows": int(qmask.sum()), "n_pinned_rows": int((~qmask).sum()),
                                         "nan_share_sampled": float(np.isnan(_sub).mean()),
                                         "nan_share_sample_rows": int(len(_sub))})

    with Phase("Predict"):
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
        if _diag.enabled():
            _pc = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            _diag.dump_json("P_scores", {"percentiles": _pc,
                                         "raw": [float(v) for v in np.percentile(scores, _pc)],
                                         "cal": [float(v) for v in np.percentile(scores_cal, _pc)],
                                         "calibrated": calibrator is not None})

    with Phase("Rescale scores to backtester UpProbability"):
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

        n_fire = int((up_pred == 1).sum())
        n_days = combined[date_col].nunique()
        logging.info(f"  fires: {n_fire:,}  avg {n_fire/max(n_days,1):.1f}/day "
                     f"over {n_days} dates")
        if _diag.enabled():
            _fb = pd.DataFrame({"Date": combined[date_col].values, "q": quality_mask.values,
                                "f": (up_pred == 1)})
            _diag.dump_parquet("P_fires_by_day",
                               _fb.groupby("Date").agg(n_rows=("q", "size"), n_quality=("q", "sum"),
                                                       n_fire=("f", "sum")).reset_index())

    raw_up = None
    if neutralize:
        with Phase("Phase 13: Pred-space factor neutralization"):
            # Neutralize the values that would actually ship (post-clip float32), so the
            # result is a permutation of the shipped per-day marginal, exactly as the old
            # standalone stage produced it by re-reading the written parquets.
            raw_up = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
            n_tick = int(combined[ticker_col].nunique())
            try:
                if n_tick < neut_min_tickers:
                    raise RuntimeError("only %d tickers scored (< %d) -- universe/panel "
                                       "looks broken" % (n_tick, neut_min_tickers))
                cov = {c: float(np.isfinite(neut_f64[c]).mean()) for c in NEUT_FACTORS}
                logging.info("  factor coverage: %s"
                             % {c: round(v, 3) for c, v in cov.items()})
                # A factor the panel doesn't carry gets median-filled to a constant and
                # contributes NOTHING to the projection. That is a silently-dead leg, so
                # say it out loud (it is not a gate failure -- the shipped/validated
                # overlay has been running this way; see the 2026-07-27 note in README).
                dead = [c for c, v in cov.items() if v <= 0.0]
                if dead:
                    logging.warning("  factors with ZERO coverage (inert, no effect on the "
                                    "projection): %s" % dead)
                new_up, meta = neutralize_upprob(
                    combined[date_col].values, combined[ticker_col].values,
                    raw_up, scores, neut_f64, **(neut_kwargs or {}))

                # GATE: the signal day must actually be neutralized (stale-panel guard)
                last_day = combined[date_col].max()
                ld = combined[date_col].values == np.datetime64(last_day)
                changed = float((np.abs(new_up[ld].astype(np.float64)
                                        - raw_up[ld].astype(np.float64)) > 1e-9).mean())
                logging.info("  signal-day (%s) neutralization: %.1f%% of %d names changed"
                             % (pd.Timestamp(last_day).date(), 100 * changed, int(ld.sum())))
                if changed < neut_min_changed:
                    raise RuntimeError("signal day looks UN-neutralized (%.1f%% < %.0f%%) "
                                       "-- factor panel stale?"
                                       % (100 * changed, 100 * neut_min_changed))

                up_prob = new_up.astype(np.float64)
                status["neutralized"] = True
                status["neut"] = meta
                logging.info("  dose %s applied over %d days" % (meta["dose_desc"],
                                                                 meta["n_days"]))
                _diag.dump_json("P_neut", dict(meta, signal_day_changed=changed, signal_day=str(last_day)[:10],
                                               factor_coverage=cov, dead_factors=dead, n_tickers=n_tick,
                                               gate_failed=False))
            except Exception as e:
                # FAIL-SAFE: ship the RAW preds and flag it. main() exits non-zero so the
                # nightly runner alerts, but the backtester still gets a usable signal.
                raw_up = None
                status["gate_failed"] = True
                status["neut_error"] = str(e)
                _diag.dump_json("P_neut", {"gate_failed": True, "error": str(e), "n_tickers": n_tick})
                logging.error("  NEUTRALIZATION GATE FAILED: %s" % e)
                logging.error("  -> shipping RAW predictions (safe degrade); "
                              "the predictor will exit non-zero.")

    pre_cm_up = None
    if conviction_momentum:
        with Phase("Phase 14: Conviction-momentum tilt"):
            # Tilt the values that would actually ship, i.e. AFTER Phase 13. The
            # standalone stage read the written parquets, so it always saw the
            # post-clip float32 value -- match that or the two disagree in the
            # last decimal and the A/B stops being an A/B.
            pre_cm_up = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
            n_tick = int(combined[ticker_col].nunique())
            ck = dict(cm_kwargs or {})
            try:
                if n_tick < cm_min_tickers:
                    raise RuntimeError("only %d tickers scored (< %d) -- universe/panel "
                                       "looks broken" % (n_tick, cm_min_tickers))
                new_up = conviction_momentum_tilt(
                    combined[date_col].values, combined[ticker_col].values,
                    pre_cm_up, **ck)

                # The standalone's two hard gates, kept verbatim in spirit: a tilt
                # that emits a non-finite value or escapes the clip is a broken
                # transform, not a weak one.
                bad_nan = int(((~np.isfinite(new_up)) & np.isfinite(pre_cm_up)).sum())
                if bad_nan:
                    raise RuntimeError("tilt introduced %d non-finite values" % bad_nan)
                oob = int(((new_up < ck.get("lo", 0.30) - 1e-6)
                           | (new_up > ck.get("hi", 0.70) + 1e-6)).sum())
                if oob:
                    raise RuntimeError("%d values escaped the clip range" % oob)

                # GATE: the signal day must actually move (short-history guard)
                last_day = combined[date_col].max()
                ld = combined[date_col].values == np.datetime64(last_day)
                changed = float((np.abs(new_up[ld].astype(np.float64)
                                        - pre_cm_up[ld].astype(np.float64)) > 1e-9).mean())
                logging.info("  signal-day (%s) tilt: %.1f%% of %d names moved"
                             % (pd.Timestamp(last_day).date(), 100 * changed, int(ld.sum())))
                if changed < cm_min_changed:
                    raise RuntimeError("signal day barely moved (%.1f%% < %.0f%%) -- inert "
                                       "transform, short history?"
                                       % (100 * changed, 100 * cm_min_changed))

                rising = float((new_up[ld] > pre_cm_up[ld]).mean())
                logging.info("  rising-conviction names today: %.1f%% | mean |delta| %.5f"
                             % (100 * rising,
                                float(np.abs(new_up[ld] - pre_cm_up[ld]).mean())))
                up_prob = new_up.astype(np.float64)
                status["conviction_momentum"] = True
                status["cm"] = dict(ck, n_tickers=n_tick, signal_day_changed=changed)
                logging.info("  applied k=%.2f spans=%s clip=[%.2f,%.2f] over %d tickers"
                             % (ck.get("k", 0.5), list(ck.get("spans", (3, 6, 12))),
                                ck.get("lo", 0.30), ck.get("hi", 0.70), n_tick))
                _diag.dump_json("P_cm", dict(status["cm"], signal_day=str(last_day)[:10], rising_share=rising,
                                             mean_abs_delta=float(np.abs(new_up[ld] - pre_cm_up[ld]).mean()),
                                             gate_failed=False))
            except Exception as e:
                # FAIL-SAFE, matching Phase 13: ship the un-tilted preds and flag it.
                pre_cm_up = None
                status["gate_failed"] = True
                status["cm_error"] = str(e)
                _diag.dump_json("P_cm", {"gate_failed": True, "error": str(e), "n_tickers": n_tick})
                logging.error("  CONVICTION-MOMENTUM GATE FAILED: %s" % e)
                logging.error("  -> shipping UN-TILTED predictions (safe degrade); "
                              "the predictor will exit non-zero.")

    if pool_top_n and pool_top_n > 0:
        with Phase("Pool cap: top-%d names per day" % pool_top_n):
            _rk = pd.Series(up_prob).groupby(combined[date_col].values).rank(ascending=False, method="first").values
            _cut = (_rk > pool_top_n) & (up_prob >= 0.40)
            up_prob = np.where(_cut, 0.30, up_prob)
            logging.info("  pool cap: floored %s rows at/above 0.40 to 0.30" % format(int(_cut.sum()), ","))

    with Phase("Write per-ticker parquets to RFpredictions"):
        combined["UpProbability"]     = np.clip(up_prob, 0.01, 0.99).astype(np.float32)
        combined["DownProbability"]   = (1.0 - combined["UpProbability"]).astype(np.float32)
        combined["PositiveThreshold"] = np.float32(1.0 - top_frac_per_day)
        combined["NegativeThreshold"] = np.float32(np.nan)
        combined["UpPrediction"]      = up_pred
        combined["raw_score"]         = scores
        if calibrator is not None:
            combined["cal_score"] = scores_cal
        if raw_up is not None:
            # Phase 13 ran: keep the un-neutralized UpProbability alongside it. This
            # replaces the old RFpredictions -> RFpredictions_raw directory swap --
            # every row now carries its own rollback value.
            combined["raw_up_prob"] = raw_up
        if pre_cm_up is not None:
            # Phase 14 ran: the post-neutralization, pre-tilt value. Same rollback
            # contract as raw_up_prob, and the same column name the standalone
            # 4.6 stage wrote, so anything reading it keeps working.
            combined["pre_cm_up_prob"] = pre_cm_up

        base_out = [date_col, "Open", "High", "Low", "Close", "Volume",
                    "UpProbability", "DownProbability",
                    "PositiveThreshold", "NegativeThreshold",
                    "UpPrediction", "raw_score"]
        optional = ["raw_up_prob", "pre_cm_up_prob", "VIX_Close", "Distance to Resistance (%)",
                    "Distance to Support (%)", "volatility"]
        out_cols = [c for c in base_out + optional if c in combined.columns]

        def _write_one(tk, sub):
            sub.to_parquet(os.path.join(output_dir, f"{tk}.parquet"), index=False)

        n_written = 0
        pbar = tqdm(total=combined[ticker_col].nunique(), desc="Writing")
        inflight = set()

        def _harvest(done):
            nonlocal n_written
            for f in done:
                try:
                    f.result()
                    n_written += 1
                except Exception as e:
                    logging.warning(f"  write failed: {e}")
                pbar.update(1)

        # Bounded parallelism: parquet write/compression releases the GIL, so
        # threads overlap I/O. Cap in-flight tasks (each holds one small
        # per-ticker copy) to keep peak memory flat.
        with ThreadPoolExecutor(max_workers=8) as ex:
            for tk, grp in combined.groupby(ticker_col, sort=False):
                inflight.add(ex.submit(_write_one, tk, grp[out_cols].copy()))
                if len(inflight) >= 32:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    _harvest(done)
            _harvest(as_completed(inflight))
        pbar.close()
        logging.info(f"  wrote {n_written:,} ticker parquets to {output_dir}"
                     + (f"  [NEUTRALIZED: dose {status['neut']['dose_desc']}]"
                        if status["neutralized"] else "  [RAW: no neutralization]"))
    return status





# -------------------------------------------------------------------------- #
# OOS decay harness - measure true out-of-sample generalization              #
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
    band) - this is the strategy-relevant number (each day you hold the band).
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
            logging.error("No data loaded - check --input_dir")
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
                            "  (data snapshot changed since training - boundaries approximate)"))

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
        logging.error("  OOS slice is empty - no data after the calib window. "
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
    lines = ["=" * 92, "OOS DECAY REPORT - true out-of-sample generalization", "=" * 92,
             f"Model:    {model_path}",
             f"Features: {len(feat_names)}   Label: topq (top_frac={args.topq_frac})   "
             f"rank = per-day percentile of model score",
             ""]
    lines += oos_band_table_lines(is_df, "IS / TRAIN (in-sample - fit-optimistic)",
                                  "_ret", "_label", "_rank", date_col)
    lines.append("")
    lines += oos_band_table_lines(calib_df, "CALIB (near-OOS - used for threshold/calib only)",
                                  "_ret", "_label", "_rank", date_col)
    lines.append("")
    lines += oos_band_table_lines(oos_df, "OOS (TRUE HOLDOUT - never seen in train or calib)",
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
# Walk-forward OOS engine - judge configs across many independent windows     #
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
    the recency-weight decay. Only the data/feature/weight treatment varies - 
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
            logging.error("No data loaded - check --input_dir")
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
             "WALK-FORWARD OOS - per-day mean return NET of bottom-50% market-drift baseline",
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
              "top1_net mean AND win% vs baseline across independent OOS windows - not one month.",
              "=" * 100]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  WF report  -> {out_report}")
    logging.info(f"  WF results -> {out_parquet}")


# -------------------------------------------------------------------------- #
# Walk-forward PARAM SWEEP - parallel, multi-seed, isolates the overfit lever  #
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
    # Faithful proxy for the LIVE shipped untuned-v2 model (Data/_ship_v2/model:
    # tuned_params=null -> argparse defaults; early-stopped ~114 trees -> fix 120).
    "ship_v2_untuned": dict(n_estimators=120, max_depth=5, learning_rate=0.05,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
                         reg_alpha=0.5, reg_lambda=2.0, gamma=0.0),
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


def parse_window_spec(spec):
    """Parse a --wf_sweep_windows token into (n_days, half_life_days).

    'full'        -> (None, 720.0)    all history up to the anchor
    'full180'     -> (None, 180.0)    all history, faster forgetting
    'w252'        -> (252,  720.0)    last 252 trading days only
    'w126hl90'    -> (126,   90.0)
    n_days=None means "no hard cutoff"; the half-life is the soft one.
    """
    s = spec.strip().lower()
    m = re.fullmatch(r"full(\d+(?:\.\d+)?)?", s)
    if m:
        return None, float(m.group(1)) if m.group(1) else 720.0
    m = re.fullmatch(r"w(\d+)(?:hl(\d+(?:\.\d+)?))?", s)
    if m:
        return int(m.group(1)), float(m.group(2)) if m.group(2) else 720.0
    raise ValueError(f"bad window spec '{spec}'. Expected full / full<HL> / w<K> / w<K>hl<HL>")


def run_walkforward_sweep():
    """Parallel, multi-seed walk-forward sweep over XGB hyperparameter configs
    CROSSED with training-window specs.

    Every (config, window) cell is scored on the same following month; only the
    hyperparameters and the training slice differ. The crossing matters because
    complexity and window length are coupled -- a longer window buys samples but
    imports more non-stationarity, so the best depth/regularization is a function
    of the window and vice versa. Sweeping either one alone (the old behaviour,
    reproduced by the default --wf_sweep_windows full720) can only see a marginal
    of that surface. Each cell is fit with `wf_seeds` seeds and averaged to beat
    the ~0.02 hist nondeterminism. Fits run concurrently in threads (XGB releases
    the GIL).
    """
    date_col, target_col = args.date_column, args.target_column
    out_report  = os.path.join(args.model_dir, "walkforward_sweep_report.txt")
    out_parquet = os.path.join(args.model_dir, "walkforward_sweep_results.parquet")
    configs = [c.strip() for c in args.wf_sweep_configs.split(",") if c.strip()]
    for c in configs:
        if c not in SWEEP_CONFIGS:
            raise ValueError(f"unknown sweep config '{c}'. Known: {list(SWEEP_CONFIGS)}")
    windows = [w.strip() for w in args.wf_sweep_windows.split(",") if w.strip()]
    window_spec = {w: parse_window_spec(w) for w in windows}
    combos = [f"{c}@{w}" for c in configs for w in windows]

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
        if args.wf_feature_file:
            keep = set(open(args.wf_feature_file).read().split())
            before = len(base_cols)
            base_cols = [c for c in base_cols if c in keep]
            logging.info(f"  --wf_feature_file: {before} -> {len(base_cols)} base features "
                         f"({len(keep - set(base_cols))} listed names absent from panel)")
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
    max_anchor = str(pd.Period(args.wf_max_anchor, freq="M")) if args.wf_max_anchor else None
    anchors = []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        if m < min_anchor or (max_anchor and m > max_anchor):
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
    # The OOS row-range depends only on the anchor; the training slice and its
    # recency weights depend on (anchor, window). Both are precomputed so the
    # threaded fits never hold the GIL building them.
    uniq_days = np.unique(dates_all)
    anchor_oos = {}    # m -> (oos_lo, oos_hi, oos_period)
    slice_info = {}    # (m, win) -> (lo, k, sw)
    for (m, anchor_date, nxt) in anchors:
        oi = np.where(ym_all == nxt)[0]
        anchor_oos[m] = (int(oi[0]), int(oi[-1]) + 1, nxt)
        k = int(np.searchsorted(dates_all, anchor_date, side="right"))
        for win in windows:
            n_days, hl = window_spec[win]
            if n_days is None:
                lo = 0
            else:
                prior = uniq_days[uniq_days <= anchor_date]
                floor = prior[-n_days] if len(prior) >= n_days else prior[0]
                lo = int(np.searchsorted(dates_all, floor, side="left"))
            sw = recency_weights(pd.Series(dates_all[lo:k]), hl)
            slice_info[(m, win)] = (lo, k, sw)

    # ANCHOR-MAJOR order: every cell of one anchor finishes before the next anchor
    # starts, so a partial/interrupted run still has COMPLETE paired anchors --
    # which is what the (config x window) interaction contrast needs. Cell-major
    # order would leave one config fully done and the rest empty: unusable.
    jobs = [(cfg, win, m, seed)
            for (m, _ad, _nxt) in anchors
            for seed in range(args.wf_seeds)
            for cfg in configs
            for win in windows]
    logging.info(f"  total fits: {len(jobs)}  "
                 f"({len(configs)} configs x {len(windows)} windows x "
                 f"{len(anchors)} anchors x {args.wf_seeds} seeds)")

    # Crash/restart resume: every finished fit is appended to a JSONL as it lands,
    # so an interrupted run (reboot, OOM, Ctrl-C) resumes instead of restarting
    # from zero. A full sweep is hours of fits; losing them to a reboot is not ok.
    ckpt_path = os.path.join(args.model_dir, "sweep_partial.jsonl")
    done_rows, done_keys = [], set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue        # torn last line from a hard kill
                k = (r["config"], r["anchor_month"], r["seed"])
                if k in done_keys:
                    continue
                done_keys.add(k)
                done_rows.append(r)
        jobs = [j for j in jobs
                if (f"{j[0]}@{j[1]}", j[2], j[3]) not in done_keys]
        logging.info(f"  RESUME: {len(done_rows)} fits already in {ckpt_path}; "
                     f"{len(jobs)} left to run")
    ckpt_lock = threading.Lock()
    ckpt_fh = open(ckpt_path, "a")

    def _one(job):
        cfg, win, m, seed = job
        olo, ohi, oos_period = anchor_oos[m]
        lo, k, sw = slice_info[(m, win)]
        params = dict(SWEEP_CONFIGS[cfg])
        n_est = params.pop("n_estimators", 300)
        clf = XGBClassifier(n_estimators=n_est, tree_method="hist",
                            objective="binary:logistic", eval_metric="aucpr",
                            device=args.wf_device, n_jobs=args.wf_threads,
                            random_state=1000 + seed, **params)
        clf.fit(Xall[lo:k], y_all[lo:k], sample_weight=sw)  # views, zero-copy
        sc = clf.predict_proba(Xall[olo:ohi])[:, 1]
        t1n, shn = _sweep_band_nets(sc, olo, ohi, dates_all, ret_all)
        return {"config": f"{cfg}@{win}", "base_config": cfg, "window": win,
                "anchor_month": m, "oos_month": oos_period, "seed": seed,
                "train_rows": int(k - lo), "top1_net": t1n, "shoulder_net": shn}

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
                    with ckpt_lock:
                        ckpt_fh.write(json.dumps(r) + "\n")
                        ckpt_fh.flush()
                        os.fsync(ckpt_fh.fileno())
                    if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                        logging.info(f"  [{done}/{len(jobs)}] {r['config']:<12} "
                                     f"{r['oos_month']} s{r['seed']}: top1_net={r['top1_net']:+.4f}")
                except Exception as e:
                    j = futs[fut]
                    logging.warning(f"  job {j[0]}@{j[2]} s{j[3]} failed: {repr(e)[:120]}")

    ckpt_fh.close()
    res = pd.DataFrame(done_rows + rows)
    if res.empty:
        logging.error("SWEEP: no results (all fits failed?)")
        return
    res.to_parquet(out_parquet, index=False)

    # Seed-average per (config, anchor), then pool across anchors.
    per_anchor = (res.groupby(["config", "oos_month"])
                     .agg(top1_net=("top1_net", "mean"),
                          top1_std=("top1_net", "std"),
                          shoulder_net=("shoulder_net", "mean"))
                     .reset_index())

    lines = ["=" * 104,
             "WALK-FORWARD PARAM SWEEP - seed-averaged top1_net (per-day mean ret NET of bot-50% baseline)",
             "=" * 104,
             f"configs={configs}  windows={windows}  anchors={len(anchors)}  "
             f"seeds={args.wf_seeds}  (complexity x training-window, crossed)",
             ""]
    # per-cell per-anchor (seed-averaged) detail
    for cell in combos:
        cfg, win = cell.split("@")
        c = per_anchor[per_anchor["config"] == cell].sort_values("oos_month")
        lines.append(f"### {cell} ###   (window: {window_spec[win]}  "
                     f"params: {SWEEP_CONFIGS[cfg]})")
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
    for cell in combos:
        c = per_anchor[per_anchor["config"] == cell]
        if len(c) == 0:
            continue
        t = c["top1_net"]
        summ.append((cell, len(c), t.mean(), t.median(), (t > 0).mean(),
                     t.std(), c["top1_std"].mean(), c["shoulder_net"].mean()))
    for cell, n, tm, tmd, tw, tstd, sds, sm in sorted(summ, key=lambda x: -x[2]):
        lines.append(f"  {cell:<13} {n:>7} {tm:>14.4f} {tmd:>9.4f} {tw*100:>5.0f}% "
                     f"{tstd:>11.4f} {sds:>13.4f} {sm:>14.4f}")
    lines += ["",
              "top1_net = strategy-relevant. avg_seed_std = the nondeterminism noise floor for a",
              "single fit; anchor_std = month-to-month dispersion. A config genuinely beats another",
              "only if the gap exceeds ~avg_seed_std/sqrt(seeds). prod_optuna = live config (to beat).",
              "=" * 104]

    # ---- COUPLING PANEL: complexity x window, and the interaction contrast ----
    # Paired per anchor (same OOS month, same seeds) so the month-to-month
    # dispersion -- which dwarfs the effect -- cancels out of every contrast.
    if len(configs) > 1 and len(windows) > 1:
        cell_mean = {}
        wide = per_anchor.pivot_table(index="oos_month", columns="config",
                                      values="top1_net")
        for cell in combos:
            cell_mean[cell] = wide[cell].mean() if cell in wide else float("nan")
        lines += ["", "=" * 104,
                  "COUPLING PANEL - pooled top1_net by (config x window)", "=" * 104,
                  "  " + " " * 14 + "".join(f"{w:>14}" for w in windows)]
        for cfg in configs:
            lines.append(f"  {cfg:<14}"
                         + "".join(f"{cell_mean[f'{cfg}@{w}']:>14.4f}" for w in windows))
        # Interaction: does the window effect depend on the config? Contrast the
        # window-delta of the most complex vs the least complex config in the run.
        w0, w1 = windows[0], windows[-1]
        c0, c1 = configs[0], configs[-1]
        if all(c in wide for c in (f"{c0}@{w0}", f"{c0}@{w1}", f"{c1}@{w0}", f"{c1}@{w1}")):
            d0 = wide[f"{c0}@{w1}"] - wide[f"{c0}@{w0}"]     # window effect for c0
            d1 = wide[f"{c1}@{w1}"] - wide[f"{c1}@{w0}"]     # window effect for c1
            inter = d1 - d0
            n_a = int(inter.notna().sum())
            t_i = (inter.mean() / (inter.std(ddof=1) / np.sqrt(n_a))) if n_a > 1 and inter.std(ddof=1) > 0 else float("nan")
            lines += ["",
                      f"  window effect ({w0} -> {w1}) for {c0:<13}: {d0.mean():+.4f}  "
                      f"(paired t={d0.mean()/(d0.std(ddof=1)/np.sqrt(max(n_a,1))):+.2f})",
                      f"  window effect ({w0} -> {w1}) for {c1:<13}: {d1.mean():+.4f}  "
                      f"(paired t={d1.mean()/(d1.std(ddof=1)/np.sqrt(max(n_a,1))):+.2f})",
                      f"  INTERACTION (difference of the two)      : {inter.mean():+.4f}  "
                      f"(paired t={t_i:+.2f}, n={n_a} anchors)",
                      "",
                      "  |t|>2 on the INTERACTION line = complexity and window are COUPLED on this",
                      "  window, i.e. sweeping either knob alone was measuring a marginal of a",
                      "  surface with a ridge in it. |t|<2 = they are separable here and the",
                      "  one-knob-at-a-time sweeps were not leaving anything on the table.",
                      "=" * 104]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  SWEEP report  -> {out_report}")
    logging.info(f"  SWEEP results -> {out_parquet}")


# -------------------------------------------------------------------------- #
# Walk-forward BAG-EVAL - does averaging K seeds' PREDICTIONS kill the per-seed #
# noise floor without killing top1_net? Fits S seeds per (config,anchor) ONCE,  #
# then evaluates prob-averaged and rank-averaged bags of size K (sweeping K).   #
# Read-only: writes only walkforward_bag_* reports, never the live model.       #
# -------------------------------------------------------------------------- #
def _perday_pctrank(vec, dvec):
    """Per-day percentile rank (pct=True) of `vec`, grouped by date vector `dvec`."""
    return (pd.DataFrame({"s": vec, "d": dvec})
            .groupby("d", sort=False)["s"]
            .rank(pct=True, method="average")
            .to_numpy(np.float64))


def run_walkforward_bag():
    date_col, target_col = args.date_column, args.target_column
    out_report  = os.path.join(args.model_dir, "walkforward_bag_report.txt")
    out_parquet = os.path.join(args.model_dir, "walkforward_bag_results.parquet")

    configs = [c.strip() for c in args.wf_bag_configs.split(",") if c.strip()]
    for c in configs:
        if c not in SWEEP_CONFIGS:
            raise ValueError(f"unknown bag config '{c}'. Known: {list(SWEEP_CONFIGS)}")
    ks = sorted({int(k) for k in args.wf_bag_ks.split(",") if k.strip()})
    S  = args.wf_bag_seeds
    B  = args.wf_bag_bags
    if max(ks) > S:
        raise ValueError(f"max bag K={max(ks)} exceeds --wf_bag_seeds S={S}")
    do_prob = args.wf_bag_mode in ("prob", "both")
    do_rank = args.wf_bag_mode in ("rank", "both")

    # ---- load + filter + feature matrix + label (ONCE) - mirrors run_walkforward_sweep ----
    with Phase("BAG: load + label all tickers"):
        parts = load_and_label_tickers(args.input_dir, target_col, date_col,
                                       args.horizon_5d, max_files=args.wf_max_tickers)
        if not parts:
            logging.error("No data loaded.")
            return
        df = pd.concat(parts, ignore_index=True)
        del parts
        df[date_col] = pd.to_datetime(df[date_col])

    with Phase("BAG: universe filter"):
        df, _ = apply_quality_filter(df)
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

    with Phase("BAG: build feature matrix (base + xs ranks)"):
        base_cols = select_base_features(df)
        if USE_XS:
            df = add_xs_rank_features(df, base_cols, date_col)
            feat_names = base_cols + [c + "_xs" for c in base_cols]
        else:
            feat_names = base_cols
        logging.info(f"  features: {len(feat_names)}")

    with Phase("BAG: arrays (one-time float32 cast)"):
        Xall = df[feat_names].to_numpy(np.float32)
        Xall[~np.isfinite(Xall)] = np.nan
        y_all = topq_label(df[target_col].values, df[date_col].values,
                           top_frac=args.topq_frac).astype(np.int8)
        ret_all   = df[target_col].to_numpy(np.float32)
        dates_all = df[date_col].values.astype("datetime64[ns]")
        ym_all    = df[date_col].dt.to_period("M").astype(str).to_numpy()
        # CRITICAL: free the wide source df (base + xs cols, ~2700 cols x 1M rows ~= tens of GB)
        # BEFORE the fit phase. Everything downstream uses only the numpy arrays above; holding df
        # alongside Xall + concurrent DMatrices thrashed RAM (51GB + swap) and stalled the fits.
        del df
        import gc; gc.collect()
        logging.info(f"  Xall {Xall.shape} ~{Xall.nbytes/1e9:.1f}GB  (source df freed)")

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
        if len(np.unique(dates_all[ym_all == nxt])) < 5:
            continue
        anchors.append((m, dates_all[ym_all == m].max(), nxt))
    if not anchors:
        logging.error("  no anchors - lower --wf_min_anchor or add data.")
        return
    logging.info(f"  anchors: {len(anchors)} ({anchors[0][0]}->{anchors[-1][0]})  "
                 f"configs={configs}  S={S} seeds  Ks={ks}  B={B}  mode={args.wf_bag_mode}")

    # per-anchor precompute: train prefix length k, recency weights, OOS range
    anchor_info = {}
    for (m, anchor_date, nxt) in anchors:
        k = int(np.searchsorted(dates_all, anchor_date, side="right"))
        sw = recency_weights(pd.Series(dates_all[:k]), 720.0)
        oi = np.where(ym_all == nxt)[0]
        anchor_info[m] = (k, sw, int(oi[0]), int(oi[-1]) + 1, nxt)

    # ---- fit all (config, anchor, seed) ONCE; keep each seed's OOS prob vector ----
    jobs = [(cfg, m, seed) for cfg in configs
            for (m, _ad, _nxt) in anchors for seed in range(S)]
    logging.info(f"  total fits: {len(jobs)}  "
                 f"({len(configs)} cfg x {len(anchors)} anchor x {S} seed)")

    def _fit_one(job):
        cfg, m, seed = job
        k, sw, olo, ohi, _oos = anchor_info[m]
        params = dict(SWEEP_CONFIGS[cfg])
        n_est = params.pop("n_estimators", 300)
        clf = XGBClassifier(n_estimators=n_est, tree_method="hist",
                            objective="binary:logistic", eval_metric="aucpr",
                            device=args.wf_device, n_jobs=args.wf_threads,
                            random_state=1000 + seed, **params)
        clf.fit(Xall[:k], y_all[:k], sample_weight=sw)        # views, zero-copy
        prob = clf.predict_proba(Xall[olo:ohi])[:, 1].astype(np.float64)
        return (cfg, m, seed, prob)

    probs = {(cfg, m): [None] * S for cfg in configs for (m, _a, _n) in anchors}
    done = 0
    with Phase(f"BAG: {len(jobs)} fits ({args.wf_workers} concurrent)"):
        with ThreadPoolExecutor(max_workers=args.wf_workers) as ex:
            futs = {ex.submit(_fit_one, j): j for j in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    cfg, m, seed, prob = fut.result()
                    probs[(cfg, m)][seed] = prob
                    if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                        logging.info(f"  [{done}/{len(jobs)}] fit {cfg} {m} s{seed}")
                except Exception as e:
                    j = futs[fut]
                    logging.warning(f"  job {j} failed: {repr(e)[:120]}")

    # ---- evaluate bags (reproducible composition) ----
    rng = np.random.RandomState(12345)
    def _bag_sets(K):
        if K == 1:
            return [[s] for s in range(S)]            # all singles = clean noise floor
        return [sorted(rng.choice(S, size=K, replace=False).tolist()) for _ in range(B)]

    rows = []
    for cfg in configs:
        for (m, _a, _n) in anchors:
            k, sw, olo, ohi, oos_period = anchor_info[m]
            seed_probs = probs[(cfg, m)]
            if any(p is None for p in seed_probs):
                logging.warning(f"  {cfg}@{m}: some seeds missing, skipping anchor")
                continue
            dvec = dates_all[olo:ohi]
            seed_ranks = [_perday_pctrank(p, dvec) for p in seed_probs] if do_rank else None
            for K in ks:
                for bag_id, bag in enumerate(_bag_sets(K)):
                    if do_prob:
                        avg_p = np.mean([seed_probs[s] for s in bag], axis=0)
                        t1, sh = _sweep_band_nets(avg_p, olo, ohi, dates_all, ret_all)
                        rows.append({"config": cfg, "oos_month": oos_period, "K": K,
                                     "bag_id": bag_id, "method": "prob",
                                     "top1_net": t1, "shoulder_net": sh})
                    if do_rank:
                        avg_r = np.mean([seed_ranks[s] for s in bag], axis=0)
                        t1, sh = _sweep_band_nets(avg_r, olo, ohi, dates_all, ret_all)
                        rows.append({"config": cfg, "oos_month": oos_period, "K": K,
                                     "bag_id": bag_id, "method": "rank",
                                     "top1_net": t1, "shoulder_net": sh})

    res = pd.DataFrame(rows)
    if res.empty:
        logging.error("  no bag results - every fit failed (see warnings above). Nothing "
                      "written. Most likely GPU OOM (use --wf_device cpu, or subsample with "
                      "--wf_max_tickers / --no_xs_features) or a data/config/target_column error.")
        return
    n_expected = len(configs) * len(anchors) * S
    n_ok = res.groupby(["config", "oos_month"]).ngroups
    if n_ok < len(configs) * len(anchors):
        logging.warning(f"  PARTIAL: {n_ok}/{len(configs)*len(anchors)} (config,anchor) cells "
                        f"produced results - some fits failed. Reporting on what completed.")
    res.to_parquet(out_parquet, index=False)

    # ---- aggregate: per (config,method,K,anchor) bag mean/std, then pool ----
    per_anchor = (res.groupby(["config", "method", "K", "oos_month"])
                     .agg(bag_mean=("top1_net", "mean"),
                          bag_std=("top1_net", "std"),
                          shoulder=("shoulder_net", "mean"))
                     .reset_index())
    pooled = (per_anchor.groupby(["config", "method", "K"])
                        .agg(top1_net=("bag_mean", "mean"),
                             seed_std=("bag_std", "mean"),
                             anchor_std=("bag_mean", "std"),
                             win=("bag_mean", lambda x: (x > 0).mean()),
                             shoulder=("shoulder", "mean"))
                        .reset_index())

    lines = ["=" * 104,
             "WALK-FORWARD BAG-EVAL -- does averaging K seeds' predictions kill seed-variance",
             "without killing top1_net?  (top-1% per-day mean ret, NET of bot-50% baseline)",
             "=" * 104,
             f"configs={configs}  anchors={len(anchors)}  S={S} seeds/anchor  Ks={ks}  "
             f"B={B} bags  mode={args.wf_bag_mode}",
             "K=1 row = single-seed baseline: top1_net = mean over seeds, seed_std = the +/- noise floor.",
             ""]
    methods = [mm for mm, on in (("prob", do_prob), ("rank", do_rank)) if on]
    for cfg in configs:
        for method in methods:
            sub = pooled[(pooled["config"] == cfg) & (pooled["method"] == method)].sort_values("K")
            if len(sub) == 0:
                continue
            has1 = (sub["K"] == 1).any()
            base_std = sub[sub["K"] == 1]["seed_std"].iloc[0] if has1 else float("nan")
            base_ret = sub[sub["K"] == 1]["top1_net"].iloc[0] if has1 else float("nan")
            lines.append(f"### {cfg}  /  {method}-averaged bag ###")
            lines.append(f"  {'K':>3} {'top1_net':>10} {'seed_std':>9} {'std_vs_K1':>10} "
                         f"{'anchor_std':>11} {'win%':>6} {'shoulder':>9}")
            for _, r in sub.iterrows():
                ratio = (r["seed_std"] / base_std) if (base_std and not np.isnan(base_std)) else float("nan")
                lines.append(f"  {int(r['K']):>3} {r['top1_net']:>10.4f} {r['seed_std']:>9.4f} "
                             f"{ratio:>9.2f}x {r['anchor_std']:>11.4f} {r['win']*100:>5.0f}% "
                             f"{r['shoulder']:>9.4f}")
            lines.append(f"  read: GOOD = top1_net holds near K=1 ({base_ret:+.4f}) while std_vs_K1 "
                         f"drops below 1.0 as K grows.")
            lines.append("")

    lines += ["=" * 104,
              "seed_std = avg across anchors of [std of top1_net across the B bags at that K]",
              "         = the run-to-run noise you'd feel at bag size K. Want it SMALL.",
              "top1_net = avg across anchors of the bag-mean. Want it to STAY near the K=1 value.",
              "SANITY: prob/K=1 and rank/K=1 top1_net must match (ranking a prob == ranking its rank).",
              "CAVEATS: (1) K>1 bags are sampled without replacement but may overlap across draws, so",
              "  seed_std is a slight LOWER bound on true independent-bag std -- raise --wf_bag_seeds to",
              "  tighten. (2) top1_net is the WF proxy; final ship call still needs a multi-seed strategy",
              "  backtest. (3) This run wrote NO live model/predictions.",
              "=" * 104]

    report = "\n".join(lines)
    with open(out_report, "w") as f:
        f.write(report)
    logging.info("\n" + report)
    logging.info(f"\n  BAG report  -> {out_report}")
    logging.info(f"  BAG results -> {out_parquet}")


# -------------------------------------------------------------------------- #
# Main                                                                       #
# -------------------------------------------------------------------------- #
def main():
    # --cm_retilt: Phase 14 on an existing pred dir, no model and no inference. Checked
    # before anything else touches disk (it must not create a model dir just to retilt).
    if args.cm_retilt:
        run_cm_retilt()
        return

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

    # --wf_bag: measure whether bagging predictions kills seed-variance without
    # killing top1_net. Read-only: writes only walkforward_bag_* reports.
    if args.wf_bag:
        run_walkforward_bag()
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
                _diag.dump_json("T_split", {
                    "reuse": True, "train_rows": len(train_df), "calib_rows": len(calib_df),
                    "train_dates": [str(train_df[args.date_column].min())[:10], str(train_df[args.date_column].max())[:10]],
                    "calib_dates": [str(calib_df[args.date_column].min())[:10], str(calib_df[args.date_column].max())[:10]],
                    "train_cols": len(train_df.columns)})
        else:
            # Read-projection: skip columns that Phase 6 would drop anyway
            # (drop_feature_patterns / drop_features_exact / drop_vol_features)
            # so wide dropped families never enter memory. Applied per-file at
            # parquet read time; the surviving feature set is identical.
            _exact   = {n.strip() for n in (args.drop_features_exact or "").split(",") if n.strip()}
            _pats    = [p.strip() for p in (args.drop_feature_patterns or "").split(",") if p.strip()]
            _protect = {args.date_column, args.ticker_column, args.target_column,
                        "Open", "High", "Low", "Close", "Volume", args.vol_col,
                        "dollar_volume_ma_10", "atr_percentage", "RSI",
                        # Phase 11/13 outputs and factors, label sources, regime columns
                        "VIX_Close", "vix_close", "Distance to Resistance (%)",
                        "Distance to Support (%)", "volatility",
                        "percent_change_close", "percent_change_Close",
                        "Market_Regime", "VIX_Regime_Numeric"} | set(NON_FEATURES) | set(NEUT_FACTORS)
            # Roster projection (2026-09-03): with --wf_feature_file, Phase 6 discards every
            # base feature outside the roster, so reading them is pure memory (a fresh-cache
            # run peaked at 47.8 GB on the 1,379-column panel and was killed by the RAM
            # watchdog). Read only roster + protected columns; the feature set Phase 6
            # builds is identical.
            _roster = (set(open(args.wf_feature_file).read().split())
                       if args.wf_feature_file else None)

            def _keep_col(c):
                if c in _protect:
                    return True
                if c in _exact:
                    return False
                if any(p in c for p in _pats):
                    return False
                if args.drop_vol_features and is_vol_feature(c):
                    return False
                if _roster is not None and c not in _roster:
                    return False
                return True

            usecols_fn = _keep_col if (_exact or _pats or args.drop_vol_features or _roster) else None

            # Phase 1+2: Load tickers + label engineering
            with Phase("Phase 1+2: Load tickers + label engineering"):
                parts = load_and_label_tickers(
                    args.input_dir, args.target_column, args.date_column,
                    args.horizon_5d, max_files=args.max_files, usecols_fn=usecols_fn,
                    read_float32=args.read_float32, book_touch_k=args.book_touch_k,
                    book_stop=args.book_stop, book_target=args.book_target, book_vol_col=args.vol_col,
                    book_beta=args.book_beta, book_min_depth=args.book_min_depth, book_max_depth=args.book_max_depth,
                    book_ratchet=args.book_ratchet,
                    load_start_date=args.load_start_date, load_end_date=args.load_end_date,
                    prefilter_quality=args.prefilter_quality)
                logging.info(f"  loaded {len(parts)} tickers, "
                             f"{sum(len(p) for p in parts):,} rows total")

            with Phase("Concatenate"):
                if not parts:
                    logging.error("No data loaded - check --input_dir")
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
                _diag.dump_json("T_split", dict(split_meta, reuse=False, train_cols=len(train_df.columns),
                                                runpercent=args.runpercent, calibpercent=args.calibpercent))
                del df

            # Save splits for --reuse on next run
            os.makedirs(prepared_dir, exist_ok=True)
            train_df.to_parquet(train_cache, index=False)
            calib_df.to_parquet(calib_cache, index=False)
            logging.info(f"  PreparedData cached -> {prepared_dir}/")

        if args.train_row_filter:
            _n0 = len(train_df)
            train_df = train_df.query(args.train_row_filter).reset_index(drop=True)
            logging.info(f"  --train_row_filter '{args.train_row_filter}': train rows {_n0:,} -> {len(train_df):,}")
            if len(train_df) < 20_000:
                raise ValueError("train_row_filter left fewer than 20,000 rows")

        # Phase 6: Feature matrix
        with Phase("Phase 6: Build feature matrix"):
            base_cols = select_base_features(train_df)
            logging.info(f"  base numeric features: {len(base_cols)}")
            _feat_steps = [("base", len(base_cols), [])]

            if args.wf_feature_file:
                keep = set(open(args.wf_feature_file).read().split())
                before = len(base_cols)
                base_cols = [c for c in base_cols if c in keep]
                logging.info(f"  --wf_feature_file: {before} -> {len(base_cols)} base features "
                             f"({len(keep - set(base_cols))} listed names absent from panel)")
                _feat_steps.append(("wf_feature_file", len(base_cols), []))

            if args.add_weekday:
                for _df in (train_df, calib_df):
                    _df["dow"] = pd.to_datetime(_df[args.date_column]).dt.dayofweek.astype(np.float32)
                if "dow" not in base_cols:
                    base_cols.append("dow")
                logging.info("  --add_weekday: day-of-week feature 'dow' added")
            if args.limit_features:
                for _df in (train_df, calib_df):
                    _new = add_limit_features(_df, args.vol_col, args.book_touch_k)
                base_cols += [c for c in _new if c not in base_cols]
                logging.info(f"  --limit_features: {_new}")
            if args.day_features:
                for _df in (train_df, calib_df):
                    _new = add_day_features(_df, args.date_column, args.vol_col, k=args.book_touch_k)
                base_cols += [c for c in _new if c not in base_cols]
                logging.info(f"  --day_features: {_new}")

            if args.drop_vol_features:
                dropped = [c for c in base_cols if is_vol_feature(c)]
                base_cols = [c for c in base_cols if not is_vol_feature(c)]
                logging.info(f"  --drop_vol_features removed {len(dropped)} cols "
                             f"(WARNING: vol features are top-ranked in winning config)")
                _feat_steps.append(("drop_vol", len(base_cols), dropped))

            if args.drop_feature_patterns:
                pats = [p.strip() for p in args.drop_feature_patterns.split(",") if p.strip()]
                dropped = [c for c in base_cols if any(p in c for p in pats)]
                base_cols = [c for c in base_cols if not any(p in c for p in pats)]
                logging.info(f"  drop_feature_patterns removed {len(dropped)} cols")
                _feat_steps.append(("drop_patterns", len(base_cols), dropped))

            if args.drop_features_exact:
                exact = {n.strip() for n in args.drop_features_exact.split(",") if n.strip()}
                dropped = [c for c in base_cols if c in exact]
                base_cols = [c for c in base_cols if c not in exact]
                logging.info(f"  drop_features_exact removed {len(dropped)} cols: {dropped}")
                _feat_steps.append(("drop_exact", len(base_cols), dropped))

            if USE_XS:
                logging.info("  --add_xs_features: appending per-day rank columns")
                train_df = add_xs_rank_features(train_df, base_cols, args.date_column)
                calib_df = add_xs_rank_features(calib_df, base_cols, args.date_column)
                feature_cols = base_cols + [c + "_xs" for c in base_cols]
            else:
                feature_cols = base_cols

            logging.info(f"  total features: {len(feature_cols)}")
            _feat_steps.append(("xs_expand" if USE_XS else "no_xs", len(feature_cols), []))
            _diag.dump_json("T_features", {"steps": [{"step": s, "n": n, "dropped": d} for s, n, d in _feat_steps],
                                           "feature_cols": list(feature_cols), "use_xs": bool(USE_XS)})
            X_train = train_df[feature_cols]
            X_calib = calib_df[feature_cols]
            if args.x_float32:
                # RAM (2026-09-05, default off): a few roster columns stay float64 for Phase 13, so
                # the mixed-dtype matrix upcast every copy (fit slice, eval set, DMatrix) to float64:
                # 49 GB at 928k rows. XGBoost hist casts inputs to float32 anyway, so a uniform
                # float32 matrix gives the same model (parity-checked). Feature columns are then
                # dropped from the frames; nothing downstream reads them (labels, dates, len only).
                X_train = X_train.astype(np.float32)
                X_calib = X_calib.astype(np.float32)
                _keep_nonfeat = [c for c in train_df.columns if c not in set(feature_cols)]
                train_df = train_df[_keep_nonfeat]
                calib_df = calib_df[_keep_nonfeat]
                logging.info(f"  --x_float32: X_train {X_train.shape} float32 ({X_train.memory_usage().sum()/1e9:.1f} GB); feature columns dropped from the frames")

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
                thresh_5d=args.thresh_5d, ratchet_win=args.book_ratchet_win,
            )
            _diag.dump_json("T_labels", {"label_mode": args.label_mode, "topq_frac": args.topq_frac,
                                         "p_train": float(np.mean(y_train)), "p_calib": float(np.mean(y_calib)),
                                         "n_train": int(len(y_train)), "n_calib": int(len(y_calib))})
            if _diag.enabled():
                _lbd = pd.DataFrame({"Date": train_df[args.date_column].values, "y": y_train})
                _diag.dump_parquet("T_label_by_day",
                                   _lbd.groupby("Date")["y"].agg(["size", "mean"]).reset_index())

        # Phase 8: Recency weights
        sw_train_full = None
        if args.recency_half_life_days:
            with Phase("Phase 8: Recency weights"):
                sw_train_full = recency_weights(
                    train_df[args.date_column], args.recency_half_life_days)
                if args.payoff_weight_cap:
                    _pw = 0.005 + np.minimum(np.abs(np.nan_to_num(train_df[args.payoff_weight_col].values)), args.payoff_weight_cap)
                    sw_train_full = (sw_train_full * (_pw / _pw.mean())).astype(np.float32)
                    logging.info(f"  --payoff_weight_cap {args.payoff_weight_cap}: weights x (0.005 + min(|ret_5d|, cap)), "
                                 f"p90/p10 = {np.percentile(sw_train_full, 90) / max(np.percentile(sw_train_full, 10), 1e-9):.1f}x")
                logging.info(f"  half_life={args.recency_half_life_days}d  "
                             f"newest:oldest ratio = "
                             f"{sw_train_full.max()/max(sw_train_full.min(),1e-9):.1f}x")
                _diag.dump_json("T_recency", {"half_life_days": args.recency_half_life_days,
                                              "ratio_newest_oldest": float(sw_train_full.max() / max(sw_train_full.min(), 1e-9)),
                                              "w_p10": float(np.percentile(sw_train_full, 10)),
                                              "w_p50": float(np.percentile(sw_train_full, 50)),
                                              "w_p90": float(np.percentile(sw_train_full, 90))})

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
                    device=args.tune_device,
                    n_gpus=args.tune_gpus,
                )
                logging.info(f"  inner-val {args.tune_objective} = {best_val:.4f}")
        else:
            logging.info("  --no_tune: skipping Optuna, using default XGB params")

        # Phase 10: Train final model
        with Phase("Phase 10: Train final XGBClassifier"):
            clf = train_bagged(
                args.bag_seeds, args.seed, bag_mode=args.bag_mode,
                X_train=X_train, y_train=y_train, tuned_params=tuned_params,
                sw_train=sw_train_full,
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
                refit_full=args.refit_full,
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

    neut_kwargs = dict(
        mode=args.neut_mode,
        dose=args.neut_dose,
        dose_above=args.neut_dose_above,
        dose_below=args.neut_dose_below,
        flag_max_stale_days=args.neut_flag_max_stale_days,
    )
    if args.neutralize:
        logging.info("--neutralize: Phase 13 ON (%s)"
                     % ("fixed dose %.2f" % args.neut_dose if args.neut_mode == "fixed"
                        else "spy200v2 %.2f above / %.2f below the SPY 200d EMA"
                             % (args.neut_dose_above, args.neut_dose_below)))
    else:
        logging.info("Phase 13 neutralization OFF (pass --neutralize for the ship config)")

    cm_kwargs = _cm_kwargs()
    if CM_ON:
        logging.info("Phase 14 conviction momentum ON (default) "
                     "(k=%.2f spans=%s clip=[%.2f,%.2f])"
                     % (args.cm_k, list(cm_kwargs["spans"]), args.cm_lo, args.cm_hi))
    else:
        logging.info("--no_conviction_momentum: Phase 14 OFF -- these are UN-TILTED preds, "
                     "NOT the ship config")

    with Phase("Phase 12: Inference - score all tickers -> RFpredictions"):
        status = run_inference(
            model_path=model_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            date_col=args.date_column,
            ticker_col=args.ticker_column,
            top_frac_per_day=args.top_frac_per_day,
            max_files=args.max_files,
            calib_path=calib_path if USE_CALIB else None,
            neutralize=args.neutralize,
            neut_kwargs=neut_kwargs,
            neut_min_tickers=args.neut_min_tickers,
            neut_min_changed=args.neut_min_changed,
            conviction_momentum=CM_ON,
            cm_kwargs=cm_kwargs,
            cm_min_tickers=args.cm_min_tickers,
            cm_min_changed=args.cm_min_changed,
            infer_tail=args.infer_tail,
            pool_top_n=args.pool_top_n,
        )

    total_min = (time.time() - run_t0) / 60
    logging.info(f"\n{'='*70}")
    logging.info(f"Pipeline done in {total_min:.1f}m")
    logging.info(f"  Model:          {model_path}")
    logging.info(f"  RFpredictions:  {args.output_dir}/")
    logging.info(f"  Neutralization: "
                 + (f"ON  ({status['neut']['dose_desc']})" if status["neutralized"]
                    else f"FAILED GATE ({status['neut_error']}) -> un-neutralized preds"
                    if status.get("neut_error") else "off (raw preds)"))
    logging.info(f"  Conviction mom: "
                 + (f"ON  (k={args.cm_k:.2f} spans={list(cm_kwargs['spans'])})"
                    if status.get("conviction_momentum")
                    else f"FAILED GATE ({status['cm_error']}) -> un-tilted preds"
                    if status.get("cm_error") else "off (--no_conviction_momentum)"))
    logging.info(f"  Next step:      python 5__NightlyBackTester.py --force")
    logging.info(f"{'='*70}")

    if status["gate_failed"]:
        # Same contract as the old 4.5 / 4.6 stages: usable preds are in place, but they
        # are NOT the ship config, so the runner must alert. Non-zero exit is the alert.
        # Name the phase that actually failed -- with Phase 14 on by default, blaming
        # neutralization for a tilt failure sends whoever reads the log the wrong way.
        why = "; ".join(
            "%s gate failed (%s)" % (ph, status[key])
            for ph, key in (("Phase 13 neutralization", "neut_error"),
                            ("Phase 14 conviction momentum", "cm_error"))
            if status.get(key)) or "unknown gate failure"
        logging.error("EXIT 1: %s. %s holds usable but NON-SHIP preds."
                      % (why, args.output_dir))
        sys.exit(1)





if __name__ == "__main__":
    main()
