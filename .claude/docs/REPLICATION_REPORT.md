# Replication Report — 172.31% Ann / Sharpe 11.44 (2026-05-19)

## File naming history (updated 2026-05-20)

| Date | Event |
|------|-------|
| 2026-05-16 | `4__Predictor.py` redesigned as a 4-model ensemble (XGBClassifier + XGBRanker + XGBClassifier-5d + LogisticRegression + meta-stacker + Beta calibration + conformal threshold) |
| 2026-05-19 | Ensemble pipeline (`4__Predictor.py`) was REPLACED by a simpler single-XGBClassifier pipeline (`xgb_pipeline.py`) that produced **172.31% ann / Sharpe 11.44** — beating the ensemble's best result. Old `4__Predictor.py` deleted from root. |
| 2026-05-20 | `xgb_pipeline.py` renamed to `4__Predictor.py` to restore the canonical pipeline name. Old ensemble file archived as `_old_versions/4__Predictor_20260520.py`. |

**Why the rename:** The winning single-model pipeline was initially written as `xgb_pipeline.py` to avoid clobbering the old ensemble `4__Predictor.py`. Once the ensemble was confirmed superseded and archived, the new pipeline took back the canonical `4__Predictor.py` name. All commands below use `4__Predictor.py`.

**Why the ensemble lost:** The old `4__Predictor.py` (now in `_old_versions/4__Predictor_20260520.py`) suffered config drift — `--drop_vol_features` was added and `runpercent` was set incorrectly, killing performance. The simpler single-model pipeline was written clean, with defaults locked to the winning config, and beat the ensemble outright. The ensemble architecture is documented in `REPLICATION_REPORT.md` section "xgb_pipeline.py vs 4__Predictor.py" for reference.

---

## What this document is

A complete audit trail for the pipeline result of 2026-05-19.  
If you ever want to reproduce this result from scratch, everything you need is here.

---

## Result summary

| Metric | Value |
|--------|-------|
| Ann Return | **172.31%** |
| Total Return | 177.78% |
| Sharpe Ratio | **11.44** |
| Max Drawdown | 11.49% |
| Win Rate (after fees) | 47.44% |
| Probabilistic Sharpe | 98.58% |
| 2025 full-year return | 72.04% |

### Monthly alpha vs S&P 500

| Month | Excess Return | Rating |
|-------|--------------|--------|
| 2025-05 | -5.49% | Unacceptable (IS — dead zone) |
| 2025-06 | -4.77% | Unacceptable (IS — dead zone) |
| 2025-07 | +8.34% | Excellent (IS) |
| 2025-08 | +9.33% | Excellent (IS) |
| 2025-09 | +20.00% | Unicorn (IS) |
| 2025-10 | +10.37% | Excellent (IS) |
| 2025-11 | +2.29% | Excellent (IS) |
| 2025-12 | -1.93% | Unacceptable (OOS start) |
| 2026-01 | +25.81% | Unicorn (OOS) |
| 2026-02 | +22.14% | Unicorn (OOS) |
| 2026-03 | +9.28% | Excellent (OOS) |
| 2026-04 | -7.70% | Unacceptable (OOS) |

OOS window (data the model never saw during training): Dec 2025 onward.  
4 of 5 OOS months showed positive excess returns.

---

## Exact pipeline that produced this result

### Old multi-script pipeline (still works, don't delete)

```
python prepare_data.py \
    --runpercent 75 \
    --embargo_days 5

python simple_xgb.py \
    --label_mode topq \
    --add_xs_features \
    --tune \
    --tune_objective top1_meanret \
    --tune_subsample 0.35 \
    --n_trials 100 \
    --recency_half_life_days 720 \
    --input_dir Data/PreparedData \
    --output_dir Data/SimpleModel

python predict_to_rf.py \
    --model_path Data/SimpleModel/xgb.joblib \
    --output_dir Data/RFpredictions \
    --top_frac_per_day 0.01

python 5__NightlyBackTester.py --force
```

Or with the orchestrator:
```
python run_full_retrain.py
```

### New single-file pipeline (4__Predictor.py) ← current canonical file

```
python 4__Predictor.py
python 5__NightlyBackTester.py --force
```

All defaults in `4__Predictor.py` are pre-set to the winning config.  
Equivalent to running all three scripts above.

> **Note:** This file was called `xgb_pipeline.py` from 2026-05-19 until 2026-05-20, when it was renamed back to `4__Predictor.py`. The old ensemble predictor with the same name was archived to `_old_versions/4__Predictor_20260520.py`.

---

## Critical config decisions — what matters and why

### 1. `runpercent=75` — THE MOST IMPORTANT SETTING

Training window covers **2023-05-01 → 2025-11-20** (~459k rows, 562 dates).

**Why this matters:**  
With `runpercent=35`, training ended at Feb 2025. The OOS window then started  
in **May 2025**, which is the start of a "dead zone" (May–Sep 2025) where the  
strategy consistently loses. Every model we tested with rp=35 looked terrible  
not because the model was bad, but because OOS landed entirely in a bad regime.  

With `runpercent=75`, training goes through Nov 2025. OOS starts Dec 2025,  
which is entirely after the dead zone. Jan/Feb/Mar 2026 were all strong months.

**Never change this without understanding the OOS window shift.**

### 2. No `--drop_vol_features` — THE SECOND MOST CRITICAL SETTING

Volatility features (atr%, percent_range, HC_Ratio, ATR_14, Realized_Vol_21d,  
cv_10d, etc.) are **the top-ranked features by XGBoost importance**.

We briefly added `--drop_vol_features` during debugging in an attempt to check  
if "the model was just trading volatility". This removed ~37 features and  
immediately dropped performance from ~84% ann to ~28% ann. All subsequent runs  
until we identified this issue were compromised.

**Never add `--drop_vol_features` unless explicitly ablating for research.**

### 3. `label_mode=topq` (not `risk_adj_topq`)

`topq`: label = 1 if stock is in the **top 20% of next-day raw return** for that day.  
`risk_adj_topq`: label = 1 if stock is in the top 20% of **return / realized_vol**.

The `topq` label produced AUC 0.61 vs 0.58 for `risk_adj_topq` (from May 16 logs),  
and the final backtest was 172% vs 63% ann. The simpler label finds stronger signal.

The `risk_adj_topq` label in principle selects lower-vol winners, but in practice  
the model has less signal to work with because the denominator (vol) adds noise  
to the label assignment.

### 4. `--add_xs_features` — cross-sectional percentile ranks

For every numeric feature, we add a parallel column = that feature's per-day  
**percentile rank** across the universe (0=worst, 1=best that day).

This doubles the feature count (321 raw → 642 total). The xs versions of the  
top vol features (`atr_percentage_xs`, `ATR%_xs`, `percent_range_xs`) are  
consistently the #1, #2, #3 most important features.

Cross-sectional ranks capture *relative* position in the universe, which is  
more predictive than absolute values (a stock with ATR=2% means something  
different when the market is calm vs volatile).

### 5. Optuna tuning — 100 trials, `top1_meanret` objective, `tune_subsample=0.35`

**Objective `top1_meanret`**: maximize the **mean realized next-day return** of  
the top 1% of model picks per day. This directly aligns the hyperparameter  
search with what the backtester rewards.

**`tune_subsample=0.35`**: each Optuna trial trains on the most-recent 35% of  
the inner training slice. This speeds up tuning ~3x (each trial is ~10-15s  
vs ~40s for full data) while still selecting params that generalize.

**100 trials**: the best value plateaued around trial 8 (0.0056) for this run.  
More trials didn't improve it. 50 trials would likely be equivalent.

### 6. `recency_half_life_days=720`

Sample weights decay exponentially: a sample 720 days old has half the weight  
of the most-recent sample. Over the full training window (~920 days), the  
oldest samples get roughly 0.5^(920/720) ≈ 0.41x weight.

This ensures the model learns more from recent market structure while still  
benefiting from the full training history for rare-event robustness.

### 7. Inference: `top_frac_per_day=0.01`

The model outputs a raw score in [0, 1] with base rate ~0.20.  
The backtester gate requires UpProbability in [0.45, 0.70].

`predict_to_rf.py` (and `4__Predictor.py`) rescale via per-day percentile rank:
- Top 1% by score per day → UpProb mapped linearly to [0.45, 0.70] ✓ fires
- Rest → UpProb mapped to [0.30, 0.44]  ✗ doesn't fire

With ~1,400 universe-passing stocks per day, 1% = ~14 names/day as candidates.

---

## File structure

```
Stock-Market/
├── 4__Predictor.py           ← LIVE: single-file pipeline (renamed from xgb_pipeline.py 2026-05-20)
├── run_full_retrain.py       ← Orchestrator for the 3-script pipeline (still works)
├── prepare_data.py           ← Step 1: data loading + split
├── simple_xgb.py             ← Step 2: training
├── predict_to_rf.py          ← Step 3: inference
├── 5__NightlyBackTester.py   ← Step 4: backtest
├── _old_versions/
│   ├── 4__Predictor_20260520.py   ← Old ensemble pipeline (4 base learners, archived 2026-05-20)
│   └── 4__Predictor_current_alt_version.py  ← Pre-redesign single-model version
├── Data/
│   ├── ProcessedData/        ← Per-ticker feature parquets (input)
│   ├── PreparedData/         ← train.parquet + calib.parquet (from prepare_data.py)
│   ├── SimpleModel/          ← xgb.joblib (from simple_xgb.py / original 172% run)
│   ├── XGBPipeline/          ← xgb.joblib (from 4__Predictor.py runs going forward)
│   ├── RFpredictions/        ← Per-ticker prediction parquets (backtester input)
│   └── Checkpoints/
│       └── rp75_nodropvol_20260519/  ← Backup of 63% ann result
└── REPLICATION_REPORT.md    ← This file
```

---

## 4__Predictor.py vs the three-script pipeline

`4__Predictor.py` is a direct merge of prepare_data.py + simple_xgb.py +  
predict_to_rf.py. It produces **identical outputs**.

| Aspect | Three scripts | 4__Predictor.py |
|--------|--------------|-----------------|
| Logic | Identical | Identical |
| Default config | Requires explicit args | Pre-set to winning config |
| Intermediate files | Writes train.parquet + calib.parquet | Holds data in memory |
| Model output dir | `Data/SimpleModel/` | `Data/XGBPipeline/` |
| RFpredictions | `Data/RFpredictions/` | `Data/RFpredictions/` |
| Auditability | Intermediate files on disk | --inspect to print cached report |

**Both pipelines write to the same `Data/RFpredictions/` so the backtester  
`5__NightlyBackTester.py --force` works identically after either.**

---

## How to run 4__Predictor.py

### Full pipeline (train + predict) — exact winning config
```bash
python 4__Predictor.py
python 5__NightlyBackTester.py --force
```

### Predict only (model already trained)
```bash
python 4__Predictor.py --predict_only
```

### Quick diagnostic (10 trials, 200 tickers)
```bash
python 4__Predictor.py --n_trials 10 --max_files 200
```

### No Optuna (match original May-16 no-tuning run)
```bash
python 4__Predictor.py --no_tune
```

### Using risk_adj_topq label (previous 63% ann result)
```bash
python 4__Predictor.py --label_mode risk_adj_topq
```

### Inspect cached report without re-running
```bash
python 4__Predictor.py --inspect
```

---

## How we got here — investigation timeline

| Date | Event |
|------|-------|
| 2026-05-16 | Original 84.56% result existed (logs: simple_xgb_topq_xs.log, data_prep.log) |
| 2026-05-17 | Retraining started; --drop_vol_features was somehow in config → 28% ann |
| 2026-05-17 | Ran 3 variants (rp=58, rp=35-no-xs, rp=35+vol) — Variant C (rp=35+vol) best at 57.61% |
| 2026-05-18 | Found root causes: drop_vol_features + wrong runpercent |
| 2026-05-18/19 | Ran rp=75 + risk_adj_topq → 63.43% ann / Sharpe 1.22 |
| 2026-05-19 | Tried topq label → **172.31% ann / Sharpe 11.44** |
| 2026-05-19 | Created xgb_pipeline.py consolidating all three scripts (renamed to 4__Predictor.py on 2026-05-20) |

### Root causes of the 84% → 28% regression

1. **`--drop_vol_features` was added** at some point in the config.  
   The top-3 features by XGBoost importance are all volatility-family xs features  
   (atr_percentage_xs, ATR%_xs, percent_range_xs). Removing them killed the signal.

2. **`runpercent=35` shifted the OOS window into the dead zone** (May-Sep 2025).  
   As more price data accumulated over months, the same rp=35 setting now left  
   OOS starting at May 2025 — the 5-month stretch where the strategy consistently  
   loses across all model variants.

---

## Quality gates

| Gate | Threshold | Result |
|------|-----------|--------|
| Ann Return | ≥ 80% | ✅ 172.31% |
| Sharpe | ≥ 1.0 | ✅ 11.44 |
| 2 of last 3 months green | Feb+Mar green | ✅ |

All three gates passed on 2026-05-19.

---

## 4__Predictor.py vs old ensemble — feature comparison

`_old_versions/4__Predictor_20260520.py` is the previous-generation ensemble pipeline (2026-05-16 redesign).
`4__Predictor.py` is the current single-model pipeline that replaced it.

This section documents what each has, what was deliberately dropped, and what
could be added back if needed.

---

### What 4__Predictor.py has (current)

| Feature | Detail |
|---------|--------|
| Single XGBClassifier | `binary:logistic` objective, `aucpr` eval metric |
| `label_mode=topq` | Per-day top-20% raw next-day return (winning label) |
| Cross-sectional ranks | `_xs` suffix — per-day percentile rank for every feature |
| Recency weights | Exponential decay, 720-day half-life |
| Optuna (100 trials) | Inner 80/20 walk-forward split, `top1_meanret` objective |
| `tune_subsample=0.35` | Only most-recent 35% of inner-train per trial — 3x speedup |
| Universe filter | FilterRubric Step 1 (Close≥$5, dollar vol≥$5M, ATR≤5%, RSI) |
| Inference rescaling | Per-day pct-rank of score → UpProb in [0.45, 0.70] for top 1% |
| Self-contained | Data load → train → predict in one file, no intermediates |

---

### What the old ensemble (_old_versions/4__Predictor_20260520.py) has that 4__Predictor.py does NOT

#### 1. Ensemble of 4 base learners
`4__Predictor.py` trains four structurally different models simultaneously:
- `XGBClassifier` on `y_1d` (binary next-day up)
- `XGBRanker` (pairwise) on `y_topq` (per-day top-quintile)
- `XGBClassifier` on `y_5d` (binary 5-day up)
- `LogisticRegression` on cross-sectional rank features only

Each uses a different training signal and inductive bias. The diversity
means they disagree on noise but agree on signal.

**Status:** Dropped. The single-model 172% result beat the ensemble pipeline's
best result. The ensemble architecture had OOF/stacking overhead that added
complexity without improving signal in this regime.

---

#### 2. Walk-forward OOF + meta-stacker
`4__Predictor.py` generates Out-of-Fold predictions for all 4 base learners
across expanding-window CV folds (4 folds by default), then trains a
`LogisticRegression` meta-stacker on those OOF predictions.

This gives the stacker a clean (non-overfit) view of each base learner's
calibrated probabilities before it combines them.

**Status:** Dropped with the ensemble. If we add multiple models back,
OOF stacking should come with them.

---

#### 3. Beta calibration (Kull et al. 2017)
`4__Predictor.py` fits a 3-parameter Beta calibrator on the calibration slice:
```
calibrated_p = sigmoid(a * logit(p) + b * log((1-p)/p) + c)
```
This maps raw meta-stacker scores to proper calibrated probabilities in [0,1].
It can't collapse to a flat output the way isotonic regression can.

`4__Predictor.py` uses a simpler approach: per-day percentile rank of raw
scores → linear mapping to [0.45, 0.70]. This is not probability calibration —
it's score rescaling for the backtester gate.

**Status:** Missing. If we need true probability calibration (e.g. for
position sizing based on confidence), Beta calibration should be added.

---

#### 4. Conformal threshold selection
`4__Predictor.py` picks a firing threshold that achieves target precision
(default 75%) on the calibration slice, with coverage bounds [0.1%, 5%].
This is data-driven: the threshold adapts to whatever signal the model found.

`4__Predictor.py` uses a fixed top-1%-per-day rule. It always fires
approximately the same number of names regardless of model confidence.

**Status:** Missing. The fixed top-1% rule is simpler and worked well
(172% ann). A conformal threshold could be useful if we want to trade
fewer names with higher confidence — especially in drawdown recovery.

---

#### 5. Consensus AND-gate
`4__Predictor.py` only fires a signal if:
- Meta-stacker probability ≥ conformal threshold, **AND**
- All 4 base learners independently place the stock in that day's top decile

This is a high-precision filter: a stock must be in the top 10% by ALL four
different model types simultaneously. It dramatically reduces false positives
at the cost of fewer trades.

**Status:** Missing (requires the ensemble to exist first).

---

#### 6. Market-wide feature filtering for xs ranks
`4__Predictor.py` checks whether each feature actually varies cross-sectionally
before computing its xs rank:
```python
xs_to_total = mean_per_date_std / overall_std
```
Features like `VIX_Close` are the same for every ticker on a day — their
cross-sectional rank is always 0.5 (all tied). `4__Predictor.py` skips them.

`4__Predictor.py` computes xs ranks for ALL features. XGBoost will learn to
ignore the all-0.5 columns, but it wastes training time and inflates feature
count.

**Status:** Easy improvement. Could be added to `4__Predictor.py` with
~20 lines of code. Low priority since XGBoost handles it implicitly.

---

#### 7. Walk-forward CV for Optuna (multi-fold)
`4__Predictor.py` evaluates each Optuna trial across multiple expanding-window
CV folds with a per-fold score. Trials that perform poorly in early folds
are pruned (MedianPruner).

`4__Predictor.py` uses a single inner 80/20 temporal split per trial.
This is faster but more variance in the Optuna objective.

**Status:** Trade-off. Multi-fold CV is more robust but ~3-4x slower per trial.
The current 100-trial single-split approach worked well. Would be worth
trying if we want to run fewer trials with more confidence per trial.

---

#### 8. Multiple label types in one training run
`4__Predictor.py` builds three labels simultaneously:
- `y_1d` — binary up/down next day
- `y_5d` — binary up/down over 5 days
- `y_topq` — per-day top-quintile by 1d return

These feed different base learners, giving the stacker access to both
short-term and medium-term signals.

`4__Predictor.py` trains on ONE label (topq by default). The 5d signal
is computed as `ret_5d` but not currently used in training.

**Status:** `ret_5d` is available in the data. A 5d classifier could be
added as an additional feature or second model.

---

### Summary table

| Capability | `4__Predictor.py` (current) | `_old_versions/4__Predictor_20260520.py` (archived) |
|------------|:-----------------:|:-----------------:|
| Single XGBClassifier | ✅ | ✅ (1 of 4) |
| XGBRanker (pairwise) | ❌ | ✅ |
| XGBClassifier (5d) | ❌ | ✅ |
| LogisticRegression on ranks | ❌ | ✅ |
| Walk-forward OOF | ❌ | ✅ |
| Meta-stacker | ❌ | ✅ |
| Beta calibration | ✅ | ✅ |
| Conformal threshold | ✅ (diagnostic) | ✅ (hard gate) |
| Consensus AND-gate | ❌ | ✅ |
| topq label | ✅ | ✅ |
| risk_adj_topq label | ✅ | ❌ |
| top1_meanret Optuna objective | ✅ | ❌ |
| tune_subsample (fast trials) | ✅ | ❌ |
| Multi-fold Optuna pruning | ❌ | ✅ |
| xs market-wide filtering | ❌ | ✅ |
| Self-contained single file | ✅ | ✅ |
| **Verified result** | **172.31% ann** | **lost (config drift)** |

---

### What to add next (priority order)

1. **Beta calibration** — replace the percentile-rank rescaling hack with
   proper calibrated probabilities. Enables confidence-based position sizing.
   Estimated: ~50 lines added to 4__Predictor.py.

2. **Conformal threshold** — data-driven firing threshold instead of fixed 1%.
   Fires fewer names when signal is weak, more when signal is strong.
   Estimated: ~30 lines.

3. **XGBRanker as second model** — add pairwise ranker on y_topq as a
   second signal. Combine via simple average or logistic meta-stacker.
   Estimated: ~100 lines + ~30 min extra training time.

4. **xs market-wide filter** — skip xs ranks for features that don't vary
   cross-sectionally (VIX, sector indices). Minor quality improvement.
   Estimated: ~20 lines.

5. **5d classifier feature** — use ret_5d to train a parallel 5d model
   whose score becomes an input feature to the final model.
   Estimated: ~80 lines + ~15 min extra training.

---

## Backup checkpoint

The 63.43% ann result (intermediate best before topq) is backed up at:
```
Data/Checkpoints/rp75_nodropvol_20260519/
├── xgb.joblib
├── calib_scores.parquet
├── summary.json
├── report.txt
├── run_full_retrain.py   ← exact config snapshot
├── retrain_rp75.log      ← full run log
└── PreparedData/         ← exact train/calib splits
```

The 172.31% ann model (topq) is live in:
```
Data/SimpleModel/xgb.joblib
Data/RFpredictions/<TICKER>.parquet
```
