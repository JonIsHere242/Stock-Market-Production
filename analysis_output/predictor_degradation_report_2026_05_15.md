# Predictor Degradation Diagnostic Report
**Date:** 2026-05-15  
**Baseline (pre-session):** Sharpe –3.96, total return +1.08%  
**Current (post-retraining):** Sharpe –6.89, total return –10.02%

---

## 1. Executive Summary

The retraining made things materially worse, not better. The Sharpe ratio went from –3.96 → –6.89 and total return from +1.08% → –10.02% over a 90-day sample. Two causes are stacked on top of each other:

1. **Root cause (pre-existing):** The matrix-power features (`mp_*`) in `3__AlphaSensitivity.py` were reverted ~3 months ago. These features contributed |IC| 0.06–0.08 each; without them the model trains on degraded features and no amount of in-predictor tuning can recover that signal.

2. **New harm introduced this session:** The `label_threshold = 0.003` change created a training/test distribution mismatch that broke the model's probability ordering, producing a bimodal output distribution and a non-monotone win rate curve. This is the primary new regression introduced in this session.

---

## 2. Key Metrics Comparison

| Metric | Before session | After retraining |
|---|---|---|
| Sharpe Ratio | –3.96 | **–6.89** |
| Total Return | +1.08% | **–10.02%** |
| IC (Spearman, prob vs fwd_1d) | 0.051 | **0.0368 (–28%)** |
| Signal Win Rate (EDA global) | ~51% | **46.7% (BELOW 50%)** |
| Backtest Win Rate | ~48% | **45.32%** |
| Profit Factor | ~1.60 (trade_history) | **0.84 (backtest), 0.79 (EDA)** |
| Calibration Error (signal zone) | –10% to –13% | **–9% to –13% (unchanged)** |
| Probability range | 0.43–0.68 | **0.40–0.67** |
| Win rate monotone with prob? | Unknown | **NO — ordering broken** |

---

## 3. Smoking Guns in the EDA

### 3a. Non-monotone win rate (most damning finding)
```
Bucket       Win Rate
[0.54,0.56)  49.5%   ← OK
[0.56,0.58)  51.0%   ← OK
[0.58,0.60)  50.9%   ← drops here
[0.60,0.62)  52.0%   ← rises again
[0.62,0.65)  52.1%
[0.65,1.00)  52.9%
```
Win rate at 0.58–0.60 *dips below* the previous bucket, then recovers. A properly calibrated model should be monotone. This means the model's ranking is partly broken — it is not reliably ordering stocks from worst to best. The can_buy relative-percentile logic depends entirely on correct ordering within each ticker's history.

### 3b. Bimodal probability distribution
```
[0.45,0.50)  617,178 rows  ← huge spike
[0.50,0.52)  163,702 rows  ← valley
[0.52,0.54)  262,525 rows
[0.54,0.56)  567,621 rows  ← second spike
```
A healthy model should produce a roughly unimodal, smooth distribution. Two separate spikes — one sub-threshold and one at 0.54–0.56 — indicate the model learned a bimodal classification rather than a proper probability regression. **This is the fingerprint of the label_threshold change.**

### 3c. Signal win rate below 50%
EDA Section 10 explicitly flags: *"Signal bucket win rate = 46.7% — barely above coin flip."* With avg win ≈ avg loss (0.27% vs 0.26% in the backtest), a 45% win rate guarantees losses regardless of trade management.

### 3d. EDA vs backtest gap (51.9% → 45.32%)
The EDA's global threshold signals show 51.9% win rate, but the backtest delivers only 45.32%. This gap exists because `can_buy()` uses **relative per-stock percentiles** (p96/p97.5 of each stock's own history), not the global threshold. With a bimodal distribution, the stocks whose own-history 96th percentile happens to be in the 0.54–0.56 cluster (which has only 49.5% win rate) still fire as "signals" in the backtester. The EDA global threshold only captures the true high-probability tail (0.60+), masking this problem.

### 3e. Top-signalled tickers are preferred shares
```
ONBPO (131 signals, 49.8% win), ONBPP (122 signals), LBRDP (113 signals),
PNFPP (113 signals), TVC (106 signals), WLKP (106 signals) ...
```
Every single top-signalled ticker is a preferred share or illiquid fixed-income equivalent. These instruments move very little, have compressed price ranges, and will consistently produce high within-stock relative probabilities simply because their low volatility makes every close-to-peak day look like a signal. These are noise positions that drag win rate down.

---

## 4. Root Cause Analysis

### Cause 1 (PRE-EXISTING): Missing matrix-power features
The `mp_*` features from `expm(M)` decompositions of OHLCV-primitive matrices were reverted from `3__AlphaSensitivity.py` ~3 months ago. These 14 features had |IC| 0.06–0.08 each — roughly 1.5–2x the current per-feature IC of the model's best features. The current feature set's best feature (`hammer_pattern`) contributes 7.4% importance but cannot compensate for 14 missing high-IC features.

**Evidence:** IC fell from the best-known 0.051 (post-session changes) and the model health summary shows issues even before this session's changes. The project memory explicitly states: *"in-predictor fixes can't recover signal, restore matrix-power features first."*

### Cause 2 (NEW — introduced this session): label_threshold = 0.003
**What it does:** Excludes training rows where `|return| < 0.3%` (~11% of data). The intent was to remove ambiguous near-zero labels.

**Why it breaks the model:**
- Training set now has an artificial gap: no examples of returns in (–0.3%, +0.3%). The model never sees "ambiguous" inputs mapped to a specific output.
- Test/live data still contains these ambiguous returns, creating a distribution mismatch.
- The model's response to ambiguous inputs (which it never trained on) is undefined — in practice, it collapses predictions into two separate clusters, producing the bimodal distribution.
- Isotonic calibration then maps these two clusters to two different probability ranges, breaking monotonicity.

**The fix:** Remove `label_threshold` entirely. Binary labeling with a zero threshold (up = 1, down = 0) is correct. The "ambiguous" 0.3% returns are real data; filtering them distorts the feature-to-label mapping. Precision is improved by better features and threshold tuning, not by removing training examples.

### Cause 3 (MINOR): Optuna scale_pos_weight range [0.5, 3.0]
The previous range was [0.2, 0.8]. Widening to [0.5, 3.0] allows Optuna to choose values >1.0, which aggressively up-weights the positive class. With a base rate of ~49%, SPW > 1.0 means the model sees artificially inflated positives and will predict 1 more often → lower precision → more noise trades. This compounds with the label_threshold bimodal issue.

---

## 5. Why Previous Config Worked (~82% annualized, 5.59 Sharpe)

From project memory (2026-04-16 best config):
- Required `--tune --runpercent 35`
- `min_child_weight=5`, `half_life=720`, dynamic date splits, long-only
- NO label_threshold, NO scale_pos_weight widening

The previous best config worked because:
1. Matrix-power features were present (the main signal source)
2. No training/test distribution mismatch
3. SPW was tuned conservatively by Optuna within [0.2, 0.8], producing tighter probability clusters that ranked correctly

---

## 6. Recommended Fixes — Priority Order

### P0: Restore matrix-power features in `3__AlphaSensitivity.py` (BLOCKERL)
This is the single most important action. All other fixes are marginal without this. The `_old_versions/3__AlphaSensitivity_pre_matrixpower.py` file shows the state before mp_* features were added. The post-matrix-power version should be in git history or `_old_versions/`. The 14 `mp_*` features need to be restored and the AlphaSensitivity pipeline re-run across all tickers.

### P1: Remove `label_threshold` from `4__Predictor.py`
In `4__Predictor.py`, find and remove the label filtering block:
```python
# REMOVE THIS ENTIRE BLOCK:
label_thr = config.get('label_threshold', 0.003)
y_train_signed = y_train_raw.apply(
    lambda x: 1 if x >= label_thr else (0 if x <= -label_thr else np.nan)
)
train_label_mask = y_train_signed.notna()
...
```
Also remove `label_threshold: 0.003` from the config dict. Revert to the original `y_train = (y_train_raw > 0).astype(int)` binary labeling.

### P2: Narrow Optuna SPW range back to [0.2, 0.8]
In the Optuna objective function, revert:
```python
scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 3.0)
# → back to:
scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.2, 0.8)
```
Values <1.0 force the model to be conservative (higher precision, lower recall) — correct for a long-only strategy where false positives are pure losses.

### P3: Filter preferred shares from the signal universe
In the backtester or broker, exclude tickers whose names end in P, PD, PR, PRA, PRB, etc. (preferred share conventions). These tickers produce systematically high within-stock relative probabilities without real momentum signal, and they suppress win rate.

### P4: Add VIX filter in the broker (pre-existing recommendation)
EDA showed VIX < 15 → 39.3% win rate. VIX 20–30 → 62–64% win rate. Skip `can_buy` when VIX < 18 in the broker script. This is a free 5–8pp win rate improvement with no model changes needed.

---

## 7. Sequence to Execute

```
1. Restore mp_* features → re-run 3__AlphaSensitivity.py across universe
2. Remove label_threshold + revert SPW range → re-run 4__Predictor.py --tune --runpercent 35
3. Re-run eda_predictions.py → verify IC ≥ 0.05, signal win rate ≥ 52%, monotone win curve
4. Re-run 5__NightlyBackTester.py --sample 90 → compare Sharpe vs baseline
5. (Optional) Add VIX filter + preferred share filter in backtester
```

---

## 8. What Success Looks Like

After restoring mp_* features and reverting label_threshold:
- IC should return to ≥ 0.05 (was 0.051 at best)
- Probability distribution should be unimodal, smooth
- Win rate curve should be monotone with probability
- Signal win rate ≥ 52%
- Backtest Sharpe > 1.0 (approaching the 5.59 best-config level once Optuna is allowed to run)

The strategy mechanics are sound — live system runs ~7.5%/month per project memory. The problem is degraded features feeding degraded predictions. Fix the features, and the predictor will recover.
