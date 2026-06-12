# Session summary (2026-05-30 → 2026-05-31) — from "high IC" to "high top-end discrimination"

## The arc (what actually happened)
1. **Goal**: lift mid-importance, per-stock features to boost the model.
2. **Attempt 1 (`_tsz`)**: added 13 within-ticker z-score / rebuild features across 4 rounds — all
   leak-safe (verified to <1e-6 vs a strictly-past recompute), all validated by strong standalone
   IC (per-day topq-label IR 0.4–0.9).
3. **Verdict**: regenerated features + retrained + backtested → **WORSE** (Sharpe 2.22 vs the
   ~4.5–6 baseline, 316 vs ~1115 trades, ~2× drawdown). **Fully reverted** (code + model + live
   signals restored from `Data/_backup_pre_tsz_20260530/`; verified pre-`_tsz` snapshot in
   `_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py`).
4. **Why it failed → the breakthrough**: high *global* IC but redundant with the model's incumbent
   volatility features; added complexity, perturbed the high-precision tail, added no edge where
   it matters.

## The conceptual shift (the real win)
| Old bar | New bar |
|---|---|
| Global per-day rank-IC (orders the *whole* universe) | **Top-decile marginal discrimination** — separate the *elite* from the *okay* among names the model already likes |
| Necessary-at-best, often misleading | Conditional on the model's own ranking + orthogonal to incumbents |
| (no stability notion) | **Tail-consistency** — reshuffle the tail *consistently* over time, not flippily |

P&L comes only from the top ~1%/day, so that is the only region worth optimizing. Global IC was
rewarding the wrong thing (it even ranks market-wide VIX features highest — they can't discriminate
between stocks at all).

## Tools built (the durable asset — all in repo root)
- `classify_features.py` — buckets all 327 features: 45 VIX / 6 non-VIX-market-wide / **267
  stock-specific** (the only discrimination levers). Output: `analysis_output/feature_classes/`.
- `eval_tail_discrimination.py` — the **TDMD screen**: marginal discrimination within the top decile,
  partialling out the incumbent score. Output: `analysis_output/tail_discrimination/`.
- `eval_best_of_best.py` — **elite-vs-okay AUC** within the trade region (top-1% cut).
  Output: `analysis_output/best_of_best/`.
- `auto_discriminator_search.py` — **automated Phase-A loop**: generates feature variants
  (over_close / over_atr / x_range / expand{3,5,10,20} / delta / accel / csz / absdev / sqdev),
  scores them, with the critical **range-orthogonality guard** + **tail-consistency** (`tail_ir`,
  `flip_rate`) gates. Output: `analysis_output/auto_search/`.
- `interaction_search.py` — **exhaustive parallel** pairwise + triple interaction search
  (~880k combos in 263s on 30 cores) with a **temporal-holdout guard** (must replicate across both
  calib halves). Output: `analysis_output/interaction_search/`.
- (plus `eda_midband_features.py`, `eda_feature_scan.py` from early exploration)

## Key findings (hard-won)
- **Monotonic transforms are no-ops for XGBoost** (it splits on rank order) — only
  non-monotonic-across-the-panel / interaction transforms move the tail picks.
- **Best-of-best = range/volatility magnitude** — but that axis is already saturated in the model
  (`ATR%` rank 10, `percent_range` 61, `High_Low` 40). That is *why* headway is brutal.
- **Normalization exposes hidden signal**: `atr_14` (absolute $) AUC 0.48 (anti-discriminates) →
  `ATR%` (÷price) AUC 0.55. Same data, better encoding (supports the "it's all in OHLCV" thesis).
- **Conjunctions win**: `min` ("both/all high") dominated every interaction sweep; products and
  divergences added nothing.
- **Anti-overfitting matters at scale**: the range-orthogonality guard and the temporal-holdout
  split caught traps that would have fooled us — `mean3` (linear average) inflating `tail_ir` via
  variance reduction, and `ATR%` leaking back in as a 14-day range feature.

## Staged for validation (Phase B, NOT yet run)
`3__AlphaSensitivity_automated.py` — a copy of the reverted `3__AlphaSensitivity.py` that:
- writes to a **separate** `Data/ProcessedData_automated/` (live baseline untouched),
- adds 3 survivor conjunctions via a cross-sectional post-pass `add_interaction_conjunction_features`:
  - `IX_pattern_energy_min3` = min3(hammer_pattern, gap_in_atr_terms, dollar_volume_ma_10) — cleanest (range_corr 0.05, redund 0.18)
  - `IX_overextension_min3`  = min3(VWAP%_from_high, HC_Predict_Regime_norm, cv_50d_percentile) — over-extension AVOID signal (most orthogonal, range_corr 0.007)
  - `IX_struct_signal_min3`  = min3(mp_d4_b185_sym_trace, trading_signal_composite, G_Momentum_Confluence_Indicator) — highest tail_ir 0.75

**Run later (separate dirs so nothing collides with the baseline):**
```
python 3__AlphaSensitivity_automated.py
python 4__Predictor.py --input_dir Data/ProcessedData_automated \
                       --model_dir Data/XGBPipeline_automated \
                       --output_dir Data/RFpredictions_automated
# then gate: top-1% precision + backtest vs the restored baseline
# (backtester reads a hardcoded Data/RFpredictions — swap in the automated preds for that one run)
```

## Honest scorecard
- **Features shipped to production: 0** — the `_tsz` batch was reverted (correctly).
- **Validated new alpha: none yet** — the 3 conjunctions are theory-coherent, holdout-replicating
  *hypotheses* pending the Phase-B gate.
- **Methodology + tooling: a genuine upgrade** — a repeatable, automated, overfitting-resistant
  pipeline that screens on the metric that actually pays (top-1% discrimination), not global IC.

## The discipline that holds
Nothing graduates without an **OOS top-1%/day precision uplift + backtest** against the restored
baseline. The screens *propose*; only the retrain *confirms*. At ~6 Sharpe / ~250% the bar is
brutal — expect most candidates to be redundant; that's the normal outcome, not a failure.

## State at end of session
- Baseline being retrained to restore ~250% / 6.0 Sharpe (independent — reads live `Data/ProcessedData`).
- Automated feature pipeline staged (on-disk code only, not run — avoiding double-retrain load).
- Memory: see `feedback_tail_discrimination_over_ic.md` (the core principle + all findings/tools)
  and `project_per_stock_feature_lift.md` (the `_tsz` revert log).
