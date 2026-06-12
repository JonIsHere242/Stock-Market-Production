# Feature changes — per-stock `_tsz` discrimination lift (2026-05-30)

## What & why

Added **13 new per-stock features** to `3__AlphaSensitivity.py` that re-base
existing mid-importance, per-stock features against **each stock's own recent
history** (a within-ticker rolling z-score, suffix `_tsz`), plus one continuous
rebuild of a coarse count. Goal: lift signal from features the model currently
under-uses, to better discriminate between stocks.

**Every original feature is left untouched** — these are additive siblings only.

### Method (validated on the live panel, metric = per-day cross-sectional IC vs the `topq` label)

- Monotonic transforms (log/winsor/clip/rank) are **no-ops** for a single feature
  into XGBoost and don't move rank-IC. The only lever that changes a single
  feature's signal for a tree is a **non-monotonic re-basing** → within-ticker
  rolling z (`_tsz`).
- Scored against the **actual `topq` label** (per-day top-20% next-day return),
  not mean return — these are tail features.
- Checked **mutual correlation** of candidate `_tsz` versions to avoid adding
  collinear duplicates.

## The shared helper

`rolling_ticker_zscore(s, window=252, min_periods=60, clip=8.0)` — defined right
after `safe_log` (~L219). Lookahead-safe (baseline uses strictly past values via
`shift(1)`; current value is known at close of day *t*, predicting *t+1*).
Clipped to ±8 to kill the `(1-0)/~0` blowup that sparse **binary** flags produce;
clipping is **rank-neutral** (XGBoost splits on order) so the validated IRs are
unchanged — it only protects the `_xs` cross-sectional dual / any linear consumer.

## The 13 added features

| Feature | Generating function | topq-IC IR (raw → tsz) |
|---|---|---|
| `dollar_volume_ratio_252d_tsz` | `calculate_liquidity_features` | 0.68 → **0.93** |
| `volume_burst_intensity_5d` † | `calculate_liquidity_features` | count 0.44 → **0.66** |
| `atr_percentile_rank_tsz` | `calculate_volatility_atr_features` | 0.02 → **0.49** |
| `atr_regime_low_tsz` | `calculate_volatility_atr_features` | -0.06 → **-0.42** |
| `atr_regime_high_tsz` | `calculate_volatility_atr_features` | 0.00 → **+0.64** |
| `G_Volume_Weighted_High_Ratio_tsz` | `calculate_genetic_indicators` | -0.46 → **-0.86** |
| `return_3d_tsz` | `calculate_price_momentum_features` | -0.08 → **-0.20** |
| `Price_Differential_Ratio_HighVolatilityRegime_tsz` | `add_volatility_regime_signals` | -0.07 → **+0.58** |
| `Price_Differential_Ratio_LowVolatilityRegime_tsz` | `add_volatility_regime_signals` | -0.09 → **-0.55** |
| `doji_pattern_tsz` | `calculate_price_action_features` | -0.51 → **+0.72** |
| `cv_10d_regime_high_tsz` | `calculate_volatility_cv_features` | 0.19 → **+0.42** |
| `cv_20d_regime_high_tsz` | `calculate_volatility_cv_features` | 0.05 → **+0.30** |
| `Complexity_Invariant_Distance_tsz` | `add_complexity_metrics` | -0.03 → **+0.25** |

† `volume_burst_intensity_5d` is not a `_tsz`; it's a continuous rebuild of the
`sustained_volume_burst_count` (0-5, 96% zeros) = 5-day mean of `volume/vol_10d_avg`,
preserving burst magnitude. 0.44 corr to both the count and `volume_spike_ratio`.

The predictor auto-generates `_xs` cross-sectional duals for all of these
(`--add_xs_features`), and picks them up with no allowlist edits
(`select_base_features` takes all numeric minus `NON_FEATURES`).

## Leak-safety

All verified to match an independent strictly-past recomputation to < 1e-6
(window `[t-252, t-1]`, excluding *t*). Warmup rows (<60 days history) are NaN,
which XGBoost handles natively.

## Analysis tooling added (not part of the pipeline)

- `eda_midband_features.py` — comprehensive EDA + transform search.
- `eda_feature_scan.py` — scan importance band for `_tsz` lift candidates.
- `classify_features.py` — bucket all features into VIX / non-VIX market-wide /
  stock-specific by cross-sectional variance share.

Outputs in `analysis_output/`.

## How to REVERT

1. **Code** — the only pipeline file changed is `3__AlphaSensitivity.py`.
   ⚠️ Do **NOT** `git checkout 3__AlphaSensitivity.py` — this file also contains
   substantial *prior* uncommitted work (GP cross-sectional alpha features) that
   predates this session; a checkout would destroy it. Instead restore the
   verified pre-`_tsz` snapshot (= current file minus only the 2026-05-30
   additions, reconstructed & verified to compile with zero `_tsz` markers):
   ```
   copy _old_versions\3__AlphaSensitivity_pre_tsz_20260530.py 3__AlphaSensitivity.py
   ```
   (Rebuilt by `_old_versions/_reconstruct_pre_tsz.py`, which removes each added
   block by exact-match assertion. Confirmed: current = snapshot + 114 added lines,
   all mine.)

2. **Trained artifacts** (if a run already overwrote them) — pre-run backups at
   `Data/_backup_pre_tsz_20260530/`:
   - `XGBPipeline/` → restore to `Data/XGBPipeline/`
   - `RFpredictions/` → restore to `Data/RFpredictions/`
   The canonical 172% model in `Data/SimpleModel/` is **not touched** by these runs.

3. **Regenerated features** — `Data/ProcessedData/` is wiped & regenerated by
   `3__AlphaSensitivity.py`. To restore the pre-change features: revert the code
   (step 1) and re-run `python 3__AlphaSensitivity.py`.

## Status

Features added & validated by IC. **Realized lift (backtest) not yet measured** —
pending `python 3__AlphaSensitivity.py` → `python 4__Predictor.py` →
`python 5__NightlyBackTester.py --force`.
