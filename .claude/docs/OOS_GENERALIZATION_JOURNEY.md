# OOS Generalization — Troubles & How We Overcame Them

**Session: 2026-06-03 → 2026-06-05.** Goal stated by user: *"hard focus on out-of-sample
generalization of `4__Predictor.py`."* This doc is the durable record of what was actually
wrong, every false lead and why it was wrong, the methodology we had to invent to stop
fooling ourselves, and the fix that shipped.

> **TL;DR.** The predictor wasn't broken. The *evaluation* was. Every "OOS collapse" we
> chased was either a misleading proxy metric or single-path noise — this strategy's monthly
> returns swing **±15pp from the RNG seed alone**, which swamps almost any real effect. The
> only change that survived honest (multi-seed, strategy-level) validation was a **seed-
> ensemble**, productionized as `4.1__Predictor.py`. It lifts OOS compounded return ~2× over
> the single model with all months green.

---

## 1. The starting symptom

The headline backtest read **172% ann / Sharpe 11.44** (the replication-report config). The
user wanted to improve OOS generalization. First real finding: **the predictor could not even
see its own OOS performance.** It only ever reported metrics on the **calibration slice**
(the 15% window right after training, Dec 2025–Mar 2026) — which happened to be the *favorable*
window. The genuinely-unseen tail (Mar→May 2026, incl. the −7.7% April) was only ever scored by
the slow full backtester, which also conflates predictor quality with strategy/execution logic.

**So step 1 had to be: make OOS measurable cheaply and honestly.**

---

## 2. The tools we built (all additive; winning pipeline untouched)

| Tool | What it does |
|---|---|
| `4__Predictor.py --oos_only` | Re-derives the EXACT train/calib/OOS date boundaries (validated against `summary.json` train_rows = EXACT) and scores the genuinely held-out tail. Rank-band × split table of per-day return + precision. |
| `--walkforward` | Train a config at monthly anchors, score the FOLLOWING month — many independent OOS windows, net of market beta. |
| `--wf_sweep` | Parallel, multi-seed hyperparameter sweep. Beats the noise floor by seed-averaging. |
| `--train_end_date` | Train through an arbitrary cutoff (for refit-cadence experiments). |
| `5__NightlyBackTester_oosvalidate.py` | A **copy** of the backtester with a read-only `--data_dir` flag, so any candidate's predictions can be backtested without clobbering live `Data/RFpredictions`. **Original backtester reverted to pristine.** |

---

## 3. The false leads (each looked compelling, each was wrong)

### 3a. "The edge migrated off the top tail to the shoulder band"
`--oos_only` showed the top-1% band (what the strategy fires) decaying to ~0 net OOS while the
0.90–0.95 "shoulder" band held +0.25%/day. Looked like: stop trading the top 1%, trade the
shoulder. **Debunked by per-month breakdown:** the shoulder's edge was *one month* (April).
Median month was a tie. The aggregate was a mirage.

### 3b. "Optuna overfits; simple params generalize better"
A clean controlled test (same rp=75 cutoff, same tail): production Optuna model's top-1% net
= **+0.015** (Sharpe 1.17) vs a simple `--no_tune` model = **+0.234** (Sharpe 4.09) — ~15× at
the **band level**. Smoking gun: Optuna fit IS harder (1.16 vs 0.88) yet generalized worse.
Looked decisive. **Debunked at the strategy level:** backtesting both on identical data,
**production WON** (208.7% ann / Sharpe 3.41 / DD 12.8% vs simple 206.5 / 3.02 / 20.7), and on
the OOS months production +34.9% compounded vs simple +1.2%. The band metric `top1_net` was a
**bad proxy**: (i) the `--oos_only` "OOS" window excluded production's huge Jan/Feb (those sat
in the calib slice), and (ii) an equal-weight top-1% basket ≠ a 4-name gated strategy with
stops/exits. **Lesson: validate at the real objective, not a proxy.**

### 3c. "The live model is stale; refit cadence will help"
Single-path refit test looked spectacular: a 1-month-fresh model beat the aging Dec model on
all 3 differing OOS months, **+32.7% vs −6.1% compounded.** **Debunked by multi-seed:** with
just 2 seeds it FLIPPED — stale beat fresh. The *same* Dec model produced March returns of
**−13.4% / +8.9% / +15.4%** across three seeds. A ~29pp swing from the seed alone. The refit
"signal" was pure noise.

### 3d. "Regularized params / recency weighting / recent-only training"
All tested in `--wf_sweep` (8 configs × 11 anchors × 4 seeds, parallel, seed-averaged). Result:
**every config landed within the seed-noise of baseline.** No architectural knob beat baseline.

---

## 4. THE CORE DISCOVERY (the thing that explains everything above)

**This strategy is noise-dominated at the single-path level.** Concentrated 4-name book +
stops/exits = monthly returns that swing **±15pp from the RNG seed alone** (the seed only
changes data-shuffle / early-stop split / subsample draws). At the annual level, 8 single seeds
of the *same config* ranged **87% → 266% ann, Sharpe 1.74 → 3.54.**

Consequences, now permanent rules:
1. **A single backtest cannot rank two models.** Any single-path comparison can flip sign on a
   re-run. Every false lead above came from trusting one path or one proxy.
2. **Any model comparison MUST be multi-seed (≥4), averaged, at the strategy level.**
3. The "172% / Sharpe 11" and "443% / Sharpe 4.34" headlines are **in-sample-dominated**;
   only the OOS months are forward-meaningful.

This is the single most valuable output of the whole investigation. It reframed the problem
from "fix the model" to "stop being fooled by the measurement, then attack the variance itself."

---

## 5. The fix that survived: a SEED-ENSEMBLE

If the problem is seed variance, the textbook fix is to **average it away**: train N models with
different seeds, average their `predict_proba` at inference, then rank/rescale as usual.
Averaging cuts the variance *by construction* — no need to prove it through the noise.

**Validated overnight (2026-06-04), strategy-level, Jan–Apr 2026 OOS.** Single-seed distribution:
ann mean 178%, Sharpe 2.79, maxDD 21.6% (range 87–266% / 1.74–3.54). Ensembles:

| | ann% | Sharpe | maxDD% |
|---|---|---|---|
| ens4 | 247 | 3.30 | 18.8 |
| ens8 | 224 | 3.32 | 18.8 |
| ens12 | 228 | 3.27 | 17.3 |
| ens16 | 254 | 3.63 | **16.8** |

The clincher: **maxDD falls monotonically with ensemble size (21.6→18.8→17.3→16.8)** — the
cleanest possible variance-reduction signature. Four sizes agreeing = signal, not luck.
Regularized params did NOT beat prod when ensembled, so: **keep prod params, just ensemble them.**

### Productionized as `4.1__Predictor.py`
Drop-in superset of `4__Predictor.py`: identical CLI/flags/output, plus `--n_seeds` (default 8)
and a `--predict` alias. Trains N diverse members (differ by `random_state` → subsample/colsample
draws), wraps them in `EnsembleModel` (predict_proba = mean; exposes feature_names_in_/
best_iteration/feature_importances_ so `evaluate_and_save`/`run_inference` work unchanged), and
saves the **plain list of members** to `xgb.joblib` (robust pickle, loads anywhere, also loads
old single `4__` models). `--n_seeds 1` reproduces `4__Predictor.py` exactly.

### First full production backtest (2026-06-04, user-run)
**454% total / 443% ann / Sharpe 4.34 / maxDD 16.76% / 302 trades / 58.9% win.** Headline is
IS-inflated (Jul–Sep 2025 unicorns are in-sample). The honest OOS read (Jan–Apr 2026):

| OOS month | Ensemble (4.1) | Single prod model |
|---|---|---|
| 2026-01 | +6.47 | +9.71 |
| 2026-02 | +26.14 | +16.42 |
| 2026-03 | +8.26 | −2.57 |
| 2026-04 | +12.62 | +8.42 |
| **compounded** | **+63.7%** | +34.9% |

Ensemble ~**2× the OOS compounded return, every month green** (single model went red in March),
Sharpe 4.34 vs 3.41. DD 16.76% is *higher* than the live model's 12.82% — but the live model was
a **lucky low-DD draw** (seed-DD mean ~21.6%); 16.8% is the *reliable* expectation. Net: a real,
shippable improvement.

---

## 6. Where it stands / what's running now

- **Intraday fill reality-check IN PROGRESS** (user logged into live TWS:7496). `2.2__Trade
  HistoryIntradayDownloader.py --trade-history Data/TradeHistory.parquet` pulls real 1-min RTH
  bars for each of the 302 new trades' hold windows → `Data/IntradayTradeSim/`. Then
  `8__IntradayFillSim.py --trade-history Data/TradeHistory.parquet` reprices entries at a real
  10:00 ET fill vs the backtest's idealized open/close. This is the honest "how much survives
  realistic fills" haircut — user rightly expects much of the headline to fall off live.
  **Caveat (known):** `trade_history` per-trade PnL doesn't perfectly reconcile with its
  `AccountValue` curve, so use the fill sim as a *trade-level fill-realism* check, not to
  reproduce the headline %.

- **`4.1__Predictor.py` is NOT yet promoted to live.** To promote: `python 4.1__Predictor.py`
  (no dir flags) overwrites `Data/XGBPipeline` + `Data/RFpredictions`. A 2nd independent 8-seed
  ensemble run landing in the same OOS ballpark would be the multi-path confirmation.

---

## 7. Standing rules earned this session (do not relearn the hard way)

1. **Validate at the strategy-level backtest, never a band/IC proxy.** Proxies (`top1_net`,
   IC, a sub-window) repeatedly said the opposite of the real objective.
2. **Multi-seed (≥4), averaged — single-path comparisons are noise.** ±15pp/month seed variance.
3. **Backtest WITHOUT `--force`** for evaluation (`--force` adds a slow finviz live-export; the
   backtest metrics are the same; ~10× faster).
4. **Touch nothing live; copy-and-edit; back up before overwriting.** Original backtester/broker
   stay pristine.
5. The seed-ensemble is the model-level win. Further gains likely live in **execution realism**
   (the intraday work) and **strategy-layer** overlays, not in more predictor knob-twiddling.

---

## Key files
- `4.1__Predictor.py` — seed-ensemble production predictor (the fix).
- `4__Predictor.py` — single-model original (unchanged; superset of its flags now in 4.1).
- `5__NightlyBackTester_oosvalidate.py` — backtester copy with `--data_dir` (original reverted).
- `2.2__TradeHistoryIntradayDownloader.py` / `8__IntradayFillSim.py` — intraday fill reality-check.
- `analysis_output/overnight_ensemble*.py`, `overnight_master*.py` — the validation harness.
- `.claude/skills/overnight-solver/` — the reusable autonomous-overnight workflow this produced.
- Memory: `project_oos_decay_harness_2026_06.md` (the running technical log).
