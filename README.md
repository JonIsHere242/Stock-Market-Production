# Stock-Market-Production

A daily-horizon US equity trading system. Every night it rebuilds a feature panel across the tradeable
universe, scores each ticker with an XGBoost classifier, backtests the resulting signal, and writes a
candidate pool. The next morning a filter funnel narrows that pool to a book of at most four names and a
broker process places the orders through Interactive Brokers.

It runs unattended on a Windows machine under Task Scheduler and trades a live account. Everything below
describes what the code actually does today, including the parts that exist only because something went
wrong once.

## What is in this repository

Code and configuration, 925 tracked files. The `Data/` tree (about 4,250 per-ticker price parquets,
processed feature panels, SEC bulk downloads, the trained model), API keys, and local lab forks are
excluded by `.gitignore`. `setup.ps1` scaffolds the directory tree and drops key templates so a fresh
clone can rebuild the data side from scratch.

```text
1__TickerDownloader.py     universe from SEC company_tickers
2__PriceDownloader.py      daily OHLCV via IBKR -> Data/PriceData/*.parquet
build_data_panels.py       SEC fundamentals, Form 4 insider, sector map
3__FeatureFramework.py     plugin feature engine -> Data/ProcessedData_v2/
4__Predictor.py            XGBoost train + inference -> Data/RFpredictions/
4.5__NeutralizePreds.py    pred-space factor neutralization
5__NightlyBackTester.py    backtrader sim -> Data/0__signals.parquet (12-name pool)
6__TickerRelator.py        cross-asset correlation clustering (weekend job)
7__MacroFilter.py          pool -> final book -> _Buy_Signals.parquet
9_SuperFastBroker.py       IBKR execution (9b__ is the paper-account twin)
8__IntradayFillSim.py      fill-realism harness (offline)
8b__BookPostmortem.py      daily winner/loser discriminator (offline)
Util.py                    shared library: can_buy(), PositionSizer, logging, calendars
trading_system.ps1         the orchestrator both scheduled tasks call
FeatureTemplates/          736 feature blocks (188 promoted, the rest candidates)
FeatureDiscovery/          paper mining and feature validation tooling
fetchers/                  raw source downloaders (SEC, FINRA, FRED, Treasury, CFTC)
auxiliary/                 correlation and market-cap side jobs
v2/                        content-addressed feature engine (scaling work, offline)
```

## Nightly run

`trading_system.ps1 -Mode evening` fires at 17:00 local. Stages run in order with a 20 second gap between
them for memory release, and a failed stage alerts without stopping the rest of the chain.

| Stage | Command |
| --- | --- |
| Ticker Downloader | `1__TickerDownloader.py --ImmediateDownload` |
| Price Downloader | `2__PriceDownloader.py --RefreshMode` |
| Data Panels | `build_data_panels.py all --refresh-all` |
| Feature Framework | `3__FeatureFramework.py --all --exclude vvg vaq_vg rvg_wl volume_spectral_splatter` |
| Predictor | `4__Predictor.py --predict_only --model_dir Data/_ship_v2/model ...` |
| Neutralize Preds | `4.5__NeutralizePreds.py --mode spy200v2 --dose_above 0.15 --dose_below 0.30` |
| Nightly BackTester | `5__NightlyBackTester.py --force` |

The four excluded feature blocks are visibility-graph and spectral families whose 29 columns the shipping
model drops at inference anyway. They cost about 16% of framework runtime for no effect on the model, so
computing them was waste.

`BT_SAMPLE_SEED=42` is exported before the backtester. Unseeded tie-breaks in the entry ranking swing
annual return by 20 to 40 percentage points on byte-identical predictions, which makes any two nightly
reports incomparable. Pinning the seed is not optional.

After the last stage the runner reads `Data/0__signals.parquet` back off disk and checks that its
`TargetDate` is dated for the next session. If it is not, the run alerts loudly instead of finishing
green.

## Morning run

`trading_system.ps1 -Mode morning` fires at 07:28 local, exits immediately on weekends, and exits with an
alert if a catch-up start lands after the 10:30 ET cutoff.

1. Pre-flight the TWS socket on port 7496, roughly 30 minutes before anything needs it.
2. Run `7__MacroFilter.py`, which self-waits to 09:35 ET internally.
3. Re-read `_Buy_Signals.parquet` and confirm its `TargetDate` is today. A book that is not fresh means no
   broker launch at all.
4. Hold until 09:57 ET, polling IBKR so a dead session surfaces before the entry window rather than during
   it.
5. Launch `9_SuperFastBroker.py`, retrying only on unambiguous connection failures.

A SPY abort is classified as a success and never retried. Re-running past that gate is exactly the manual
override the intraday study prices at Sharpe -6.50.

## Feature framework

`3__FeatureFramework.py` is a plugin host. Each file in `FeatureTemplates/` declares `METADATA` (name,
description, `requires`, `produces`, tags) and a `compute(df) -> df` function. The host imports every
non-underscore file, topologically sorts blocks by their declared dependencies, and runs the resulting
order across the universe in a process pool. Adding a feature means copying
`FeatureTemplates/__example_template.py` and writing one function; the new file is picked up on the next
run without touching any registry or import list.

The underscore prefix carves out the namespace. Double-underscore files are tooling and are never imported
as blocks. Single-underscore files are either shared helpers (`_marketcap`, `_indexes`, `_fundamentals`,
`_insider`) or candidate blocks that a production build deliberately skips. 188 blocks are promoted,
377 are machine-generated candidates from the feature factory, and 86 are ports of published research.

Candidates are gated before promotion rather than after. `FeatureTemplates/__tail_screen.py` measures
marginal lift in the top decile, which is the only region the strategy trades; global rank IC rewards
features that sort the middle of the book and the middle of the book is never held. A candidate has to
survive at least four seeds to move. Single-seed wins are noise, and enough of them have been chased here
to say that with confidence.

Non-price panels come from `build_data_panels.py`, which merges three former standalone builders over SEC
XBRL companyfacts, SEC Form 4, and the sector map. Every SEC fact carries both the date it describes and
the date it became public. The merges key on the filing date via a backward `merge_asof`, so a feature can
never see a filing before the market could. Keying on the period end date instead is silent lookahead
leakage and it does not show up in out-of-sample IC.

## Predictor

`4__Predictor.py` is a single-file XGBoost pipeline: load, label, filter the universe, shuffle within each
date, split with a 5 day embargo, build the feature matrix, apply recency weights, tune with Optuna, train,
calibrate, and score.

The label is `topq`, a binary flag for the top 20% of next-day returns within each trading day. Attempts to
make the target cleverer have been tried and rolled back. Residual regression targets, ranking objectives,
and target ensembles all looked better in-sample and none of them survived a pinned as-of backtest. One of
those experiments turned out to have tomorrow's return sitting in the feature matrix, which was invisible
in out-of-sample IC and obvious in the calibration AUC.

Nightly inference runs `--predict_only` against a pinned model directory (`Data/_ship_v2/model`) with
explicit `--drop_features_exact` and `--drop_feature_patterns` lists. The feature framework evolves faster
than the model gets retrained, so live inference is pinned to the exact column set the model was trained
on. Drift between the two silently degrades predictions rather than raising an error.

## Prediction neutralization

`4.5__NeutralizePreds.py` sits between prediction and backtest. For each day it projects the cross-section
of raw scores onto beta to SPY, ATR percentage, and log dollar volume, subtracts a fraction of that
projection, then quantile-maps the result back onto the original probability values for that day. The
per-day marginal distribution is preserved exactly, so only the within-day ordering changes and every
downstream threshold keeps its meaning.

The dose is regime-adaptive: 0.15 when SPY closes at or above its 200 day EMA, 0.30 below. It has zero
fitted parameters. Any gate failure (too few output files, a factor panel that is stale on the signal day,
fewer than 5% of names changed) leaves the raw predictions in place and exits non-zero, so the pipeline
alerts and the system trades the plain signal instead of a broken one.

## Filter funnel

`7__MacroFilter.py` narrows a 12 name pool to at most 4, ordered by cost so the paid step only ever sees
what survived the free ones.

- Stage 0: align the pool to the next NYSE session, attach price history.
- Stage 1: hard exclusions. Price floor, micro-cap, weekly volatility cliff, UpProbability floor at 0.40,
  ideological quarantine list.
- Stage 2: soft flags. Penny/illiquid, a gap-then-volume-collapse signature that usually means a pending
  acquisition at a fixed price, and RSI above 80. These de-prioritize and escalate to the LLM.
- Stage 3: an LLM pass with web search over the survivors only, checking for an active M&A target or a
  material crisis. If the API key is unfunded the stage skips cleanly and the mechanical funnel still
  produces a book.
- Stage 4: rank by UpProbability under an industry concentration cap, relaxing soft flags and then the cap
  if a clean book cannot be filled.

An RSI 30-40 "death zone" exclusion used to live in Stage 1. A 17,600 candidate study found that band was
the best-performing one in the sample and the gate was the only screen that was net negative at book level.
It was removed.

The write is guarded by provenance. A book stamped `VetSource='manual'` is never replaced automatically, a
book already vetted by the LLM exits without spending anything, and a run whose LLM died mid-flight refuses
to overwrite an existing same-session book. That guard exists because on 2026-07-02 a rerun clobbered a
hand-vetted book with a mechanical fallback while the API key was unfunded.

## Execution

`9_SuperFastBroker.py` reads the narrowed book, connects to TWS through `ib_insync`, and places IBKR
Adaptive limit orders. Sizing is four equal slots against account value with a 10% cash buffer and lot
rounding to avoid odd lots.

Entry timing comes from an intraday study over 1,231 signal-day observations. Buying at the open gives
Sharpe 0.31; waiting until 10:00 ET gives 1.11. Conditioning on SPY at 10:00 splits that sharply, from
Sharpe +6.77 when SPY is up more than 0.5% to -6.50 when it is down more than 0.5%. The broker therefore
holds until 10:00 ET, aborts the entire day if SPY is at or below -0.5% from the open, skips any individual
name that gapped more than 1.5% above its open, and carries a 1.9% hard stop.

That stop used to be 0.5%. A name with 2% daily volatility entered at 10:00 has a very high chance of
touching -0.5% before the close even on a good day, so the old setting exited most positions on noise and
the backtest win rate never described the system that was actually running.

## Fail-safes

Most of these are dated because each one is a repair.

- Single-instance lock file. Concurrent runs corrupt shared parquet writes and produce phantom backtest
  results.
- Partial-bar guard. A catch-up start inside market hours would pull an in-progress daily bar into the
  feature build, so the evening mode exits instead.
- Book staleness abort in the broker, measured in trading days so weekends do not count. On 2026-06-26 it
  filled a four-day-old book. On 2026-07-07 the one-day "warn but proceed" path traded a stale book after
  the nightly pipeline had silently stopped running for four days, so the tolerance is now zero.
- Exit-code checks that do not trust exit codes. `2__PriceDownloader.py` swallows exceptions and exits 0,
  so the runner greps its transcript for "No tickers were successfully processed".
- Alerts that are hard to miss. Console, log file, `BROKER_ALERT.txt`, three beeps, and a non-blocking
  popup, because during the 2026-07 incident the first three went unnoticed for four consecutive days.

## Offline research code

`8__IntradayFillSim.py` replays signals against 5 minute bars to check that backtest fills are reachable in
reality. `8b__BookPostmortem.py` runs a daily discriminator over winners and losers in the live book.
`FeatureDiscovery/` mines papers into candidate feature specs and validates them for causality before they
reach `FeatureTemplates/`. `v2/` holds a content-addressed rewrite of the feature engine that caches by
content hash and generates block code, used for scaling experiments and not wired into the nightly run.

None of these are in the live path and none of them are allowed to write to the live signal files.

## Setup

Windows, Python 3.13, and a running TWS or IB Gateway on port 7496.

```powershell
.\setup.ps1            # venv, dependencies, Data/ tree, key templates, import check
.\setup.ps1 -Gpu       # force the CUDA torch wheel
.\setup.ps1 -ColdStart # then run stages 1-5 end to end (hours)
```

`requirements.txt` pins the packages the code imports. Two are installed separately by `setup.ps1` because
they are platform or source specific: torch, which needs the correct CPU or CUDA wheel index, and
backtrader, which is installed from pinned git commits. API keys go in the files templated by
`Claud-API-KEY.txt.example` and `.fred_api_key.example`, both of which are gitignored in their real form.

Manual invocation:

```powershell
.\trading_system.ps1 -Mode evening   # rebuild data and signals after the close
.\trading_system.ps1 -Mode morning   # funnel and broker before the open
.\trading_system.ps1                 # picks a mode from the current hour
```

## On the performance numbers

This repository does not publish a headline return figure, and the numbers embedded in older docstrings
should be read as historical run records rather than claims. A previous 172% annualized / Sharpe 11.4
checkpoint was later decomposed and attributed mostly to hyperparameter tuning, the training window, and
the random seed, and it did not reproduce out of sample.

What the code does instead is make results falsifiable: pinned as-of backtests so a "validation" cannot
peek past its cutoff, a fixed tie-break seed, a minimum of four seeds before any comparison counts, a 5 day
embargo between training and calibration, point-in-time merges on SEC filing dates, and a habit of
re-running the thing that looked good until it stops looking good. A fair number of features and ideas in
the history of this repo died that way.

## Author

[@JonIsHere242](https://github.com/JonIsHere242)

## License

The Old Secret Mission CIA Edition.
