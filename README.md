# Stock-Market-Production

A daily-horizon US equity trading system. Every night it rebuilds a feature panel across roughly 4,200
tradeable tickers, scores each one with an XGBoost classifier, backtests the resulting signal, and writes a
36 name candidate pool. The next morning a broker process rests dip-limit orders on that pool through
Interactive Brokers and manages the exits.

It runs unattended on a Windows machine under Task Scheduler and trades a live account. Everything below
describes what the code does today, including the parts that exist only because something went wrong once.

## What is in this repository

The pipeline and the libraries it imports, 80 files. What is kept out on purpose: the `Data/` tree
(per-ticker price parquets, processed feature panels, SEC bulk downloads, the trained model), API keys,
the A/B rigs under `experimental/`, the diagnostics suite, and the feature library. `setup.ps1` scaffolds
the directory tree and drops key templates so a fresh clone can rebuild the data side from scratch.

The feature library is the omission worth naming. `FeatureTemplates/` holds close to 2,000 blocks locally,
and none of the promoted ones are published, because those are the model's edge. What ships is the block
contract, `FeatureTemplates/__example_template.py`, plus ten candidate blocks as worked examples of the
format. The framework that loads them is here in full.

```text
1__TickerDownloader.py     universe from SEC company_tickers
2__PriceDownloader.py      daily OHLCV via IBKR -> Data/PriceData/*.parquet
2__MacroFetcher.py         declarative manifest for every non-price source (not yet wired
                           into the runner; the nightly still calls the two stages it wraps)
fetchers/                  raw source downloaders (SEC, FINRA, FRED, Treasury, CFTC, Ken French,
                           Shiller, index lakes) plus refresh_market_data.py, the nightly caller
build_data_panels.py       SEC fundamentals, Form 4 insider, sector map
builders/                  the filing-meta and filing-calendar panel builders it shells out to
3__FeatureFramework.py     plugin feature engine -> Data/ProcessedData_v2/
FeatureTemplates/          the block contract plus ten candidate blocks as examples; the
                           promoted library is not published
4__Predictor.py            XGBoost train and inference, plus pred-space neutralization and the
                           conviction tilt -> Data/RFpredictions/
5__NightlyBackTester.py    backtrader sim of the live configuration -> Data/0__Signals.parquet,
                           the 36 name pool in UpProbability order. Canonical source since
                           2026-08-30; a bare hand run overwrites the live pool, so use
                           run_optimal_backtest.py for sandboxed runs.
signal_filter.py           mechanical screen that writes the MechExclude column on the pool
rubric_veto.py             the operator's shallow veto, at most 4 names, as RubricExclude
7__MacroFilter.py          pool -> narrowed book -> _Buy_Signals.parquet. Out of the default
                           live path since 2026-08-28; see below.
9_SuperFastBroker.py       IBKR execution
auxiliary/trigger_entry.py the one place a limit-entry trigger price is computed
auxiliary/bracket_config.py the one place the exit bracket is defined
launchers/                 the .bat files that start the broker (dry, live, paper)
Util.py                    shared library: can_buy(), PositionSizer, logging, calendars
trading_system.ps1         the orchestrator both scheduled tasks call
FeatureDiscovery/          the candidate gate (validate_feature.py) and the audit tools
8__IntradayFillSim.py      fill realism against 5 minute bars, under broker rules
backtest_diagnostics.py    trade-book autopsy, imported lazily by 5__
tools/attic.py             moves dead code to _ATTIC/, reversibly, with a manifest
client/                    packager for a thin execution client: the broker, the book and a
                           pre-trade guardrail, without the model, the data lake or the keys
6__TickerRelator.py        cross-asset correlation clustering (no caller, kept for reference)
```

Files 1 through 9 are the scheduled path. Everything else is a library the path imports, a screen that
annotates the pool, or an offline harness. Nothing outside 1 through 9 writes to the live signal files.

## Nightly run

`trading_system.ps1 -Mode evening` fires at 17:00 local. Stages run in order with a 20 second gap between
them for memory release, and a failed stage alerts without stopping the rest of the chain.

| Stage | Command |
| --- | --- |
| Ticker Downloader | `1__TickerDownloader.py --ImmediateDownload` |
| Price Downloader | `2__PriceDownloader.py --RefreshMode` |
| Market Data Refresh | `fetchers/refresh_market_data.py` |
| Data Panels | `build_data_panels.py all --refresh-all` |
| Feature Framework | `3__FeatureFramework.py --all --exclude vvg vaq_vg rvg_wl volume_spectral_splatter --workers 16` |
| Predictor | `4__Predictor.py --predict_only --input_dir Data/ProcessedData_v2 --model_dir Data/_ship_v2/model --neutralize --neut_mode spy200v2 --neut_dose_above 0.15 --neut_dose_below 0.30 --cm_k 0.5 --cm_spans 3,6,12` |
| Nightly BackTester | `5__NightlyBackTester.py --force` |

The four excluded feature blocks are visibility-graph and spectral families whose 29 columns the shipping
model drops at inference anyway. They cost about 16% of framework runtime for no effect on the model.
Worker count is pinned at 16 because the default of 32 exhausted memory partway through the universe.

`BT_SAMPLE_SEED=42` and `BT_SELRULE=low_atr` are exported before the backtester. Unseeded tie-breaks in the
entry ranking swing annual return by 20 to 40 percentage points on byte-identical predictions, which makes
any two nightly reports incomparable. Pinning both is not optional.

The Market Data Refresh stage was added on 2026-07-28 because nothing in the pipeline had ever refreshed
the market and macro lakes. The Data Panels stage only pulls the three SEC sources, so `Data/Indexes` had
been frozen since the last time a human ran the fetchers by hand. That took roughly 88 index-relative panel
columns to all-NaN and left the predictor's neutralization beta leg inert, with no error raised anywhere.
The refresh skips lakes that are already current and exits non-zero if a critical lake is still stale. Its
freshness predicate is per file, because an earlier version took the newest file in a directory as the
directory's age and skipped a lake whose siblings were six sessions behind. Census any time with
`python auxiliary/lake_freshness.py --strict`.

After the last stage the runner reads `Data/0__Signals.parquet` back off disk and checks that its
`TargetDate` is dated for the next session. If it is not, the run alerts loudly instead of finishing green.

## Morning run

`trading_system.ps1 -Mode morning` fires at 07:28 local, exits immediately on weekends, and exits with an
alert if a catch-up start lands after the 10:30 ET cutoff.

1. Pre-flight the TWS socket on port 7496, roughly 30 minutes before anything needs it.
2. Run `7__MacroFilter.py`, which self-waits to 09:35 ET internally.
3. Re-read `_Buy_Signals.parquet` and confirm its `TargetDate` is today. A book that is not fresh means no
   broker launch at all.
4. Hold until 09:57 ET, polling IBKR so a dead session surfaces before the entry window rather than during
   it.
5. Launch `9_SuperFastBroker.py --exp-stop-limit --exp-moc-exit`, retrying only on unambiguous connection
   failures.

A SPY abort is classified as a success and never retried. Re-running past that gate is the manual override
the intraday study prices at Sharpe -6.50.

Step 3 is worth a note. Since the 2026-08-28 changeover the broker arms off the wide pool and does not read
`_Buy_Signals.parquet` at all unless it is launched with `--no-trigger-entry`, so the funnel's narrowed
book is no longer traded. The stage stays in the morning chain because its book is still the freshness gate
that proves the nightly pipeline ran, and because its Stage 1 mechanical screens are the rollback path.

## Feature framework

`3__FeatureFramework.py` is a plugin host. Each file in `FeatureTemplates/` declares `METADATA` (name,
description, `requires`, `produces`, tags) and a `compute(df) -> df` function. The host imports every
non-underscore file, topologically sorts blocks by their declared dependencies, and runs the resulting
order across the universe in a process pool. Adding a feature means copying
`FeatureTemplates/__example_template.py` and writing one function. The new file is picked up on the next
run without touching any registry or import list.

The underscore prefix carves out the namespace. Double-underscore files are tooling and are never imported
as blocks. Single-underscore files are either shared helpers (`_marketcap`, `_indexes`, `_fundamentals`,
`_insider`) or candidate blocks that a production build skips. Locally that comes to 191 promoted blocks,
1,667 machine-generated candidates from the feature factory, 92 ports of published research, and 10
tooling files. The ten `_cand_` blocks published here are examples of the format, drawn from the ungated
pool.

Candidates are gated before promotion rather than after. `FeatureDiscovery/validate_feature.py` runs four
stages, cheapest and most decisive first: a static leakage scan plus a causality test that recomputes the
block on truncated price series and requires past values to be identical with or without future bars; a
pooled in-sample against out-of-sample IC durability check; a redundancy correlation against the existing
feature set; then a verdict of PASS, WEAK or FAIL. For machine-generated code a high IC is usually a leak
rather than alpha, which is why the causality test runs first.

The screen that follows the gate measures marginal lift in the top decile, which is the only region the
strategy trades. Global rank IC rewards features that sort the middle of the book, and the middle of the
book is never held. A candidate has to survive at least four seeds to move. Single-seed wins are noise, and
enough of them have been chased here to say that with confidence.

Non-price panels come from `build_data_panels.py`, which merges SEC XBRL companyfacts, SEC Form 4, and the
sector map. Every SEC fact carries both the date it describes and the date it became public. The merges key
on the filing date via a backward `merge_asof`, so a feature cannot see a filing before the market could.
Keying on the period end date instead is lookahead leakage that does not show up in out-of-sample IC.

## Predictor

`4__Predictor.py` is a single-file XGBoost pipeline: load, label, filter the universe, shuffle within each
date, split with a 5 day embargo, build the feature matrix, apply recency weights, tune with Optuna, train,
calibrate, and score.

The label is `topq`, a binary flag for the top 20% of next-day returns within each trading day. Attempts to
make the target cleverer have been tried and rolled back. Residual regression targets, ranking objectives,
and target ensembles all looked better in-sample and none survived a pinned as-of backtest. One of those
experiments turned out to have tomorrow's return in the feature matrix, which was invisible in
out-of-sample IC and obvious in the calibration AUC. Longer-horizon and payoff-shaped labels (`topq_5d`,
`thresh_5d`, the `book_*` family) are implemented and selectable, and remain research options rather than
the shipped configuration.

Nightly inference runs `--predict_only` against a pinned model directory (`Data/_ship_v2/model`) with
explicit `--drop_features_exact` and `--drop_feature_patterns` lists. The feature framework evolves faster
than the model is retrained, so live inference is pinned to the exact column set the model was trained on.
Drift between the two degrades predictions without raising an error.

### Prediction neutralization

Phase 13, enabled with `--neutralize`. For each day it projects the cross-section of raw scores onto beta
to SPY, ATR percentage, and log dollar volume, subtracts a fraction of that projection, then quantile-maps
the result back onto the original probability values for that day. The per-day marginal distribution is
preserved exactly, so only the within-day ordering changes and every downstream threshold keeps its
meaning.

The dose is regime-adaptive: 0.15 when SPY closes at or above its 200 day EMA, 0.30 below. It has zero
fitted parameters. Any gate failure (too few tickers scored, a factor panel that is stale on the signal
day, fewer than 5% of names changed) writes the raw predictions and exits non-zero, so the pipeline alerts
and the system trades the plain signal instead of a broken one. Every row keeps its un-neutralized value in
`raw_up_prob`, so rollback needs no second directory.

The phase logs a warning naming any factor with zero coverage rather than filling it and moving on. That
check exists because the beta leg was inert for weeks while the index lake sat stale, which was a
prediction problem before it was a neutralization problem.

### Conviction momentum

Phase 14, and the last thing that touches a probability before it is written. It runs on by default;
`--no_conviction_momentum` is the ablation switch. Per ticker, causally, on the final post-neutralization
value:

```text
up' = clip(up + k * mean_over_spans(up - EMA_span(up)), 0.30, 0.70)      k=0.5, spans 3/6/12
```

That is a first derivative of model conviction, which the flat level does not expose. It tilts toward names
whose conviction is rising and away from names where it is fading. It adds no new data and drops no
tickers. The reason it pays is mechanical: `can_buy` is a within-ticker spike detector, firing when a name
clears its own rolling 90th or 95th percentile of recent UpProbability, so amplifying a name's deviation
from its own trend makes rising-conviction moves clear that bar. The exact inverse, EMA smoothing of the
same series, lost 27 to 32 percentage points in full sim, so the sign is well identified.

Full sim on the main 252 day window took annual return from 59.4% to 98.7%, Sharpe from 1.48 to 2.40, and
max drawdown from 27.1% to 12.2%. The 8 seed baseline null spans 59.4% to 62.1%, so the tilt clears the
best baseline seed with no overlap on any metric, and it sits on a parameter plateau over k of 0.5 or 1.0
crossed with spans 3, 6 and 12.

Three things about it are still open. The three validation windows overlap, which is closer to two
independent periods than three. Pre-2023 slices are unreachable because `Data/RFpredictions` only goes back
to 2023-08-30. It is also regime dependent: it wins while the level signal is degraded (January to June
2026, baseline -7.9% against +58.7% tilted) and loses while the level signal is healthy (September to
December 2025, baseline +68.4% against +16.1% tilted). If the healthy regime returns, lower `--cm_k` or
turn the phase off. Rank IC falls as returns rise, from 0.18 to 0.11 to 0.06, so IC would have rejected
this and the full sim is the only metric that judges it.

The A/B path is `--cm_retilt`, which tilts predictions already on disk:

```powershell
python 4__Predictor.py --cm_retilt Data/RFpredictions --cm_k 1.0 --cm_retilt_out Data/_cm_k1
```

It reuses the same transform Phase 14 calls, so the two cannot drift, and it bases the tilt on
`pre_cm_up_prob` when that column is present, so pointing it at the live directory re-tilts from the
original values rather than compounding a second tilt. The source directory is never touched without
`--cm_retilt_apply`.

## Backtester

`5__NightlyBackTester.py` is the canonical source for both research and production. The chain that used to
generate it from a research fork is retired. One file now holds every experimental knob behind an
environment variable, with each production value set by `os.environ.setdefault` at the top, so a bare run
gets the live configuration and the A/B rig overrides it per arm. Shipping a change means flipping one
setdefault.

The eleven live defaults are the trade-history and signals paths, the tie-break seed, trigger K of 1.5,
oversubscription of 4, the Yang-Zhang volatility estimator, the cash-adjustment fix, production-only exits,
the gap-frequency pool gate (2 gaps above 4% in 20 bars), and a pool size of 36.

A hardcode pass on 2026-08-31 found that 101 of the file's 135 environment knobs were never set by
anything, so each only ever evaluated its shipped branch. Those are now literals and their dead branches
are gone, which removed about 500 lines along with several whole features that had been carried as
unreachable alternatives. Trade books were byte-identical across the change.

Two things about running it. A bare hand run writes the live pool, so use `run_optimal_backtest.py` for
sandboxed runs. And two copies must never run at once, because they contend on the same log directory and
the same live files.

The backtester cannot separate ranking rules. Seeded tie-break jitter alone spans a range far wider than
any ranker difference measured here, so single-run ranker comparisons are not interpretable. Selection
comparisons need paired seeds, and a minimum of four.

## From pool to orders

The nightly writes a 36 name pool to `Data/0__Signals.parquet` in UpProbability order. Two annotators run
on top of it, and neither drops rows:

- `signal_filter.py` writes `MechExclude`: price below $2, market cap below $952M, weekly volatility above
  5%. The broker's trigger path enforces this, because it bypasses `7__MacroFilter` and would otherwise
  have no hard exclusions. An RSI 30 to 40 exclusion is implemented and off by default, because on 2.53M
  scored rows that band is the most profitable one in the book.
- `rubric_veto.py` writes `RubricExclude`: the operator's morning judgement, marking at most 4 names. The
  ceiling is enforced in the broker rather than here, and a veto deeper than the ceiling is refused rather
  than trimmed, on the grounds that a rubric flagging 20 of 36 is malfunctioning and its first four flags
  are not trustworthy either. A verdict stamped for a previous session is reported as stale and reset.

Before the changeover the rubric picked the best three names and the broker traded those. Trigger entry
arms off the wide pool, so a rubric that writes a three name book is a rubric whose output goes nowhere.
The job is now to shave rather than to choose.

## Execution

`9_SuperFastBroker.py` connects to TWS through `ib_insync`, sizes three equal slots against account value
with a 10% cash buffer, and rounds lots to avoid odd lots. Three slots came from a cost calibration on 362
real fills: commission is a flat $1.00 floor on 98.9% of orders, so cost as a percentage of a trade is a
function of position size, and concentrating the same capital into fewer names cuts the drag by the same
factor it raises notional. The count must stay in sync with `Util.STRATEGY_PARAMS['max_positions']`.

Entry timing comes from an intraday study over 1,231 signal-day observations. Buying at the open gives
Sharpe 0.31; waiting until 10:00 ET gives 1.11. Conditioning on SPY at 10:00 splits that sharply, from
Sharpe +6.77 when SPY is up more than 0.5% to -6.50 when it is down more than 0.5%. So the broker holds
until 10:00 ET, aborts the whole day if SPY is at or below -0.5% from the open, and skips any name that
gapped more than 4% above its open.

Since 2026-08-28 the default entry is a dip limit rather than a marketable order.
`auxiliary/trigger_entry.py` computes one trigger price per name:

```text
depth   = clamp(K * beta**BETA_EXP * vol_20d, MIN_DEPTH, MAX_DEPTH)
trigger = prior_close * (1 - depth)
```

`vol_20d` is an equal-weighted 20 day close-to-close standard deviation taken from 20 closes and 19
returns, and beta is a rolling 60 day beta against SPY read `asof` the last close, defaulting to 1.0 when a
name is missing. Scaling depth to the name's own beta and volatility asks every name for a dip of
comparable rarity. A flat percentage limit almost never fills on a quiet name and fills constantly on a
noisy one, which turns it into a volatility screen. The broker arms up to 12 candidates so that about 3
fill, holds the table in memory, and sends one real order the moment a name reaches its trigger, so at most
`free_slots` orders exist at once. That virtual arming is not the arm that was backtested: the measured
version rests a real bid and earns the spread, and the difference is unmeasured.

What the trigger arm is worth: ten shuffle seeds put it at +0.7730% per trade against a shipped control of
+0.7360%, an edge of +0.0370pp at t 0.53, which is inside noise and must not be sized as real. What is not
inside noise is the risk profile. All ten runs landed max drawdown between 17.83% and 23.92% against the
control's 33.25%, with zero overlap, on 33.4% to 47.5% of capital deployed against 58.48%. Every one of
those numbers is gross of spread. Nothing in the backtester models the spread.

The exit bracket lives in `auxiliary/bracket_config.py`, which the broker, the backtester and the signals
file all read, so the backtest simulates what the broker sends. An audit in July 2026 found five
disagreeing bracket definitions in the pipeline, one of which was what the headline annual return had been
built on. The current shape:

- Hard stop 3.0%, target sized at 2:1 on the risk. Widening the stop while pinning the target shrinks the
  reward-to-risk ratio and made the strong half of the sample worse at every width tested, so both legs
  move together or neither does.
- Runner exits on, which means no take-profit leg at all. Instead a plain GTC stop is re-pegged each
  morning to prior close times (1 - 3.0%) and is never lowered. That is an end-of-day ratchet, not an
  exchange trail order, and it is the exit that most positions actually end on.
- At +15% unrealized, sell half and let the runner ride the repegged stop for up to 15 days. This replaced
  selling 80% at +10% on 2026-08-29.
- Age exit at 5 days for a normal position, 15 for a runner.
- Exchange trailing stops are off, and that is a measured decision rather than an omission. Every trailing
  setting tested lost, on all books. Ratcheting the stop up intraday cuts winners before they reach the
  target and removes the payoff asymmetry the strategy runs on.

The stop was 1.9% until the bracket consolidation. Before that it was 0.5%, which a name with 2% daily
volatility entered at 10:00 will touch before the close even on a good day, so the old setting exited most
positions on noise and the backtest win rate never described the system that was running. The broker prints
a loud banner whenever the bracket is not the live-validated 1.9/3.5 pair, because this module is shared
and widening it for research widens it for the next live run.

## Fail-safes

Most of these are dated because each one is a repair.

- Single-instance lock file. Concurrent runs corrupt shared parquet writes and produce phantom backtest
  results.
- Partial-bar guard. A catch-up start inside market hours would pull an in-progress daily bar into the
  feature build, so the evening mode exits instead.
- Free-memory precondition before the predictor. The stage waits for 28 GB and alerts rather than dying
  with an allocation error partway through the feature matrix.
- Book staleness abort in the broker, measured in trading days so weekends do not count. On 2026-06-26 it
  filled a four-day-old book. On 2026-07-07 the one-day warn-but-proceed path traded a stale book after the
  nightly pipeline had stopped running for four days, so the tolerance is now zero.
- Exit-code checks that do not trust exit codes. `2__PriceDownloader.py` swallows exceptions and exits 0,
  so the runner greps its transcript for "No tickers were successfully processed".
- Gate censuses. Several screens have failed open for weeks at a time: the micro-cap gate when FinViz
  started raising and a null cap passed `pd.notna`, the 52-week gate when an empty slice made it a no-op,
  the mechanical filter when its module was never committed and the caller skipped it behind a bare `if`.
  Gates now print evaluated, no-input and fired counts every run, so the next such failure shows up in one
  line instead of not at all.
- Alerts that are hard to miss. Console, log file, `BROKER_ALERT.txt`, three beeps, and a non-blocking
  popup, because during the July 2026 incident the first three went unnoticed for four consecutive days.

## Offline research code

`8__IntradayFillSim.py` replays the backtester's chosen trades against 5 minute bars under the live
broker's rules. A companion harness replays the same trades under the backtester's own rules, so running
both on one trade list prices the rule set at identical resolution on identical data, with neither side
contaminated by the daily-versus-intraday resolution gap. That comparison found the rule set, not the bar
resolution, is what separates the backtest from live.

`backtest_diagnostics.py` is the trade-book autopsy. `experimental/` holds the A/B rigs and is not tracked.

None of these are in the live path and none may write to the live signal files.

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
`auxiliary/Claud-API-KEY.txt.example` and `.fred_api_key.example`, both gitignored in their real form. The
Anthropic key file is optional: an `ANTHROPIC_API_KEY` environment variable or an OAuth profile also works.

Manual invocation:

```powershell
.\trading_system.ps1 -Mode evening   # rebuild data and signals after the close
.\trading_system.ps1 -Mode morning   # funnel and broker before the open
.\trading_system.ps1                 # picks a mode from the current hour
python run_optimal_backtest.py       # sandboxed backtest of the shipped configuration
```

## On the performance numbers

This repository does not publish a headline return figure, and numbers embedded in older docstrings should
be read as historical run records rather than claims. A previous 172% annualized and Sharpe 11.4 checkpoint
was later decomposed and attributed mostly to hyperparameter tuning, the training window, and the random
seed, and it did not reproduce out of sample. A separate audit found the backtester had been
double-counting every gain and loss since May 2026, which inflated returns by a factor between 2.4 and 2.9
and doubled the reported drawdown. That bug did not change selection, so paired comparisons made under it
still stand, but any single figure quoted before the fix is wrong.

What the code does instead is make results falsifiable: pinned as-of backtests so a validation cannot see
past its cutoff, a fixed tie-break seed, a minimum of four seeds before any comparison counts, a 5 day
embargo between training and calibration, point-in-time merges on SEC filing dates, gate censuses so a
screen cannot fail open unnoticed, and a habit of re-running the thing that looked good until it stops
looking good. A fair number of features and ideas in the history of this repo died that way.

## Author

[@JonIsHere242](https://github.com/JonIsHere242)

## License

The Old Secret Mission CIA Edition.
