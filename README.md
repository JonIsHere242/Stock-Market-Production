# Stock-Market-Production

A daily-horizon US equity trading system. It scores about 4,200 tickers every night, picks a candidate
pool, and rests limit orders on that pool through Interactive Brokers the next morning. It runs unattended
on Windows under Task Scheduler and trades a live account.

`trading_system.ps1` is the orchestrator. Both scheduled tasks call it.

## The run

**Evening, 17:00 local.** `trading_system.ps1 -Mode evening`

| # | Stage | What it does |
| --- | --- | --- |
| 1 | `1__TickerDownloader.py` | Universe from the SEC company tickers file |
| 2 | `2__PriceDownloader.py` | Daily OHLCV from IBKR, then `fetchers/refresh_market_data.py` for the macro and index lakes and `build_data_panels.py` for SEC fundamentals, Form 4 insider and the sector map |
| 3 | `3__FeatureFramework.py` | Runs every feature block over the universe, writes the panel |
| 4 | `4__Predictor.py` | XGBoost inference against a pinned model, writes an UpProbability per ticker |
| 5 | `5__NightlyBackTester.py` | Backtests the live configuration, writes the 36 name pool to `Data/0__Signals.parquet` |

**Overnight.** Nothing runs. The pool is dated for the next session, and the runner alerts if it is not.

**Morning, 07:28 local.** `trading_system.ps1 -Mode morning`

| # | Stage | What it does |
| --- | --- | --- |
| 7 | `7__MacroFilter.py` | Waits to 09:35 ET, screens the pool, writes a narrowed book. The broker no longer trades that book, so this is mainly the gate that proves the nightly ran |
| 9 | `9_SuperFastBroker.py` | Waits to 10:00 ET, checks SPY, rests dip limits, manages the exits |

Stages 6 and 8 are not scheduled. `6__TickerRelator.py` is correlation clustering kept for reference, and
`8__IntradayFillSim.py` replays trades against 5 minute bars to check that backtest fills are reachable.

## Stage 9 in detail

This is where the money moves, so it is worth spelling out.

1. Hold until 10:00 ET. An intraday study over 1,231 signal days puts buying at the open at Sharpe 0.31 and
   waiting until 10:00 at 1.11.
2. Check SPY. If it is at or below -0.5% from the open, abort the whole day. That same study splits at
   10:00 from Sharpe +6.77 when SPY is up over 0.5% to -6.50 when it is down over 0.5%.
3. Skip any name that gapped more than 4% above its open.
4. Rest a dip limit on each survivor at `prior_close * (1 - K * beta * vol_20d)`, so every name is asked for
   a dip of comparable rarity rather than a flat percentage. 12 candidates are armed so that about 3 fill.
5. Size three equal slots against account value, keeping a 10% cash buffer.
6. Exits, all defined in `auxiliary/bracket_config.py`: a 3.0% hard stop, no take-profit leg, a stop
   re-pegged each morning to prior close minus 3% and never lowered, sell half at +15%, and an age exit at
   5 days, or 15 for a runner.

## Design notes

- The backtester's tie-break seed is pinned. Unseeded runs swing annual return by 20 to 40 points on
  byte-identical predictions, which makes any two nightly reports incomparable.
- Inference is pinned to the model's exact column set. The feature framework changes faster than the model
  is retrained, and drift between them degrades predictions without raising an error.
- SEC merges key on the filing date, not the period end date, so a feature cannot see a filing before the
  market could.
- Every gate prints a census. Several screens have failed open for weeks at a time: a market-cap gate when
  a null cap passed `pd.notna`, a 52-week gate when an empty slice made it a no-op, a mechanical filter
  whose module was never committed. Evaluated, no-input and fired counts now surface it in one line.
- A stale book means no orders at all. The tolerance is zero trading days, because a one-day
  warn-and-proceed path traded a four-day-old book once.

## Feature framework

`3__FeatureFramework.py` is a plugin host. Each file in `FeatureTemplates/` declares a `METADATA` dict and
a `compute(df) -> df` function. The host imports every block, sorts them by their declared dependencies,
and runs the result across the universe in a process pool. Adding a feature means copying
`FeatureTemplates/__example_template.py` and writing one function.

Candidates are gated before promotion. `FeatureDiscovery/validate_feature.py` scans for leakage, recomputes
each block on truncated price series to prove it is causal, checks that out-of-sample signal survives, and
correlates against the existing set for redundancy. For machine-generated code a high IC is usually a leak
rather than alpha, which is why the causality test runs first.

## What is not in this repository

The `Data/` tree, API keys, the A/B rigs, the diagnostics suite, and the feature library. The promoted
blocks are the model's edge, so what ships is the block contract plus ten candidate blocks as worked
examples of the format.

## Setup

Windows, Python 3.13, and a running TWS or IB Gateway on port 7496.

```powershell
.\setup.ps1                          # venv, dependencies, Data/ tree, key templates
.\trading_system.ps1 -Mode evening   # rebuild data and signals after the close
.\trading_system.ps1 -Mode morning   # screen and trade before the open
python run_optimal_backtest.py       # sandboxed backtest, does not touch the live pool
```

`requirements.txt` pins what the code imports. torch and backtrader are installed separately by `setup.ps1`
because they are platform or source specific.

## On the performance numbers

This repository publishes no headline return figure. An earlier 172% annualized checkpoint was later
decomposed and attributed mostly to hyperparameter tuning, the training window, and the random seed, and it
did not reproduce out of sample. A separate audit found the backtester had been double-counting every gain
and loss for months.

What the code does instead is make results falsifiable: pinned as-of backtests, a fixed tie-break seed, a
minimum of four seeds before any comparison counts, a 5 day embargo between training and calibration,
point-in-time merges, and a habit of re-running the thing that looked good until it stops looking good. A
fair number of ideas in the history of this repo died that way.

## Author

[@JonIsHere242](https://github.com/JonIsHere242)

## License

The Old Secret Mission CIA Edition.
