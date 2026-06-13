# Stock Market Trading System

XGBoost-based pipeline that predicts next-day upward stock moves and executes trades through Interactive Brokers. The full cycle runs nightly and each morning via a PowerShell orchestrator.

## Pipeline

### 1. Ticker Downloader
Pulls the current tradeable universe from the SEC. Runs nightly to keep the ticker list current and prune delisted names.

### 2. Price Downloader
Fetches daily OHLCV bars for the full universe through IBKR. Stores per-ticker parquet files in `Data/PriceData/`.

### 3. Alpha Sensitivity
The feature engineering stage. Runs in parallel across ~4000 tickers and produces roughly 300 features per ticker per day. The more interesting parts:

**Visibility graph features.** Each rolling price window is converted into a Natural Visibility Graph: each bar is a node, and two bars share an edge if no intermediate bar blocks the line of sight between them. The stage builds two graphs per window, one on the raw detrended log-price series and one on its inverse, then extracts the average shortest path length from each. The difference between the two paths encodes directional momentum structure in a way that standard indicators cannot.

**Matrix power features.** 14 `mp_*` features are derived by building a DxD matrix M from sign-weighted, panel-normalized OHLCV primitives over a rolling window, computing the matrix exponential `expm(M)` via scaling-and-squaring with Taylor expansion, and extracting a scalar summary (trace, Frobenius norm, or top-left element). The geometry of the matrix exponential captures interaction effects between price, volume, volatility, and higher moments that are invisible to single-variable indicators. The 14 specs are stored in `Data/matrix_power_spec.json` and were validated against held-out IC targets.

**GP cross-sectional features.** After the parallel per-ticker pass completes, a second pass runs across the full universe panel grouped by date. This applies genetic-programming-discovered formulas that combine cross-sectionally ranked inputs: rolling Higuchi fractal dimension, Lyapunov exponent estimates, DFA scaling, spectral entropy, time-between-extremes distributions, and volume entropy. The formulas were evolved to discriminate the top percentile of next-day movers rather than maximize global rank IC.

**VIX-adaptive dynamic RSI.** RSI window length is driven by the current VIX regime. Rather than recomputing RSI from scratch at each row, the stage precomputes a full 26-column RSI matrix (windows 5-30) and selects the appropriate column per row via numpy indexing.

Standard indicators also included: rolling Hurst exponent, ATR percentile ranks, DVAMR probability, beta to major indexes, volume spectral analysis, and a suite of genetic autocorrelation features.

### 4. Predictor
Trains and runs an XGBoost binary classifier on the processed features. Outputs a calibrated 0-1 probability per ticker per day to `Data/RFpredictions/`. Run with `--predict_only` nightly to skip retraining.

### 5. Nightly Backtester
Scores the current signals against historical data and writes a candidate pool to `Data/0__signals.parquet`. Runs after each nightly predict cycle.

### 6. Ticker Relator
Computes rolling cross-asset correlations across the universe. Run on weekends. Output feeds the macro filter and position sizing.

### 7. Macro Filter
Narrows the candidate pool to a final book using a cost-ordered funnel: mechanical filters first, then macro regime checks, then an LLM pass on the remainder. Writes `_Buy_Signals.parquet`. Run each morning before the broker.

### 8. Broker
`9_SuperFastBroker.py` connects to TWS, reads the filtered signal book, and places IBKR Adaptive limit orders. Includes a SPY-based market conditions gate that skips trading on statistically unfavorable days.

## Orchestration

`trading_system.ps1` runs everything. It auto-detects the time of day or accepts a `-Mode` flag.

```powershell
.\trading_system.ps1 -Mode evening   # runs stages 1-5 after market close
.\trading_system.ps1 -Mode morning   # runs macro filter + broker before open
.\trading_system.ps1                 # auto-selects based on current hour
```

Evening mode runs stages 1 through 5 sequentially with 20-second gaps for memory cleanup. Morning mode runs the macro filter funnel, waits until the configured launch time, then hands off to the broker with connection-failure retry logic and a hard cutoff.

A file lock prevents concurrent runs from corrupting shared parquet writes.

## Experimental

Scripts in `experimental/` are not part of the live pipeline. They include alternative downloaders, intraday tools, and predictor variants under development.

## Author

[@JonIsHere242](https://github.com/JonIsHere242)

## License

The Old Secret Mission CIA Edition.
