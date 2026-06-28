"""
Regime stickiness and age features per-ticker proxy derived from:
  "Predicting Risk-adjusted Returns using an Asset Independent Regime-switching Model"
  (hal:3751910).

The paper constructs an HMM with "sticky features that directly affect the regime
stickiness and thereby changing turnover levels", distinguishing bull, bear, and
high-volatility periods over ~20 years of daily data.  The key signal: HOW LONG the
current regime has been active (regime age) and how the risk-adjusted performance
within that regime age cohort compares to the broader history.

Per-ticker OHLCV proxy: We define three causal states using lagged price/vol signals,
then at each bar:
  - Count consecutive days in the current state (regime age)
  - Compute the Sharpe within the current regime stint (risk-adj return of current run)
  - Estimate the hazard rate of leaving the current state (empirical, expanding window)
  - Compare current Sharpe vs the historic average Sharpe for same-state stints

Regime definition (fully causal at each bar t, using only data ≤ t-1):
  0 = bear   : rolling 20d SMA < rolling 60d SMA  AND  recent 10d return < 0
  1 = high-vol: realized 10d vol > 1.5× its own rolling 60d mean
  2 = bull   : otherwise (SMA uptrend, normal vol)

States are assigned from lagged (t-1) signals so the label at t is known before
trading at t.

Features emitted (prefix `rag_`):
  rag_regime_age         -- consecutive days in current regime state (raw count)
  rag_regime_age_norm    -- regime age / historic median age for this state (expanding)
  rag_stint_sharpe       -- Sharpe of log-returns within the current regime stint
  rag_stint_vs_hist      -- current stint Sharpe minus historic mean stint Sharpe (same state)
  rag_exit_hazard        -- empirical probability of leaving current state next day (expanding)
  rag_state              -- current regime state id (0=bear, 1=high-vol, 2=bull)

All rolling, causal, no lookahead.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name":        "_paper_hal_3751910_regime_stickiness_age",
    "description": (
        "Regime age and stickiness features: consecutive days in current bull/bear/high-vol "
        "state, stint Sharpe, vs-historic comparison, and empirical exit hazard; "
        "proxy for hal:3751910 sticky HMM regime-switching signal."
    ),
    "requires":    ["Close", "High", "Low"],
    "produces":    [
        "rag_regime_age",
        "rag_regime_age_norm",
        "rag_stint_sharpe",
        "rag_stint_vs_hist",
        "rag_exit_hazard",
        "rag_state",
    ],
    "tags":        ["experimental", "market_regime", "mean_reversion"],
    "version":     "1.0",
    "author":      "paper-mining slate 1",
}

_SMA_FAST    = 20
_SMA_SLOW    = 60
_VOL_SHORT   = 10
_VOL_LONG    = 60
_RET_SHORT   = 10
_WARMUP      = _SMA_SLOW + 1   # rows before any output is emitted
_MIN_STINTS  = 3                # need at least this many completed stints to compare


def _rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean; NaN during warmup."""
    out = np.full(len(arr), np.nan)
    cs = np.nancumsum(arr)
    for i in range(window - 1, len(arr)):
        slc = arr[i - window + 1: i + 1]
        valid = slc[np.isfinite(slc)]
        if len(valid) == window:
            out[i] = valid.mean()
    return out


def _rolling_std_np(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std (ddof=1); NaN during warmup."""
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        slc = arr[i - window + 1: i + 1]
        valid = slc[np.isfinite(slc)]
        if len(valid) >= 2:
            out[i] = valid.std(ddof=1)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].to_numpy(dtype=np.float64)
    high  = df["High"].to_numpy(dtype=np.float64)
    low   = df["Low"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Step 1: causal daily log-returns
    # ------------------------------------------------------------------ #
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # ------------------------------------------------------------------ #
    #  Step 2: regime signals (all lagged by 1 so state[t] known at t)
    # ------------------------------------------------------------------ #
    # SMA
    sma_fast = _rolling_mean_np(close, _SMA_FAST)
    sma_slow = _rolling_mean_np(close, _SMA_SLOW)

    # Rolling 10d realized vol (std of log-returns over 10 bars) — Parkinson proxy
    # Use GK-style: sqrt(0.5 * mean(log(H/L)^2))
    hl_sq = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12)) ** 2
    vol_short = np.full(n, np.nan)
    for i in range(_VOL_SHORT - 1, n):
        sl = hl_sq[i - _VOL_SHORT + 1: i + 1]
        if np.isfinite(sl).all():
            vol_short[i] = np.sqrt(0.5 * sl.mean() * 252)

    vol_long_mean = _rolling_mean_np(vol_short, _VOL_LONG)

    # 10d cumulative return (causal: sum of last 10 log-returns up to and including t-1)
    ret_short = np.full(n, np.nan)
    for i in range(_RET_SHORT, n):
        sl = log_ret[i - _RET_SHORT: i]        # [t-10 .. t-1], NOT including t
        valid = sl[np.isfinite(sl)]
        if len(valid) >= _RET_SHORT:
            ret_short[i] = valid.sum()

    # Lagged signals (shift by 1 so state at t uses info through t-1)
    def _lag1(arr):
        out = np.empty(n, dtype=np.float64)
        out[0] = np.nan
        out[1:] = arr[:-1]
        return out

    sma_fast_lag   = _lag1(sma_fast)
    sma_slow_lag   = _lag1(sma_slow)
    vol_short_lag  = _lag1(vol_short)
    vol_long_lag   = _lag1(vol_long_mean)
    ret_short_lag  = _lag1(ret_short)

    # Assign causal state array (int8 -1=unknown, 0=bear, 1=high-vol, 2=bull)
    state = np.full(n, -1, dtype=np.int8)
    for i in range(_WARMUP, n):
        sf = sma_fast_lag[i]
        ss = sma_slow_lag[i]
        vs = vol_short_lag[i]
        vl = vol_long_lag[i]
        rs = ret_short_lag[i]
        if not (np.isfinite(sf) and np.isfinite(ss)):
            continue
        # high-vol regime takes precedence
        if np.isfinite(vs) and np.isfinite(vl) and vl > 1e-12:
            if vs > 1.5 * vl:
                state[i] = 1
                continue
        # bear: SMA downtrend AND recent negative return
        if sf < ss and np.isfinite(rs) and rs < 0:
            state[i] = 0
        else:
            state[i] = 2   # bull

    # ------------------------------------------------------------------ #
    #  Step 3: walk forward to compute regime age and stint features
    # ------------------------------------------------------------------ #
    rag_age         = np.full(n, np.nan)
    rag_age_norm    = np.full(n, np.nan)
    rag_sharpe      = np.full(n, np.nan)
    rag_vs_hist     = np.full(n, np.nan)
    rag_exit_hazard = np.full(n, np.nan)
    rag_state_out   = np.full(n, np.nan)

    # Track current stint
    cur_state   = -1
    cur_start   = -1
    cur_rets    = []

    # Completed stints per state: list of (length, sharpe)
    completed   = {0: [], 1: [], 2: []}   # state → list of (age, sharpe)
    # Total transitions for exit-hazard
    state_days  = {0: 0, 1: 0, 2: 0}     # days spent in each state
    state_exits = {0: 0, 1: 0, 2: 0}     # number of exits

    for t in range(_WARMUP, n):
        s = int(state[t])
        r = log_ret[t]

        if s < 0:
            # Unknown state — reset streak
            if cur_state >= 0 and len(cur_rets) >= 2:
                # Close out stint
                _close_stint(cur_state, cur_rets, completed, state_exits, state_days)
            cur_state = -1
            cur_start = -1
            cur_rets  = []
            continue

        if s != cur_state:
            # Regime change — close previous stint
            if cur_state >= 0 and len(cur_rets) >= 2:
                _close_stint(cur_state, cur_rets, completed, state_exits, state_days)
            cur_state = s
            cur_start = t
            cur_rets  = []

        # Accumulate return for this bar into current stint
        if np.isfinite(r):
            cur_rets.append(r)

        # Update day count
        state_days[s] += 1

        # Regime age (days since stint started, 1-indexed)
        age = t - cur_start + 1
        rag_age[t] = float(age)
        rag_state_out[t] = float(s)

        # Normalized age: age / median historic age for this state
        hist_ages = [x[0] for x in completed[s]]
        if len(hist_ages) >= _MIN_STINTS:
            med_age = np.median(hist_ages)
            rag_age_norm[t] = age / med_age if med_age > 0 else np.nan

        # Stint Sharpe (returns accumulated so far in current stint)
        if len(cur_rets) >= 3:
            arr_r = np.array(cur_rets, dtype=np.float64)
            mu  = arr_r.mean()
            sig = arr_r.std(ddof=1)
            if sig > 1e-12:
                rag_sharpe[t] = mu / sig

        # Stint Sharpe vs historic mean Sharpe for same state
        hist_sharpes = [x[1] for x in completed[s] if np.isfinite(x[1])]
        if len(hist_sharpes) >= _MIN_STINTS and np.isfinite(rag_sharpe[t]):
            rag_vs_hist[t] = rag_sharpe[t] - float(np.mean(hist_sharpes))

        # Empirical exit hazard: exits / days in state → P(exit | in state)
        if state_days[s] > 5:
            rag_exit_hazard[t] = state_exits[s] / state_days[s]

    df["rag_regime_age"]      = rag_age
    df["rag_regime_age_norm"] = rag_age_norm
    df["rag_stint_sharpe"]    = rag_sharpe
    df["rag_stint_vs_hist"]   = rag_vs_hist
    df["rag_exit_hazard"]     = rag_exit_hazard
    df["rag_state"]           = rag_state_out

    return df


def _close_stint(state, rets, completed, state_exits, state_days):
    """Record a completed regime stint."""
    state_exits[state] += 1
    if len(rets) < 2:
        completed[state].append((len(rets), np.nan))
        return
    arr = np.array(rets, dtype=np.float64)
    mu  = arr.mean()
    sig = arr.std(ddof=1)
    sh  = mu / sig if sig > 1e-12 else np.nan
    completed[state].append((len(rets), sh))
