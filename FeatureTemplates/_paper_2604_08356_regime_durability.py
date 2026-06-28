"""
_paper_2604_08356_regime_durability.py  --  Per-ticker regime-durability / fragility proxy.

Inspired by: "Measuring Strategy-Decay Risk: Minimum Regime Performance and the
Durability of Systematic Investing" (arXiv 2604.08356).

The paper defines MINIMUM REGIME PERFORMANCE (MRP) — the worst risk-adjusted return
across distinct market regimes — as a measure of robustness vs fragility.  Here we
translate that concept into a CAUSAL, PER-TICKER, OHLCV-only block:

  Regimes are assigned using PAST data only:
    - Volatility tercile of trailing 60-day realized vol (low=0 / mid=1 / high=2).
    - Trend state of trailing 20-day SMA slope vs a 60-day SMA (up=0 / down=1).
  Combined: 6 cells (0-5), but often 2-3 are occupied at any given point.

  Within a 252-day trailing window we collect per-day returns and their causal
  regime labels, then compute the per-regime Sharpe (mean/std of returns).

  Features emitted (`rdr_` prefix):
    rdr_mrp             -- Minimum Regime Performance: lowest per-regime Sharpe in window.
    rdr_fragility       -- Spread of per-regime Sharpes (max − min); higher = more fragile.
    rdr_regime_std      -- Std-dev of per-regime Sharpes; another fragility measure.
    rdr_vs_worst_gap    -- Current-regime Sharpe minus the worst-regime Sharpe; 0 if current IS worst.
    rdr_consistency     -- Fraction of occupied regimes where mean return > 0 (0..1).
    rdr_current_regime  -- Current regime id (0-5); NaN until warmup complete.
    rdr_n_regimes_seen  -- Number of distinct regimes observed in the trailing window.

NOTE: This is an UNPROVEN candidate block (leading underscore = hidden from
auto-discovery).  IC / model-contribution validation required before promotion.
"""

import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "_paper_2604_08356_regime_durability",
    "description": (
        "Per-ticker regime-durability proxy (MRP, fragility, consistency) derived from "
        "causal vol-tercile + trend-state regimes; inspired by arXiv 2604.08356 MRP concept."
    ),
    "requires":    ["Close"],
    "produces":    [
        "rdr_mrp",
        "rdr_fragility",
        "rdr_regime_std",
        "rdr_vs_worst_gap",
        "rdr_consistency",
        "rdr_current_regime",
        "rdr_n_regimes_seen",
    ],
    "tags":        ["market_regime", "volatility", "trend", "experimental"],
    "version":     "1.0",
    "author":      "paper arXiv 2604.08356 — per-ticker OHLCV-only MRP proxy",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VOL_WINDOW   = 20    # trailing window for realized vol (daily std of log-returns)
_SMA_FAST     = 20    # short SMA for trend state
_SMA_SLOW     = 60    # long SMA for trend state
_REGIME_WIN   = 252   # trailing window for per-regime Sharpe computation
_MIN_PERIODS  = 60    # minimum rows before emitting any value
_MIN_OBS_REG  = 5     # minimum observations in a regime to trust its Sharpe


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker regime-durability features.

    All regime labels are assigned using ONLY past (causal) information so that
    the label at row t depends solely on rows < t.  The rolling Sharpe aggregation
    then collects (label, return) pairs over a 252-day trailing window — again,
    using only data available at or before t.

    Leading NaNs (before _MIN_PERIODS rows) are expected and left as-is.
    """
    n = len(df)

    # Pre-allocate output arrays (all NaN by default)
    mrp           = np.full(n, np.nan)
    fragility     = np.full(n, np.nan)
    regime_std    = np.full(n, np.nan)
    vs_worst_gap  = np.full(n, np.nan)
    consistency   = np.full(n, np.nan)
    current_reg   = np.full(n, np.nan)
    n_regimes     = np.full(n, np.nan)

    if n < _MIN_PERIODS:
        _assign_outputs(df, mrp, fragility, regime_std, vs_worst_gap,
                        consistency, current_reg, n_regimes)
        return df

    close = df["Close"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1 — Compute causal log-returns (return at t = log(C_t / C_{t-1}))
    # ------------------------------------------------------------------
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # ------------------------------------------------------------------
    # Step 2 — Causal volatility regime (vol tercile label 0/1/2)
    # Using a fixed trailing _VOL_WINDOW of daily std of log-returns.
    # Tercile thresholds are computed from the PAST _REGIME_WIN days,
    # shifted by 1 so today's value is excluded from the labelling.
    # ------------------------------------------------------------------
    # Rolling realized vol (shift-1 so label is fully causal)
    vol_series = _rolling_std(log_ret, _VOL_WINDOW)   # shape (n,), NaN in warmup
    # Shift vol by 1 — the vol we "see" on day t is computed through day t-1
    vol_lagged = np.empty(n)
    vol_lagged[0] = np.nan
    vol_lagged[1:] = vol_series[:-1]

    # ------------------------------------------------------------------
    # Step 3 — Causal trend regime (0=up, 1=down)
    # SMA slope: sma_fast > sma_slow → up (0), else down (1).
    # Both SMAs use shift-1 so that t's label only uses prices t-1 and earlier.
    # ------------------------------------------------------------------
    sma_fast_series = _rolling_mean(close, _SMA_FAST)
    sma_slow_series = _rolling_mean(close, _SMA_SLOW)
    # Shift by 1 to make causal
    sma_fast_lag = np.empty(n)
    sma_fast_lag[0] = np.nan
    sma_fast_lag[1:] = sma_fast_series[:-1]

    sma_slow_lag = np.empty(n)
    sma_slow_lag[0] = np.nan
    sma_slow_lag[1:] = sma_slow_series[:-1]

    # ------------------------------------------------------------------
    # Step 4 — Assign combined regime id at each row using lagged signals.
    # vol tercile label (0/1/2) × trend label (0/1) → regime 0-5
    # All NaN until both vol and sma signals are available.
    # ------------------------------------------------------------------
    # We do this row-by-row only for the window aggregation, so we build arrays
    # of per-row vol-tercile breakpoints from the PAST _REGIME_WIN days.
    # To keep it vectorized and fast we compute rolling 33rd / 67th pctile of
    # vol_lagged over the _REGIME_WIN window (shift-1 baked in already).

    vol_p33 = _rolling_percentile(vol_lagged, _REGIME_WIN, 33.33)
    vol_p67 = _rolling_percentile(vol_lagged, _REGIME_WIN, 66.67)

    # Vol tercile for each day (causal: uses lagged vol vs lagged thresholds)
    # 0=low, 1=mid, 2=high
    vol_tercile = np.where(
        np.isnan(vol_lagged) | np.isnan(vol_p33),
        np.nan,
        np.where(
            vol_lagged <= vol_p33, 0.0,
            np.where(vol_lagged <= vol_p67, 1.0, 2.0)
        )
    )

    # Trend label (causal): 0=up, 1=down
    trend_label = np.where(
        np.isnan(sma_fast_lag) | np.isnan(sma_slow_lag),
        np.nan,
        np.where(sma_fast_lag >= sma_slow_lag, 0.0, 1.0)
    )

    # Combined regime id: vol_tercile * 2 + trend_label → 0..5
    regime_id = np.where(
        np.isnan(vol_tercile) | np.isnan(trend_label),
        np.nan,
        vol_tercile * 2.0 + trend_label
    )

    # ------------------------------------------------------------------
    # Step 5 — Rolling per-regime Sharpe over trailing _REGIME_WIN window.
    # At each row t we look back at rows [t - _REGIME_WIN, t-1] (fully causal,
    # exclusive of row t itself so we shift returns by 1 as well).
    # We collect (regime_id[s], log_ret[s]) for those s, group by regime,
    # compute mean/std, then derive MRP / fragility / consistency.
    #
    # Python loop per row is O(n * _REGIME_WIN) which at n=700, win=252 is
    # 176 400 iterations — very fast in NumPy slice terms.
    # ------------------------------------------------------------------
    # Use shifted return: the return we "know" at t is log_ret[t-1] (yesterday's).
    # Regime label at t is already causal (vol and sma lagged).
    # For the window at t we look at indices [max(0, t-win) .. t-1] inclusive.

    ret_arr = log_ret          # shape (n,)
    reg_arr = regime_id        # shape (n,), causal label at each day

    for t in range(_MIN_PERIODS, n):
        start = max(0, t - _REGIME_WIN)
        end   = t  # exclusive; indices [start, end)

        ret_win = ret_arr[start:end]
        reg_win = reg_arr[start:end]

        # Drop rows where either is NaN
        valid_mask = np.isfinite(ret_win) & np.isfinite(reg_win)
        if valid_mask.sum() < _MIN_OBS_REG * 2:
            continue

        ret_valid = ret_win[valid_mask]
        reg_valid = reg_win[valid_mask].astype(np.int8)

        unique_regs = np.unique(reg_valid)
        n_seen = len(unique_regs)
        if n_seen < 2:
            # Need at least 2 regimes to compute dispersion; still emit what we can
            current_reg[t] = regime_id[t] if np.isfinite(regime_id[t]) else np.nan
            n_regimes[t]   = float(n_seen)
            continue

        # Per-regime Sharpe: mean / std (annualised sqrt252 cancels in ratio)
        sharpes: List[float] = []
        for r in unique_regs:
            mask_r = reg_valid == r
            rets_r = ret_valid[mask_r]
            if len(rets_r) < _MIN_OBS_REG:
                continue
            mu  = rets_r.mean()
            sig = rets_r.std(ddof=1)
            if sig > 1e-12:
                sharpes.append(mu / sig)
            # If sig == 0 (all returns identical), skip this regime — degenerate

        if len(sharpes) < 2:
            current_reg[t] = regime_id[t] if np.isfinite(regime_id[t]) else np.nan
            n_regimes[t]   = float(n_seen)
            continue

        sh_arr = np.array(sharpes)
        sh_min = sh_arr.min()
        sh_max = sh_arr.max()

        mrp[t]         = sh_min
        fragility[t]   = sh_max - sh_min
        regime_std[t]  = sh_arr.std(ddof=1) if len(sh_arr) > 1 else 0.0
        current_reg[t] = regime_id[t] if np.isfinite(regime_id[t]) else np.nan
        n_regimes[t]   = float(n_seen)

        # Current regime Sharpe vs worst
        cur_r = regime_id[t]
        if np.isfinite(cur_r):
            cur_int = int(cur_r)
            if cur_int in unique_regs:
                mask_cur = reg_valid == cur_int
                rets_cur = ret_valid[mask_cur]
                if len(rets_cur) >= _MIN_OBS_REG:
                    mu_cur  = rets_cur.mean()
                    sig_cur = rets_cur.std(ddof=1)
                    sh_cur  = mu_cur / sig_cur if sig_cur > 1e-12 else np.nan
                    if np.isfinite(sh_cur):
                        vs_worst_gap[t] = max(0.0, sh_cur - sh_min)

        # Consistency: fraction of occupied regimes with positive mean return
        pos_count = 0
        total_valid_regs = 0
        for r in unique_regs:
            mask_r = reg_valid == r
            rets_r = ret_valid[mask_r]
            if len(rets_r) < _MIN_OBS_REG:
                continue
            total_valid_regs += 1
            if rets_r.mean() > 0:
                pos_count += 1
        if total_valid_regs > 0:
            consistency[t] = float(pos_count) / float(total_valid_regs)

    _assign_outputs(df, mrp, fragility, regime_std, vs_worst_gap,
                    consistency, current_reg, n_regimes)
    return df


# ---------------------------------------------------------------------------
# Helper: assign output columns
# ---------------------------------------------------------------------------
def _assign_outputs(df, mrp, fragility, regime_std, vs_worst_gap,
                    consistency, current_reg, n_regimes):
    idx = df.index
    df["rdr_mrp"]            = pd.array(mrp,          dtype="Float64")[:]
    df["rdr_fragility"]      = pd.array(fragility,     dtype="Float64")[:]
    df["rdr_regime_std"]     = pd.array(regime_std,    dtype="Float64")[:]
    df["rdr_vs_worst_gap"]   = pd.array(vs_worst_gap,  dtype="Float64")[:]
    df["rdr_consistency"]    = pd.array(consistency,   dtype="Float64")[:]
    df["rdr_current_regime"] = pd.array(current_reg,   dtype="Float64")[:]
    df["rdr_n_regimes_seen"] = pd.array(n_regimes,     dtype="Float64")[:]

    # Re-align to df.index (pd.array above is positional; re-assign with index)
    for col in METADATA["produces"]:
        df[col] = pd.Series(df[col].to_numpy(), index=idx)


# ---------------------------------------------------------------------------
# Rolling helpers (fully causal, NaN in warmup)
# ---------------------------------------------------------------------------
def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing std of `arr` with `window` bars (ddof=1). NaN for warmup rows."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        sl = arr[i - window + 1: i + 1]
        valid = sl[np.isfinite(sl)]
        if len(valid) >= max(2, window // 2):
            result[i] = valid.std(ddof=1)
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of `arr` with `window` bars. NaN for warmup rows."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        sl = arr[i - window + 1: i + 1]
        valid = sl[np.isfinite(sl)]
        if len(valid) >= window:
            result[i] = valid.mean()
    return result


def _rolling_percentile(arr: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Trailing percentile of `arr` with `window` bars. NaN for warmup rows."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        sl = arr[i - window + 1: i + 1]
        valid = sl[np.isfinite(sl)]
        if len(valid) >= window // 2:
            result[i] = np.percentile(valid, pct)
    return result
