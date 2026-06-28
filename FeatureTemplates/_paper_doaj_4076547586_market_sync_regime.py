"""
Market synchronization regime per-ticker proxy derived from:
  "Stock Market Synchronization and Stock Volatility: The Case of an Emerging Market"
  (doaj:4076547586e3498f9ffaa8cceab8e88d).

The paper computes cross-sectional Minimum Spanning Tree Length (MSTL) as a market-wide
synchronization measure and shows that higher synchronization predicts higher realized
volatility in the following month.  The cross-sectional MSTL cannot be computed per-ticker,
but the per-ticker analog is:

  1. sync_z_20_252: z-score of the current 20-day rolling corr vs SPY relative to its
     own 252-day history.  A high z-score means this ticker is unusually synchronised with
     the market right now (a regime shift signal).
  2. sync_z_5_63: shorter-term version (5-day corr, 63-day history) for detecting
     sudden synchronization spikes.
  3. sync_corr_rise_22: 22-day change in the 20-day rolling corr — capturing the
     direction and speed of synchronization movement.
  4. sync_hi_regime_63: fraction of last 63 days where 20-day corr exceeded its own
     63-day rolling 75th-pct — a regime-in-high-sync indicator.

The paper's mechanism: high synchronization → lower diversification → higher idiosyncratic
vol in the following period.  The z-score and regime features capture this directly.
All rolling, causal, no lookahead.
"""
import numpy as np
import pandas as pd

try:
    from _indexes import index_close as _index_close
    _SPY_CLOSE = _index_close("SPY")
except Exception:
    _SPY_CLOSE = pd.Series(dtype="float64")

METADATA = {
    "name":        "_paper_doaj_4076547586_market_sync_regime",
    "description": (
        "Per-ticker market synchronization regime: z-score of rolling corr vs SPY "
        "relative to its own history, synchronization momentum, and high-sync regime "
        "fraction; proxy for MSTL-based synchronization from DOAJ stock market sync paper."
    ),
    "requires":    ["Close", "Date"],
    "produces":    [
        "sync_z_20_252",      # z-score of 20d corr(ticker,SPY) vs 252d history
        "sync_z_5_63",        # z-score of 5d corr vs 63d history
        "sync_corr_rise_22",  # 22-day change in 20d rolling corr
        "sync_hi_regime_63",  # fraction of 63d in high-sync state
    ],
    "tags":        ["experimental", "market_regime", "synchronization", "volatility"],
    "version":     "1.0",
    "author":      "paper-mining slate 6",
}

_SHORT_WIN   = 20   # short rolling corr window
_TINY_WIN    = 5    # very short corr window
_HIST_LONG   = 252  # history for z-scoring
_HIST_SHORT  = 63   # shorter history
_REGIME_WIN  = 63   # window for high-sync regime fraction
_RISE_WIN    = 22   # window for corr change


def _rolling_corr_with_spy(log_ret: np.ndarray,
                            spy_ret: np.ndarray,
                            window: int,
                            min_p: int) -> np.ndarray:
    """Rolling Pearson correlation between two return series."""
    n = len(log_ret)
    ts = pd.Series(log_ret)
    ss = pd.Series(spy_ret)
    corr = ts.rolling(window, min_periods=min_p).corr(ss)
    return corr.values


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker market synchronization regime features."""
    dates = pd.to_datetime(df["Date"])
    close = df["Close"].values.astype(float)
    n     = len(close)

    # ── Log returns of ticker ─────────────────────────────────────────────────
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # ── SPY log returns aligned ───────────────────────────────────────────────
    if len(_SPY_CLOSE) > 0:
        spy_s  = _SPY_CLOSE.reindex(dates).ffill()
        spy_arr = spy_s.values.astype(float)
    else:
        spy_arr = np.full(n, np.nan)

    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = np.log(np.where(spy_arr[:-1] > 0, spy_arr[1:] / spy_arr[:-1], np.nan))

    # ── Rolling correlations ─────────────────────────────────────────────────
    corr20 = _rolling_corr_with_spy(log_ret, spy_ret, _SHORT_WIN, 10)
    corr5  = _rolling_corr_with_spy(log_ret, spy_ret, _TINY_WIN,  3)

    # Clip to [-1, 1]
    corr20 = np.clip(corr20, -1.0, 1.0)
    corr5  = np.clip(corr5, -1.0, 1.0)

    # ── Z-scores of correlation vs its own history ────────────────────────────
    c20_s = pd.Series(corr20)
    c5_s  = pd.Series(corr5)

    roll_mean_252 = c20_s.rolling(_HIST_LONG, min_periods=60).mean().values
    roll_std_252  = c20_s.rolling(_HIST_LONG, min_periods=60).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        z_20_252 = np.where(
            roll_std_252 > 1e-6,
            (corr20 - roll_mean_252) / roll_std_252,
            np.nan
        )
    z_20_252 = np.clip(z_20_252, -4.0, 4.0)

    roll_mean_63 = c5_s.rolling(_HIST_SHORT, min_periods=20).mean().values
    roll_std_63  = c5_s.rolling(_HIST_SHORT, min_periods=20).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        z_5_63 = np.where(
            roll_std_63 > 1e-6,
            (corr5 - roll_mean_63) / roll_std_63,
            np.nan
        )
    z_5_63 = np.clip(z_5_63, -4.0, 4.0)

    # ── 22-day change in 20d rolling corr ────────────────────────────────────
    corr20_series = pd.Series(corr20)
    corr_rise_22  = (corr20_series - corr20_series.shift(_RISE_WIN)).values

    # ── High-sync regime fraction ─────────────────────────────────────────────
    # Rolling 75th percentile of 20-day corr over 63d window
    roll_q75 = c20_s.rolling(_REGIME_WIN, min_periods=20).quantile(0.75).values
    # Is current corr above that threshold?
    in_hi_sync = (corr20 > roll_q75).astype(float)
    # Replace NaN-contaminated bars
    in_hi_sync = np.where(np.isfinite(corr20) & np.isfinite(roll_q75), in_hi_sync, np.nan)
    # Rolling fraction of days in high-sync state
    hi_regime_frac = pd.Series(in_hi_sync).rolling(_REGIME_WIN, min_periods=20).mean().values

    # ── Write outputs ─────────────────────────────────────────────────────────
    df["sync_z_20_252"]     = z_20_252
    df["sync_z_5_63"]       = z_5_63
    df["sync_corr_rise_22"] = corr_rise_22
    df["sync_hi_regime_63"] = hi_regime_frac

    return df
