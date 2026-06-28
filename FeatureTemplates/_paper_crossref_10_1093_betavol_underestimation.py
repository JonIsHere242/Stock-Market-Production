"""
Beta-volatility (rolling std of rolling beta) per-ticker proxy derived from:
  "Asset Prices When Investors Underestimate Discount Rate Dynamics"
  (crossref_journals:10.1093/rapstu/raag005).

The paper shows that analysts systematically underestimate the VOLATILITY of discount
rates (CAPM betas), and that high beta-volatility stocks are systematically mispriced.
Their key signal: stocks where rolling beta has been more volatile (higher std of beta
over a long lookback) earn lower subsequent CAPM alphas — an overconfidence anomaly.

Per-ticker OHLCV proxy:
  1. Compute rolling 40-day OLS beta of ticker vs SPY (using log returns).
  2. Compute the rolling 252-day std of that beta series (meta-rolling volatility-of-beta).
  3. Also compute the 126-day version and the ratio of current beta to its 252d mean
     (which captures whether beta is currently elevated vs its own history).

Distinct from `fli_vol_60/120` (which uses sub-window chunking within a single window)
— here we roll a 40-day beta first, then apply a long 252-day std on the resulting series.
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
    "name":        "_paper_crossref_10_1093_betavol_underestimation",
    "description": (
        "Meta-rolling volatility of beta: rolling std of rolling 40-day OLS beta vs SPY "
        "over 252-day and 126-day windows; captures discount-rate-volatility underestimation "
        "per Rapstu 2021."
    ),
    "requires":    ["Close", "Date"],
    "produces":    [
        "bvol_beta40_252d",      # 252-day rolling std of 40-day rolling beta
        "bvol_beta40_126d",      # 126-day rolling std of 40-day rolling beta
        "bvol_beta40_cur_vs_mean",  # current 40d beta / 252d mean of 40d beta
        "bvol_beta40_level",     # current 40-day rolling beta (level)
    ],
    "tags":        ["experimental", "beta", "market_regime", "risk"],
    "version":     "1.0",
    "author":      "paper-mining slate 6",
}

_BETA_WIN    = 40    # inner window for rolling beta
_META_LONG   = 252   # outer window for std of beta
_META_SHORT  = 126   # shorter outer window
_MIN_INNER   = 20    # min periods for inner beta estimation
_MIN_OUTER   = 60    # min periods for outer std


def _rolling_beta_vs_spy(ticker_log_ret: np.ndarray,
                         spy_log_ret: np.ndarray,
                         window: int,
                         min_periods: int) -> np.ndarray:
    """
    Rolling OLS beta of ticker on SPY using a sliding window.
    Returns array of same length as inputs (NaN for early bars).
    """
    n = len(ticker_log_ret)
    betas = np.full(n, np.nan)
    for i in range(window - 1, n):
        lo = max(0, i - window + 1)
        tx = ticker_log_ret[lo : i + 1]
        sx = spy_log_ret[lo : i + 1]
        mask = np.isfinite(tx) & np.isfinite(sx)
        if mask.sum() < min_periods:
            continue
        tm = tx[mask]; sm = sx[mask]
        sm_dm = sm - sm.mean()
        denom = np.dot(sm_dm, sm_dm)
        if denom < 1e-14:
            continue
        betas[i] = np.dot(sm_dm, tm - tm.mean()) / denom
    return betas


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute meta-rolling volatility-of-beta features."""
    # ── Align SPY to ticker dates ────────────────────────────────────────────
    dates = pd.to_datetime(df["Date"])
    close = df["Close"].values.astype(float)

    # Log returns of ticker
    log_ret_ticker = np.full(len(close), np.nan)
    log_ret_ticker[1:] = np.log(close[1:] / close[:-1])

    # SPY log returns aligned to ticker dates
    if len(_SPY_CLOSE) > 0:
        spy_ser = _SPY_CLOSE.reindex(dates).ffill()
        spy_arr = spy_ser.values.astype(float)
    else:
        spy_arr = np.full(len(close), np.nan)

    log_ret_spy = np.full(len(spy_arr), np.nan)
    log_ret_spy[1:] = np.log(spy_arr[1:] / np.where(spy_arr[:-1] > 0, spy_arr[:-1], np.nan))

    # ── Inner rolling beta (40-day window) ───────────────────────────────────
    beta40 = _rolling_beta_vs_spy(log_ret_ticker, log_ret_spy, _BETA_WIN, _MIN_INNER)

    # Clip to sane range
    beta40 = np.clip(beta40, -5.0, 5.0)

    # ── Outer meta-rolling std of beta ───────────────────────────────────────
    beta40_s = pd.Series(beta40)

    std_252 = beta40_s.rolling(_META_LONG, min_periods=_MIN_OUTER).std().values
    std_126 = beta40_s.rolling(_META_SHORT, min_periods=_MIN_OUTER // 2).std().values

    # Current beta vs its 252d rolling mean
    mean_252 = beta40_s.rolling(_META_LONG, min_periods=_MIN_OUTER).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_vs_mean = np.where(
            np.abs(mean_252) > 0.05,
            beta40 / mean_252,
            np.nan
        )
    beta_vs_mean = np.clip(beta_vs_mean, -5.0, 5.0)

    # ── Write outputs ─────────────────────────────────────────────────────────
    df["bvol_beta40_252d"]        = std_252
    df["bvol_beta40_126d"]        = std_126
    df["bvol_beta40_cur_vs_mean"] = beta_vs_mean
    df["bvol_beta40_level"]       = beta40

    return df
