"""
_paper_2605_15767_excess_volatility.py  --  Excess-volatility / variance-bound proxies.

Candidate block (underscore prefix = hidden from auto-discovery, UNPROVEN).

Source paper: "Market Makers and Risk Aversion: A Hamiltonian Approach to the
              Excess Volatility Puzzle", arXiv 2605.15767.

Concept (Shiller's excess-volatility puzzle):
  Realized price volatility is far larger than justified by fundamentals or a
  smooth long-run-trend view of value.  This block operationalises that idea
  in a PURE OHLCV, per-ticker, look-ahead-safe way via five families:

  1. Vol term-structure ratio (exv_vol_ratio_*)
       Short-window realized vol / long-window realized vol.
       >>1 = short-horizon "excess" overreaction; ~1 = steady regime.

  2. Variance-ratio overreaction score (exv_vr_*)
       k-day return variance vs k × 1-day return variance.
       Under a random walk this = 1; >1 = trending/excess vol;
       <1 = mean-reverting / under-reaction.
       Framed as the SIGNED DEVIATION from 1 (not the raw ratio) to
       separate the excess-volatility direction from Lo-MacKinlay VR tests.

  3. Price-vs-smooth-trend dispersion (exv_trend_disp_*)
       Detrended price std (rolling, look-ahead-free) / price EWMA.
       Measures how much prices "overshoot" a smooth trend anchor.

  4. Overshoot magnitude (exv_overshoot_*)
       Absolute z-score of price from its long EWMA — how far above/below
       the smooth trend the price currently sits.

  5. Mean-reversion speed proxy (exv_rev_speed_*)
       When price is far from trend, how strongly does it snap back?
       Estimated as the rolling correlation of (price-deviation at t) with
       (next price-deviation at t+1) on a PAST window (no lookahead).
       A large negative value = fast reversion (Shiller "correcting" phase).

All columns prefixed `exv_`.  Leading NaNs are expected as windows fill.
No look-ahead: every window uses only rows <= t.  No inf emitted.

NOTE: "per-ticker excess-volatility / variance-bound proxy — pure OHLCV,
no fundamentals, no external data."
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2605_15767_excess_volatility",
    "description": (
        "Per-ticker excess-volatility / variance-bound proxies: vol term-structure "
        "ratio, variance-ratio overreaction score, price-vs-smooth-trend dispersion, "
        "overshoot z-score, and mean-reversion-speed proxy — pure OHLCV, no external data."
    ),
    "requires": ["Close"],
    "produces": [
        # 1. Vol term-structure ratios (short/long realized vol)
        "exv_vol_ratio_5_63",    # 5-day rv / 63-day rv  (1-week vs 1-quarter)
        "exv_vol_ratio_10_126",  # 10-day rv / 126-day rv (2-week vs 6-month)
        "exv_vol_ratio_21_252",  # 21-day rv / 252-day rv (1-month vs 1-year)

        # 2. Variance-ratio overreaction score (deviation from random-walk baseline = 1)
        "exv_vr_dev_5",   # VR(5)  - 1  :  k=5,  signed deviation
        "exv_vr_dev_10",  # VR(10) - 1  :  k=10
        "exv_vr_dev_21",  # VR(21) - 1  :  k=21

        # 3. Price-vs-smooth-trend dispersion
        "exv_trend_disp_63",   # rolling 63-day detrended std / EWMA(63)
        "exv_trend_disp_126",  # rolling 126-day detrended std / EWMA(126)

        # 4. Overshoot: z-score of price from its long EWMA
        "exv_overshoot_63",   # (Close - EWMA_63) / rolling_std(63)
        "exv_overshoot_126",  # (Close - EWMA_126) / rolling_std(126)

        # 5. Mean-reversion-speed proxy
        "exv_rev_speed_63",   # rolling autocorr(lag=1) of deviation, 63-bar window
        "exv_rev_speed_126",  # rolling autocorr(lag=1) of deviation, 126-bar window
    ],
    "tags": ["volatility", "mean_reversion", "experimental"],
    "version": "1.0",
    "author": "paper arXiv:2605.15767 — excess volatility / Hamiltonian market-maker model",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series) -> pd.Series:
    """Log returns; NaN at first position."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(close / close.shift(1))
    # guard against inf that could arise from zero-price edge case
    return lr.replace([np.inf, -np.inf], np.nan)


def _realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Annualised realized vol over `window` days (no lookahead)."""
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(252)


def _variance_ratio_deviation(log_ret: pd.Series, k: int, base_window: int) -> pd.Series:
    """
    Variance-ratio overreaction score = VR(k) - 1.

    VR(k) = Var(k-day return) / (k * Var(1-day return))
          where both variances are computed over a rolling window of `base_window`
          days using PAST data only (no lookahead).

    Under a random walk VR = 1.  VR > 1 => trending / excess vol;
    VR < 1 => mean-reverting / under-reaction.

    We emit VR - 1 (signed deviation) so the signal is zero-centred and
    directional — not a plain VR clone.

    Vectorised implementation: rolling variance of 1-day and k-day returns,
    both computed with pandas .rolling().var() (causal, ddof=1).
    """
    # 1-day rolling variance over base_window bars
    var_1d = log_ret.rolling(base_window, min_periods=max(10, base_window // 2)).var()

    # k-day returns = non-overlapping would be ideal but overlapping rolling sum
    # is standard and consistent with Lo-MacKinlay (1988) convention.
    kday_ret = log_ret.rolling(k, min_periods=k).sum()
    var_kd = kday_ret.rolling(base_window, min_periods=max(5, base_window // 4)).var()

    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.where(var_1d > 0, var_kd / (k * var_1d), np.nan)

    vr_dev = pd.Series(vr, index=log_ret.index) - 1.0
    # Where either input was NaN the subtraction propagates NaN correctly;
    # replace any inf just in case of pathological zeros.
    return vr_dev.replace([np.inf, -np.inf], np.nan)


def _trend_dispersion(close: pd.Series, span: int) -> pd.Series:
    """
    Rolling detrended std / EWMA — how much the price "overshoots" its smooth trend.

    EWMA(span) is used as the smooth-trend anchor.
    Detrended std = rolling std of (Close - EWMA) over the same span.
    Ratio = detrended_std / EWMA  (dimensionless, scale-free).

    CAUSALITY: EWMA at t uses all past rows including t (pandas ewm default).
    Rolling std uses the same window.  Both are causal.
    """
    ewma = close.ewm(span=span, min_periods=span // 2, adjust=True).mean()
    deviation = close - ewma
    roll_std_dev = deviation.rolling(span, min_periods=span // 2).std()
    denom = ewma.abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, roll_std_dev / denom, np.nan)
    return pd.Series(ratio, index=close.index)


def _overshoot_z(close: pd.Series, span: int) -> pd.Series:
    """
    Z-score of Close from its EWMA: (Close - EWMA) / rolling_std(Close, span).

    Positive = price is above smooth trend (stretched up);
    Negative = price is below (stretched down).
    Large absolute values characterise the "excess" departure.
    """
    ewma = close.ewm(span=span, min_periods=span // 2, adjust=True).mean()
    roll_std = close.rolling(span, min_periods=span // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(roll_std > 0, (close - ewma) / roll_std, np.nan)
    return pd.Series(z, index=close.index)


def _rev_speed(close: pd.Series, span: int) -> pd.Series:
    """
    Rolling lag-1 autocorrelation of (Close - EWMA) over `span` bars.

    A large negative value means deviations strongly revert (fast mean-reversion).
    Near zero = random-walk deviations.
    Near +1 = deviations trend (excess persistence / momentum).

    Computed entirely from past data: at row i we use the deviation series
    for rows [i-span+1 .. i] (causal).

    Vectorised via pandas rolling().corr() between dev and dev.shift(1).
    This computes Pearson correlation over each rolling window — equivalent
    to the lag-1 autocorrelation of the deviation series.
    """
    ewma = close.ewm(span=span, min_periods=span // 2, adjust=True).mean()
    dev = close - ewma
    dev_lag1 = dev.shift(1)
    # rolling Pearson corr between dev[t] and dev[t-1] over `span` bars
    # min_periods guards early NaN rows
    ac = dev.rolling(span, min_periods=max(10, span // 4)).corr(dev_lag1)
    return ac.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute excess-volatility / variance-bound proxy columns.

    Per-ticker, ascending by Date, OHLCV columns present.
    Adds 12 columns prefixed `exv_`.  Never modifies existing columns.
    """
    close = df["Close"]
    lr = _log_returns(close)

    # ------------------------------------------------------------------
    # 1. Vol term-structure ratios
    # ------------------------------------------------------------------
    rv5   = _realized_vol(lr, 5)
    rv10  = _realized_vol(lr, 10)
    rv21  = _realized_vol(lr, 21)
    rv63  = _realized_vol(lr, 63)
    rv126 = _realized_vol(lr, 126)
    rv252 = _realized_vol(lr, 252)

    def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        return pd.Series(r, index=df.index)

    df["exv_vol_ratio_5_63"]   = _safe_ratio(rv5,  rv63)
    df["exv_vol_ratio_10_126"] = _safe_ratio(rv10, rv126)
    df["exv_vol_ratio_21_252"] = _safe_ratio(rv21, rv252)

    # ------------------------------------------------------------------
    # 2. Variance-ratio overreaction scores
    # ------------------------------------------------------------------
    df["exv_vr_dev_5"]  = _variance_ratio_deviation(lr, k=5,  base_window=63)
    df["exv_vr_dev_10"] = _variance_ratio_deviation(lr, k=10, base_window=126)
    df["exv_vr_dev_21"] = _variance_ratio_deviation(lr, k=21, base_window=252)

    # ------------------------------------------------------------------
    # 3. Price-vs-smooth-trend dispersion
    # ------------------------------------------------------------------
    df["exv_trend_disp_63"]  = _trend_dispersion(close, span=63)
    df["exv_trend_disp_126"] = _trend_dispersion(close, span=126)

    # ------------------------------------------------------------------
    # 4. Overshoot z-scores
    # ------------------------------------------------------------------
    df["exv_overshoot_63"]  = _overshoot_z(close, span=63)
    df["exv_overshoot_126"] = _overshoot_z(close, span=126)

    # ------------------------------------------------------------------
    # 5. Mean-reversion speed proxies
    # ------------------------------------------------------------------
    df["exv_rev_speed_63"]  = _rev_speed(close, span=63)
    df["exv_rev_speed_126"] = _rev_speed(close, span=126)

    return df
