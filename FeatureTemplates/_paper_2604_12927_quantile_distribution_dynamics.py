"""
_paper_2604_12927_quantile_distribution_dynamics.py

Per-ticker proxy for quantile distribution dynamics, inspired by:
  "Forecasting Oil Prices Across the Distribution: A Quantile VAR Approach"
  arXiv 2604.12927

The paper models the CONDITIONAL DISTRIBUTION of returns across quantiles
rather than just the mean, finding that predictor effects differ strongly
across quantiles and that uncertainty/financial-condition variables especially
predict DOWNSIDE (left-tail) risk.

This block captures the SHAPE, ASYMMETRY, and DYNAMICS of the rolling
single-stock return distribution — distinct from a plain volatility block.
The focus is on quantile geometry: tail width, skew direction, downside
risk level and trend, where today's return sits in its own history, whether
the distribution is widening, and relative tail thickness.

NOTE: This is an UNPROVEN candidate block (leading underscore). All features
are honest per-ticker rolling quantile-distribution dynamics proxies; no
cross-sectional or look-ahead information is used.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_2604_12927_quantile_distribution_dynamics",
    "description": (
        "Rolling quantile-distribution shape features for per-ticker returns: "
        "IQR width, wide-band width, Bowley skewness proxy, downside VaR level "
        "and trend, rolling return-rank, width trend, and tail-thickness ratio "
        "(upside vs downside), at 60-day and 120-day windows."
    ),
    "requires":    ["Close"],
    "produces":    [
        # 60-day window
        "qdd_iqr_60",           # IQR (q75 - q25) of returns — distributional width
        "qdd_wide_60",          # Wide band (q90 - q10) of returns
        "qdd_bowley_60",        # Bowley skewness proxy: (q90-q50)-(q50-q10) / (q90-q10)
        "qdd_var5_60",          # 5th-percentile return (downside VaR proxy)
        "qdd_var5_chg_60",      # Change in downside VaR (5 bars): is tail risk intensifying?
        "qdd_ret_rank_60",      # Rolling percentile rank of today's return in its 60-day window
        "qdd_width_trend_60",   # Trend of wide-band width (today vs 20-bar mean of width)
        "qdd_tail_ratio_60",    # Tail-thickness ratio: (q90-q50) / (q50-q10); >1 = fatter upside
        # 120-day window
        "qdd_iqr_120",
        "qdd_wide_120",
        "qdd_bowley_120",
        "qdd_var5_120",
        "qdd_var5_chg_120",
        "qdd_ret_rank_120",
        "qdd_width_trend_120",
        "qdd_tail_ratio_120",
    ],
    "tags":        ["volatility", "distribution", "tail_risk", "experimental"],
    "version":     "1.0",
    "author":      "paper arXiv 2604.12927 — quantile VAR distribution dynamics proxy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Trailing-window quantile ending at t (inclusive), min_periods=window."""
    return series.rolling(window, min_periods=window).quantile(q)


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """
    For each row t, fraction of the preceding `window` values (t-window+1 .. t)
    that are STRICTLY LESS THAN series[t]. Returns [0, 1).
    Uses .apply(raw=True) on a rolling window — causal (includes row t itself,
    which mirrors what any real VaR/rank metric does: rank today in rolling history
    that includes today, consistent with rolling.quantile behaviour).
    """
    def _rank(arr: np.ndarray) -> float:
        v = arr[-1]
        # fraction of past window values strictly below current value
        return float(np.sum(arr[:-1] < v) / max(len(arr) - 1, 1))

    return series.rolling(window, min_periods=window).apply(_rank, raw=True)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quantile-distribution dynamics features for one ticker.

    All features are derived from daily log-returns computed from Close prices.
    Two windows are used: 60 and 120 trading days.
    The first (window - 1) rows will be NaN for each feature — expected.
    """

    # ---- 1. Log returns (NaN at row 0) -------------------------------------
    # Use log returns for better distributional properties (additive, symmetric).
    ret = np.log(df["Close"] / df["Close"].shift(1))  # pd.Series, NaN at index 0

    # ---- 2. Per-window features --------------------------------------------
    for w in (60, 120):
        p = f"_{w}"  # suffix shorthand

        # -- Trailing quantiles of returns (causal: window ends at t inclusive) --
        q10 = _rolling_quantile(ret, w, 0.10)
        q25 = _rolling_quantile(ret, w, 0.25)
        q50 = _rolling_quantile(ret, w, 0.50)
        q75 = _rolling_quantile(ret, w, 0.75)
        q90 = _rolling_quantile(ret, w, 0.90)
        q05 = _rolling_quantile(ret, w, 0.05)

        # -- Distributional WIDTH -------------------------------------------
        iqr   = q75 - q25                      # inter-quartile range
        wide  = q90 - q10                      # wide 80-percentile band

        df[f"qdd_iqr{p}"]  = iqr
        df[f"qdd_wide{p}"] = wide

        # -- Bowley skewness proxy: (upper spread - lower spread) / wide band
        #    = [(q90-q50) - (q50-q10)] / (q90-q10)
        #    Positive → right-skewed (fat upside), negative → left-skewed (fat downside)
        upper_spread = q90 - q50
        lower_spread = q50 - q10
        denom_bowley = wide.replace(0, np.nan)   # guard zero-band
        df[f"qdd_bowley{p}"] = (upper_spread - lower_spread) / denom_bowley

        # -- Downside VaR (5th percentile return) ----------------------------
        df[f"qdd_var5{p}"] = q05

        # 5-bar change in the downside quantile (is left tail moving down = worsening?)
        df[f"qdd_var5_chg{p}"] = q05 - q05.shift(5)

        # -- Rolling return rank: where does today's return sit in its own window? --
        df[f"qdd_ret_rank{p}"] = _rolling_rank(ret, w)

        # -- Width trend: today's wide-band vs its own 20-bar trailing mean ---
        # Use shift(1) so the mean is purely from past values, excluding today's
        # band value (avoids trivial self-reference).
        wide_mean_20 = wide.shift(1).rolling(20, min_periods=10).mean()
        denom_wtrend = wide_mean_20.replace(0, np.nan)
        df[f"qdd_width_trend{p}"] = (wide - denom_wtrend) / denom_wtrend.abs().clip(lower=1e-8)

        # -- Tail-thickness ratio: upper half-range vs lower half-range -------
        # >1 means upper tail is fatter (skew positive); <1 means lower tail fatter
        denom_tail = lower_spread.replace(0, np.nan)
        df[f"qdd_tail_ratio{p}"] = upper_spread / denom_tail

    # ---- Always return df --------------------------------------------------
    return df
