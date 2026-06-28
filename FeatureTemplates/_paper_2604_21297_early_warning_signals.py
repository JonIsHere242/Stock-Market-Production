"""
_paper_2604_21297_early_warning_signals.py  --  Critical Slowing Down / DNM early-warning signals.

Paper: "Identifying dynamical network markers of financial market instability"
       arXiv 2604.21297  (Dynamical Network Markers / DNM theory)

The paper detects impending regime shifts in a *multivariate* participant network via
rising autocorrelation, variance, and cross-correlation — the universal Critical Slowing
Down (CSD) fingerprint that appears as a system nears a bifurcation point.

PER-TICKER PROXY NOTE: This implementation adapts the CSD idea to a single price series.
The multi-participant cross-correlation component is replaced by a within-series AR(1)
coefficient and a flickering/skewness measure.  Results should be interpreted as
"single-ticker CSD proxies" only — they do NOT capture cross-stock network effects.

Signals produced:
  ews_ac1_21d          — rolling 21-day lag-1 autocorrelation of returns (core CSD indicator)
  ews_ac1_trend_10d    — 10-day slope of ews_ac1_21d (rising = warning)
  ews_var_21d          — rolling 21-day variance of returns (CSD: variance rises near transition)
  ews_var_trend_10d    — 10-day slope of ews_var_21d (rising = warning)
  ews_skew_42d         — rolling 42-day skewness of returns (flickering / asymmetry build-up)
  ews_kurt_42d         — rolling 42-day excess kurtosis of returns (fat tails = flickering)
  ews_ar1_coef_21d     — rolling OLS AR(1) coefficient of returns on lag-1 (CSD: |coef| → 1)
  ews_ar1_delta_10d    — 10-day change in ews_ar1_coef_21d (rising = warning)
  ews_composite        — z-scored composite of the four rising-trend indicators (causal)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_2604_21297_early_warning_signals",
    "description": (
        "Per-ticker Critical Slowing Down proxy: rolling AC1, variance, skewness, kurtosis, "
        "AR(1) coefficient, their trend signals, and a composite early-warning score "
        "(inspired by DNM theory, arXiv 2604.21297; single-series, not cross-stock network)."
    ),
    "requires":    ["Close"],
    "produces":    [
        "ews_ac1_21d",
        "ews_ac1_trend_10d",
        "ews_var_21d",
        "ews_var_trend_10d",
        "ews_skew_42d",
        "ews_kurt_42d",
        "ews_ar1_coef_21d",
        "ews_ar1_delta_10d",
        "ews_composite",
    ],
    "tags":        ["volatility", "market_regime", "experimental", "paper"],
    "version":     "1.0",
    "author":      "arXiv 2604.21297 — Critical Slowing Down / DNM theory (single-ticker proxy)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_lag1_ac(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling lag-1 autocorrelation of `returns` over `window` rows (causal).

    For each position t, computes corr(r[t-window+2 : t+1], r[t-window+1 : t])
    using the Pearson formula on two overlapping windows of length (window-1).
    Returns NaN for positions with insufficient history.
    """
    n = len(returns)
    out = np.full(n, np.nan)
    # Need at least window points to compute the window-1 lag-pair correlation
    if window < 3:
        return out

    # Vectorised: build lag-0 and lag-1 sub-windows using a stride trick
    # then compute per-row Pearson correlation.
    # We use pd.Series.rolling with a custom apply that is causal.
    s = pd.Series(returns)

    def _ac1(x):
        # x is the rolling window of length `window`
        if len(x) < 3:
            return np.nan
        y0 = x[1:]      # lags 0 (current half of pair)
        y1 = x[:-1]     # lags 1 (lagged half of pair)
        m0 = y0.mean()
        m1 = y1.mean()
        num = ((y0 - m0) * (y1 - m1)).sum()
        d0 = np.sqrt(((y0 - m0) ** 2).sum())
        d1 = np.sqrt(((y1 - m1) ** 2).sum())
        denom = d0 * d1
        if denom < 1e-14:
            return np.nan
        return num / denom

    result = s.rolling(window, min_periods=window).apply(_ac1, raw=True)
    return result.to_numpy()


def _rolling_ols_ar1(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OLS AR(1) coefficient over `window` bars (causal).

    Regresses r[t] ~ a + b * r[t-1] within the window; returns b.
    Uses the closed-form OLS slope: b = Cov(x,y)/Var(x).
    """
    n = len(returns)
    out = np.full(n, np.nan)
    if window < 3 or n < window + 1:
        return out

    s = pd.Series(returns)
    s_lag = s.shift(1)

    def _ols_slope(vals):
        # vals = stacked [y, x] via the combined rolling frame trick
        # We get a flat array of length window; split in half.
        half = len(vals) // 2
        y = vals[:half]
        x = vals[half:]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            return np.nan
        xv, yv = x[valid], y[valid]
        xm, ym = xv.mean(), yv.mean()
        xd = xv - xm
        denom = (xd ** 2).sum()
        if denom < 1e-14:
            return np.nan
        return (xd * (yv - ym)).sum() / denom

    # Stack y and x (lagged) into a single rolling window via a DataFrame
    combo = pd.concat([s, s_lag], axis=1).to_numpy(dtype=np.float64)

    # Roll manually: iterate is slow; use a vectorised approach with stride trick.
    # Fall back to a vectorised Pearson/Var formula applied row-by-row via rolling:
    # slope = sum((x - xbar)(y - ybar)) / sum((x - xbar)^2)
    #       = [E(xy) - E(x)E(y)] / [E(x^2) - E(x)^2]

    # y = s (returns), x = s_lag (lagged returns)
    xy = s * s_lag          # element-wise product
    x2 = s_lag ** 2

    roll_kwargs = dict(window=window, min_periods=window)
    Exy  = xy.rolling(**roll_kwargs).mean()
    Ex   = s_lag.rolling(**roll_kwargs).mean()
    Ey   = s.rolling(**roll_kwargs).mean()
    Ex2  = x2.rolling(**roll_kwargs).mean()

    num   = Exy - Ex * Ey
    denom = Ex2 - Ex ** 2

    slope = np.where(np.abs(denom) > 1e-14, num / denom, np.nan)
    # First window-1 positions of s_lag are NaN (due to shift(1)) — already NaN via denom
    return slope


def _rolling_slope(series: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OLS slope (trend) of `series` over `window` bars (causal).

    Uses the analytical formula slope = (n*sum(i*y) - sum(i)*sum(y)) / (n*sum(i^2) - sum(i)^2)
    where i = 0..window-1 (time index within window).
    Implemented via a vectorised rolling approach.
    """
    n = len(series)
    out = np.full(n, np.nan)
    if window < 2:
        return out

    s = pd.Series(series, dtype=np.float64)
    idx = np.arange(window, dtype=np.float64)
    # Precompute fixed sums for the time index (same for every window)
    sum_i  = idx.sum()
    sum_i2 = (idx ** 2).sum()
    denom_fixed = window * sum_i2 - sum_i ** 2  # scalar, >0 for window>=2

    if abs(denom_fixed) < 1e-14:
        return out

    # For each window position, slope = (window * sum(i*y_i) - sum_i * sum(y_i)) / denom_fixed
    # sum(y_i) is just the rolling sum.
    # sum(i * y_i): weight i=0 maps to oldest, i=window-1 maps to newest.
    # We compute this by multiplying with a linearly increasing weight array.
    # Trick: build a rolling weighted sum using the fact that weight[k] = k for k=0..w-1.
    # Equivalent to sum_{j=0}^{w-1} j * y[t-w+1+j] = sum_{j} (j+1)*y - sum(y)
    # = rolling sum of (j+1)*y where j+1 = 1..window, minus nothing.

    # Use cumsum trick: rolling sum of (w-position)*y.
    # Directly: roll apply is clean enough here (window is small: 10 or 21).
    weights = np.arange(window, dtype=np.float64)

    def _slope_apply(y):
        if not np.all(np.isfinite(y)):
            return np.nan
        num = window * (weights * y).sum() - sum_i * y.sum()
        return num / denom_fixed

    result = s.rolling(window, min_periods=window).apply(_slope_apply, raw=True)
    return result.to_numpy()


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Critical Slowing Down early-warning signals as single-ticker proxies
    of the DNM indicators described in arXiv 2604.21297.

    All indicators are causal (depend only on rows <= t).  Leading NaNs are
    expected and left in place.  No inf values are emitted.
    """

    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ------------------------------------------------------------------
    # 1. Log returns (causal: r[t] = log(C[t]/C[t-1]), r[0] = NaN)
    # ------------------------------------------------------------------
    log_ret = np.full(n, np.nan)
    valid_close = np.isfinite(close)
    for i in range(1, n):
        if valid_close[i] and valid_close[i - 1] and close[i - 1] > 0:
            log_ret[i] = np.log(close[i] / close[i - 1])

    # Vectorised version (equivalent, safer):
    with np.errstate(divide="ignore", invalid="ignore"):
        c_prev = np.roll(close, 1)
        c_prev[0] = np.nan
        log_ret_vec = np.where(
            np.isfinite(close) & np.isfinite(c_prev) & (c_prev > 0),
            np.log(close / c_prev),
            np.nan,
        )
    log_ret = log_ret_vec   # use the vectorised version

    ret_s = pd.Series(log_ret, index=df.index)

    # ------------------------------------------------------------------
    # 2. Rolling lag-1 autocorrelation of returns (window = 21d)
    #    The canonical CSD indicator: rises toward 1 near a transition.
    # ------------------------------------------------------------------
    ac1_21 = _rolling_lag1_ac(log_ret, window=21)
    df["ews_ac1_21d"] = pd.Series(ac1_21, index=df.index)

    # ------------------------------------------------------------------
    # 3. Rolling variance of returns (window = 21d)
    #    CSD: variance rises near a bifurcation / regime shift.
    # ------------------------------------------------------------------
    var_21 = ret_s.rolling(21, min_periods=21).var().to_numpy()
    df["ews_var_21d"] = pd.Series(var_21, index=df.index)

    # ------------------------------------------------------------------
    # 4. Trend (10-day slope) of AC1 and Variance
    #    Rising trend = early warning of approaching transition.
    # ------------------------------------------------------------------
    df["ews_ac1_trend_10d"] = pd.Series(
        _rolling_slope(ac1_21, window=10), index=df.index
    )
    df["ews_var_trend_10d"] = pd.Series(
        _rolling_slope(var_21, window=10), index=df.index
    )

    # ------------------------------------------------------------------
    # 5. Rolling skewness and kurtosis (window = 42d)
    #    Flickering / asymmetry build-up near critical transitions.
    # ------------------------------------------------------------------
    df["ews_skew_42d"] = ret_s.rolling(42, min_periods=42).skew()

    # pandas rolling().kurt() returns excess kurtosis (Fisher definition)
    df["ews_kurt_42d"] = ret_s.rolling(42, min_periods=42).kurt()

    # ------------------------------------------------------------------
    # 6. Rolling AR(1) coefficient of returns (window = 21d)
    #    CSD theory: |AR1 coef| → 1 as system approaches tipping point.
    # ------------------------------------------------------------------
    ar1_coef = _rolling_ols_ar1(log_ret, window=21)
    df["ews_ar1_coef_21d"] = pd.Series(ar1_coef, index=df.index)

    # ------------------------------------------------------------------
    # 7. 10-day change in AR(1) coefficient
    #    Rapid rise = warning signal.
    # ------------------------------------------------------------------
    df["ews_ar1_delta_10d"] = pd.Series(ar1_coef, index=df.index).diff(10)

    # ------------------------------------------------------------------
    # 8. Composite early-warning score (causal z-scored average)
    #    Components: ews_ac1_trend_10d, ews_var_trend_10d,
    #                ews_ar1_delta_10d, ews_ac1_21d
    #    Each z-scored over a 126-day rolling window (causal), then averaged.
    #    Higher = stronger early-warning signal.
    # ------------------------------------------------------------------
    def _causal_zscore(series: pd.Series, window: int = 126) -> pd.Series:
        """Rolling z-score: (x - rolling_mean) / rolling_std, causal."""
        m = series.rolling(window, min_periods=max(10, window // 4)).mean()
        s = series.rolling(window, min_periods=max(10, window // 4)).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(s > 1e-14, (series - m) / s, np.nan)
        return pd.Series(z, index=series.index)

    composite_components = [
        _causal_zscore(df["ews_ac1_trend_10d"]),
        _causal_zscore(df["ews_var_trend_10d"]),
        _causal_zscore(df["ews_ar1_delta_10d"]),
        _causal_zscore(df["ews_ac1_21d"]),
    ]

    composite_df = pd.concat(composite_components, axis=1)
    # Row-wise mean (ignores NaN components via skipna=True, but require >= 2 valid)
    valid_count = composite_df.notna().sum(axis=1)
    raw_composite = composite_df.mean(axis=1, skipna=True)
    composite = np.where(valid_count >= 2, raw_composite, np.nan)

    df["ews_composite"] = pd.Series(composite, index=df.index)

    # ------------------------------------------------------------------
    # Sanitise: replace any inf with NaN (defensive; should not occur)
    # ------------------------------------------------------------------
    for col in METADATA["produces"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
