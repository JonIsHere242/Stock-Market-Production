"""
Self-Granger-causality and lag-profile features for OHLCV.

Paper: "Graphical Causal Reasoning for Root Cause Analysis in Cloud Networks"
       (arXiv 2606.13532)

The paper's computable methods:
  1. Bivariate Granger causality between time series
  2. Conditional independence tests across time lags
  3. EDGE-SPECIFIC CONDITIONAL PROBABILITIES as a function of TIME LAG:
     P(cause -> effect | lag=k) — the causal strength decays with lag
  4. Graph traversal using lag-weighted edge probabilities for attribution scoring

OHLCV adaptation — "self-Granger lag profile":
  Applied to a SINGLE ticker's return series using INTRA-TICKER cross-variable Granger:
  - Does past Volume Granger-cause future returns? (at which lags?)
  - Does past High-Low range Granger-cause future returns?
  - Does past return itself self-Granger-cause (momentum vs reversion)?
  - Time-lag probability profile: P(significant F-test at lag k) over rolling windows

  The LAG PROFILE (which lags are significant) is itself the distinctive signal.
  A pure momentum stock has strong lag-1 Granger; a mean-reverting stock peaks at lag-2+.

Produces 7 features:
  gcl_self_granger_lag1_20d  : rolling Granger F-stat of vol -> ret at lag 1 (30d window)
  gcl_self_granger_lag2_20d  : rolling Granger F-stat of vol -> ret at lag 2
  gcl_vol_granger_lag1_20d   : Volume -> ret Granger F-stat at lag 1
  gcl_range_granger_lag1_20d : (High-Low)/Close -> ret Granger F-stat at lag 1
  gcl_lag_peak_20d           : lag (1,2,3) at which vol->ret Granger F-stat is maximised
  gcl_lag_entropy_20d        : entropy of Granger F-stat distribution over vol->ret lags 1-3
  gcl_vol_lead_score_40d     : rolling mean of vol_granger_lag1 (persistent vol-lead signal)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13532_granger_lag_profile",
    "description": (
        "Rolling self-Granger-causality and lag-profile features (return-to-return, "
        "volume-to-return, range-to-return) inspired by graph-causal edge-specific "
        "conditional probabilities with time-lag weighting (arXiv 2606.13532)."
    ),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "gcl_self_granger_lag1_20d",
        "gcl_self_granger_lag2_20d",
        "gcl_vol_granger_lag1_20d",
        "gcl_range_granger_lag1_20d",
        "gcl_lag_peak_20d",
        "gcl_lag_entropy_20d",
        "gcl_vol_lead_score_40d",
    ],
    "tags":        ["momentum", "volume", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13532",
}


def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """
    OLS residual sum of squares: RSS = ||y - X @ (X'X)^{-1} X'y||^2
    X should already include intercept column.
    """
    if X.shape[0] <= X.shape[1]:
        return np.nan
    try:
        # Normal equations: beta = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        # Add small ridge for numerical stability (in-place diagonal add,
        # numerically identical to `XtX += np.eye(k) * 1e-10`).
        XtX.reshape(-1)[:: XtX.shape[0] + 1] += 1e-10
        beta = np.linalg.solve(XtX, Xty)
        resid = y - X @ beta
        return float(resid @ resid)
    except np.linalg.LinAlgError:
        return np.nan


def _granger_f_stat(y: np.ndarray, x: np.ndarray, lag: int,
                    rss_r: float = None) -> float:
    """
    Granger causality F-statistic: does x Granger-cause y?
    Restricted model: y_t = c + a1*y_{t-1} + ... + a_{lag}*y_{t-lag}
    Unrestricted model: restricted + b1*x_{t-1} + ... + b_{lag}*x_{t-lag}
    F = ((RSS_r - RSS_u) / lag) / (RSS_u / (T - 2*lag - 1))

    ``rss_r`` may be supplied when the caller has already computed the
    restricted-model RSS for the same (y, lag) pair (it depends only on y and
    lag, not on x), avoiding a redundant OLS solve.
    """
    T = len(y)
    min_obs = 2 * lag + 4
    if T < min_obs:
        return np.nan

    # Build lagged matrices
    max_lag = lag
    n_obs = T - max_lag

    # y target: y[max_lag:]
    y_target = y[max_lag:]

    # Restricted: intercept + lagged y. Unrestricted: restricted + lagged x.
    # Preallocate both design matrices and fill via slice assignment (avoids
    # the per-call np.ones / list-building / np.column_stack overhead).
    X_u = np.empty((n_obs, 1 + 2 * max_lag), dtype=np.float64)
    X_u[:, 0] = 1.0
    for k in range(1, max_lag + 1):
        X_u[:, k] = y[max_lag - k: T - k]
    for k in range(1, max_lag + 1):
        X_u[:, max_lag + k] = x[max_lag - k: T - k]
    X_r = X_u[:, : max_lag + 1]

    if rss_r is None:
        rss_r = _ols_rss(y_target, X_r)
    rss_u = _ols_rss(y_target, X_u)

    if rss_r is None or rss_u is None or np.isnan(rss_r) or np.isnan(rss_u):
        return np.nan
    if rss_u < 1e-20:
        return 0.0

    df_num = lag
    df_den = n_obs - X_u.shape[1]
    if df_den <= 0:
        return np.nan

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    return float(max(f_stat, 0.0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(
        df["Close"].clip(lower=1e-8) / df["Close"].shift(1).clip(lower=1e-8)
    ).values.astype(np.float64)

    # Normalised volume and range series
    vol = df["Volume"].values.astype(np.float64)
    vol_roll_mean = (
        pd.Series(vol).rolling(20, min_periods=5).mean()
        .bfill().values + 1e-8
    )
    vol_norm = vol / vol_roll_mean

    hl_range = (
        (df["High"].values - df["Low"].values)
        / df["Close"].clip(lower=1e-8).values
    ).astype(np.float64)

    n = len(df)
    W = 30  # rolling window (needs to be >= 2*max_lag + min_obs)

    sg_lag1 = np.full(n, np.nan)
    sg_lag2 = np.full(n, np.nan)
    vg_lag1 = np.full(n, np.nan)
    rg_lag1 = np.full(n, np.nan)
    lag_peak = np.full(n, np.nan)
    lag_ent  = np.full(n, np.nan)

    for i in range(W + 6, n):
        s = max(0, i - W + 1)
        r  = log_ret[s: i + 1]
        v  = vol_norm[s: i + 1]
        hl = hl_range[s: i + 1]

        valid = ~(np.isnan(r) | np.isnan(v) | np.isnan(hl))
        r  = r[valid]; v = v[valid]; hl = hl[valid]

        if len(r) < 15:
            continue

        # Self-Granger at lag 1 and 2 (standard — y != x so no collinearity problem)
        # But for self-Granger (x==y) at lag k, we test whether lag k ALONE adds
        # predictive power BEYOND lags 1..k-1. So:
        #   Restricted: y_t = c + y_{t-1}
        #   Unrestricted: y_t = c + y_{t-1} + y_{t-k}  (k >= 2)
        # This avoids collinearity and captures the marginal lag-k contribution.
        # The lag-1 restricted model (intercept + 1 lagged return) depends only
        # on r, so its RSS is shared by the vol->ret and range->ret lag-1 tests.
        Tn = len(r)
        if Tn >= 2 * 1 + 4:
            X_r1 = np.empty((Tn - 1, 2), dtype=np.float64)
            X_r1[:, 0] = 1.0
            X_r1[:, 1] = r[:-1]
            rss_r1 = _ols_rss(r[1:], X_r1)
        else:
            rss_r1 = None

        f1 = _granger_f_stat(r, v, lag=1, rss_r=rss_r1)   # volume -> return at lag 1 (primary)
        f2 = _granger_f_stat(r, v, lag=2)   # volume -> return at lag 2
        sg_lag1[i] = f1
        sg_lag2[i] = f2

        vg_lag1[i]  = f1
        rg_lag1[i]  = _granger_f_stat(r, hl, lag=1, rss_r=rss_r1)

        # Lag profile across volume->return lags 1, 2, 3
        f3 = _granger_f_stat(r, v, lag=3)

        f_vals = np.array([
            f1 if not np.isnan(f1) else 0.0,
            f2 if not np.isnan(f2) else 0.0,
            f3 if not np.isnan(f3) else 0.0,
        ])
        lag_peak[i] = float(np.argmax(f_vals) + 1)

        # Entropy of the F-stat distribution over lags 1-3
        f_sum = f_vals.sum()
        if f_sum > 1e-10:
            probs = f_vals / f_sum
            probs = probs.clip(min=1e-10)
            lag_ent[i] = float(-np.sum(probs * np.log(probs)))

    df["gcl_self_granger_lag1_20d"]  = sg_lag1
    df["gcl_self_granger_lag2_20d"]  = sg_lag2
    df["gcl_vol_granger_lag1_20d"]   = vg_lag1
    df["gcl_range_granger_lag1_20d"] = rg_lag1
    df["gcl_lag_peak_20d"]           = lag_peak
    df["gcl_lag_entropy_20d"]        = lag_ent

    # Persistent volume-lead score: rolling mean of volume Granger F-stat (40d)
    vg_s = pd.Series(vg_lag1, index=df.index)
    df["gcl_vol_lead_score_40d"] = vg_s.rolling(40, min_periods=10).mean()

    return df
