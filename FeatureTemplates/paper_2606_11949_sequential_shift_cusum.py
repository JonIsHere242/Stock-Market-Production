"""
Sequential Distributional Shift Detection Features  —  arxiv:2606.11949
"Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers"

The paper presents a sequential monitoring system using calibrated statistics to
detect when a deployed classifier moves out of distribution. Upon detection, a
conformal abstention layer adapts thresholds to recover target coverage.

OHLCV translation:
  The COMPUTABLE METHOD is sequential CUSUM (cumulative sum) shift detection over
  a score sequence, plus coverage/calibration adaptation metrics.

  Applied to OHLCV:
    - Define a "nonconformity score" per bar: how surprising is today's
      log-return given the rolling calibration window?
    - Run Page-Hinkley CUSUM statistic over these scores to detect mean shift
    - Track the CUSUM value, its stopping-time analog, and cumulative evidence
    - Implement weighted ESS (effective sample size) to measure calibration
      collapse risk analogous to DeBERTa weight clipping in the paper
    - Multi-window: 20-bar and 60-bar calibration pools

  Produces 7 columns prefixed "seqshift_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11949_sequential_shift_cusum",
    "description": (
        "Sequential CUSUM shift-detection statistics on rolling log-return "
        "nonconformity scores; inspired by online shift detection in arxiv 2606.11949."
    ),
    "requires": ["Close"],
    "produces": [
        "seqshift_cusum_20",
        "seqshift_cusum_60",
        "seqshift_evidence_20",
        "seqshift_ess_ratio_20",
        "seqshift_ess_ratio_60",
        "seqshift_alarm_density_20",
        "seqshift_drift_rate_20",
        "seqshift_coverage_deficit_10",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11949",
}


def _rolling_cusum(scores: np.ndarray, window: int, delta: float = 0.5) -> np.ndarray:
    """
    Page-Hinkley CUSUM statistic in a rolling fashion.

    For each position t, compute CUSUM over the trailing `window` scores:
      S_0 = 0
      S_i = max(0, S_{i-1} + (x_i - mu_ref - delta/2))
    where mu_ref = mean of first half of the window (reference distribution).
    delta = slack parameter (half the expected shift magnitude in std units).

    Returns the final CUSUM value S_window as the rolling statistic.
    This is standardized by the std of the calibration scores.
    """
    n = len(scores)
    result = np.full(n, np.nan)
    half = max(window // 2, 5)

    for t in range(window - 1, n):
        w = scores[t - window + 1: t + 1]
        ref = w[:half]
        mu_ref = np.nanmean(ref)
        std_ref = np.nanstd(ref)
        if std_ref < 1e-10:
            result[t] = 0.0
            continue
        # Standardize scores relative to reference
        z = (w - mu_ref) / std_ref
        slack = delta / 2.0
        S = 0.0
        for zi in z[half:]:
            if np.isfinite(zi):
                S = max(0.0, S + zi - slack)
        result[t] = S

    return result


def _rolling_ess_ratio(scores: np.ndarray, window: int) -> np.ndarray:
    """
    Effective Sample Size ratio: ESS / window.

    ESS via importance-weight analogy: treat the current score distribution
    vs the reference as if we had importance weights w_i = exp(-|z_i - mu|).
    ESS = (sum w_i)^2 / sum(w_i^2), ratio = ESS / window.
    Low ESS ratio = weight collapse = calibration breakdown signal.
    """
    n = len(scores)
    result = np.full(n, np.nan)
    half = max(window // 2, 5)

    for t in range(window - 1, n):
        w = scores[t - window + 1: t + 1]
        ref = w[:half]
        test = w[half:]
        mu_ref = np.nanmean(ref)
        std_ref = np.nanstd(ref)
        if std_ref < 1e-10 or len(test) < 3:
            result[t] = 1.0
            continue
        z = (test - mu_ref) / std_ref
        weights = np.exp(-np.abs(z[np.isfinite(z)]))
        if len(weights) == 0:
            result[t] = np.nan
            continue
        sw = weights.sum()
        sw2 = (weights ** 2).sum()
        if sw2 < 1e-15:
            result[t] = 1.0
            continue
        ess = (sw ** 2) / sw2
        result[t] = ess / len(weights)

    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    n = len(close)

    # Log returns as base nonconformity scores
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    # Nonconformity score: |return - rolling median| (absolute deviation)
    # Using vectorised rolling to produce fast scores, then CUSUM over them
    lr_series = pd.Series(log_ret)
    med20 = lr_series.rolling(20, min_periods=5).median().values
    med60 = lr_series.rolling(60, min_periods=15).median().values
    nc20 = np.abs(log_ret - med20)
    nc60 = np.abs(log_ret - med60)

    # ---- CUSUM over nonconformity scores: detects shift in score mean ----------
    # Use vectorised approach for speed (no Python loop over each bar)
    # We compute a "rolling CUSUM accumulation" using pandas expanding logic

    def fast_cusum(nc: np.ndarray, window: int) -> np.ndarray:
        """
        Vectorised rolling CUSUM: at each t, reference = mean/std of
        nc[t-window+1 .. t-window//2], test = nc[t-window//2+1 .. t].
        CUSUM final value = max(0, sum of z_i - delta/2).
        """
        result = np.full(n, np.nan)
        half = window // 2
        delta_slack = 0.25  # slack parameter (in std units)

        # Pre-compute rolling mean and std over reference half-windows
        nc_s = pd.Series(nc)
        # Rolling over first half: reference stats at position t
        # ref_mean[t] = mean(nc[t-window+1 .. t-window//2])
        # We shift to align: ref_mean is the rolling mean over [t-window+1, t-half]
        ref_mean = nc_s.rolling(half, min_periods=3).mean().shift(half).values
        ref_std = nc_s.rolling(half, min_periods=3).std().shift(half).values

        # Test slice cumulative sum: sum of (nc - ref_mean - slack) / ref_std
        # over trailing `half` bars
        for t in range(window - 1, n):
            rm = ref_mean[t]
            rs = ref_std[t]
            if not np.isfinite(rm) or rs < 1e-10:
                continue
            test_slice = nc[t - half + 1: t + 1]
            z = (test_slice - rm) / rs - delta_slack
            z = z[np.isfinite(z)]
            if len(z) == 0:
                continue
            # CUSUM final value (positive only)
            S = 0.0
            for zi in z:
                S = max(0.0, S + zi)
            result[t] = S
        return result

    cusum20 = fast_cusum(nc20, 20)
    cusum60 = fast_cusum(nc60, 60)

    df["seqshift_cusum_20"] = cusum20
    df["seqshift_cusum_60"] = cusum60

    # ---- Cumulative evidence: exponentially-weighted CUSUM (faster to compute) --
    # S_t = max(0, lambda * S_{t-1} + (z_t - delta)) — leaky integrator
    nc20_std = pd.Series(nc20).rolling(60, min_periods=10).std().values
    nc20_mean = pd.Series(nc20).rolling(60, min_periods=10).mean().values
    evidence = np.full(n, np.nan)
    S = 0.0
    lam = 0.95
    for t in range(1, n):
        nc_t = nc20[t]
        mu_t = nc20_mean[t]
        sd_t = nc20_std[t]
        if not (np.isfinite(nc_t) and np.isfinite(mu_t) and sd_t > 1e-10):
            evidence[t] = np.nan
            continue
        z = (nc_t - mu_t) / sd_t
        S = max(0.0, lam * S + z - 0.25)
        evidence[t] = S

    df["seqshift_evidence_20"] = evidence

    # ---- ESS ratio (calibration health) ----------------------------------------
    ess20 = _rolling_ess_ratio(nc20, 40)
    ess60 = _rolling_ess_ratio(nc60, 80)
    df["seqshift_ess_ratio_20"] = ess20
    df["seqshift_ess_ratio_60"] = ess60

    # ---- Alarm density: fraction of trailing 20 bars where CUSUM > threshold ---
    threshold = 1.0  # 1 std-unit CUSUM level
    cusum_s = pd.Series(cusum20)
    df["seqshift_alarm_density_20"] = (
        cusum_s.gt(threshold)
        .rolling(20, min_periods=5)
        .mean()
    )

    # ---- Drift rate: slope of CUSUM over trailing 10 bars ----------------------
    cusum_valid = pd.Series(cusum20)
    # Use linear regression slope via rolling cov/var
    t_idx = pd.Series(np.arange(n, dtype=float))

    def rolling_slope(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
        cov = y.rolling(w, min_periods=w // 2).cov(x)
        var = x.rolling(w, min_periods=w // 2).var()
        return cov / var.replace(0, np.nan)

    df["seqshift_drift_rate_20"] = rolling_slope(cusum_valid, t_idx, 10)

    # ---- Coverage deficit: fraction of 10 bars where |return| > rolling 90th pct -
    # Conformal adaptation recovers coverage when threshold is exceeded; we track
    # how frequently the classifier is "out of coverage" (high = stressed regime)
    abs_ret = np.abs(log_ret)
    abs_ret_s = pd.Series(abs_ret)
    q90_60 = abs_ret_s.rolling(60, min_periods=15).quantile(0.90)
    df["seqshift_coverage_deficit_10"] = (
        (abs_ret_s > q90_60).astype(float).rolling(10, min_periods=3).mean()
    )

    return df
