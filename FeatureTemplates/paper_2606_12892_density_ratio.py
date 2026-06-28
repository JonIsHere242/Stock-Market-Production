"""
Density ratio / importance weight features for OHLCV covariate shift detection.

Paper: "Prediction-Powered Causal Inference by Automatic Debiased Machine Learning
        and Semi-Supervised Riesz Regression" (arXiv 2606.12892)

The paper has NO directly OHLCV-computable method (it is a semiparametric causal
inference framework using Riesz representers for density ratio estimation). SKIPPED.

REPLACEMENT feature family — "covariate density ratio / importance weighting":
Inspired by the paper's core object: the RIESZ REPRESENTER w(x) = p_source(x) / p_target(x),
which measures how a current observation reweights historical observations.

For OHLCV:
  - "Source" distribution = recent W_short bars (today's regime)
  - "Target" distribution = longer W_long bars (historical baseline)
  - Density ratio proxy: Parzen-window (Gaussian kernel) estimate of density at the
    current return value under both distributions → ratio = local density shift
  - This answers: "How much does today's return stand out relative to recent vs
    historical context?" (importance weight = how much to upweight this observation
    in the recent vs the historical frame)

This is DISTINCT from plain volatility: two distributions can have the same std
but different density ratios at the tails.

Produces 7 features:
  drz_density_ratio_10v60d   : KDE density ratio at current return: p_10d / p_60d
  drz_density_ratio_5v20d    : KDE density ratio at current return: p_5d / p_20d
  drz_log_density_ratio_10v60d: log of the 10v60 ratio (signed importance weight)
  drz_kde_percentile_10d     : CDF rank of today's return in 10d distribution
  drz_kde_percentile_60d     : CDF rank of today's return in 60d distribution
  drz_cdf_gap_10v60d         : difference in CDF ranks (shift in tail positioning)
  drz_importance_weight_20d  : rolling normalised density ratio (reweighting signal)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_12892_density_ratio",
    "description": (
        "Kernel density ratio / importance weight features measuring covariate shift "
        "between short and long return windows, inspired by Riesz representer density "
        "ratios in prediction-powered causal inference (arXiv 2606.12892)."
    ),
    "requires":    ["Close"],
    "produces": [
        "drz_density_ratio_10v60d",
        "drz_density_ratio_5v20d",
        "drz_log_density_ratio_10v60d",
        "drz_kde_percentile_10d",
        "drz_kde_percentile_60d",
        "drz_cdf_gap_10v60d",
        "drz_importance_weight_20d",
    ],
    "tags":        ["volatility", "market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.12892",
}


def _gauss_kde_density(x_eval: float, samples: np.ndarray, bw: float) -> float:
    """Gaussian KDE density estimate at x_eval from samples with bandwidth bw."""
    if bw < 1e-14 or len(samples) < 2:
        return 1e-14
    diffs = (x_eval - samples) / bw
    # Gaussian kernel: K(u) = exp(-0.5 u^2) / sqrt(2pi)
    weights = np.exp(-0.5 * diffs ** 2)
    return float(weights.mean() / (bw * np.sqrt(2.0 * np.pi))) + 1e-14


def _silverman_bw(samples: np.ndarray) -> float:
    """Silverman's rule of thumb bandwidth."""
    n = len(samples)
    if n < 2:
        return 1.0
    std = samples.std()
    if std < 1e-12:
        return 1.0
    return 1.06 * std * (n ** (-0.2))


def _ecdf_rank(x_eval: float, samples: np.ndarray) -> float:
    """Fraction of samples <= x_eval (empirical CDF)."""
    return float((samples <= x_eval).mean())


def _windowed_kde_and_ecdf(log_ret: np.ndarray, win: int, lo: int):
    """Vectorized per-bar KDE density + ECDF rank.

    For every bar i in [lo, n), evaluate, over the `win`-length window of
    log_ret ENDING at i-1 (i.e. log_ret[i-win : i], excluding the current bar):
      * Silverman bandwidth bw_i (population std, ddof=0)
      * Gaussian-KDE density of the current return r_cur=log_ret[i]
      * empirical CDF rank of r_cur

    Returns (p, ecdf) arrays of length n with NaN before `lo`.

    Mirrors the original element-by-element math exactly, including the
    1e-14 floors and the degenerate-window fallbacks:
      bw -> 1.0 if std < 1e-12 (else 1.06*std*win**-0.2)
      density -> 1e-14 if bw < 1e-14 (never here) else mean(K)/(bw*sqrt(2pi))+1e-14
    """
    n = log_ret.shape[0]
    p = np.full(n, np.nan)
    ecdf = np.full(n, np.nan)
    if n <= lo:
        return p, ecdf
    # sliding windows: row j corresponds to log_ret[j : j+win]; the window for
    # bar i is the one ending at i-1, i.e. starting index i-win -> row (i-win).
    sw = np.lib.stride_tricks.sliding_window_view(log_ret, win)  # (n-win+1, win)
    # rows valid for bars i in [lo, n): row index = i-win, for i in [lo, n)
    r0 = lo - win
    rows = sw[r0: n - win]                 # (n-lo, win)
    r_cur = log_ret[lo:n]                   # (n-lo,)

    # Silverman bandwidth (population std, matches ndarray.std()/ddof=0)
    std = rows.std(axis=1)
    bw = np.where(std < 1e-12, 1.0, 1.06 * std * (win ** (-0.2)))

    # Gaussian KDE density at r_cur
    diffs = (r_cur[:, None] - rows) / bw[:, None]
    weights = np.exp(-0.5 * diffs * diffs)
    dens = weights.mean(axis=1) / (bw * np.sqrt(2.0 * np.pi)) + 1e-14

    # ECDF rank
    rank = (rows <= r_cur[:, None]).mean(axis=1)

    valid = ~np.isnan(r_cur)
    out_idx = np.arange(lo, n)
    p[out_idx[valid]] = dens[valid]
    ecdf[out_idx[valid]] = rank[valid]
    return p, ecdf


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(
        df["Close"].clip(lower=1e-8) / df["Close"].shift(1).clip(lower=1e-8)
    ).values.astype(np.float64)
    n = len(df)

    dr_10v60   = np.full(n, np.nan)
    dr_5v20    = np.full(n, np.nan)
    ldr_10v60  = np.full(n, np.nan)
    pct_10d    = np.full(n, np.nan)
    pct_60d    = np.full(n, np.nan)
    cdf_gap    = np.full(n, np.nan)
    iw_20d     = np.full(n, np.nan)

    # The original loop runs for i in [60, n) and, since the smallest window
    # start is i-59 >= 1, never includes log_ret[0] (the only NaN). Thus every
    # window is full-length with no internal NaNs and all the len(...) guards
    # (9>=5, 59>=20, 4>=3, 19>=10) are always satisfied. So the per-bar work is
    # purely the KDE/ECDF over the four fixed-size windows (4, 9, 19, 59).
    if n > 60:
        p5,  _      = _windowed_kde_and_ecdf(log_ret, 4,  60)
        p10, ec10   = _windowed_kde_and_ecdf(log_ret, 9,  60)
        p20, _      = _windowed_kde_and_ecdf(log_ret, 19, 60)
        p60, ec60   = _windowed_kde_and_ecdf(log_ret, 59, 60)

        valid = ~np.isnan(log_ret)
        valid[:60] = False

        ratio = p10 / p60
        dr_10v60[valid]  = ratio[valid]
        ldr_10v60[valid] = np.log(ratio[valid])
        pct_10d[valid]   = ec10[valid]
        pct_60d[valid]   = ec60[valid]
        cdf_gap[valid]   = ec10[valid] - ec60[valid]
        dr_5v20[valid]   = (p5 / p20)[valid]
        iw_20d[valid]    = np.log((p20 / p60)[valid])

    # Smooth the density ratio signals with short rolling average (reduces noise)
    ldr_s   = pd.Series(ldr_10v60, index=df.index)
    cdf_s   = pd.Series(cdf_gap,   index=df.index)
    dr5v20_s = pd.Series(dr_5v20,  index=df.index)

    # Smoothed log-density-ratio (5d EWM reduces single-bar noise)
    df["drz_density_ratio_10v60d"]     = ldr_s.ewm(span=5, min_periods=2).mean()
    df["drz_density_ratio_5v20d"]      = dr5v20_s
    df["drz_log_density_ratio_10v60d"] = ldr_10v60
    df["drz_kde_percentile_10d"]       = pct_10d
    df["drz_kde_percentile_60d"]       = pct_60d

    # Smoothed CDF gap
    df["drz_cdf_gap_10v60d"] = cdf_s.ewm(span=5, min_periods=2).mean()

    # Rolling z-score of importance weight to normalise across regime changes
    iw_s = pd.Series(iw_20d, index=df.index)
    # EWM smooth first to reduce per-bar noise
    iw_smooth = iw_s.ewm(span=5, min_periods=2).mean()
    roll_m = iw_smooth.rolling(60, min_periods=15).mean()
    roll_s = iw_smooth.rolling(60, min_periods=15).std().clip(lower=1e-8)
    df["drz_importance_weight_20d"] = (iw_smooth - roll_m) / roll_s

    return df
