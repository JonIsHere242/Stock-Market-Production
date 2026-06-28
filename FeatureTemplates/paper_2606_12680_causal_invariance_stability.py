"""
Causal Invariance / Cross-Domain Stability Features  —  arxiv:2606.12680
"How Useful is Causal Invariance for Domain Adaptation in Finite-Sample Settings?"

The paper explores causal invariance — a predictor is "causal-invariant" if it
maintains stable risk across different domain shifts. Invariant predictors use only
the causal feature subset whose risk margin separates them from spurious alternatives.

OHLCV translation:
  The COMPUTABLE METHOD is measuring RISK STABILITY ACROSS DOMAINS (time windows).
  Applied to OHLCV, we treat each non-overlapping sub-period (week, month, quarter)
  as a "domain" and measure how stable the return prediction signal is across them.

  Specifically:
    - Compute the Spearman rank of a signal (Close position in rolling window)
      across multiple temporal sub-domains and measure variance
    - Causal-invariant features have LOW cross-domain variance in their predictive
      relationship with next-period returns
    - We implement: cross-window risk margin (gap between best/worst period IC),
      domain-shift risk variance, and invariant residual tracking

  Features:
    - cinv_ret_stability_20: std of 10-bar rolling mean returns over 20d window
      (stable return = low std = invariant regime)
    - cinv_vol_invariance: ratio of short/long-window vol (near 1 = invariant)
    - cinv_rank_margin_20: max - min of 5-bar rolling median return rank
    - cinv_domain_shift_mag: magnitude of mean-shift between 4 non-overlapping
      weekly sub-windows over the last 20 bars
    - cinv_candidate_margin: gap between top/bottom quartile candidates in
      20-bar return distribution (the "finite-sample margin" from the paper)
    - cinv_bet_stability: stability of bull/bear signals across sub-periods

  Produces 6 columns prefixed "cinv_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12680_causal_invariance_stability",
    "description": (
        "Cross-domain (cross-temporal-window) risk stability features measuring "
        "causal invariance of price signals; inspired by arxiv 2606.12680 "
        "(Causal Invariance for Domain Adaptation)."
    ),
    "requires": ["Close", "High", "Low", "Volume"],
    "produces": [
        "cinv_ret_stability_20",
        "cinv_vol_invariance",
        "cinv_rank_margin_20",
        "cinv_domain_shift_mag",
        "cinv_candidate_margin",
        "cinv_bet_stability",
    ],
    "tags": ["momentum", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12680",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    vol = df["Volume"].values.astype(float)
    n = len(close)

    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    lr = pd.Series(log_ret)

    # ---- cinv_ret_stability_20: std of 5-bar rolling mean returns in 20d window -
    # Stable return profile (low domain-to-domain variance) = stable signal
    rolling_mean_5 = lr.rolling(5, min_periods=2).mean()
    # std of those 5-bar means over 20-bar window = cross-week return variance
    df["cinv_ret_stability_20"] = rolling_mean_5.rolling(20, min_periods=6).std()

    # ---- cinv_vol_invariance: ratio short/long realised vol (near 1 = invariant) -
    rv5 = lr.rolling(5, min_periods=2).std()
    rv20 = lr.rolling(20, min_periods=6).std()
    # Clamp ratio to [0.1, 10] to avoid explosion; near 1.0 = vol-invariant regime
    ratio = rv5 / rv20.replace(0, np.nan)
    df["cinv_vol_invariance"] = ratio.clip(lower=0.1, upper=10.0)

    # ---- cinv_rank_margin_20: IQR of 5-bar cumulative returns in 20d window ------
    # "Margin" between domain candidates: spread of 5-bar return blocks across time
    # Q75 - Q25 of 5-bar cumulative returns = cross-domain candidate spread
    cum5 = lr.rolling(5, min_periods=2).sum()
    df["cinv_rank_margin_20"] = (
        cum5.rolling(20, min_periods=6).quantile(0.75)
        - cum5.rolling(20, min_periods=6).quantile(0.25)
    )

    # ---- cinv_domain_shift_mag: mean-shift across 4 weekly sub-windows in 20d ----
    # For each t, split [t-19 .. t] into 4 blocks of 5; compute block means,
    # then return the range (max-min) of those means as "domain shift magnitude"
    domain_shift = np.full(n, np.nan)
    for t in range(19, n):
        block = log_ret[t - 19: t + 1]
        if np.sum(np.isfinite(block)) < 12:
            continue
        b1 = np.nanmean(block[0:5])
        b2 = np.nanmean(block[5:10])
        b3 = np.nanmean(block[10:15])
        b4 = np.nanmean(block[15:20])
        vals = [b1, b2, b3, b4]
        if all(np.isfinite(v) for v in vals):
            domain_shift[t] = max(vals) - min(vals)
    df["cinv_domain_shift_mag"] = domain_shift

    # ---- cinv_candidate_margin: Q75 - Q25 of returns in 40-bar window -----------
    # Paper: gains achievable when margin between candidates is large
    df["cinv_candidate_margin"] = (
        lr.rolling(40, min_periods=12).quantile(0.75)
        - lr.rolling(40, min_periods=12).quantile(0.25)
    )

    # ---- cinv_bet_stability: how consistently bullish/bearish over sub-windows ---
    # Sign of 5-bar mean return — stability = fraction of agreement across 4 windows
    # Compute via rolling sign agreement
    sign5 = np.sign(rolling_mean_5.values)  # +1, 0, -1 per 5-bar sub-period
    sign_s = pd.Series(sign5)
    # fraction of last 20 bars where sign equals today's sign
    def sign_agreement(x):
        if len(x) < 4 or not np.isfinite(x[-1]):
            return np.nan
        today = x[-1]
        if today == 0:
            return 0.5
        return float(np.sum(x[:-1] == today)) / max(len(x) - 1, 1)

    df["cinv_bet_stability"] = sign_s.rolling(20, min_periods=6).apply(
        sign_agreement, raw=True
    )

    return df
