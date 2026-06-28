"""
Calibration-Based Market Uncertainty Features.

PAPER: "Human-AI Teaming Through the Lens of Calibration" (arxiv 2606.10906).
       Paper about human-AI prediction delegation via statistical calibration —
       no directly extractable OHLCV feature method.

SUBSTITUTE (same theme — statistical calibration / conformal prediction):

The paper's core idea: a calibrated predictor produces intervals with correct
empirical coverage.  For OHLCV we measure how well the return distribution is
calibrated vs the historical quantile envelope — deviations signal stress.

Features (all vectorized for speed):
  1. cal_exceedance_21d: fraction of 21d returns exceeding ±1σ of 63d baseline.
  2. cal_exceedance_63d: fraction of 63d returns exceeding ±1σ of 126d baseline.
  3. cal_pinball_q10: asymmetric quantile loss at tau=0.10 (lower tail calibration).
  4. cal_pinball_q90: asymmetric quantile loss at tau=0.90 (upper tail calibration).
  5. cal_interval_width_norm: ratio of current 90-10% pct-range vs 126d ewm median.
  6. cal_coverage_gap: |empirical 80% coverage - 0.80| over 21d vs 63d quantiles.
  7. cal_nc_zscore: z-score of current |return - rolling median| (conformal score).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_10906_calibration_uncertainty",
    "description": (
        "Calibration-based market uncertainty: empirical vs nominal coverage, "
        "asymmetric quantile (pinball) loss, conformal nonconformity z-score, "
        "and quantile interval width ratio. Inspired by statistical calibration "
        "theory for human-AI teaming (arxiv 2606.10906)."
    ),
    "requires": ["Close"],
    "produces": [
        "cal_exceedance_21d",
        "cal_exceedance_63d",
        "cal_pinball_q10",
        "cal_pinball_q90",
        "cal_interval_width_norm",
        "cal_coverage_gap",
        "cal_nc_zscore",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.10906",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    n = len(close)

    # Log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_raw = np.log(close[1:] / close[:-1])
    ret = np.concatenate([[np.nan], ret_raw])
    ret_s = pd.Series(ret)

    # Rolling statistics (vectorized pandas)
    mu_21 = ret_s.rolling(21, min_periods=10).mean()
    std_21 = ret_s.rolling(21, min_periods=10).std()
    mu_63 = ret_s.rolling(63, min_periods=20).mean()
    std_63 = ret_s.rolling(63, min_periods=20).std()
    mu_126 = ret_s.rolling(126, min_periods=40).mean()
    std_126 = ret_s.rolling(126, min_periods=40).std()

    # --- 1. Exceedance rate: fraction of recent bars beyond ±1 std of longer window ---
    # For 21d: how many of last 21 returns exceed ±std_63 at each point
    # Vectorized: use shift trick — for each bar i, track whether ret[i] > std_63[i]
    # Then take rolling sum
    exceed_63 = (ret_s.abs() > std_63).astype(float)
    exceed_126 = (ret_s.abs() > std_126).astype(float)

    cal_exc_21 = exceed_63.rolling(21, min_periods=10).mean()
    cal_exc_63 = exceed_126.rolling(63, min_periods=20).mean()

    # --- 2. Pinball loss at tau=0.10 and tau=0.90 (current return vs rolling quantile) ---
    # Rolling quantile (pandas 1.1+)
    q10_63 = ret_s.rolling(63, min_periods=20).quantile(0.10)
    q90_63 = ret_s.rolling(63, min_periods=20).quantile(0.90)

    # Pinball at tau=0.10: L = tau*(y-q) if y>=q else (1-tau)*(q-y)
    tau = 0.10
    diff_10 = ret_s - q10_63
    pinball_q10 = np.where(diff_10 >= 0, tau * diff_10, (1 - tau) * (-diff_10))
    tau = 0.90
    diff_90 = ret_s - q90_63
    pinball_q90 = np.where(diff_90 >= 0, tau * diff_90, (1 - tau) * (-diff_90))

    # --- 3. Interval width ratio ---
    # Current 90-10 pct range (21d) vs EWM of 90-10 pct range
    q10_21 = ret_s.rolling(21, min_periods=10).quantile(0.10)
    q90_21 = ret_s.rolling(21, min_periods=10).quantile(0.90)
    iw_21 = q90_21 - q10_21

    # EWM of interval width as "baseline"
    iw_ewm = iw_21.ewm(span=126, min_periods=40).mean()
    cal_iw_ratio = iw_21 / (iw_ewm.abs() + 1e-10)

    # --- 4. Coverage gap ---
    # Empirical 80% coverage: fraction of 21d returns inside [q10_63, q90_63]
    in_band = ((ret_s >= q10_63) & (ret_s <= q90_63)).astype(float)
    emp_cov = in_band.rolling(21, min_periods=10).mean()
    cal_cov_gap = (emp_cov - 0.80).abs()

    # --- 5. Conformal nonconformity z-score ---
    # Nonconformity = |ret - rolling_median| (residual from naive median forecast)
    med_63 = ret_s.rolling(63, min_periods=20).median()
    nc = (ret_s - med_63).abs()
    nc_mu = nc.rolling(63, min_periods=20).mean()
    nc_std = nc.rolling(63, min_periods=20).std()
    cal_nc_z = (nc - nc_mu) / (nc_std + 1e-12)

    df["cal_exceedance_21d"] = cal_exc_21.values
    df["cal_exceedance_63d"] = cal_exc_63.values
    df["cal_pinball_q10"] = pinball_q10
    df["cal_pinball_q90"] = pinball_q90
    df["cal_interval_width_norm"] = cal_iw_ratio.values
    df["cal_coverage_gap"] = cal_cov_gap.values
    df["cal_nc_zscore"] = cal_nc_z.values

    return df
