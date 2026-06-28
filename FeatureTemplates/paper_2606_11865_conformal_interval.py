"""
Conformal prediction interval width features.

PAPER: "Conformal Bayes under Label Shift: Post-Hoc Calibration vs.
In-Training Adaptation" (arxiv 2606.11865) — pure ML-calibration theory,
no extractable OHLCV feature. SKIPPED as-is.

SUBSTITUTE (same theme — conformal / nonconformity score / coverage width):
  Implements a DATA-DRIVEN CONFORMAL WIDTH family for OHLCV.

  Conformal prediction quantifies uncertainty via "nonconformity scores" and
  sets prediction intervals whose width reflects the *current* difficulty of
  prediction. We emulate this without a model by using the empirical quantile
  spread of the rolling residuals from a naïve predictor (yesterday's close).

  Concretely:
    nonconf score_t = |log_ret_t - median(log_ret over trailing 20/60 days)|
  then we track:
    - rolling (90th - 10th) pct quantile spread of these scores  → interval "width"
    - rolling 90th pct of scores                                 → upper nonconformity
    - ratio of short/long-window interval widths                 → regime shift signal
    - rank of today's nonconformity among recent scores          → surprise rank

  These are orthogonal to standard vol (ATR) because they focus on the
  *distribution of deviations from the recent median*, not just the std.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11865_conformal_interval",
    "description": (
        "Conformal-style nonconformity scores and rolling interval widths on "
        "daily log returns; inspired by conformal calibration in arxiv 2606.11865 "
        "(Conformal Bayes under Label Shift)."
    ),
    "requires": ["Close"],
    "produces": [
        "conf_width_20",
        "conf_width_60",
        "conf_q90_20",
        "conf_width_ratio",
        "conf_surprise_rank_20",
        "conf_surprise_rank_60",
    ],
    "tags": ["volatility", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11865",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # ---- Nonconformity score: |residual from rolling median| ------------------
    med20 = log_ret.rolling(20, min_periods=5).median()
    med60 = log_ret.rolling(60, min_periods=15).median()

    nc20 = (log_ret - med20).abs()
    nc60 = (log_ret - med60).abs()

    # ---- Conformal interval width: 90th - 10th percentile of recent scores ----
    def width(s: pd.Series, w: int) -> pd.Series:
        q90 = s.rolling(w, min_periods=w // 3).quantile(0.90)
        q10 = s.rolling(w, min_periods=w // 3).quantile(0.10)
        return q90 - q10

    df["conf_width_20"] = width(nc20, 20)
    df["conf_width_60"] = width(nc60, 60)

    # ---- 90th percentile nonconformity (upper tail) ---------------------------
    df["conf_q90_20"] = nc20.rolling(20, min_periods=7).quantile(0.90)

    # ---- Ratio: short / long width (regime shift) ----------------------------
    df["conf_width_ratio"] = df["conf_width_20"] / df["conf_width_60"].replace(0, np.nan)

    # ---- Surprise rank: percentile of today's score in trailing window --------
    def surprise_rank(nc: pd.Series, w: int) -> pd.Series:
        def _rank(x):
            if len(x) < 3 or not np.isfinite(x[-1]):
                return np.nan
            return float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1)
        return nc.rolling(w, min_periods=w // 3).apply(_rank, raw=True)

    df["conf_surprise_rank_20"] = surprise_rank(nc20, 20)
    df["conf_surprise_rank_60"] = surprise_rank(nc60, 60)

    return df
