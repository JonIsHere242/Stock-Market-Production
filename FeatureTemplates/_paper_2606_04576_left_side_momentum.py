"""
Left-side (negative-tail) momentum features per-ticker proxy derived from:
  "ReSGA: A Large Tail Risk Model for Learning Value-at-Risk and Expected Shortfall"
  (arxiv:2606.04576).

The paper introduces a "size-enhanced left-side momentum strategy" built on the
left tail of the return distribution (negative returns only).  The key insight is
that the SEQUENCE of left-tail returns carries its own momentum signal distinct
from full-distribution momentum: stocks with persistent negative tail experiences
differ in behaviour from those whose bad returns are isolated.

Per-ticker proxy: at each bar t we collect only the subset of daily log-returns
that are NEGATIVE over a rolling window, then compute:
  - mean (intensity of recent negative experiences)
  - std (volatility of bad days)
  - count / fraction (frequency of bad days)
  - momentum slope via linear trend of negative-day returns (are they worsening?)
  - autocorrelation of negative returns (clustering of bad streaks)
  - left-tail Sharpe: |mean_neg| / std_neg (risk per unit of negative experience)

Multiple windows (20, 60, 120 days) capture short- vs medium-term left-tail dynamics.
All rolling, causal, no lookahead.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name":        "_paper_2606_04576_left_side_momentum",
    "description": (
        "Left-side negative-return momentum: rolling statistics computed exclusively "
        "on the negative-return subset (frequency, intensity, trend, autocorrelation) "
        "across 20/60/120-day windows; proxy for arXiv 2606.04576 left-side momentum."
    ),
    "requires":    ["Close"],
    "produces":    [
        "lsm_neg_mean_20",
        "lsm_neg_mean_60",
        "lsm_neg_mean_120",
        "lsm_neg_std_20",
        "lsm_neg_std_60",
        "lsm_neg_frac_20",
        "lsm_neg_frac_60",
        "lsm_neg_frac_120",
        "lsm_neg_trend_60",
        "lsm_neg_acf1_60",
        "lsm_neg_sharpe_60",
    ],
    "tags":        ["experimental", "momentum", "tail_risk", "returns"],
    "version":     "1.0",
    "author":      "paper-mining slate 1",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute left-side negative-return momentum features.

    log_ret[t] = log(Close[t] / Close[t-1]) — causal daily log-return.
    At each row t we look at the trailing window of log-returns and
    isolate the subset where log_ret < 0 (negative days).  All statistics
    are derived from that left-tail subset.
    """
    n = len(df)
    close = df["Close"].to_numpy(dtype=np.float64)

    # Causal log-returns: ret[t] uses Close[t] and Close[t-1]
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # --- Pre-allocate output arrays (NaN) ---
    neg_mean_20  = np.full(n, np.nan)
    neg_mean_60  = np.full(n, np.nan)
    neg_mean_120 = np.full(n, np.nan)
    neg_std_20   = np.full(n, np.nan)
    neg_std_60   = np.full(n, np.nan)
    neg_frac_20  = np.full(n, np.nan)
    neg_frac_60  = np.full(n, np.nan)
    neg_frac_120 = np.full(n, np.nan)
    neg_trend_60 = np.full(n, np.nan)   # OLS slope of negative-day returns (60d)
    neg_acf1_60  = np.full(n, np.nan)   # lag-1 autocorrelation of neg-day returns (60d)
    neg_sharpe60 = np.full(n, np.nan)   # |mean_neg| / std_neg (60d)

    MIN_NEG = 3  # min negative-day observations needed to compute a stat

    for t in range(1, n):
        # ---- 20-day window ----
        if t >= 20:
            sl20 = log_ret[t - 20 + 1: t + 1]        # 20 returns ending at t (inclusive)
            valid20 = sl20[np.isfinite(sl20)]
            neg20 = valid20[valid20 < 0.0]
            if len(valid20) > 0:
                neg_frac_20[t] = len(neg20) / len(valid20)
            if len(neg20) >= MIN_NEG:
                neg_mean_20[t] = neg20.mean()
                if neg20.std(ddof=1) > 1e-12:
                    neg_std_20[t] = neg20.std(ddof=1)

        # ---- 60-day window ----
        if t >= 60:
            sl60 = log_ret[t - 60 + 1: t + 1]
            valid60 = sl60[np.isfinite(sl60)]
            neg60 = valid60[valid60 < 0.0]
            if len(valid60) > 0:
                neg_frac_60[t] = len(neg60) / len(valid60)
            if len(neg60) >= MIN_NEG:
                m60 = neg60.mean()
                s60 = neg60.std(ddof=1) if len(neg60) > 1 else np.nan
                neg_mean_60[t] = m60
                if np.isfinite(s60) and s60 > 1e-12:
                    neg_std_60[t] = s60
                    neg_sharpe60[t] = abs(m60) / s60   # |mean| / std

                # OLS slope of the sequence of negative-day returns (are bad days worsening?)
                # Use positional index within the negative subset
                if len(neg60) >= 4:
                    xs = np.arange(len(neg60), dtype=np.float64)
                    xs_c = xs - xs.mean()
                    ys_c = neg60 - neg60.mean()
                    denom = (xs_c * xs_c).sum()
                    if denom > 1e-12:
                        neg_trend_60[t] = (xs_c * ys_c).sum() / denom

                # Lag-1 autocorrelation of negative-day returns
                if len(neg60) >= 4:
                    y0 = neg60[:-1]
                    y1 = neg60[1:]
                    y0c = y0 - y0.mean()
                    y1c = y1 - y1.mean()
                    num = (y0c * y1c).sum()
                    den = np.sqrt((y0c * y0c).sum() * (y1c * y1c).sum())
                    if den > 1e-12:
                        neg_acf1_60[t] = num / den

        # ---- 120-day window ----
        if t >= 120:
            sl120 = log_ret[t - 120 + 1: t + 1]
            valid120 = sl120[np.isfinite(sl120)]
            neg120 = valid120[valid120 < 0.0]
            if len(valid120) > 0:
                neg_frac_120[t] = len(neg120) / len(valid120)
            if len(neg120) >= MIN_NEG:
                neg_mean_120[t] = neg120.mean()

    df["lsm_neg_mean_20"]   = neg_mean_20
    df["lsm_neg_mean_60"]   = neg_mean_60
    df["lsm_neg_mean_120"]  = neg_mean_120
    df["lsm_neg_std_20"]    = neg_std_20
    df["lsm_neg_std_60"]    = neg_std_60
    df["lsm_neg_frac_20"]   = neg_frac_20
    df["lsm_neg_frac_60"]   = neg_frac_60
    df["lsm_neg_frac_120"]  = neg_frac_120
    df["lsm_neg_trend_60"]  = neg_trend_60
    df["lsm_neg_acf1_60"]   = neg_acf1_60
    df["lsm_neg_sharpe_60"] = neg_sharpe60

    return df
