"""
Candidate block: xdom_mutual_info_vol
Rolling mutual information between log-volume and |return|,
discretized into terciles via rolling quantiles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_mutual_info_vol",
    "description": (
        "Per-ticker proxy for rolling volume<->volatility mutual information (MI). "
        "Within a 100-day rolling window, discretizes log-volume and |log-return| each "
        "into terciles using rolling quantile boundaries, then computes the empirical "
        "joint MI = sum p(x,y)*log(p(x,y)/(p(x)*p(y))) over the 3x3 joint distribution. "
        "Captures nonlinear volume-volatility coupling beyond linear correlation. "
        "Inherently a per-ticker time-series rolling estimate (no cross-section needed). "
        "Also produces a 20d slope of MI to detect regime changes in coupling strength."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "xdom_mutual_info_vol_mi100",     # rolling 100d MI between log-vol and |ret|
        "xdom_mutual_info_vol_mi100_slope",  # 20d slope of MI (trend in coupling)
        "xdom_mutual_info_vol_lincorr100",   # linear corr as a baseline contrast
    ],
    "tags": ["volume", "volatility", "mutual_information", "information_theory", "cross_domain"],
    "version": "1.0",
    "author": (
        "Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "spec: Volume<->volatility mutual information (information-theoretic dependence)"
    ),
}

_WINDOW = 100
_BINS = 3  # terciles


def _rolling_mi_lincorr(log_vol: np.ndarray, abs_ret: np.ndarray, window: int, bins: int):
    """
    Compute rolling mutual information and linear correlation for two 1-D arrays.
    Returns arrays of length n (NaN for first window-1 entries).
    Uses a Python loop over windows -- O(n * window * bins^2) but window=100, bins=3,
    n~700 is well within 100ms budget.
    """
    n = len(log_vol)
    mi_out = np.full(n, np.nan)
    corr_out = np.full(n, np.nan)

    for i in range(window - 1, n):
        x = log_vol[i - window + 1: i + 1]   # log volume slice
        y = abs_ret[i - window + 1: i + 1]    # |return| slice

        # Check sufficient valid data
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < window // 2:
            continue

        xv = x[valid]
        yv = y[valid]

        # Rolling quantile boundaries for tercile discretization
        x_q1, x_q2 = np.percentile(xv, [100 / bins, 200 / bins])
        y_q1, y_q2 = np.percentile(yv, [100 / bins, 200 / bins])

        # Digitize into 0/1/2 bins
        xb = np.digitize(xv, [x_q1, x_q2])   # 0,1,2
        yb = np.digitize(yv, [y_q1, y_q2])   # 0,1,2

        m = len(xv)
        # Joint and marginal counts
        joint = np.zeros((bins, bins), dtype=np.float64)
        for a, b in zip(xb, yb):
            joint[a, b] += 1.0

        joint /= m
        px = joint.sum(axis=1)   # marginal over y
        py = joint.sum(axis=0)   # marginal over x

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for a in range(bins):
            for b in range(bins):
                pxy = joint[a, b]
                denom = px[a] * py[b]
                if pxy > 0.0 and denom > 0.0:
                    mi += pxy * np.log(pxy / denom)

        mi_out[i] = mi

        # Linear correlation for contrast
        if xv.std() > 0 and yv.std() > 0:
            corr_out[i] = np.corrcoef(xv, yv)[0, 1]

    return mi_out, corr_out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # --- inputs ---
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # log-return (|ret|) -- no lookahead: use only past prices
    # ret[t] = log(Close[t] / Close[t-1]); first entry is NaN
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(
            np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
        )
    abs_ret = np.abs(log_ret)

    # log-volume
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.where(volume > 0, np.log(volume), np.nan)

    # --- rolling MI and linear correlation ---
    mi100, lincorr100 = _rolling_mi_lincorr(log_vol, abs_ret, _WINDOW, _BINS)

    # --- 20d slope of MI ---
    mi_series = pd.Series(mi100)
    slope20 = mi_series.rolling(20, min_periods=10).apply(
        lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if np.isfinite(w).sum() >= 5 else np.nan,
        raw=True,
    ).to_numpy()

    # Guard: replace inf
    mi100 = np.where(np.isfinite(mi100), mi100, np.nan)
    slope20 = np.where(np.isfinite(slope20), slope20, np.nan)
    lincorr100 = np.where(np.isfinite(lincorr100), lincorr100, np.nan)

    df["xdom_mutual_info_vol_mi100"] = mi100
    df["xdom_mutual_info_vol_mi100_slope"] = slope20
    df["xdom_mutual_info_vol_lincorr100"] = lincorr100

    return df
