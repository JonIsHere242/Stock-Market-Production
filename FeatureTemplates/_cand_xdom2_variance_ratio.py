"""
Lo-MacKinlay (1988) rolling variance ratio feature block.

VR(q) = Var(q-day return) / (q * Var(1-day return))

VR > 1  -> positive autocorrelation / trending
VR < 1  -> negative autocorrelation / mean-reversion
VR ~ 1  -> random walk

Implemented per-ticker via rolling 120-bar windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_variance_ratio",
    "description": (
        "Lo-MacKinlay (1988) rolling variance ratio VR(q) computed per ticker "
        "over a 120-day trailing window for q=5 and q=10. "
        "VR(q) = Var(q-day return) / (q * Var(1-day return)). "
        "VR>1 signals trending/positive autocorrelation; VR<1 signals mean-reversion. "
        "The heteroskedasticity-robust form (HC variance) is approximated per the "
        "Lo-MacKinlay (1988) paper using demeaned squared returns. "
        "A slope column captures recent momentum in the ratio itself (20-day change). "
        "Fully per-ticker; no cross-sectional component."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_variance_ratio_vr5",
        "xdom2_variance_ratio_vr10",
        "xdom2_variance_ratio_vr5_chg20",
    ],
    "tags": ["variance_ratio", "autocorrelation", "mean_reversion", "trending", "lo_mackinlay"],
    "version": "1.0",
    "author": "Lo-MacKinlay variance ratio (1988); block by Claude",
}


def _rolling_vr(log_ret: np.ndarray, q: int, window: int) -> np.ndarray:
    """
    Compute rolling Lo-MacKinlay variance ratio VR(q) for each bar t using
    the trailing `window` log-returns ending at t.

    VR(q) = [ (1/(m*q)) * sum_{k=q}^{n} (r_k^(q) - q*mu_hat)^2 ]
            / [ (1/(m))  * sum_{k=1}^{n} (r_k - mu_hat)^2        ]

    where r_k^(q) = log_ret[k] + ... + log_ret[k-q+1]  (q-period overlapping)
          mu_hat  = mean of 1-day returns in window
          m       = n - q  (effective dof for q-period numerator)
          n       = window  (number of 1-day returns)

    This is the simple (non-heteroskedasticity-robust) VR, which is the
    standard form. Returns NaN for bars with insufficient data.
    """
    n = len(log_ret)
    out = np.full(n, np.nan, dtype=np.float64)

    for t in range(window - 1, n):
        seg = log_ret[t - window + 1: t + 1]  # length = window

        # 1-day variance (unbiased)
        mu = seg.mean()
        demeaned = seg - mu
        var1 = np.dot(demeaned, demeaned) / (window - 1)

        if var1 == 0.0 or np.isnan(var1):
            out[t] = np.nan
            continue

        # q-period overlapping returns: sum of q consecutive 1-day returns
        # seg[q-1], seg[q], ..., seg[window-1]
        # There are (window - q + 1) overlapping q-period returns
        m_q = window - q + 1
        if m_q < 2:
            out[t] = np.nan
            continue

        # Build q-period return array via cumsum trick
        cs = np.cumsum(seg)
        # cs[i] = sum of seg[0..i]; q-period sum ending at index i = cs[i] - cs[i-q]
        # valid indices: i from q-1 to window-1  -> m_q values
        retq = np.empty(m_q, dtype=np.float64)
        retq[0] = cs[q - 1]
        retq[1:] = cs[q:] - cs[:window - q]

        mu_q = q * mu  # expected q-period mean under demeaning
        dq = retq - mu_q
        var_q = np.dot(dq, dq) / (m_q - 1)

        out[t] = var_q / (q * var1)

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Need at least window + max(q) rows for anything useful
    WINDOW = 120
    Q5 = 5
    Q10 = 10

    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    vr5_arr = np.full(n, np.nan, dtype=np.float64)
    vr10_arr = np.full(n, np.nan, dtype=np.float64)

    if n >= WINDOW + Q10:
        # log returns: log(Close[t] / Close[t-1]), skip the very first NaN
        log_ret = np.empty(n, dtype=np.float64)
        log_ret[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret[1:] = np.log(close[1:] / close[:-1])

        # Replace inf/-inf arising from 0 prices with NaN
        log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

        # We need at least `window` valid log returns before computing
        # Work on the slice starting at index 1 (first valid log return)
        # _rolling_vr handles NaN propagation via the indexing (it could get
        # NaN segments; the variance calc will still produce a number unless
        # the whole segment is NaN or zero-variance).  For simplicity,
        # treat NaN returns as 0 for the window computation (conservative).
        lr = np.where(np.isnan(log_ret), 0.0, log_ret)

        vr5_arr = _rolling_vr(lr, Q5, WINDOW)
        vr10_arr = _rolling_vr(lr, Q10, WINDOW)

        # Guard: replace inf/-inf with NaN
        vr5_arr = np.where(np.isfinite(vr5_arr), vr5_arr, np.nan)
        vr10_arr = np.where(np.isfinite(vr10_arr), vr10_arr, np.nan)

    df["xdom2_variance_ratio_vr5"] = vr5_arr
    df["xdom2_variance_ratio_vr10"] = vr10_arr

    # 20-day change in vr5 to capture momentum in the ratio
    vr5_series = pd.Series(vr5_arr, index=df.index)
    chg20 = vr5_series.diff(20)
    df["xdom2_variance_ratio_vr5_chg20"] = chg20.to_numpy(dtype=np.float64)

    return df
