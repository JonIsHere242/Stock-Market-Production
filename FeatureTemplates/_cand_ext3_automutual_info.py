"""
Auto-mutual-information (AMI) of returns — nonlinear serial dependence.
Captures nonlinear autocorrelation that linear autocorrelation misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_automutual_info",
    "description": (
        "Rolling 120-day auto-mutual-information (AMI) of log-returns at lag-1 and "
        "lag-2, computed via tercile discretisation and empirical joint/marginal "
        "probability tables. MI = sum p(x,y)*log(p(x,y)/(p(x)*p(y))). Captures "
        "NONLINEAR serial dependence that linear autocorrelation misses. "
        "Pure per-ticker OHLCV proxy. Produces: lag-1 AMI, lag-2 AMI, and their "
        "ratio (lag1/lag2 momentum of nonlinear memory)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_automutual_info_lag1",
        "ext3_automutual_info_lag2",
        "ext3_automutual_info_ratio",
    ],
    "tags": ["nonlinear", "autocorrelation", "entropy", "information-theory", "returns"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom_allan_variance)",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rolling_ami(ret: np.ndarray, window: int, lag: int) -> np.ndarray:
    """
    Compute rolling AMI(ret[t], ret[t-lag]) over `window` observations.

    Discretisation: terciles computed from the rolling window itself (no
    lookahead — only past observations enter each window).
    MI = sum_{i,j} p_ij * log(p_ij / (p_i * p_j))
    Result placed at the right edge of each window (index window-1 onwards).
    """
    n = len(ret)
    out = np.full(n, np.nan)

    # minimum usable length: window observations + lag for the lagged series
    min_len = window + lag

    for t in range(min_len - 1, n):
        # window of current values (end inclusive) — length `window`
        x = ret[t - window + 1 : t + 1]          # shape (window,)
        # lagged counterpart: same positions but shifted back `lag` steps
        y = ret[t - window + 1 - lag : t + 1 - lag]  # shape (window,)

        # guard: skip if any NaN
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 9:   # need at least 9 obs to fill a 3×3 table
            continue

        xv = x[mask]
        yv = y[mask]

        # tercile boundaries from current window (causal)
        xq = np.percentile(xv, [33.33, 66.67])
        yq = np.percentile(yv, [33.33, 66.67])

        # digitise into 0,1,2
        xi = np.searchsorted(xq, xv, side="right")   # 0,1,2
        yi = np.searchsorted(yq, yv, side="right")

        m = len(xv)
        # joint frequencies (3×3)
        joint = np.zeros((3, 3), dtype=np.float64)
        for a, b in zip(xi, yi):
            joint[a, b] += 1.0
        joint /= m

        # marginals
        px = joint.sum(axis=1)   # shape (3,)
        py = joint.sum(axis=0)   # shape (3,)

        # MI
        mi = 0.0
        for a in range(3):
            for b in range(3):
                pij = joint[a, b]
                denom = px[a] * py[b]
                if pij > 0.0 and denom > 0.0:
                    mi += pij * np.log(pij / denom)

        out[t] = max(mi, 0.0)   # MI >= 0 by definition; clamp float noise

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext3_automutual_info_lag1, _lag2, _ratio columns."""

    WINDOW = 120
    EPS = 1e-10

    close = df["Close"].to_numpy(dtype=np.float64)

    # log returns (causal: ret[t] = log(Close[t]/Close[t-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            close[1:] > 0,
            np.log(np.where(close[1:] > 0, close[1:], np.nan) /
                   np.where(close[:-1] > 0, close[:-1], np.nan)),
            np.nan,
        )
    # prepend NaN for the first row so length == len(df)
    ret = np.empty(len(df), dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = log_ret

    ami1 = _rolling_ami(ret, WINDOW, lag=1)
    ami2 = _rolling_ami(ret, WINDOW, lag=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ami2 > EPS, ami1 / ami2, np.nan)

    df["ext3_automutual_info_lag1"] = ami1
    df["ext3_automutual_info_lag2"] = ami2
    df["ext3_automutual_info_ratio"] = ratio

    return df
