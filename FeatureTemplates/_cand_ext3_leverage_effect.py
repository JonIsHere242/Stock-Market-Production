"""
Leverage Effect — return-volatility asymmetry (per-ticker OHLCV proxy).

The classical leverage effect: negative equity returns predict higher future
realised volatility.  Within a rolling look-back window we correlate today's
log-return with the NEXT day's absolute return (both past-only at evaluation
time), giving a causal, per-ticker estimate of this asymmetry.

Produced columns
----------------
ext3_leverage_effect_corr60  : rolling 60-day corr(r[t], |r[t+1]|) — 1-day-ahead vol predictor
ext3_leverage_effect_corr5d  : rolling 60-day corr(r[t], avg|r[t+1..t+5]|) — 5-day-ahead vol predictor
ext3_leverage_effect_ch20    : 20-day change in ext3_leverage_effect_corr60 (momentum of the effect)

Causality note:
  |r[t+1]| for bar t is computed as abs(shift(-1)) on the FULL series, which
  would be lookahead on the last bar.  To avoid this, we instead pair
  r[t] with |r[t+1]| where "t+1" means the *next available past bar*
  at each rolling evaluation point.  Concretely we construct the lagged
  series via shift(+1) on |r| (so it aligns |r[t]| with r[t-1]), then
  within each window of size W ending at time T, the pairs are
  (r[t-1], |r[t]|) for t in [T-W+1, T].  This is identical in content to
  (r[t], |r[t+1]|) within that window but uses only bars <= T.  No future
  information leaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_leverage_effect",
    "description": (
        "Rolling return-volatility asymmetry (leverage effect). "
        "Computes rolling 60-day Pearson correlation between the log-return "
        "at bar t-1 and the absolute log-return at bar t (a causal shift so "
        "no future bars are used), capturing whether negative returns predict "
        "elevated next-day realised vol.  A second column extends this to a "
        "5-day-ahead vol proxy.  A third column is the 20-day change in the "
        "primary correlation, capturing whether the asymmetry is strengthening "
        "or fading.  Pure OHLCV, no cross-sectional dependencies."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_leverage_effect_corr60",
        "ext3_leverage_effect_corr5d",
        "ext3_leverage_effect_ch20",
    ],
    "tags": ["volatility", "asymmetry", "leverage_effect", "correlation", "ohlcv"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom2_downside_beta); spec: ext3_leverage_effect",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Log returns                                                        #
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # Guard against zero / negative prices
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # ret has length n-1; pad front with NaN to align with df rows
    ret_full = np.empty(n, dtype=np.float64)
    ret_full[0] = np.nan
    ret_full[1:] = ret

    abs_ret = np.abs(ret_full)

    # ------------------------------------------------------------------ #
    # 2. Causal shift: pair ret[t-1] with |ret[t]|                        #
    #    so the "predictor" r[t-1] and the "response" |r[t]| are both     #
    #    available at time t.                                               #
    # ------------------------------------------------------------------ #
    # x[t] = ret[t-1]  (return of previous bar, available at t)
    x = np.empty(n, dtype=np.float64)
    x[0] = np.nan
    x[1:] = ret_full[:-1]

    # y1[t] = |ret[t]|  (1-day-ahead abs return, now in the past at t)
    y1 = abs_ret.copy()

    # y5[t] = mean(|ret[t]|, |ret[t+1]|, ..., |ret[t+4]|) but causal:
    # at bar T the 5 bars [T-4 .. T] are all past, so we use a
    # backward rolling mean of |ret| with window=5.
    # Then we shift x by 4 more to align it with the START of that 5-bar
    # window (x lags y5 by 4 additional bars).
    # Concretely: x5[t] = ret[t-5], y5[t] = mean(|ret[t-4..t]|)
    y5 = _rolling_mean(abs_ret, 5)   # backward rolling mean, no lookahead
    x5 = np.empty(n, dtype=np.float64)
    x5[:5] = np.nan
    x5[5:] = ret_full[:-5]           # ret[t-5] aligned with y5[t]

    # ------------------------------------------------------------------ #
    # 3. Rolling 60-day Pearson correlation                                #
    # ------------------------------------------------------------------ #
    WIN = 60
    corr60 = _rolling_corr(x, y1, WIN)
    corr5d = _rolling_corr(x5, y5, WIN)

    # 20-day change in the primary correlation
    ch20 = np.empty(n, dtype=np.float64)
    ch20[:20] = np.nan
    ch20[20:] = corr60[20:] - corr60[:-20]

    # ------------------------------------------------------------------ #
    # 4. Assign to df                                                       #
    # ------------------------------------------------------------------ #
    df["ext3_leverage_effect_corr60"] = corr60
    df["ext3_leverage_effect_corr5d"] = corr5d
    df["ext3_leverage_effect_ch20"] = ch20

    return df


# ------------------------------------------------------------------ #
# Helpers (module-level, pure functions, no globals mutated)          #
# ------------------------------------------------------------------ #

def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Backward rolling mean, length-preserving, leading NaNs."""
    out = np.full(len(a), np.nan, dtype=np.float64)
    # Use cumsum trick for speed
    cs = np.nancumsum(a)
    cnt = np.nancumsum(~np.isnan(a)).astype(np.float64)
    for i in range(w - 1, len(a)):
        if i >= w:
            s = cs[i] - cs[i - w]
            c = cnt[i] - cnt[i - w]
        else:
            s = cs[i]
            c = cnt[i]
        out[i] = s / c if c >= 2 else np.nan
    return out


def _rolling_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorised rolling Pearson correlation between x and y with window w.
    Uses the sum-of-squares formula to avoid O(n^2) Python loops.
    Returns array of same length; leading w-1 positions are NaN.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)

    # Build valid-pair mask: both x and y must be finite
    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    vf = valid.astype(np.float64)

    # Prefix sums
    n_   = np.cumsum(vf)
    sx   = np.cumsum(xv)
    sy   = np.cumsum(yv)
    sxx  = np.cumsum(xv * xv)
    syy  = np.cumsum(yv * yv)
    sxy  = np.cumsum(xv * yv)

    for i in range(w - 1, n):
        j = i - w  # exclusive start of window
        if j >= 0:
            cnt = n_[i]  - n_[j]
            _sx  = sx[i]  - sx[j]
            _sy  = sy[i]  - sy[j]
            _sxx = sxx[i] - sxx[j]
            _syy = syy[i] - syy[j]
            _sxy = sxy[i] - sxy[j]
        else:
            cnt = n_[i]
            _sx  = sx[i]
            _sy  = sy[i]
            _sxx = sxx[i]
            _syy = syy[i]
            _sxy = sxy[i]

        if cnt < 5:
            continue

        mean_x = _sx / cnt
        mean_y = _sy / cnt
        cov   = _sxy / cnt - mean_x * mean_y
        var_x = _sxx / cnt - mean_x * mean_x
        var_y = _syy / cnt - mean_y * mean_y

        denom = np.sqrt(var_x * var_y)
        if denom > 1e-12:
            out[i] = np.clip(cov / denom, -1.0, 1.0)

    return out
