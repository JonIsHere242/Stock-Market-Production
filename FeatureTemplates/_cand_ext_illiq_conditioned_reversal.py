"""
Illiquidity-conditioned return autocorrelation.

Computes daily Amihud illiquidity |ret| / (Close * Volume), splits the
trailing 90-day window into terciles, and produces:
  - lag-1 return autocorrelation among HIGH-illiquidity days
  - high-minus-low illiquidity autocorrelation spread

The spread captures the CGW insight that illiquid stocks exhibit stronger
short-horizon reversal (negative autocorrelation) than liquid stocks.
Per-ticker proxy -- cross-sectional ranking of the spread is done downstream.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_illiq_conditioned_reversal",
    "description": (
        "Amihud-illiquidity-conditioned lag-1 return autocorrelation. "
        "Splits trailing 90-day window by Amihud illiquidity terciles, then "
        "computes lag-1 autocorr among HIGH-illiq days and among LOW-illiq "
        "days separately. Produces (a) high-illiq autocorr and (b) the "
        "high-minus-low spread (CGW: illiquid names reverse more strongly). "
        "Per-ticker rolling proxy; no cross-sectional data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_illiq_conditioned_reversal_hi_autocorr",
        "ext_illiq_conditioned_reversal_spread",
    ],
    "tags": ["reversal", "illiquidity", "amihud", "autocorrelation", "microstructure"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension of xdom2_autocorr_volume_return (gate-validated winner). "
        "Theoretical basis: Amihud (2002) illiquidity ratio; "
        "CGW = Chordia, Goyal, Wahal (illiquid stocks reverse more)."
    ),
}

_WINDOW = 90       # rolling look-back (days)
_MIN_OBS = 12      # minimum observations per tercile group to emit a value


def _lag1_autocorr(arr: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D array; NaN if too few points."""
    n = len(arr)
    if n < 3:
        return np.nan
    x = arr[:-1]
    y = arr[1:]
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    denom = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    if denom == 0.0:
        return np.nan
    return num / denom


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n = len(df)

    # Daily log return (NaN for row 0)
    ret = np.empty(n, dtype=float)
    ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.log(close[1:] / close[:-1])

    # Amihud illiquidity = |ret| / (Close * Volume); 0 denom -> NaN
    dollar_vol = close * volume
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = np.where(dollar_vol > 0.0, np.abs(ret) / dollar_vol, np.nan)

    hi_autocorr_vals = np.full(n, np.nan)
    spread_vals = np.full(n, np.nan)

    for t in range(_WINDOW - 1, n):
        # Slice the window [t-WINDOW+1 .. t] inclusive
        w_start = t - _WINDOW + 1
        w_ret = ret[w_start : t + 1]       # length = _WINDOW
        w_amihud = amihud[w_start : t + 1]

        # Only keep rows where both ret and amihud are finite
        mask = np.isfinite(w_ret) & np.isfinite(w_amihud)
        r_valid = w_ret[mask]
        a_valid = w_amihud[mask]

        if len(r_valid) < _MIN_OBS * 2:
            continue  # not enough data

        # Tercile boundaries (nanpercentile on valid subset)
        t33 = np.percentile(a_valid, 100.0 / 3.0)
        t67 = np.percentile(a_valid, 200.0 / 3.0)

        hi_mask = a_valid >= t67
        lo_mask = a_valid <= t33

        r_hi = r_valid[hi_mask]
        r_lo = r_valid[lo_mask]

        if len(r_hi) >= _MIN_OBS:
            hi_autocorr_vals[t] = _lag1_autocorr(r_hi)

        if len(r_hi) >= _MIN_OBS and len(r_lo) >= _MIN_OBS:
            lo_ac = _lag1_autocorr(r_lo)
            hi_ac = hi_autocorr_vals[t]
            if np.isfinite(hi_ac) and np.isfinite(lo_ac):
                spread_vals[t] = hi_ac - lo_ac

    df["ext_illiq_conditioned_reversal_hi_autocorr"] = hi_autocorr_vals
    df["ext_illiq_conditioned_reversal_spread"] = spread_vals
    return df
