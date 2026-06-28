"""
Short-term reversal feature block.

Spec ID : osap_streversal
Source  : OpenSourceAP (Chen-Zimmermann); method from Jegadeesh (1990)
          "Evidence of Predictable Behavior of Security Returns", JF 45(3).

The canonical cross-sectional signal is last-month total return (approximately
t-21 to t-1 trading days), predicted sign -1 (past losers outperform past
winners in the following month).  Here we implement the signal per-ticker
directly — no cross-sectional rank is possible at compute time — and add a
volume-weighted variant and a rolling z-score of the reversal signal to
capture the strength/persistence of the mean-reversion pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA: dict = {
    "name": "osap_streversal",
    "description": (
        "Short-term (1-month) price reversal factor per Jegadeesh (1990). "
        "osap_streversal_ret1m: raw 21-trading-day log return (predicted sign -1 "
        "for next-period outperformance of losers). "
        "osap_streversal_volwt: volume-weighted version — weights each daily "
        "return by relative volume within the 21-day window, emphasising days "
        "with heavier participation. "
        "osap_streversal_zscore: rolling 252-day z-score of ret1m, capturing how "
        "extreme the current reversal signal is relative to its own history. "
        "Per-ticker proxy; the original is cross-sectional but the economic "
        "signal (recent-return mean reversion) is captured faithfully within "
        "each series."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_streversal_ret1m",
        "osap_streversal_volwt",
        "osap_streversal_zscore",
    ],
    "tags": ["reversal", "momentum", "price", "short_term", "osap"],
    "version": "1.0.0",
    "author": "Jegadeesh 1990 / OpenSourceAP Chen-Zimmermann; block by Claude",
}

_WINDOW = 21       # ~1 calendar month
_ZS_WIN = 252      # rolling z-score look-back


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Same frame with three new columns appended.
    """
    close = df["Close"].astype(np.float64)
    volume = df["Volume"].astype(np.float64)

    # Guard: replace non-positive close with NaN to avoid log(<=0)
    close = close.where(close > 0, np.nan)

    # ------------------------------------------------------------------
    # 1.  21-day log return  (approx 1 calendar month)
    #     log(Close_t / Close_{t-21})
    #     shift(21) uses only past data — no lookahead.
    # ------------------------------------------------------------------
    log_close = np.log(close)
    ret1m = log_close - log_close.shift(_WINDOW)

    # ------------------------------------------------------------------
    # 2.  Volume-weighted 21-day log return
    #     daily_ret_i * (volume_i / sum(volume_{t-20..t}))  summed over window
    #
    #     daily_log_ret[t] = log(Close[t] / Close[t-1])
    #     weight[t]        = volume[t] / rolling_21d_sum_volume
    #     volwt_ret[t]     = rolling sum of (daily_log_ret * weight) — but
    #                        the weight denominator changes per window, so we
    #                        compute: sum(vol_i * dlr_i) / sum(vol_i) over 21d
    # ------------------------------------------------------------------
    daily_log_ret = log_close.diff(1)           # log(Close_t / Close_{t-1})

    # numerator: rolling sum of volume * daily_log_ret
    vol_x_ret = volume * daily_log_ret          # element-wise; NaN where daily_ret is NaN
    roll_vol = volume.rolling(_WINDOW, min_periods=_WINDOW).sum()
    roll_vxr = vol_x_ret.rolling(_WINDOW, min_periods=_WINDOW).sum()

    # Avoid division by zero
    volwt = roll_vxr / roll_vol.where(roll_vol > 0, np.nan)

    # ------------------------------------------------------------------
    # 3.  Rolling 252-day z-score of ret1m
    #     How extreme is this month's return relative to its own history?
    # ------------------------------------------------------------------
    roll_mean = ret1m.rolling(_ZS_WIN, min_periods=20).mean()
    roll_std  = ret1m.rolling(_ZS_WIN, min_periods=20).std(ddof=1)
    zscore = (ret1m - roll_mean) / roll_std.where(roll_std > 0, np.nan)

    # ------------------------------------------------------------------
    # Assign — replace any stray inf with NaN
    # ------------------------------------------------------------------
    df["osap_streversal_ret1m"]  = ret1m.replace([np.inf, -np.inf], np.nan)
    df["osap_streversal_volwt"]  = volwt.replace([np.inf, -np.inf], np.nan)
    df["osap_streversal_zscore"] = zscore.replace([np.inf, -np.inf], np.nan)

    return df
