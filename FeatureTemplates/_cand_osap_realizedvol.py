"""
Realized Volatility (osap_realizedvol)
--------------------------------------
Source: OpenSourceAP (Chen-Zimmermann); Ang, Hodrick, Xing, Zhang (2006)
"The Cross-Section of Volatility and Expected Returns", Journal of Finance.

Realized volatility of daily returns predicts returns negatively in the cross-section
(high-vol stocks earn lower future returns -- the "idiosyncratic volatility puzzle").
The canonical measure is the annualized standard deviation of daily log-returns
over the past month (~21 trading days). We also include:
  - A medium-term (63-day) variant for a slower-moving signal.
  - A vol trend (ratio of short / medium vol) capturing whether volatility is rising
    or falling; rising vol is associated with negative momentum.

Per-ticker proxy: this is straightforwardly per-ticker (no cross-section needed).
Predicted sign: -1 (high realized vol -> lower future returns).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_realizedvol",
    "description": (
        "Realized daily-return volatility: annualized std-dev of log-returns over "
        "21-day (1-month) and 63-day (3-month) rolling windows, plus a vol-trend ratio. "
        "High realized vol predicts lower returns (idiosyncratic vol puzzle). "
        "Per-ticker implementation; no cross-section needed. "
        "Source: OpenSourceAP (Chen-Zimmermann); Ang, Hodrick, Xing, Zhang (2006)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_realizedvol_21d",    # annualized realized vol, 21-day window
        "osap_realizedvol_63d",    # annualized realized vol, 63-day window
        "osap_realizedvol_trend",  # ratio 21d/63d: >1 = vol expanding, <1 = contracting
    ],
    "tags": ["volatility", "risk", "osap", "liquidity", "trading"],
    "version": "1.0",
    "author": "Ang, Hodrick, Xing, Zhang (2006) via OpenSourceAP (Chen-Zimmermann); block by Claude",
}

_SQRT252 = np.sqrt(252.0)
_MIN_OBS_21 = 15   # require at least this many valid returns in the 21-day window
_MIN_OBS_63 = 42   # require at least this many valid returns in the 63-day window


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute realized volatility features per ticker."""
    close = df["Close"].to_numpy(dtype=np.float64)

    # Log daily returns; first return is NaN (no prior bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # Prepend NaN for alignment with original index
    log_ret = np.concatenate([[np.nan], log_ret])

    n = len(df)
    ret_series = pd.Series(log_ret, index=df.index)

    # --- 21-day realized vol (annualized) ---
    vol_21 = (
        ret_series
        .rolling(window=21, min_periods=_MIN_OBS_21)
        .std(ddof=1)
        * _SQRT252
    )

    # --- 63-day realized vol (annualized) ---
    vol_63 = (
        ret_series
        .rolling(window=63, min_periods=_MIN_OBS_63)
        .std(ddof=1)
        * _SQRT252
    )

    # --- Vol trend: ratio of short/medium term vol ---
    # Guard against zero / NaN denominators
    vol_trend = np.where(
        (vol_63.to_numpy() > 0) & np.isfinite(vol_63.to_numpy()) & np.isfinite(vol_21.to_numpy()),
        vol_21.to_numpy() / vol_63.to_numpy(),
        np.nan,
    )

    df["osap_realizedvol_21d"] = vol_21.to_numpy()
    df["osap_realizedvol_63d"] = vol_63.to_numpy()
    df["osap_realizedvol_trend"] = vol_trend

    return df
