"""
Garman-Klass vs close-to-close volatility divergence regime block.

GK vol uses the full OHLC range; CC vol uses only close-to-close moves.
Their ratio captures whether intraday range is large relative to overnight
moves -- a regime / microstructure efficiency signal.

Per-ticker implementation (no cross-sectional ranking needed; all math is
own OHLCV time-series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_garman_klass_regime",
    "description": (
        "Rolling 20-day Garman-Klass volatility divided by close-to-close "
        "volatility (gk_cc_ratio_20). GK > CC means intraday range is wide "
        "relative to the open-to-open drift component -- a proxy for "
        "mean-reverting intraday churn / micro-structure friction. Produces "
        "the ratio level and its 20-day change (regime shift). Per-ticker "
        "only; no cross-sectional component."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "xdom_garman_klass_regime_ratio20",   # GK_vol / CC_vol, 20-day rolling
        "xdom_garman_klass_regime_delta20",   # 20-day change in the ratio
        "xdom_garman_klass_regime_gk20",      # GK vol itself (annualised), useful standalone
    ],
    "tags": ["volatility", "regime", "microstructure", "cross-domain", "range-estimator"],
    "version": "1.0.0",
    "author": (
        "Range-estimator divergence (Garman-Klass vs close-to-close vol); "
        "SOURCE: Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "Spec: xdom_garman_klass_regime"
    ),
}

_WINDOW = 20
_TRADING_DAYS = 252


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Garman-Klass / close-to-close vol ratio on a 20-day rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock panel, ascending Date, columns include OHLCV.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    log_hl = np.log(
        df["High"].replace(0, np.nan) / df["Low"].replace(0, np.nan)
    )
    log_co = np.log(
        df["Close"].replace(0, np.nan) / df["Open"].replace(0, np.nan)
    )

    # Garman-Klass daily variance estimate (per-bar):
    #   gk_daily = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
    gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    # Clamp negatives (can occur for tiny intraday ranges) to zero before sqrt
    gk_daily = gk_daily.clip(lower=0.0)

    # Rolling mean of per-day GK variance, then annualise
    gk_var_20 = gk_daily.rolling(window=_WINDOW, min_periods=_WINDOW).mean()
    gk_vol_20 = np.sqrt(gk_var_20 * _TRADING_DAYS)  # annualised GK vol

    # Close-to-close log return
    log_ret = np.log(
        df["Close"].replace(0, np.nan) / df["Close"].replace(0, np.nan).shift(1)
    )

    # Rolling CC variance (sample, ddof=1) then annualise
    cc_var_20 = log_ret.rolling(window=_WINDOW, min_periods=_WINDOW).var(ddof=1)
    cc_vol_20 = np.sqrt(cc_var_20 * _TRADING_DAYS)

    # Ratio: GK_vol / CC_vol  -- guard against zero denominator
    denom = cc_vol_20.replace(0.0, np.nan)
    ratio20 = gk_vol_20 / denom
    # Clip extreme outliers (e.g. near-zero CC vol days; ratio can blow up)
    ratio20 = ratio20.clip(upper=20.0)

    # 20-day change in the ratio (regime shift signal)
    delta20 = ratio20 - ratio20.shift(_WINDOW)

    df["xdom_garman_klass_regime_ratio20"] = ratio20
    df["xdom_garman_klass_regime_delta20"] = delta20
    df["xdom_garman_klass_regime_gk20"] = gk_vol_20

    return df
