"""
Uncertainty-First Forecasting of the South African Equity Market Using Deep
Learning and Temporal Conformal Prediction
DOI: 10.3390/bdcc10030093  (Big Data and Cognitive Computing, 2026)

The paper uses Variational Mode Decomposition (VMD) to isolate latent frequency
components, feeds them into a GRU, then wraps the result in a Temporal Conformal
Prediction (TCP) framework to produce distribution-free prediction INTERVALS.

PROXY IMPLEMENTED: Multi-scale decomposition via moving-average (MA) residuals at
three frequency bands (short / medium / long), mirroring what VMD separates.
Rolling prediction interval WIDTH is proxied by EWMA conditional volatility
(|residual|) at each scale. A volatility-regime indicator flags high-uncertainty
regimes where interval width is elevated — the paper's chief contribution is
distinguishing calm vs stressed regimes.

DROPPED: VMD algorithm (requires scipy.signal.windows + iterative optimisation
beyond pure numpy), GRU model, conformal calibration (requires held-out labels).
MA residuals are the closest causal OHLCV stand-in for VMD modes.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_bdcc10030093_uncertainty_south",
    "description": (
        "Multi-scale MA-residual decomposition (short/medium/long bands) + EWMA "
        "conditional-vol interval-width proxies for the VMD-GRU-TCP uncertainty "
        "framework of Govender et al. (2026); VMD, GRU, and conformal calibration "
        "dropped — pure OHLCV MA residuals only."
    ),
    "requires": ["Close"],
    "produces": [
        "ufs_ma_resid_short_10",
        "ufs_ma_resid_medium_30",
        "ufs_ma_resid_long_60",
        "ufs_ewma_vol_short_10",
        "ufs_ewma_vol_medium_30",
        "ufs_ewma_vol_long_60",
        "ufs_interval_width_proxy_10",
        "ufs_interval_width_proxy_30",
        "ufs_vol_regime_flag",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "proxy: paper DOI:10.3390/bdcc10030093 (Govender et al., 2026); VMD/GRU/TCP dropped",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-scale decomposition proxies.

    Each MA residual = Close - MA(window), i.e. the deviation of price from the
    trend component captured by the MA, analogous to a VMD mode at that frequency.
    EWMA of |residual| is a causal, rolling estimate of conditional volatility at
    each scale.  The interval-width proxy = 2 * 1.96 * ewma_vol (a 95% Gaussian
    interval from that scale's vol) mimicking TCP interval widths.
    Vol regime flag: 1 if short-scale vol is above its own 60-day rolling median.
    """
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))  # daily log return, causal

    for window in (10, 30, 60):
        ma     = close.rolling(window, min_periods=window).mean()
        resid  = (close - ma) / ma.replace(0.0, float("nan"))  # normalised residual

        # EWMA conditional vol of the residual at this scale
        abs_resid = resid.abs()
        ewma_vol  = abs_resid.ewm(span=window, min_periods=window // 2, adjust=False).mean()

        tag = {10: "short_10", 30: "medium_30", 60: "long_60"}[window]
        df[f"ufs_ma_resid_{tag}"]  = resid
        df[f"ufs_ewma_vol_{tag}"]  = ewma_vol

    # Interval-width proxy = 2 * 1.96 * ewma_vol  (95% symmetric interval)
    df["ufs_interval_width_proxy_10"] = 2.0 * 1.96 * df["ufs_ewma_vol_short_10"]
    df["ufs_interval_width_proxy_30"] = 2.0 * 1.96 * df["ufs_ewma_vol_medium_30"]

    # Volatility regime flag: 1 when short-scale vol > its 60-day rolling median
    vol_short  = df["ufs_ewma_vol_short_10"]
    median_60  = vol_short.rolling(60, min_periods=30).median()
    df["ufs_vol_regime_flag"] = (vol_short > median_60).astype(float)
    # Where median is NaN, set flag to NaN
    df.loc[median_60.isna(), "ufs_vol_regime_flag"] = float("nan")

    return df
