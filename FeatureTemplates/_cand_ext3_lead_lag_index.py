"""
ext3_lead_lag_index — Lead-lag relationship between the stock and SPY (Lo-MacKinlay).

Per-ticker, rolling 60-day cross-correlations:
  lag_coeff : corr(SPY_ret[t-1], stock_ret[t])  — stock lagging the market
  lead_coeff: corr(stock_ret[t-1], SPY_ret[t])  — stock leading the market
  net_lead_lag: lag_coeff - lead_coeff (positive = stock lags SPY; negative = stock leads)

Method follows Lo & MacKinlay (1990) lead-lag effect. Per-ticker OHLCV proxy using
_indexes.index_close("SPY") for market returns, merged backward-safe with merge_asof.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never via package import)
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_lead_lag_index",
    "description": (
        "Rolling 60-day cross-correlations capturing the Lo-MacKinlay lead-lag "
        "relationship between SPY and the stock. "
        "lag_coeff = corr(SPY_ret[t-1], stock_ret[t]) (stock lags market); "
        "lead_coeff = corr(stock_ret[t-1], SPY_ret[t]) (stock leads market); "
        "net_lead_lag = lag_coeff - lead_coeff. "
        "Cross-sectional lead-lag ranking is not feasible per-ticker; this is the "
        "closest faithful per-ticker proxy of the Lo-MacKinlay statistic."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_lead_lag_index_lag_coeff",
        "ext3_lead_lag_index_lead_coeff",
        "ext3_lead_lag_index_net",
    ],
    "tags": ["lead_lag", "market", "cross_correlation", "index", "lo_mackinklay"],
    "version": "1.0.0",
    "author": "Round-4 expansion spec (xdom2_autocorr_volume_return); Lo & MacKinlay (1990)",
}

# ---------------------------------------------------------------------------
# Rolling correlation helper (vectorised with stride tricks)
# ---------------------------------------------------------------------------
_WINDOW = 60
_MIN_OBS = 30  # require at least half the window to emit a value


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """
    Compute rolling Pearson correlation between x and y over `window` bars.
    x and y must be 1-D arrays of the same length, with NaN where data is missing.
    Returns an array of the same length (NaN for insufficient data).
    """
    n = len(x)
    out = np.full(n, np.nan)

    for i in range(window - 1, n):
        xs = x[i - window + 1 : i + 1]
        ys = y[i - window + 1 : i + 1]
        # mask NaNs jointly
        mask = np.isfinite(xs) & np.isfinite(ys)
        cnt = mask.sum()
        if cnt < min_obs:
            continue
        xm = xs[mask]
        ym = ys[mask]
        xd = xm - xm.mean()
        yd = ym - ym.mean()
        denom = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
        if denom == 0.0:
            continue
        out[i] = (xd * yd).sum() / denom

    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    nan_series = pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------
    # 1. Fetch SPY close, merge backward-safe
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # Series, DatetimeIndex
    except Exception:
        df["ext3_lead_lag_index_lag_coeff"] = nan_series
        df["ext3_lead_lag_index_lead_coeff"] = nan_series
        df["ext3_lead_lag_index_net"] = nan_series
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # restore original order
    work = work.set_index(df.index)

    # ------------------------------------------------------------------
    # 2. Compute daily returns (pct change; no lookahead — shift(1) is past)
    # ------------------------------------------------------------------
    stock_ret = work["Close"].pct_change()           # r_t
    spy_ret = work["spy_close"].pct_change()         # R_t

    # lagged series (t-1)
    stock_ret_lag = stock_ret.shift(1)               # r_{t-1}
    spy_ret_lag = spy_ret.shift(1)                   # R_{t-1}

    s_arr = stock_ret.to_numpy(dtype=float)
    R_arr = spy_ret.to_numpy(dtype=float)
    s_lag_arr = stock_ret_lag.to_numpy(dtype=float)
    R_lag_arr = spy_ret_lag.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 3. Rolling cross-correlations
    #    lag_coeff  : corr(R_{t-1}, r_t)  — SPY leads stock
    #    lead_coeff : corr(r_{t-1}, R_t)  — stock leads SPY
    # ------------------------------------------------------------------
    lag_coeff = _rolling_corr(R_lag_arr, s_arr, _WINDOW, _MIN_OBS)
    lead_coeff = _rolling_corr(s_lag_arr, R_arr, _WINDOW, _MIN_OBS)
    net = np.where(
        np.isfinite(lag_coeff) & np.isfinite(lead_coeff),
        lag_coeff - lead_coeff,
        np.nan,
    )

    df["ext3_lead_lag_index_lag_coeff"] = lag_coeff
    df["ext3_lead_lag_index_lead_coeff"] = lead_coeff
    df["ext3_lead_lag_index_net"] = net

    return df
