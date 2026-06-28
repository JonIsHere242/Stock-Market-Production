"""
Kyle-lambda price-impact elasticity block.

Spec: ext_volume_price_impact
Source: Extension of xdom2_autocorr_volume_return (gate-validated winner).
Method: Rolling 60-day OLS slopes capturing PRICE IMPACT (elasticity), orthogonal
to the parent's contemporaneous correlation signal.

  1. ext_volume_price_impact_abs_slope  -- 60d OLS beta of |return| ~ detrended log-vol
     (Kyle-lambda proxy: price impact per unit volume shock)
  2. ext_volume_price_impact_signed_slope -- 60d OLS beta of signed return ~ signed vol
     (direction-conditional impact: does big buying actually push price up?)
  3. ext_volume_price_impact_slope_chg  -- 20d change in abs_slope
     (is market-impact rising = thinning liquidity / smart-money accumulating?)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext_volume_price_impact",
    "description": (
        "Kyle-lambda price-impact elasticity (slope, not correlation). "
        "Rolling 60d OLS slope of |return| on detrended log-volume gives the "
        "absolute price-impact coefficient (Kyle-lambda proxy). Rolling OLS slope "
        "of signed return on signed volume captures direction-conditional impact. "
        "A 20d change in the abs slope measures liquidity-thinning dynamics. "
        "All three are per-ticker time-series and orthogonal to the contemporaneous "
        "correlation captured by xdom2_autocorr_volume_return (the parent feature). "
        "Per-ticker proxy -- no cross-sectional ranking needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_volume_price_impact_abs_slope",
        "ext_volume_price_impact_signed_slope",
        "ext_volume_price_impact_slope_chg",
    ],
    "tags": ["volume", "price_impact", "liquidity", "kyle_lambda", "ols", "slope"],
    "version": "1.0.0",
    "author": (
        "Spec: ext_volume_price_impact / extension of xdom2_autocorr_volume_return "
        "(gate-validated winner). Kyle-lambda concept: Kyle (1985) 'Continuous Auctions "
        "and Insider Trading'."
    ),
}

# Window constants
_LONG_WIN = 60    # OLS regression window (days)
_TREND_WIN = 20   # Volume detrending window (rolling mean of log-vol)
_CHNG_WIN  = 20   # Lookback for slope-change feature


def _rolling_ols_slope(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling OLS slope of y ~ x (with intercept) over `window` bars.
    Returns an array of length len(y) with NaN for the first (window-1) entries.
    Fully vectorised via sliding_window_view -- O(n * window) but numpy-fast.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    y_wins = sliding_window_view(y, window)   # shape (n-window+1, window)
    x_wins = sliding_window_view(x, window)

    # OLS slope = cov(x,y) / var(x) for each window
    x_mean = x_wins.mean(axis=1)
    y_mean = y_wins.mean(axis=1)

    xd = x_wins - x_mean[:, None]
    yd = y_wins - y_mean[:, None]

    cov_xy = (xd * yd).sum(axis=1)
    var_x  = (xd * xd).sum(axis=1)

    # Guard divide-by-zero
    slope = np.where(var_x == 0, np.nan, cov_xy / var_x)

    out[window - 1:] = slope
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Need at least 2 rows to compute returns
    n = len(df)

    abs_slope_col    = "ext_volume_price_impact_abs_slope"
    signed_slope_col = "ext_volume_price_impact_signed_slope"
    slope_chg_col    = "ext_volume_price_impact_slope_chg"

    if n < _LONG_WIN + 1:
        df[abs_slope_col]    = np.nan
        df[signed_slope_col] = np.nan
        df[slope_chg_col]    = np.nan
        return df

    close  = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # --- Daily log return (t vs t-1), length n with NaN at index 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.where(close > 0, np.log(close), np.nan)
    ret = np.empty(n)
    ret[0] = np.nan
    ret[1:] = log_close[1:] - log_close[:-1]

    # --- Signed volume: positive when return > 0, negative otherwise
    sign_ret = np.sign(ret)          # {-1, 0, 1}, NaN propagates as 0 via np.sign(nan)
    # Fix: np.sign(nan) gives 0; make it nan explicitly
    sign_ret = np.where(np.isnan(ret), np.nan, sign_ret)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.where(volume > 0, np.log(volume), np.nan)

    signed_vol = sign_ret * volume    # signed raw volume

    # --- Detrend log-volume: subtract rolling mean (TREND_WIN) to remove secular trend
    log_vol_series = pd.Series(log_vol)
    log_vol_trend  = log_vol_series.rolling(_TREND_WIN, min_periods=_TREND_WIN).mean().to_numpy()
    detrended_log_vol = log_vol - log_vol_trend   # NaN where trend not yet available

    # --- Absolute return
    abs_ret = np.abs(ret)

    # Replace any inf that snuck through
    abs_ret          = np.where(np.isinf(abs_ret), np.nan, abs_ret)
    detrended_log_vol = np.where(np.isinf(detrended_log_vol), np.nan, detrended_log_vol)
    signed_vol       = np.where(np.isinf(signed_vol), np.nan, signed_vol)
    ret              = np.where(np.isinf(ret), np.nan, ret)

    # --- Rolling OLS: |ret| ~ detrended_log_vol  (Kyle-lambda abs proxy)
    # Substitute NaN with 0 for regression windows -- but we want them to propagate.
    # Strategy: pass arrays as-is; _rolling_ols_slope uses numpy ops that propagate NaN
    # naturally through mean/sum when the window contains any NaN (acceptable; the first
    # ~TREND_WIN+1 rows will be NaN anyway).
    abs_slope    = _rolling_ols_slope(abs_ret, detrended_log_vol, _LONG_WIN)

    # --- Rolling OLS: signed_ret ~ signed_vol  (direction-conditional impact)
    # signed_vol in raw-volume units is huge; normalise by rolling std to keep numerics clean
    sv_series = pd.Series(signed_vol)
    sv_std    = sv_series.rolling(_LONG_WIN, min_periods=_LONG_WIN).std().to_numpy()
    sv_std    = np.where(sv_std == 0, np.nan, sv_std)
    signed_vol_norm = signed_vol / sv_std        # z-scored within the window lookback

    signed_slope = _rolling_ols_slope(ret, signed_vol_norm, _LONG_WIN)

    # --- 20d change in abs_slope (liquidity-thinning / regime shift)
    abs_slope_series = pd.Series(abs_slope)
    slope_chg = (
        abs_slope_series
        - abs_slope_series.shift(_CHNG_WIN)
    ).to_numpy()

    # Final inf guard
    abs_slope    = np.where(np.isinf(abs_slope), np.nan, abs_slope)
    signed_slope = np.where(np.isinf(signed_slope), np.nan, signed_slope)
    slope_chg    = np.where(np.isinf(slope_chg), np.nan, slope_chg)

    df[abs_slope_col]    = abs_slope
    df[signed_slope_col] = signed_slope
    df[slope_chg_col]    = slope_chg

    return df
