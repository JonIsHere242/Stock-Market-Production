from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282347f_volume_price_unexpected_vol_return_reaction_40",
    "description": (
        "Signed-return reaction to unexpected volume over a 40-day rolling window. "
        "Computes the log1p-volume z-score (vs 40-day trailing mean/std) and same-day "
        "signed return; then estimates the OLS slope of return ~ vol_zscore using a "
        "rolling 40-bar window. Positive slope = surprise volume accompanies up moves "
        "(directional accumulation); near-zero = two-sided disagreement proxy. "
        "Three columns: slope (OLS beta), correlation (Pearson r of the same pair, "
        "sign-preserving), and a 10-bar slope for short-horizon regime changes. "
        "Per-ticker proxy; no cross-sectional data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282347f_volume_price_unexpected_vol_return_reaction_40_slope",
        "ff06282347f_volume_price_unexpected_vol_return_reaction_40_corr",
        "ff06282347f_volume_price_unexpected_vol_return_reaction_40_slope10",
    ],
    "tags": ["volume", "price", "reaction", "zscore", "regression", "disagreement"],
    "version": "1.0.0",
    "author": "feature-factory ff06282347f",
}

_WINDOW = 40
_WINDOW_SHORT = 10
_COL_SLOPE = "ff06282347f_volume_price_unexpected_vol_return_reaction_40_slope"
_COL_CORR = "ff06282347f_volume_price_unexpected_vol_return_reaction_40_corr"
_COL_SLOPE10 = "ff06282347f_volume_price_unexpected_vol_return_reaction_40_slope10"


def _rolling_ols_slope(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling OLS slope of y ~ x using a vectorised sliding approach."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Use cumulative sums for O(n) rolling moments
    # sx = sum(x), sy = sum(y), sxy = sum(x*y), sxx = sum(x^2)
    # where any nan in the window collapses that window to nan

    # Replace nans with a sentinel then mask after
    valid = np.isfinite(x) & np.isfinite(y)

    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    vc = valid.astype(np.float64)

    # Prefix sums
    csx = np.cumsum(xv)
    csy = np.cumsum(yv)
    csxy = np.cumsum(xv * yv)
    csxx = np.cumsum(xv * xv)
    cvc = np.cumsum(vc)

    for i in range(window - 1, n):
        j = i - window  # start-1 index
        cnt = cvc[i] - (cvc[j] if j >= 0 else 0.0)
        if cnt < window * 0.8:          # allow up to 20% NaN in window
            continue
        sx = csx[i] - (csx[j] if j >= 0 else 0.0)
        sy = csy[i] - (csy[j] if j >= 0 else 0.0)
        sxy = csxy[i] - (csxy[j] if j >= 0 else 0.0)
        sxx = csxx[i] - (csxx[j] if j >= 0 else 0.0)
        denom = cnt * sxx - sx * sx
        if denom == 0.0:
            continue
        out[i] = (cnt * sxy - sx * sy) / denom

    return out


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Pearson correlation in a rolling window."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    vc = valid.astype(np.float64)

    csx = np.cumsum(xv)
    csy = np.cumsum(yv)
    csxy = np.cumsum(xv * yv)
    csxx = np.cumsum(xv * xv)
    csyy = np.cumsum(yv * yv)
    cvc = np.cumsum(vc)

    for i in range(window - 1, n):
        j = i - window
        cnt = cvc[i] - (cvc[j] if j >= 0 else 0.0)
        if cnt < window * 0.8:
            continue
        sx = csx[i] - (csx[j] if j >= 0 else 0.0)
        sy = csy[i] - (csy[j] if j >= 0 else 0.0)
        sxy = csxy[i] - (csxy[j] if j >= 0 else 0.0)
        sxx = csxx[i] - (csxx[j] if j >= 0 else 0.0)
        syy = csyy[i] - (csyy[j] if j >= 0 else 0.0)
        num = cnt * sxy - sx * sy
        var_x = cnt * sxx - sx * sx
        var_y = cnt * syy - sy * sy
        denom = var_x * var_y
        if denom <= 0.0:
            continue
        out[i] = num / np.sqrt(denom)

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path
    df[_COL_SLOPE] = np.nan
    df[_COL_CORR] = np.nan
    df[_COL_SLOPE10] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # --- Signed return (today's close vs yesterday's close) ---
    close = df["Close"].values.astype(np.float64)
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    # Guard against zero/nan denominator
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(prev_close > 0.0, (close - prev_close) / prev_close, np.nan)

    # --- Log1p volume z-score vs trailing 40-day window ---
    vol = df["Volume"].values.astype(np.float64)
    log_vol = np.log1p(np.where(vol >= 0.0, vol, np.nan))

    # Rolling mean and std of log_vol using pandas for clarity/correctness
    lv_s = pd.Series(log_vol)
    roll_mean = lv_s.rolling(window=_WINDOW, min_periods=int(_WINDOW * 0.8)).mean().values
    roll_std = lv_s.rolling(window=_WINDOW, min_periods=int(_WINDOW * 0.8)).std().values

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = np.where(roll_std > 0.0, (log_vol - roll_mean) / roll_std, np.nan)

    # --- Rolling 40-bar OLS slope: signed_return ~ vol_z_score ---
    slope40 = _rolling_ols_slope(vol_z, ret, _WINDOW)
    corr40 = _rolling_corr(vol_z, ret, _WINDOW)
    slope10 = _rolling_ols_slope(vol_z, ret, _WINDOW_SHORT)

    # Clip extreme regression artefacts (±10 is already huge for a return/z relationship)
    slope40 = np.clip(slope40, -10.0, 10.0)
    slope10 = np.clip(slope10, -10.0, 10.0)

    df[_COL_SLOPE] = slope40
    df[_COL_CORR] = corr40
    df[_COL_SLOPE10] = slope10

    return df
