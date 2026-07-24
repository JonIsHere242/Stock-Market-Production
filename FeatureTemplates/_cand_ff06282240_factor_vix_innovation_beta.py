"""
VIX Innovation Beta feature block.

Beta of stock returns to VIX innovations (the residual of VIX daily change
after removing its own AR(1) component). Captures sensitivity to unexpected
volatility shocks, which is orthogonal to raw VIX-level exposure.

Splits beta by sign of shock to measure asymmetric response (fear vs. calm).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Load _indexes helper (by file path -- never import from FeatureTemplates)
# --------------------------------------------------------------------------- #
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# --------------------------------------------------------------------------- #
METADATA = {
    "name": "ff06282240_factor_vix_innovation_beta",
    "description": (
        "Beta to VIX innovations. dVIX = VIX_close.diff(); innovation = "
        "residual of AR(1) regression of dVIX on its own one-day lag, "
        "computed over a 120-day trailing window. Full-sample OLS slope of "
        "stock daily return on the innovation series gives vix_innov_beta "
        "(level). Subsetting to days where dVIX>0 and dVIX<=0 gives "
        "vix_innov_beta_up and vix_innov_beta_dn; asymmetry = up - down. "
        "Recomputed on a fixed-from-start stride=5 grid and forward-filled. "
        "Per-ticker proxy of sensitivity to unexpected volatility shocks."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282240_factor_vix_innovation_beta_level",
        "ff06282240_factor_vix_innovation_beta_up",
        "ff06282240_factor_vix_innovation_beta_dn",
        "ff06282240_factor_vix_innovation_beta_asym",
    ],
    "tags": ["vix", "beta", "volatility", "innovation", "factor", "asymmetry"],
    "version": "1.0.0",
    "author": "feature-factory ff06282240",
}

_WINDOW = 120
_STRIDE = 5
_VAR_EPS = 1e-12
_PRODUCED = METADATA["produces"]


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x (no intercept guard, but denominator protected)."""
    denom = np.dot(x, x)
    if denom < _VAR_EPS or len(y) < 4:
        return np.nan
    return float(np.dot(x, y) / denom)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front so every code path is
    # covered (including empty df, missing VIX, or insufficient history).
    for col in _PRODUCED:
        df[col] = np.nan

    if len(df) < _WINDOW + 5:
        return df

    # ------------------------------------------------------------------ #
    # 1. Pull VIX daily close and align to this stock's dates via
    #    merge_asof (backward = no lookahead).
    # ------------------------------------------------------------------ #
    try:
        vix_df = _indexes.vix_daily_close()  # DataFrame[["Date","vix_close"]]
    except Exception:
        return df

    if vix_df is None or vix_df.empty:
        return df

    stock = df[["Date", "Close"]].copy()
    stock["Date"] = pd.to_datetime(stock["Date"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df = vix_df.sort_values("Date")

    merged = pd.merge_asof(
        stock.sort_values("Date"),
        vix_df[["Date", "vix_close"]],
        on="Date",
        direction="backward",
    )
    # Restore original order (df is already ascending by Date per contract)
    merged = merged.reset_index(drop=True)

    vix_close = merged["vix_close"].values.astype(float)
    close = merged["Close"].values.astype(float)

    n = len(df)

    # ------------------------------------------------------------------ #
    # 2. Compute VIX daily change and AR(1) innovation series.
    #    dVIX[t] = VIX[t] - VIX[t-1]
    #    innovation[t] = dVIX[t] - ar1_coeff * dVIX[t-1]
    #    Both ar1_coeff and the innovation are computed per-window to stay
    #    fully causal.  The ar1 is the OLS slope of dVIX on dVIX.shift(1)
    #    within the same window.
    # ------------------------------------------------------------------ #
    dvix = np.empty(n)
    dvix[:] = np.nan
    dvix[1:] = np.diff(vix_close)

    ret = np.empty(n)
    ret[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ret[1:] = np.diff(np.log(np.where(close > 0, close, np.nan)))

    # ------------------------------------------------------------------ #
    # 3. Stride loop: compute at indices where i % stride == 0
    #    (anchored from the series start, so truncation doesn't re-anchor).
    # ------------------------------------------------------------------ #
    level_arr = np.full(n, np.nan)
    up_arr = np.full(n, np.nan)
    dn_arr = np.full(n, np.nan)
    asym_arr = np.full(n, np.nan)

    for i in range(0, n):
        if i % _STRIDE != 0:
            continue
        start = i - _WINDOW + 1
        if start < 2:
            # Need at least one lag for AR(1), so minimum start is 2
            continue

        # Window slices (indices start..i inclusive)
        w_dvix = dvix[start : i + 1]       # shape (WINDOW,)
        w_dvix_lag = dvix[start - 1 : i]   # one-step-lagged, same length
        w_ret = ret[start : i + 1]

        # Drop any NaN rows (first bar always NaN)
        valid = (
            np.isfinite(w_dvix)
            & np.isfinite(w_dvix_lag)
            & np.isfinite(w_ret)
        )
        if valid.sum() < 10:
            continue

        dv = w_dvix[valid]
        dvl = w_dvix_lag[valid]
        wr = w_ret[valid]

        # AR(1): slope of dVIX on dVIX.lag (with intercept via de-meaning)
        dv_dm = dv - dv.mean()
        dvl_dm = dvl - dvl.mean()
        ar1_denom = np.dot(dvl_dm, dvl_dm)
        if ar1_denom < _VAR_EPS:
            ar1 = 0.0
        else:
            ar1 = float(np.dot(dvl_dm, dv_dm) / ar1_denom)

        # Innovation = residual from AR(1) (de-meaned x)
        innov = dv - ar1 * dvl

        # Guard: need non-trivial innovation variance
        if np.var(innov) < _VAR_EPS:
            continue

        # De-mean innovation for OLS (intercept absorbed by centering)
        innov_dm = innov - innov.mean()

        # Full-sample beta
        level_arr[i] = _ols_slope(wr, innov_dm)

        # Positive VIX shock subset (up days)
        mask_up = dv > 0
        if mask_up.sum() >= 4:
            up_arr[i] = _ols_slope(wr[mask_up], innov_dm[mask_up])

        # Non-positive VIX shock subset (down/flat days)
        mask_dn = ~mask_up
        if mask_dn.sum() >= 4:
            dn_arr[i] = _ols_slope(wr[mask_dn], innov_dm[mask_dn])

        u = up_arr[i]
        d = dn_arr[i]
        if np.isfinite(u) and np.isfinite(d):
            asym_arr[i] = u - d

    # ------------------------------------------------------------------ #
    # 4. Forward-fill (carry stride values between compute points).
    # ------------------------------------------------------------------ #
    def _ffill(arr: np.ndarray) -> np.ndarray:
        s = pd.Series(arr)
        return s.ffill().values

    df["ff06282240_factor_vix_innovation_beta_level"] = _ffill(level_arr)
    df["ff06282240_factor_vix_innovation_beta_up"] = _ffill(up_arr)
    df["ff06282240_factor_vix_innovation_beta_dn"] = _ffill(dn_arr)
    df["ff06282240_factor_vix_innovation_beta_asym"] = _ffill(asym_arr)

    return df
