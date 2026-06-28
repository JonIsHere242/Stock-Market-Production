"""
Frazzini-Pedersen Beta (osap_betafp)
Source: OpenSourceAP (Chen-Zimmermann), based on Frazzini and Pedersen (2014)
"Betting Against Beta", Journal of Financial Economics.

Method:
  - Define tempR_i = r_t + r_{t-1} + r_{t-2}  (3-day overlapping stock return)
  - Define tempR_m = rm_t + rm_{t-1} + rm_{t-2} (3-day overlapping market return, SPY proxy)
  - Regress tempR_i on tempR_m using a rolling 5-year window (min 3 years = 756 trading days)
  - BetaFP = sqrt(R²) * (sigma_i / sigma_m)
    where sigma_i = std(daily stock returns), sigma_m = std(daily market returns)
    both measured over the same rolling window

Per-ticker note: This is implemented as a rolling per-ticker computation using SPY as
the market proxy. The overlapping-return construction reduces asynchronous-trading bias
as in the original paper. Predicted sign is +1 (high-beta stocks have higher expected
returns in the BAB cross-section).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load _indexes helper
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "osap_betafp",
    "description": (
        "Frazzini-Pedersen (2014) beta computed per ticker via rolling OLS of 3-day overlapping "
        "stock returns on 3-day overlapping SPY returns. BetaFP = sqrt(R²) * (sigma_i / sigma_m). "
        "Window: 5-year rolling (min 3 years). Produces level, slope (12m change), and "
        "idiosyncratic-vol ratio. Cross-sectional ranking is not available per-ticker; "
        "the level itself is the BAB signal (high beta = positive predicted return, sign=+1)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_betafp_level",     # rolling FP beta
        "osap_betafp_slope",     # 12-month change in beta (beta trend)
        "osap_betafp_idvol_ratio",  # sigma_i / sigma_m (idiosyncratic vol ratio component)
    ],
    "tags": ["beta", "risk", "frazzini-pedersen", "bab", "market-beta", "osap"],
    "version": "1.0",
    "author": "Frazzini and Pedersen (2014) via OpenSourceAP (Chen-Zimmermann); block by Claude",
}

# Rolling window parameters (trading days)
_WINDOW_LONG = 1260   # ~5 years
_WINDOW_MIN  = 756    # ~3 years minimum
_WINDOW_MED  = 252    # ~1 year for slope lookback


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < _WINDOW_MIN + 10:
        df["osap_betafp_level"]      = np.nan
        df["osap_betafp_slope"]      = np.nan
        df["osap_betafp_idvol_ratio"] = np.nan
        return df

    # --- 1. Get SPY market returns -------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    if spy_close is None or len(spy_close) == 0:
        df["osap_betafp_level"]      = np.nan
        df["osap_betafp_slope"]      = np.nan
        df["osap_betafp_idvol_ratio"] = np.nan
        return df

    # Align SPY to df dates via merge_asof (backward = lookahead-safe)
    df_dates = df[["Date"]].copy()
    df_dates["Date"] = pd.to_datetime(df_dates["Date"])

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date")

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df,
        on="Date",
        direction="backward",
    )
    # Restore original row order
    merged.index = df_dates.sort_values("Date").index
    spy_aligned = merged["spy_close"].reindex(df.index)

    # --- 2. Daily returns ---------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    spy   = spy_aligned.values.astype(np.float64)

    n = len(close)
    r_i = np.empty(n)
    r_m = np.empty(n)
    r_i[0] = np.nan
    r_m[0] = np.nan

    # log returns (robust to negative prices edge cases via max)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_i[1:] = np.where(close[:-1] > 0, np.log(close[1:] / close[:-1]), np.nan)
        r_m[1:] = np.where(spy[:-1]   > 0, np.log(spy[1:]   / spy[:-1]),   np.nan)

    # --- 3. 3-day overlapping (sum) returns ---------------------------------------
    # tempR[t] = r[t] + r[t-1] + r[t-2]
    # Implemented as rolling sum over 3 days; first 2 rows = NaN
    def _rolling_sum3(arr):
        out = np.full(n, np.nan)
        for t in range(2, n):
            vals = arr[t-2:t+1]
            if np.all(np.isfinite(vals)):
                out[t] = vals.sum()
        return out

    # Use vectorised approach for speed
    def _vec_sum3(arr):
        # convolution-style: shift and add
        a0 = arr
        a1 = np.concatenate([[np.nan], arr[:-1]])
        a2 = np.concatenate([[np.nan, np.nan], arr[:-2]])
        with np.errstate(invalid="ignore"):
            out = a0 + a1 + a2
        # Any NaN in the 3 inputs => NaN out
        mask = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(a2)
        out[~mask] = np.nan
        return out

    tempR_i = _vec_sum3(r_i)
    tempR_m = _vec_sum3(r_m)

    # --- 4. Rolling FP beta -------------------------------------------------------
    # BetaFP[t] = sqrt(R²) * (sigma_i / sigma_m)
    # where R² comes from OLS of tempR_i on tempR_m over past [WINDOW_MIN, WINDOW_LONG] days
    # sigma_i, sigma_m = std of DAILY returns over same window (as in original paper)

    beta_fp     = np.full(n, np.nan)
    idvol_ratio = np.full(n, np.nan)

    # To keep this O(n * window) but avoid Python row-loops, we use pandas rolling
    # with a custom apply. For n~700 and window 1260 this is fast enough.
    s_i = pd.Series(tempR_i)
    s_m = pd.Series(tempR_m)
    s_ri_daily = pd.Series(r_i)
    s_rm_daily = pd.Series(r_m)

    def _fp_beta(idx_arr):
        # idx_arr is the index array passed by rolling apply -- not useful directly
        # We use a different approach below (manual loop with numpy slice)
        pass

    # Manual rolling: iterate only over valid end-points.
    # For ~700 rows this is at most 700 iterations, each numpy op on ≤1260 rows -> fast.
    for t in range(_WINDOW_MIN, n):
        start = max(0, t - _WINDOW_LONG + 1)
        xi = tempR_i[start:t+1]
        xm = tempR_m[start:t+1]
        ri_d = r_i[start:t+1]
        rm_d = r_m[start:t+1]

        # Require at least WINDOW_MIN valid pairs
        mask = np.isfinite(xi) & np.isfinite(xm)
        if mask.sum() < _WINDOW_MIN:
            continue

        xi_v = xi[mask]
        xm_v = xm[mask]

        # OLS: regress xi on xm (with intercept)
        # R² = (cov(xi,xm)^2) / (var(xi)*var(xm))
        # slope = cov(xi,xm)/var(xm)
        xm_mean = xm_v.mean()
        xi_mean = xi_v.mean()
        xm_c = xm_v - xm_mean
        xi_c = xi_v - xi_mean

        var_m = np.dot(xm_c, xm_c)
        cov_im = np.dot(xi_c, xm_c)
        var_i = np.dot(xi_c, xi_c)

        if var_m <= 0 or var_i <= 0:
            continue

        r2 = (cov_im ** 2) / (var_m * var_i)
        # R² in [0,1] by construction; clamp for numerical safety
        r2 = min(max(r2, 0.0), 1.0)

        # sigma_i / sigma_m from daily returns (same window)
        mask_d = np.isfinite(ri_d) & np.isfinite(rm_d)
        if mask_d.sum() < 20:
            continue
        sig_i = ri_d[mask_d].std()
        sig_m = rm_d[mask_d].std()

        if sig_m <= 0:
            continue

        ratio = sig_i / sig_m
        beta_fp[t] = np.sqrt(r2) * ratio
        idvol_ratio[t] = ratio

    # --- 5. 12-month slope (change in beta) ---------------------------------------
    beta_series = pd.Series(beta_fp)
    beta_lag252 = beta_series.shift(_WINDOW_MED)
    with np.errstate(invalid="ignore"):
        beta_slope = (beta_series - beta_lag252).values

    # --- 6. Assign outputs --------------------------------------------------------
    df["osap_betafp_level"]       = beta_fp
    df["osap_betafp_slope"]       = beta_slope
    df["osap_betafp_idvol_ratio"] = idvol_ratio

    return df
