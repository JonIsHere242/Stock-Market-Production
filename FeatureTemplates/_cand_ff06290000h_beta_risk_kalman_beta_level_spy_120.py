from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Load index helper (SPY close)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_beta_risk_kalman_beta_level_spy_120",
    "description": (
        "Adaptive Kalman-filtered beta vs SPY over a 120-day window. "
        "A scalar random-walk state-space model tracks beta_t each day "
        "(process_var=1e-4, obs_var=rolling residual variance over 120 bars). "
        "Produced columns: "
        "(1) ff06290000h_kalman_beta_level: the final filtered beta_t, a time-varying "
        "estimate of systematic exposure; "
        "(2) ff06290000h_kalman_ols_drift: difference between filtered beta_t and the "
        "120-day OLS beta -- captures how much the adaptive estimate has drifted from "
        "the static estimate (the primary spec signal); "
        "(3) ff06290000h_kalman_beta_vol: rolling 20-day std of the filtered beta series, "
        "measuring instability of systematic exposure. "
        "Per-ticker causal proxy; no cross-sectional data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06290000h_kalman_beta_level",
        "ff06290000h_kalman_ols_drift",
        "ff06290000h_kalman_beta_vol",
    ],
    "tags": ["beta", "kalman", "spy", "systematic_risk", "adaptive"],
    "version": "1.0.0",
    "author": "feature-factory ff06290000h",
}

_WINDOW = 120
_PROCESS_VAR = 1e-4
_MIN_OBS_VAR = 1e-12
_BETA_VOL_WIN = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path
    df["ff06290000h_kalman_beta_level"] = np.nan
    df["ff06290000h_kalman_ols_drift"] = np.nan
    df["ff06290000h_kalman_beta_vol"] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY daily close and align to this ticker's dates
    # ------------------------------------------------------------------
    try:
        spy_raw = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_raw is None or len(spy_raw) == 0:
        return df

    spy_ser = spy_raw.rename("spy_close")
    spy_df = spy_ser.reset_index()
    spy_df.columns = ["Date", "spy_close"]

    # Normalise date types for merge_asof
    df_dates = df[["Date"]].copy()
    df_dates["Date"] = pd.to_datetime(df_dates["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    merged = merged.set_index(df_dates.sort_values("Date").index).reindex(df.index)

    spy_close = merged["spy_close"].values.astype(np.float64)
    stock_close = df["Close"].values.astype(np.float64)

    n = len(df)

    # Log returns (t vs t-1); index 0 = NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret = np.empty(n)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.where(
            stock_close[:-1] > 0,
            np.log(stock_close[1:] / stock_close[:-1]),
            np.nan,
        )

        spy_ret = np.empty(n)
        spy_ret[0] = np.nan
        spy_ret[1:] = np.where(
            spy_close[:-1] > 0,
            np.log(spy_close[1:] / spy_close[:-1]),
            np.nan,
        )

    # ------------------------------------------------------------------
    # Kalman filter: scalar random-walk beta
    #   State: beta_t ~ N(beta_{t-1}, Q)   Q = process_var
    #   Obs:   r_t = beta_t * m_t + eps_t  R_t = rolling residual var
    # ------------------------------------------------------------------
    kalman_beta = np.full(n, np.nan)
    ols_beta = np.full(n, np.nan)

    # State estimate and error covariance
    beta_est = 1.0  # initial guess
    P_est = 1.0     # initial estimation error covariance

    # We need at least _WINDOW returns before the first estimate
    # Compute rolling obs_var from residuals using previous beta estimate
    # We use a simple online approach: for each bar t >= _WINDOW, compute
    # obs_var from the last _WINDOW residuals using current beta estimate.

    # Pre-compute rolling OLS beta for reference (vectorised)
    # OLS beta_t = cov(r, m) / var(m) over [t-WINDOW+1 .. t]
    # We compute this via rolling sums of r*m and m^2

    # Pad with NaN-safe rolling
    ret_s = pd.Series(stock_ret)
    mkt_s = pd.Series(spy_ret)

    # Rolling cross-products over _WINDOW
    roll_rm = (ret_s * mkt_s).rolling(_WINDOW, min_periods=_WINDOW).mean()
    roll_m2 = (mkt_s * mkt_s).rolling(_WINDOW, min_periods=_WINDOW).mean()
    roll_r  = ret_s.rolling(_WINDOW, min_periods=_WINDOW).mean()
    roll_m  = mkt_s.rolling(_WINDOW, min_periods=_WINDOW).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        cov_rm = roll_rm.values - roll_r.values * roll_m.values
        var_m  = roll_m2.values - roll_m.values ** 2
        ols_beta_arr = np.where(var_m > _MIN_OBS_VAR, cov_rm / var_m, np.nan)

    # Kalman pass -- sequential, but only O(n) scalar ops
    for t in range(1, n):
        if np.isnan(stock_ret[t]) or np.isnan(spy_ret[t]):
            # propagate state but leave estimate as NaN for output
            P_est = P_est + _PROCESS_VAR
            continue

        # Predict
        P_pred = P_est + _PROCESS_VAR

        # Observation variance: rolling residual variance over up to _WINDOW bars
        # Use a lightweight estimate: residuals from current beta_est
        lo = max(1, t - _WINDOW + 1)
        seg_r = stock_ret[lo: t + 1]
        seg_m = spy_ret[lo: t + 1]
        valid = ~(np.isnan(seg_r) | np.isnan(seg_m))
        if valid.sum() >= 5:
            resid = seg_r[valid] - beta_est * seg_m[valid]
            obs_var = float(np.var(resid))
        else:
            obs_var = 0.0
        obs_var = max(obs_var, _MIN_OBS_VAR)

        # Kalman gain
        m_t = spy_ret[t]
        innov_var = P_pred * m_t * m_t + obs_var
        if innov_var < _MIN_OBS_VAR:
            K = 0.0
        else:
            K = P_pred * m_t / innov_var

        # Update
        innov = stock_ret[t] - beta_est * m_t
        beta_est = beta_est + K * innov
        P_est = (1.0 - K * m_t) * P_pred

        if t >= _WINDOW:
            kalman_beta[t] = beta_est

    # OLS beta (already vectorised above)
    ols_beta = ols_beta_arr

    # Drift = kalman - ols
    with np.errstate(invalid="ignore"):
        drift = np.where(
            ~np.isnan(kalman_beta) & ~np.isnan(ols_beta),
            kalman_beta - ols_beta,
            np.nan,
        )

    # Rolling vol of kalman beta
    kalman_beta_s = pd.Series(kalman_beta)
    beta_vol = kalman_beta_s.rolling(_BETA_VOL_WIN, min_periods=_BETA_VOL_WIN).std().values

    df["ff06290000h_kalman_beta_level"] = kalman_beta
    df["ff06290000h_kalman_ols_drift"] = drift
    df["ff06290000h_kalman_beta_vol"] = beta_vol

    return df
