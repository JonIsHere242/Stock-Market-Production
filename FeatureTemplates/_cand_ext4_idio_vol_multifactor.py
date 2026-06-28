"""
ext4_idio_vol_multifactor
Idiosyncratic volatility from a rolling 3-index (SPY, QQQ, IWM) OLS model.
Cleaner than a 1-factor CAPM residual because it strips out size/growth/value tilts.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load _indexes helper ──────────────────────────────────────────────────────
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ext4_idio_vol_multifactor",
    "description": (
        "Rolling 120-day 3-factor OLS of stock returns on SPY, QQQ, and IWM daily returns. "
        "Produces (1) idiosyncratic volatility = annualised residual std, "
        "(2) systematic R^2 (fraction of variance explained by the 3 indices), "
        "and (3) 20-day z-score of idiosyncratic vol to capture its trend. "
        "Per-ticker proxy; cross-sectional rank is applied externally. "
        "Lookahead-safe: merge_asof backward on Date, rolling window only uses past bars."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_idio_vol_multifactor_ivol",   # annualised idiosyncratic vol
        "ext4_idio_vol_multifactor_r2",     # rolling 3-factor R^2 (systematic share)
        "ext4_idio_vol_multifactor_ivol_z", # 20-day z-score of ivol (trend signal)
    ],
    "tags": ["volatility", "idiosyncratic", "multi-factor", "OLS", "residual"],
    "version": "1.0.0",
    "author": "Round-5 expansion (osap_idiovolaht) — extends osap_idiovolaht 1-factor to 3-index OLS",
}

_WINDOW = 120      # rolling OLS window (trading days)
_IVOL_Z_WINDOW = 20
_ANN = np.sqrt(252)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── 1. Stock daily returns ────────────────────────────────────────────────
    ret = df["Close"].pct_change()  # length-N, NaN at t=0

    # ── 2. Fetch index returns ────────────────────────────────────────────────
    df_work = df[["Date"]].copy()
    df_work["_ret"] = ret.values

    def _get_idx_ret(sym: str) -> pd.Series:
        try:
            s = _indexes.index_close(sym)  # DatetimeIndex → float
            idx_df = s.reset_index()
            idx_df.columns = ["Date", "_close"]
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df["_iret"] = idx_df["_close"].pct_change()
            # merge_asof requires sorted keys
            merged = pd.merge_asof(
                df_work[["Date"]].sort_values("Date"),
                idx_df[["Date", "_iret"]].sort_values("Date"),
                on="Date",
                direction="backward",
            )
            # restore original order
            merged = merged.set_index("Date").reindex(df["Date"]).reset_index()
            return merged["_iret"].values
        except Exception:
            return np.full(len(df), np.nan)

    spy_ret = _get_idx_ret("SPY")
    qqq_ret = _get_idx_ret("QQQ")
    iwm_ret = _get_idx_ret("IWM")

    n = len(df)
    ivol   = np.full(n, np.nan)
    r2_arr = np.full(n, np.nan)

    ret_arr = ret.values  # numpy array for speed

    # ── 3. Rolling 120-day multivariate OLS ─────────────────────────────────
    # For each bar t >= WINDOW-1 we regress:
    #   ret[t-W+1 : t+1] = a + b1*spy + b2*qqq + b3*iwm + eps
    # Residual std = idio vol; R^2 = 1 - var(eps)/var(y).
    # Vectorised using numpy sliding_window_view where possible,
    # but the OLS itself is a small 120×4 solve — fast.

    W = _WINDOW
    for t in range(W - 1, n):
        y  = ret_arr[t - W + 1 : t + 1]
        x1 = spy_ret[t - W + 1 : t + 1]
        x2 = qqq_ret[t - W + 1 : t + 1]
        x3 = iwm_ret[t - W + 1 : t + 1]

        # stack regressors; intercept column
        X = np.column_stack([np.ones(W), x1, x2, x3])

        # drop rows with any NaN
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < 20:          # need at least 20 clean obs
            continue

        y_  = y[mask]
        X_  = X[mask]

        # OLS via normal equations (120×4 is tiny)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_, y_, rcond=None)
        except np.linalg.LinAlgError:
            continue

        resid = y_ - X_ @ coeffs
        ss_res = float(np.dot(resid, resid))
        y_dm   = y_ - y_.mean()
        ss_tot = float(np.dot(y_dm, y_dm))

        if ss_tot < 1e-20:
            continue

        ivol[t]   = np.std(resid, ddof=4) * _ANN   # annualised
        r2_arr[t] = max(0.0, 1.0 - ss_res / ss_tot)

    # ── 4. Z-score of ivol over trailing 20 bars ─────────────────────────────
    ivol_s  = pd.Series(ivol, index=df.index)
    roll    = ivol_s.rolling(_IVOL_Z_WINDOW, min_periods=10)
    ivol_z  = (ivol_s - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)

    # ── 5. Attach columns ────────────────────────────────────────────────────
    df["ext4_idio_vol_multifactor_ivol"]   = ivol
    df["ext4_idio_vol_multifactor_r2"]     = r2_arr
    df["ext4_idio_vol_multifactor_ivol_z"] = ivol_z.values

    return df
