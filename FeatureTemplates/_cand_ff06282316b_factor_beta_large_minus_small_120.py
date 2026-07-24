"""
ff06282316b_factor_beta_large_minus_small_120
Size-tilt loading: two-variable OLS of ticker returns on [SPY, DIA-IWM] over 120-day window.
The LMS (large-minus-small) coefficient isolates the ticker's sensitivity to the
size spread, net of broad market exposure. Companion columns: market beta (SPY loading)
and regression R-squared.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path -- never via package import)
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_factor_beta_large_minus_small_120",
    "description": (
        "Per-ticker size-tilt factor loading via two-variable OLS (120-day trailing window). "
        "Regresses daily log-returns on [SPY_return, DIA_return - IWM_return] using normal "
        "equations with epsilon stabilisation. Produces: LMS beta (size-tilt net of market), "
        "market beta (SPY loading), and OLS R-squared. Proxy: cross-sectional ranks are not "
        "available; this captures the same economic signal as a Fama-French SMB loading but "
        "from a single-ticker OLS perspective."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_beta_large_minus_small_120_lms_beta",
        "ff06282316b_factor_beta_large_minus_small_120_mkt_beta",
        "ff06282316b_factor_beta_large_minus_small_120_rsq",
    ],
    "tags": ["factor", "size", "beta", "regression", "lms"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
_WINDOW = 120
_EPS = 1e-12

_COL_LMS = "ff06282316b_factor_beta_large_minus_small_120_lms_beta"
_COL_MKT = "ff06282316b_factor_beta_large_minus_small_120_mkt_beta"
_COL_RSQ = "ff06282316b_factor_beta_large_minus_small_120_rsq"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN so all code paths produce the columns.
    df[_COL_LMS] = np.nan
    df[_COL_MKT] = np.nan
    df[_COL_RSQ] = np.nan

    if len(df) < _WINDOW + 2:
        return df

    # ------------------------------------------------------------------
    # Fetch index series
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        dia_close = _indexes.index_close("DIA")
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df

    if spy_close is None or dia_close is None or iwm_close is None:
        return df

    # ------------------------------------------------------------------
    # Build a date-aligned frame for the index returns
    # ------------------------------------------------------------------
    dates = pd.to_datetime(df["Date"])

    spy_ret = spy_close.pct_change()
    dia_ret = dia_close.pct_change()
    iwm_ret = iwm_close.pct_change()
    lms_ret = dia_ret - iwm_ret  # large-minus-small spread

    idx_df = pd.DataFrame({
        "Date": spy_close.index,
        "_spy_r": spy_ret.values,
        "_lms_r": lms_ret.reindex(spy_close.index).values,
    })
    idx_df["Date"] = pd.to_datetime(idx_df["Date"])

    merged = pd.merge_asof(
        df[["Date"]].assign(Date=dates).sort_values("Date"),
        idx_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    merged = merged.set_index(df.index)

    spy_r = merged["_spy_r"].values.astype(float)
    lms_r = merged["_lms_r"].values.astype(float)

    # Ticker log-returns (daily)
    close_vals = df["Close"].values.astype(float)
    tkr_r = np.empty(len(close_vals))
    tkr_r[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        tkr_r[1:] = np.where(
            close_vals[:-1] > 0,
            np.log(close_vals[1:] / close_vals[:-1]),
            np.nan,
        )

    n = len(df)
    lms_beta_out = np.full(n, np.nan)
    mkt_beta_out = np.full(n, np.nan)
    rsq_out = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Rolling two-variable OLS via normal equations.
    # At each bar t (0-indexed), use rows [t-window+1 .. t].
    # We solve [X'X] [b0,b1]' = X'y  where X = [spy_r | lms_r] (no intercept
    # kept, but we demean to mimic an intercept implicitly -- actually we use
    # a 3-column design with intercept for correctness).
    # Design: Z = [1, spy_r, lms_r], shape (W,3).  Use normal eqs Z'Z b = Z'y.
    # ------------------------------------------------------------------
    for t in range(_WINDOW - 1, n):
        sl = slice(t - _WINDOW + 1, t + 1)
        y = tkr_r[sl]
        x1 = spy_r[sl]
        x2 = lms_r[sl]

        # Drop NaN rows
        mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        ny = y[mask]; nx1 = x1[mask]; nx2 = x2[mask]

        if ny.shape[0] < 10:
            continue

        # Design matrix [1, spy, lms]
        ones = np.ones(ny.shape[0])
        Z = np.column_stack([ones, nx1, nx2])  # (m, 3)

        ZtZ = Z.T @ Z            # (3,3)
        Zty = Z.T @ ny           # (3,)

        # Tikhonov stabilisation
        ZtZ_reg = ZtZ + np.eye(3) * _EPS

        try:
            coeffs = np.linalg.solve(ZtZ_reg, Zty)
        except np.linalg.LinAlgError:
            continue

        b_intercept, b_mkt, b_lms = coeffs

        mkt_beta_out[t] = b_mkt
        lms_beta_out[t] = b_lms

        # R-squared
        y_hat = Z @ coeffs
        ss_res = np.sum((ny - y_hat) ** 2)
        ss_tot = np.sum((ny - ny.mean()) ** 2)
        if ss_tot > _EPS:
            rsq_out[t] = 1.0 - ss_res / ss_tot
        else:
            rsq_out[t] = np.nan

    df[_COL_LMS] = lms_beta_out
    df[_COL_MKT] = mkt_beta_out
    df[_COL_RSQ] = rsq_out

    return df
