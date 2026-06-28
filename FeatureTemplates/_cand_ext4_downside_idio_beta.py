"""
Downside beta of the idiosyncratic component.

For each day, uses a trailing 120-day OLS market model (stock returns ~ SPY returns)
to extract residuals (idiosyncratic returns).  From those residuals we produce:

  1. ext4_downside_idio_beta_dsratio  -- downside semi-deviation / upside semi-deviation
       of idiosyncratic residuals.  Ratio > 1 means the stock's idiosyncratic shocks
       are fatter on the downside than the upside (crash asymmetry).

  2. ext4_downside_idio_beta_crashcov -- rolling covariance of idiosyncratic residuals
       with SPY returns specifically on days when SPY is in its bottom quartile
       (extreme-down-market days).  Captures idiosyncratic crash co-movement:
       how much of the stock's unexplained move co-moves with market stress.

  3. ext4_downside_idio_beta_slope -- 20-day change in dsratio; measures whether
       downside asymmetry of the idiosyncratic component is worsening.

Per-ticker proxy; cross-sectional ranking not possible in this framework.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# --- load _indexes helper ---
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name": "ext4_downside_idio_beta",
    "description": (
        "Downside beta of the idiosyncratic component.  Fits a 120-day rolling OLS "
        "market model (stock ~ SPY) and extracts residuals.  Produces: "
        "(1) downside/upside semi-deviation ratio of residuals (crash asymmetry), "
        "(2) residual covariance with SPY on extreme-down-market days (crash co-move), "
        "(3) 20-day slope of the semi-dev ratio.  Per-ticker proxy -- no cross-section."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_downside_idio_beta_dsratio",
        "ext4_downside_idio_beta_crashcov",
        "ext4_downside_idio_beta_slope",
    ],
    "tags": ["idiosyncratic", "downside", "risk", "residual", "beta", "crash"],
    "version": "1.0.0",
    "author": "Round-5 expansion (osap_idiovolaht); spec: ext4_downside_idio_beta",
}

# Rolling OLS window and sub-windows
_MARKET_WINDOW = 120   # days for OLS market model
_CRASH_QUANTILE = 0.25  # bottom quartile of SPY = "extreme down"
_SLOPE_WINDOW = 20      # days for slope of dsratio


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Pre-allocate output columns with NaN
    df["ext4_downside_idio_beta_dsratio"] = np.nan
    df["ext4_downside_idio_beta_crashcov"] = np.nan
    df["ext4_downside_idio_beta_slope"] = np.nan

    if n < _MARKET_WINDOW + 1:
        return df

    # ------------------------------------------------------------------ #
    # 1.  Align SPY series to df via merge_asof (backward, lookahead-safe)
    # ------------------------------------------------------------------ #
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        return df

    merged = pd.merge_asof(
        df[["Date"]].assign(Date=pd.to_datetime(df["Date"])).sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.set_index(df.index)

    spy_px = merged["spy_close"].values.astype(float)

    # Stock and market daily log-returns
    stock_px = df["Close"].values.astype(float)

    stock_ret = np.full(n, np.nan)
    mkt_ret = np.full(n, np.nan)

    stock_ret[1:] = np.where(
        stock_px[:-1] > 0,
        np.log(np.where(stock_px[:-1] > 0, stock_px[1:] / stock_px[:-1], np.nan)),
        np.nan,
    )
    mkt_ret[1:] = np.where(
        spy_px[:-1] > 0,
        np.log(np.where(spy_px[:-1] > 0, spy_px[1:] / spy_px[:-1], np.nan)),
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # 2.  Rolling 120-day OLS -> residuals (vectorised via stride)
    # ------------------------------------------------------------------ #
    # We compute OLS coefficients analytically:
    #   beta = Cov(y,x) / Var(x);  alpha = mean(y) - beta * mean(x)
    #   resid = y - alpha - beta * x

    W = _MARKET_WINDOW

    dsratio = np.full(n, np.nan)
    crashcov = np.full(n, np.nan)

    # bottom-quartile threshold for SPY over the same 120-day window
    for t in range(W, n):
        y = stock_ret[t - W + 1 : t + 1]
        x = mkt_ret[t - W + 1 : t + 1]

        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < W // 2:
            continue

        y_m = y[mask]
        x_m = x[mask]

        mean_x = np.mean(x_m)
        mean_y = np.mean(y_m)
        var_x = np.mean((x_m - mean_x) ** 2)

        if var_x == 0 or not np.isfinite(var_x):
            continue

        beta_hat = np.sum((x_m - mean_x) * (y_m - mean_y)) / (var_x * len(x_m))
        alpha_hat = mean_y - beta_hat * mean_x

        resid = y_m - alpha_hat - beta_hat * x_m  # idiosyncratic returns

        # --- downside / upside semi-deviation ratio ---
        down_mask = resid < 0
        up_mask = resid > 0
        n_down = down_mask.sum()
        n_up = up_mask.sum()

        if n_down > 0 and n_up > 0:
            down_sd = np.sqrt(np.mean(resid[down_mask] ** 2))
            up_sd = np.sqrt(np.mean(resid[up_mask] ** 2))
            if up_sd > 0 and np.isfinite(up_sd) and np.isfinite(down_sd):
                dsratio[t] = down_sd / up_sd
            # else leave NaN

        # --- idiosyncratic crash covariance ---
        # x_m (mkt returns in window); find bottom-quartile threshold
        q25 = np.percentile(x_m, 100 * _CRASH_QUANTILE)
        crash_days = x_m <= q25
        n_crash = crash_days.sum()
        if n_crash > 2:
            # residuals on crash days (same mask-aligned)
            resid_crash = resid[crash_days]
            mkt_crash = x_m[crash_days]
            # covariance of idiosyncratic return with market on crash days
            cov_val = np.mean(
                (resid_crash - np.mean(resid_crash))
                * (mkt_crash - np.mean(mkt_crash))
            )
            crashcov[t] = cov_val if np.isfinite(cov_val) else np.nan

    # ------------------------------------------------------------------ #
    # 3.  Slope: 20-day change in dsratio
    # ------------------------------------------------------------------ #
    slope = np.full(n, np.nan)
    sw = _SLOPE_WINDOW
    for t in range(sw, n):
        if np.isfinite(dsratio[t]) and np.isfinite(dsratio[t - sw]):
            slope[t] = dsratio[t] - dsratio[t - sw]

    df["ext4_downside_idio_beta_dsratio"] = dsratio
    df["ext4_downside_idio_beta_crashcov"] = crashcov
    df["ext4_downside_idio_beta_slope"] = slope

    return df
