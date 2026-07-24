from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional helper: market index closes
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282328d_cross_asset_jensen_alpha_level_spy_120",
    "description": (
        "Rolling 120-day OLS of ticker daily returns on SPY daily returns. "
        "Produces (1) Jensen alpha annualised (intercept * 252), (2) rolling OLS "
        "beta, and (3) a 20-bar slope of the annualised alpha to capture "
        "alpha momentum. Per-ticker proxy -- no cross-sectional ranking required. "
        "SPY returns come from the _indexes helper; missing SPY dates handled via "
        "backward merge_asof. Causally strided alpha-slope is computed on a fixed "
        "grid anchored to bar 0 to avoid lookahead under truncation."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282328d_cross_asset_jensen_alpha_level_spy_120_alpha",
        "ff06282328d_cross_asset_jensen_alpha_level_spy_120_beta",
        "ff06282328d_cross_asset_jensen_alpha_level_spy_120_alpha_slope",
    ],
    "tags": ["cross_asset", "regression", "alpha", "beta", "spy", "jensen"],
    "version": "1.0",
    "author": "feature-factory ff06282328d",
}

_WINDOW = 120
_SLOPE_WINDOW = 20
_COL_ALPHA = "ff06282328d_cross_asset_jensen_alpha_level_spy_120_alpha"
_COL_BETA = "ff06282328d_cross_asset_jensen_alpha_level_spy_120_beta"
_COL_SLOPE = "ff06282328d_cross_asset_jensen_alpha_level_spy_120_alpha_slope"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN on EVERY code path
    df[_COL_ALPHA] = np.nan
    df[_COL_BETA] = np.nan
    df[_COL_SLOPE] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # -----------------------------------------------------------------------
    # 1. Fetch SPY closes and align to this ticker's dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
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
    # Restore original order
    work = work.set_index(df.index)

    spy_c = work["spy_close"].values.astype(float)
    tkr_c = work["Close"].values.astype(float)
    n = len(df)

    # -----------------------------------------------------------------------
    # 2. Daily returns (log-return, shifted so ret[t] = log(C[t]/C[t-1]))
    # -----------------------------------------------------------------------
    tkr_ret = np.empty(n, dtype=float)
    spy_ret = np.empty(n, dtype=float)
    tkr_ret[0] = np.nan
    spy_ret[0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        tkr_ret[1:] = np.where(
            tkr_c[:-1] > 0, np.log(tkr_c[1:] / tkr_c[:-1]), np.nan
        )
        spy_ret[1:] = np.where(
            spy_c[:-1] > 0, np.log(spy_c[1:] / spy_c[:-1]), np.nan
        )

    # -----------------------------------------------------------------------
    # 3. Rolling OLS: alpha (annualised) and beta over _WINDOW bars
    #    Using vectorised prefix sums for O(n) computation.
    # -----------------------------------------------------------------------
    alpha_arr = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)

    # We need at least _WINDOW valid return pairs; start from index _WINDOW
    # (returns are available from index 1 onward, so window [t-_WINDOW+1..t])
    for t in range(_WINDOW, n):
        y = tkr_ret[t - _WINDOW + 1 : t + 1]
        x = spy_ret[t - _WINDOW + 1 : t + 1]

        # Drop NaN pairs
        mask = np.isfinite(x) & np.isfinite(y)
        ym = y[mask]
        xm = x[mask]

        if len(xm) < max(10, _WINDOW // 2):
            continue

        xbar = xm.mean()
        ybar = ym.mean()
        xc = xm - xbar
        yc = ym - ybar
        xx = np.dot(xc, xc)
        xy = np.dot(xc, yc)

        if xx < 1e-12:
            # Near-zero SPY variance: set beta=1, alpha=0 per spec guidance
            beta_arr[t] = 1.0
            alpha_arr[t] = 0.0
        else:
            b = xy / xx
            a = ybar - b * xbar
            beta_arr[t] = b
            alpha_arr[t] = a * 252.0  # annualise

    df[_COL_ALPHA] = alpha_arr
    df[_COL_BETA] = beta_arr

    # -----------------------------------------------------------------------
    # 4. Rolling slope of annualised alpha over _SLOPE_WINDOW bars
    #    (OLS slope of alpha vs bar index -- causal, no lookahead)
    # -----------------------------------------------------------------------
    slope_arr = np.full(n, np.nan)
    x_idx = np.arange(_SLOPE_WINDOW, dtype=float)
    x_idx_c = x_idx - x_idx.mean()
    xx_s = np.dot(x_idx_c, x_idx_c)

    for t in range(_WINDOW + _SLOPE_WINDOW - 1, n):
        window = alpha_arr[t - _SLOPE_WINDOW + 1 : t + 1]
        if np.sum(np.isfinite(window)) < _SLOPE_WINDOW // 2:
            continue
        # Replace NaN with window mean for slope calc (conservative)
        wmean = np.nanmean(window)
        w_filled = np.where(np.isfinite(window), window, wmean)
        yc_s = w_filled - w_filled.mean()
        if xx_s < 1e-12:
            continue
        slope_arr[t] = np.dot(x_idx_c, yc_s) / xx_s

    df[_COL_SLOPE] = slope_arr

    return df
