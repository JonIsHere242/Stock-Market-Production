from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# Load index helper
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_beta_risk_beta_vol_of_vol_spy_120",
    "description": (
        "Beta-volatility (vol-of-beta) feature. "
        "Computes rolling 20-day OLS SPY-beta at every 5th bar (fixed-from-start stride), "
        "then takes the std of first-differences of those strided beta estimates over a "
        "trailing 120-bar span (~24 beta observations), normalised by the mean rolling-20d "
        "SPY return vol to isolate beta-churn from market-vol churn. "
        "Produces: (1) raw beta-vol-of-vol, (2) normalised beta-vol-of-vol, "
        "(3) the most-recent strided OLS beta itself. Per-ticker proxy; no cross-section."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_vol_raw",
        "ff06282316b_beta_vol_norm",
        "ff06282316b_beta_current",
    ],
    "tags": ["beta", "vol_of_vol", "spy", "risk", "rolling_ols"],
    "version": "1.0.0",
    "author": "feature-factory",
}

# Constants
_WIN_BETA = 20     # OLS window for beta
_STRIDE = 5        # compute beta every 5 bars (fixed from series start)
_SPAN = 120        # trailing bars to collect strided betas
_EPS = 1e-8


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-initialise all produced columns to NaN
    df["ff06282316b_beta_vol_raw"] = np.nan
    df["ff06282316b_beta_vol_norm"] = np.nan
    df["ff06282316b_beta_current"] = np.nan

    n = len(df)
    if n < _WIN_BETA + _STRIDE:
        return df

    # ------------------------------------------------------------------ #
    # 1.  Merge SPY close causal (backward merge_asof on Date)            #
    # ------------------------------------------------------------------ #
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "_spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(work.sort_values("Date"), spy_df.sort_values("Date"),
                         on="Date", direction="backward")
    # Restore original index alignment
    work.index = df.index

    spy_arr = work["_spy_close"].to_numpy(dtype=float)
    stk_arr = work["Close"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 2.  Compute per-bar SPY and stock log-returns                       #
    # ------------------------------------------------------------------ #
    spy_ret = np.empty(n)
    spy_ret[0] = np.nan
    spy_ret[1:] = np.log(spy_arr[1:] / np.where(spy_arr[:-1] == 0, np.nan, spy_arr[:-1]))

    stk_ret = np.empty(n)
    stk_ret[0] = np.nan
    stk_ret[1:] = np.log(stk_arr[1:] / np.where(stk_arr[:-1] == 0, np.nan, stk_arr[:-1]))

    # ------------------------------------------------------------------ #
    # 3.  Compute rolling-20d SPY return volatility (std of spy_ret)     #
    # ------------------------------------------------------------------ #
    spy_ret_s = pd.Series(spy_ret)
    spy_vol_rolling = spy_ret_s.rolling(_WIN_BETA, min_periods=_WIN_BETA).std().to_numpy()

    # ------------------------------------------------------------------ #
    # 4.  At each stride bar, compute OLS beta over trailing _WIN_BETA    #
    #     bars using vectorised numpy. Grid: i % _STRIDE == 0 from start. #
    # ------------------------------------------------------------------ #
    beta_at = np.full(n, np.nan)   # beta stored at its computation bar
    vol_at = np.full(n, np.nan)    # spy rolling vol at that bar

    # Stride grid fixed from start: bars 0, 5, 10, ...
    # We need at least _WIN_BETA observations ending at bar i.
    for i in range(n):
        if i % _STRIDE != 0:
            continue
        if i < _WIN_BETA - 1:
            continue
        # Window [i - _WIN_BETA + 1 .. i]
        x = spy_ret[i - _WIN_BETA + 1: i + 1]
        y = stk_ret[i - _WIN_BETA + 1: i + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        xv = x[valid]
        yv = y[valid]
        xbar = xv.mean()
        ybar = yv.mean()
        denom = np.sum((xv - xbar) ** 2)
        if denom < _EPS:
            continue
        beta_at[i] = np.sum((xv - xbar) * (yv - ybar)) / denom
        vol_at[i] = spy_vol_rolling[i]  # may be nan if spy_vol insufficient

    # ------------------------------------------------------------------ #
    # 5.  For each bar, look back over _SPAN bars and collect strided     #
    #     beta estimates; compute vol-of-beta (std of first diffs).       #
    #     We also compute the mean SPY vol over that same window.         #
    # ------------------------------------------------------------------ #
    beta_vol_raw = np.full(n, np.nan)
    beta_vol_norm = np.full(n, np.nan)
    beta_current = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - _SPAN + 1)
        # Collect stride-grid betas in [start..i] where grid bar has valid beta
        betas = []
        spy_vols_here = []
        for j in range(start, i + 1):
            if j % _STRIDE == 0 and np.isfinite(beta_at[j]):
                betas.append(beta_at[j])
                if np.isfinite(vol_at[j]):
                    spy_vols_here.append(vol_at[j])
        if len(betas) < 3:
            continue
        b_arr = np.array(betas)
        # Most recent valid beta
        beta_current[i] = b_arr[-1]
        # Std of first differences = vol of beta
        diffs = np.diff(b_arr)
        if len(diffs) == 0:
            continue
        bvol = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
        beta_vol_raw[i] = bvol
        # Normalise by mean SPY vol
        if len(spy_vols_here) >= 1:
            mean_spy_vol = np.mean(spy_vols_here)
            beta_vol_norm[i] = bvol / (mean_spy_vol + _EPS)

    df["ff06282316b_beta_vol_raw"] = beta_vol_raw
    df["ff06282316b_beta_vol_norm"] = beta_vol_norm
    df["ff06282316b_beta_current"] = beta_current

    return df
