"""
Bawa-Lindenberg LPM beta vs SPY (target = mean SPY return over 250 days).
Downside days = days where SPY return < rolling mean SPY return.
LPM-beta = E[(r-mean_r)(m-mean_m)|down] / E[(m-mean_m)^2|down].
Feature = LPM_beta - full OLS beta (downside excess co-movement vs market mean).
Per-ticker proxy; causal; no cross-sectional data required.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: market index
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06290014i_beta_risk_lpm_beta_target_meanret_spy_250",
    "description": (
        "Bawa-Lindenberg LPM beta vs SPY with downside defined as SPY daily "
        "return below its rolling 250-day mean. "
        "ff06290014i_lpm_beta: raw downside beta; "
        "ff06290014i_lpm_beta_excess: LPM-beta minus full OLS beta "
        "(downside excess co-movement signal); "
        "ff06290014i_lpm_down_frac: fraction of days classified as downside. "
        "All computed in causal 250-day rolling windows, min 40 observations."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_lpm_beta",
        "ff06290014i_lpm_beta_excess",
        "ff06290014i_lpm_down_frac",
    ],
    "tags": ["beta", "downside_risk", "lpm", "spy", "risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06290014i",
}

# ---------------------------------------------------------------------------
_WINDOW = 250
_MIN_OBS = 40
_MIN_DOWN = 20
_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up-front (required on every path)
    for col in METADATA["produces"]:
        df[col] = np.nan

    if len(df) < 2:
        return df

    # --- SPY returns (market) via index helper ---
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Align SPY to this ticker's dates via backward merge_asof
    dates = df["Date"].copy()
    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    dates_df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    merged = pd.merge_asof(
        dates_df.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.set_index(dates_df.sort_values("Date").index).reindex(df.index)

    spy_px = merged["spy_close"].values.astype(float)
    # SPY daily log returns
    m = np.full(len(df), np.nan)
    m[1:] = np.diff(np.log(np.where(spy_px > 0, spy_px, np.nan)))

    # Stock daily log returns
    close = df["Close"].values.astype(float)
    r = np.full(len(df), np.nan)
    r[1:] = np.diff(np.log(np.where(close > 0, close, np.nan)))

    n = len(df)
    lpm_beta_arr = np.full(n, np.nan)
    lpm_excess_arr = np.full(n, np.nan)
    down_frac_arr = np.full(n, np.nan)

    # Rolling computation using a stride-aligned approach:
    # For each bar t (0-indexed), use the window [max(0, t-W+1) .. t]
    # We need at least MIN_OBS total and MIN_DOWN downside bars.
    # To avoid O(n^2) python loops we vectorise using numpy striding where
    # possible, but window computations require per-bar stats.  The window
    # is large (250) and n ~700, so the total work is manageable with numpy
    # slicing (no pure Python per-row loop over individual elements).

    for t in range(_WINDOW - 1, n):
        i0 = t - _WINDOW + 1
        r_win = r[i0 : t + 1]
        m_win = m[i0 : t + 1]

        # Drop NaN pairs
        mask = np.isfinite(r_win) & np.isfinite(m_win)
        rw = r_win[mask]
        mw = m_win[mask]
        if len(rw) < _MIN_OBS:
            continue

        mean_r = rw.mean()
        mean_m = mw.mean()

        dr = rw - mean_r
        dm = mw - mean_m

        # Full OLS beta = cov(r,m) / var(m)
        var_m = (dm * dm).mean()
        if var_m < _EPS:
            continue
        ols_beta = (dr * dm).mean() / var_m

        # Downside: m < mean_m
        down_mask = mw < mean_m
        n_down = down_mask.sum()
        down_frac = n_down / len(rw)

        if n_down < _MIN_DOWN:
            # Not enough downside days; store NaN for LPM metrics
            down_frac_arr[t] = down_frac
            continue

        dr_down = dr[down_mask]
        dm_down = dm[down_mask]

        denom_lpm = (dm_down * dm_down).mean()
        if denom_lpm < _EPS:
            continue

        lpm_b = (dr_down * dm_down).mean() / denom_lpm

        lpm_beta_arr[t] = lpm_b
        lpm_excess_arr[t] = lpm_b - ols_beta
        down_frac_arr[t] = down_frac

    df["ff06290014i_lpm_beta"] = lpm_beta_arr
    df["ff06290014i_lpm_beta_excess"] = lpm_excess_arr
    df["ff06290014i_lpm_down_frac"] = down_frac_arr

    return df
