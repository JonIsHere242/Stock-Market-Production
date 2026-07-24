"""
ff06282240_factor_beta_magnitude_gap
Conditional beta split by market-move MAGNITUDE (not sign).
beta_turbulent = cov(stock,SPY)/var(SPY) on high-|SPY_ret| days
beta_calm      = same on low-|SPY_ret| days
beta_magnitude_gap = turbulent - calm
Positive gap = convex stress beta (stock amplifies exposure on big-move days).
Window=150d, stride=5 (i%5==0 from series start), forward-filled.
Requires >=25 obs per leg, else NaN.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load _indexes helper ---
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282240_factor_beta_magnitude_gap",
    "description": (
        "Conditional beta split by market-move MAGNITUDE over a 150-day trailing "
        "window. Days are split by whether |SPY_ret| is above or below the window "
        "median. beta_turbulent = cov(stock,SPY)/var(SPY) on high-|move| days; "
        "beta_calm on low-|move| days; beta_magnitude_gap = turbulent - calm. "
        "Positive = stock exposure amplifies on big-move days (convex stress beta). "
        "Recomputed every 5 bars (causal grid from series start), forward-filled. "
        "Minimum 25 observations per leg required else NaN."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282240_factor_beta_magnitude_gap_turbulent",
        "ff06282240_factor_beta_magnitude_gap_calm",
        "ff06282240_factor_beta_magnitude_gap_gap",
    ],
    "tags": ["beta", "factor", "conditional", "magnitude", "market"],
    "version": "1.0.0",
    "author": "feature-factory ff06282240",
}

_WINDOW = 150
_STRIDE = 5
_MIN_OBS = 25


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-initialise all produced columns to NaN
    df["ff06282240_factor_beta_magnitude_gap_turbulent"] = np.nan
    df["ff06282240_factor_beta_magnitude_gap_calm"] = np.nan
    df["ff06282240_factor_beta_magnitude_gap_gap"] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # Pull SPY close and align to df dates
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build SPY return series aligned to df
    spy_ret_series = spy_close.pct_change()

    # Ensure df has a DatetimeIndex for merge_asof; work on a temp column approach
    df_dates = pd.to_datetime(df["Date"])
    spy_df = spy_ret_series.rename("spy_ret").reset_index()
    spy_df.columns = ["Date", "spy_ret"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    # merge_asof to attach spy_ret to each row (backward = no lookahead)
    df_tmp = df[["Date"]].copy()
    df_tmp["Date"] = df_dates
    df_tmp = pd.merge_asof(
        df_tmp.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # restore original order
    df_tmp = df_tmp.set_index(df.index)

    spy_ret_aligned = df_tmp["spy_ret"].values

    # Stock returns
    close = df["Close"].values.astype(np.float64)
    stk_ret = np.empty(len(close), dtype=np.float64)
    stk_ret[0] = np.nan
    stk_ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] == 0, np.nan, close[:-1])

    n = len(df)
    beta_turb = np.full(n, np.nan)
    beta_calm = np.full(n, np.nan)

    # Compute only on causal fixed-grid stride bars (i % stride == 0 from series start)
    # We start at the first bar where we have enough history: i >= WINDOW
    for i in range(n):
        if i < _WINDOW:
            continue
        if i % _STRIDE != 0:
            continue

        spy_w = spy_ret_aligned[i - _WINDOW: i]
        stk_w = stk_ret[i - _WINDOW: i]

        # mask out NaN pairs
        mask = np.isfinite(spy_w) & np.isfinite(stk_w)
        spy_w_v = spy_w[mask]
        stk_w_v = stk_w[mask]

        if len(spy_w_v) < 2 * _MIN_OBS:
            continue

        abs_spy = np.abs(spy_w_v)
        med = np.median(abs_spy)

        high_mask = abs_spy >= med
        low_mask = ~high_mask

        spy_high = spy_w_v[high_mask]
        stk_high = stk_w_v[high_mask]
        spy_low = spy_w_v[low_mask]
        stk_low = stk_w_v[low_mask]

        if len(spy_high) < _MIN_OBS or len(spy_low) < _MIN_OBS:
            continue

        # beta = cov(stk, spy) / var(spy)
        var_high = np.var(spy_high, ddof=1)
        var_low = np.var(spy_low, ddof=1)

        if var_high == 0 or not np.isfinite(var_high):
            b_turb = np.nan
        else:
            b_turb = np.cov(stk_high, spy_high, ddof=1)[0, 1] / var_high

        if var_low == 0 or not np.isfinite(var_low):
            b_calm = np.nan
        else:
            b_calm = np.cov(stk_low, spy_low, ddof=1)[0, 1] / var_low

        beta_turb[i] = b_turb
        beta_calm[i] = b_calm

    # Forward-fill the stride-computed values
    def _ffill(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        last = np.nan
        for k in range(len(out)):
            if np.isfinite(out[k]):
                last = out[k]
            else:
                out[k] = last
        return out

    beta_turb_ff = _ffill(beta_turb)
    beta_calm_ff = _ffill(beta_calm)

    with np.errstate(invalid="ignore"):
        beta_gap = np.where(
            np.isfinite(beta_turb_ff) & np.isfinite(beta_calm_ff),
            beta_turb_ff - beta_calm_ff,
            np.nan,
        )

    df["ff06282240_factor_beta_magnitude_gap_turbulent"] = beta_turb_ff
    df["ff06282240_factor_beta_magnitude_gap_calm"] = beta_calm_ff
    df["ff06282240_factor_beta_magnitude_gap_gap"] = beta_gap

    return df
