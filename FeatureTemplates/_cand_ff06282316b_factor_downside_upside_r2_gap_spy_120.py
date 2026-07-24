"""
Downside vs Upside R² Gap vs SPY (120-day rolling).

Splits the trailing 120-day window into SPY-down days and SPY-up days,
computes the R² of regressing ticker returns on SPY returns separately
within each group, then returns downside_R2 - upside_R2.

A positive gap means the ticker co-moves more tightly with SPY during
market declines than during advances — a classical "downside beta dominance"
asymmetry that is distinct from beta itself. R² measures co-movement quality,
not magnitude.

Per-ticker proxy: fully causal (merge_asof backward on Date).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_factor_downside_upside_r2_gap_spy_120",
    "description": (
        "Downside R² minus Upside R² of ticker-vs-SPY return regression over a "
        "trailing 120-day window. Days are split by SPY return sign. Requires >=20 "
        "observations per side; otherwise NaN. Captures asymmetric market co-movement "
        "quality (tracking tightness on down days vs up days). Companion column: the "
        "downside R² and upside R² separately."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_downside_upside_r2_gap_spy_120_gap",
        "ff06282316b_factor_downside_upside_r2_gap_spy_120_down_r2",
        "ff06282316b_factor_downside_upside_r2_gap_spy_120_up_r2",
    ],
    "tags": ["factor", "spy", "r2", "asymmetry", "downside", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

_WINDOW = 120
_MIN_OBS = 20
_PRODUCED = METADATA["produces"]


def _r2_ols(x: np.ndarray, y: np.ndarray) -> float:
    """
    R² of regressing y on x (with intercept) using closed-form OLS.
    Returns NaN if degenerate.
    """
    n = len(x)
    if n < 2:
        return np.nan
    x_m = x - x.mean()
    y_m = y - y.mean()
    ss_xx = (x_m * x_m).sum()
    ss_yy = (y_m * y_m).sum()
    if ss_xx == 0.0 or ss_yy == 0.0:
        return np.nan
    beta = (x_m * y_m).sum() / ss_xx
    y_hat = beta * x_m  # centred
    ss_res = ((y_m - y_hat) ** 2).sum()
    r2 = 1.0 - ss_res / ss_yy
    # clamp to [0,1] — numerics can push slightly outside
    return float(np.clip(r2, 0.0, 1.0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (required on every path)
    for col in _PRODUCED:
        df[col] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY closes and merge
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    tmp = df[["Date", "Close"]].copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = pd.merge_asof(
        tmp.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    tmp = tmp.set_index(df.index)

    spy_vals = tmp["spy_close"].values
    close_vals = df["Close"].values
    n = len(df)

    # Pre-compute daily log returns (shift-1 = previous bar, causal)
    # ticker returns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk_ret = np.empty(n)
        tk_ret[0] = np.nan
        tk_ret[1:] = np.log(
            np.where(close_vals[:-1] != 0, close_vals[1:] / close_vals[:-1], np.nan)
        )

        spy_ret = np.empty(n)
        spy_ret[0] = np.nan
        spy_ret[1:] = np.log(
            np.where(spy_vals[:-1] != 0, spy_vals[1:] / spy_vals[:-1], np.nan)
        )

    gap_arr = np.full(n, np.nan)
    down_arr = np.full(n, np.nan)
    up_arr = np.full(n, np.nan)

    # Rolling window: for bar t (0-indexed), use bars [t-_WINDOW+1 .. t]
    # We need at least _WINDOW bars (index _WINDOW-1 onward)
    for t in range(_WINDOW - 1, n):
        w_spy = spy_ret[t - _WINDOW + 1 : t + 1]
        w_tk = tk_ret[t - _WINDOW + 1 : t + 1]

        # Drop NaN pairs
        mask = np.isfinite(w_spy) & np.isfinite(w_tk)
        xs = w_spy[mask]
        ys = w_tk[mask]

        if len(xs) < _MIN_OBS * 2:
            continue

        down_mask = xs < 0.0
        up_mask = xs >= 0.0

        n_down = down_mask.sum()
        n_up = up_mask.sum()

        if n_down < _MIN_OBS or n_up < _MIN_OBS:
            continue

        r2_down = _r2_ols(xs[down_mask], ys[down_mask])
        r2_up = _r2_ols(xs[up_mask], ys[up_mask])

        if np.isnan(r2_down) or np.isnan(r2_up):
            continue

        down_arr[t] = r2_down
        up_arr[t] = r2_up
        gap_arr[t] = r2_down - r2_up

    df[_PRODUCED[0]] = gap_arr
    df[_PRODUCED[1]] = down_arr
    df[_PRODUCED[2]] = up_arr

    return df
