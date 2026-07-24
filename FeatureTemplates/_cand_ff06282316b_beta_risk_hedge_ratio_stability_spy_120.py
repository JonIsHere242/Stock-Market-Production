"""
Feature block: ff06282316b_beta_risk_hedge_ratio_stability_spy_120
Vein: beta_risk

Rolling hedge-ratio instability: computes a 30-day OLS beta vs SPY on a stride-5
grid over a 120-bar trailing window, then returns the std-dev of those short-window
betas normalised by mean-absolute-beta. Lower = stable, reliably hedgeable exposure.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helper: SPY index
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282316b_beta_risk_hedge_ratio_stability_spy_120",
    "description": (
        "Rolling hedge-ratio instability score. "
        "At each bar on a stride-5 fixed-from-start grid, compute the 30-day OLS beta "
        "of ticker log-returns vs SPY log-returns using only past data. "
        "Over a 120-bar trailing window of those stride-5 beta estimates, compute "
        "std(betas) / (mean(|betas|) + 1e-6). Forward-fill between stride bars. "
        "Also emits the most-recent beta level and the slope (beta trend) over the "
        "same 120-bar window. Per-ticker proxy; no cross-sectional ranking needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_risk_hedge_ratio_stability_spy_120_instability",
        "ff06282316b_beta_risk_hedge_ratio_stability_spy_120_beta",
        "ff06282316b_beta_risk_hedge_ratio_stability_spy_120_beta_slope",
    ],
    "tags": ["beta", "hedge_ratio", "stability", "spy", "risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

_PRODUCED = METADATA["produces"]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_STRIDE = 5          # compute OLS beta every 5 bars (fixed-from-start)
_BETA_WIN = 30       # bars for each short-window OLS beta
_POOL_WIN = 120      # trailing number of bars over which to pool stride-5 betas
_EPS = 1e-6


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """Return OLS slope of y on x; NaN if degenerate."""
    if len(y) < 5:
        return np.nan
    xd = x - x.mean()
    denom = (xd * xd).sum()
    if denom < _EPS:
        return np.nan
    return float((xd * y).sum() / denom)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path
    for col in _PRODUCED:
        df[col] = np.nan

    n = len(df)
    if n < _BETA_WIN + 1:
        return df

    # ------------------------------------------------------------------
    # 1. Align SPY log-returns to ticker dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    # Build a small DataFrame from ticker dates so we can merge_asof
    ticker_dates = pd.DataFrame({"Date": df["Date"].values})
    if not pd.api.types.is_datetime64_any_dtype(ticker_dates["Date"]):
        ticker_dates["Date"] = pd.to_datetime(ticker_dates["Date"])

    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    if not pd.api.types.is_datetime64_any_dtype(spy_df["Date"]):
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        ticker_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df row order
    merged = merged.set_index(ticker_dates.sort_values("Date").index)
    merged = merged.reindex(df.index)

    spy_arr = merged["spy_close"].values.astype(float)
    close_arr = df["Close"].values.astype(float)

    # Log-returns (shift by 1 so return[t] = log(close[t]/close[t-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ticker_ret = np.where(
            close_arr[:-1] > _EPS,
            np.log(close_arr[1:] / close_arr[:-1]),
            np.nan,
        )
        spy_ret = np.where(
            spy_arr[:-1] > _EPS,
            np.log(spy_arr[1:] / spy_arr[:-1]),
            np.nan,
        )

    # Pad so ret[i] corresponds to df row i (ret at row 0 is NaN)
    ticker_ret = np.concatenate([[np.nan], ticker_ret])
    spy_ret = np.concatenate([[np.nan], spy_ret])

    # ------------------------------------------------------------------
    # 2. On a fixed-from-start stride-5 grid, compute 30-bar OLS betas
    # ------------------------------------------------------------------
    # beta_arr[i] = OLS beta using rows [i-BETA_WIN+1 .. i] (causal)
    beta_at_stride = {}  # row_index -> beta value

    for i in range(n):
        if i % _STRIDE != 0:
            continue
        if i < _BETA_WIN - 1:
            continue
        start = i - _BETA_WIN + 1
        y_win = ticker_ret[start : i + 1]
        x_win = spy_ret[start : i + 1]
        # Require enough non-NaN pairs
        valid = ~(np.isnan(y_win) | np.isnan(x_win))
        if valid.sum() < 5:
            continue
        beta_at_stride[i] = _ols_beta(y_win[valid], x_win[valid])

    if not beta_at_stride:
        return df

    # ------------------------------------------------------------------
    # 3. For each stride bar, pool the last POOL_WIN bars worth of stride
    #    betas (i.e., the last POOL_WIN // STRIDE stride observations)
    # ------------------------------------------------------------------
    n_pool_strides = max(1, _POOL_WIN // _STRIDE)  # = 24 strides

    stride_indices = sorted(beta_at_stride.keys())

    # Arrays to fill (only at stride positions; forward-filled later)
    instab_arr = np.full(n, np.nan)
    beta_arr = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)

    for pos, idx in enumerate(stride_indices):
        pool_start = max(0, pos - n_pool_strides + 1)
        pool_idxs = stride_indices[pool_start : pos + 1]
        pool_betas = np.array([beta_at_stride[k] for k in pool_idxs], dtype=float)
        pool_betas = pool_betas[~np.isnan(pool_betas)]

        if len(pool_betas) < 2:
            continue

        std_b = float(np.std(pool_betas, ddof=1))
        mean_abs_b = float(np.mean(np.abs(pool_betas)))
        instability = std_b / (mean_abs_b + _EPS)

        instab_arr[idx] = instability
        beta_arr[idx] = beta_at_stride[idx]

        # Slope of betas across pool window (simple linear trend vs position)
        if len(pool_betas) >= 3:
            xs = np.arange(len(pool_betas), dtype=float)
            xs -= xs.mean()
            denom = (xs * xs).sum()
            if denom > _EPS:
                slope_arr[idx] = float((xs * pool_betas).sum() / denom)

    # ------------------------------------------------------------------
    # 4. Forward-fill between stride bars
    # ------------------------------------------------------------------
    def _ffill(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        last = np.nan
        for i in range(len(out)):
            if not np.isnan(out[i]):
                last = out[i]
            else:
                out[i] = last
        return out

    instab_arr = _ffill(instab_arr)
    beta_arr = _ffill(beta_arr)
    slope_arr = _ffill(slope_arr)

    df[_PRODUCED[0]] = instab_arr
    df[_PRODUCED[1]] = beta_arr
    df[_PRODUCED[2]] = slope_arr

    return df
