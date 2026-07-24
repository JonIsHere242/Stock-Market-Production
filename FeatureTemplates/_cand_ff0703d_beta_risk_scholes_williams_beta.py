"""
_cand_ff0703d_beta_risk_scholes_williams_beta.py -- Scholes-Williams thin-trading-adjusted beta.

METHOD
------
Classic Scholes-Williams (1977) beta correction for non-synchronous trading: instead of a
single contemporaneous OLS beta, sum a lagged beta, a contemporaneous beta and a leading beta
of the stock's return against the market (SPY) return, then rescale by (1 + 2*rho) where rho
is the market return's own lag-1 autocorrelation. This down-weights spurious beta inflation/
deflation caused by stale prices.

CAUSALITY
---------
The "lead" term, cov(r_i,t, r_m,t+1)/var(r_m,t+1), literally needs tomorrow's market return
for the LAST observation in any window ending at the current bar -- so that final observation
is dropped from the lead-beta pairing (x = r_i[:-1] paired with y = r_m[1:] within the window,
i.e. every pair used is (return at bar u, return at bar u+1) for u strictly before the last
bar in the window). No value outside [window_start, current_bar] is ever touched.

To keep this affordable and strictly causal under truncation, the rolling 120-bar estimation
is only refreshed on a FIXED-FROM-START grid (row_position % 5 == 0) and forward-filled to the
bars in between -- never anchored to the last bar, so it is stable under any truncation of the
tail of the series (required by the causal gate).

PER-TICKER PROXY NOTE
----------------------
This is computed per-ticker against SPY as "the market" (the only broad-market index available
in this per-ticker pipeline), which is the standard faithful proxy for CRSP-value-weighted-index
based Scholes-Williams studies.

PRODUCES
--------
  ff0703d_sw_beta        : level -- the SW-adjusted beta, refreshed every 5 bars, ffilled.
  ff0703d_sw_beta_chg20  : 20-bar change in ff0703d_sw_beta (thin-trading-adjusted beta drift).
  ff0703d_sw_adj         : ff0703d_sw_beta - beta_con (how much the correction moves the naive
                            contemporaneous beta; sign/magnitude ~ non-synchronous-trading effect).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import importlib.util as _ilu
from pathlib import Path as _P

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff0703d_beta_risk_scholes_williams_beta",
    "description": (
        "Scholes-Williams (1977) thin-trading-adjusted beta vs SPY: sums lagged, "
        "contemporaneous and (causally truncated) leading betas over a rolling 120-bar "
        "window, rescaled by 1+2*rho(market lag-1 autocorr); refreshed on a fixed 5-bar "
        "grid from series start and forward-filled. Faithful per-ticker proxy using SPY "
        "as 'the market' (no CRSP value-weighted index available in this pipeline)."
    ),
    "requires": ["Close"],
    "produces": ["ff0703d_sw_beta", "ff0703d_sw_beta_chg20", "ff0703d_sw_adj"],
    "tags": ["beta_risk", "market_microstructure", "candidate"],
    "version": "1.0",
    "author": "feature-factory (auto-generated, faithful per-ticker SPY-beta proxy)",
}

_WINDOW = 120
_STRIDE = 5
_MIN_OBS = 20


def _beta_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """cov(x,y)/var(y) with pairwise-finite masking and a var floor guard."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < _MIN_OBS:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    yv_var = float(np.var(yv, ddof=1))
    if not np.isfinite(yv_var) or yv_var < 1e-10:
        return np.nan
    cov = float(np.cov(xv, yv, ddof=1)[0, 1])
    if not np.isfinite(cov):
        return np.nan
    return cov / yv_var


def _lag1_autocorr(y: np.ndarray) -> float:
    if y.shape[0] < _MIN_OBS + 1:
        return np.nan
    x = y[1:]
    y0 = y[:-1]
    mask = np.isfinite(x) & np.isfinite(y0)
    if int(mask.sum()) < _MIN_OBS:
        return np.nan
    xv = x[mask]
    yv = y0[mask]
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return np.nan
    corr = float(np.corrcoef(xv, yv)[0, 1])
    return corr if np.isfinite(corr) else np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    sw_beta = np.full(n, np.nan, dtype="float64")
    beta_con_arr = np.full(n, np.nan, dtype="float64")

    if n == 0:
        df["ff0703d_sw_beta"] = pd.Series(dtype="float64")
        df["ff0703d_sw_beta_chg20"] = pd.Series(dtype="float64")
        df["ff0703d_sw_adj"] = pd.Series(dtype="float64")
        return df

    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype="float64")
    r_i = np.full(n, np.nan, dtype="float64")
    if n > 1:
        prev = close[:-1]
        cur = close[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_i[1:] = np.where(prev != 0, (cur - prev) / prev, np.nan)

    # Market return via backward merge_asof of SPY close onto this ticker's Date grid.
    spy_close = _indexes.index_close("SPY")
    r_m = np.full(n, np.nan, dtype="float64")
    if len(spy_close) > 0 and "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        spy_df = spy_df.sort_values("Date")
        left = pd.DataFrame({"Date": dates}).reset_index(drop=True)
        left_sorted = left.sort_values("Date")
        merged = pd.merge_asof(left_sorted, spy_df, on="Date", direction="backward")
        merged = merged.reindex(left_sorted.index).sort_index()
        spy_aligned = merged["spy_close"].to_numpy(dtype="float64")
        if n > 1:
            prev_m = spy_aligned[:-1]
            cur_m = spy_aligned[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                r_m[1:] = np.where(
                    np.isfinite(prev_m) & (prev_m != 0),
                    (cur_m - prev_m) / prev_m,
                    np.nan,
                )

    have_market = np.isfinite(r_m).sum() >= _MIN_OBS

    if have_market:
        grid_idx = np.arange(0, n, _STRIDE)
        for i in grid_idx:
            start = max(0, i - _WINDOW + 1)
            ri_w = r_i[start:i + 1]
            rm_w = r_m[start:i + 1]

            if rm_w.shape[0] < _MIN_OBS:
                continue

            rm_finite = rm_w[np.isfinite(rm_w)]
            if rm_finite.shape[0] < _MIN_OBS:
                continue
            rm_var_full = float(np.var(rm_finite, ddof=1))
            if not np.isfinite(rm_var_full) or rm_var_full < 1e-10:
                continue

            # contemporaneous
            beta_con = _beta_ratio(ri_w, rm_w)
            # lagged: (r_i[t], r_m[t-1])
            if ri_w.shape[0] >= 2:
                beta_lag = _beta_ratio(ri_w[1:], rm_w[:-1])
            else:
                beta_lag = np.nan
            # leading, causal: drop final window observation -> (r_i[t], r_m[t+1]) for t up to i-1
            if ri_w.shape[0] >= 2:
                beta_lead = _beta_ratio(ri_w[:-1], rm_w[1:])
            else:
                beta_lead = np.nan

            rho = _lag1_autocorr(rm_w)

            beta_con_arr[i] = beta_con

            if (
                np.isfinite(beta_con)
                and np.isfinite(beta_lag)
                and np.isfinite(beta_lead)
                and np.isfinite(rho)
            ):
                denom = 1.0 + 2.0 * rho
                if abs(denom) >= 1e-4:
                    sw_beta[i] = (beta_lag + beta_con + beta_lead) / denom

    sw_beta_s = pd.Series(sw_beta, index=df.index).ffill()
    beta_con_s = pd.Series(beta_con_arr, index=df.index).ffill()

    df["ff0703d_sw_beta"] = sw_beta_s
    df["ff0703d_sw_beta_chg20"] = sw_beta_s - sw_beta_s.shift(20)
    df["ff0703d_sw_adj"] = sw_beta_s - beta_con_s

    return df
