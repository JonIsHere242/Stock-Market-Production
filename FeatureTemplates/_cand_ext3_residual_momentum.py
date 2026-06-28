"""
Residual (idiosyncratic) momentum feature block.
Blitz, Huij & Martens (2011) – residual momentum is the 11-month (skip last
month) cumulative idiosyncratic return scaled by residual volatility, where
idiosyncratic returns come from a trailing 120-day market-model regression
(stock daily return ~ SPY daily return).  A 6-month variant is also produced.

Per-ticker proxy: fully causal (rolling OLS approximated with a fast
vectorised closed-form over a 120-day window on every bar), so no cross-
sectional data is needed.  SPY is loaded via the _indexes helper.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path; no package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_indexes)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_residual_momentum",
    "description": (
        "Residual (idiosyncratic) momentum (Blitz, Huij & Martens 2011). "
        "At each bar a 120-day rolling market-model regression (stock daily "
        "return ~ SPY daily return) is estimated in closed form; its residuals "
        "are the idiosyncratic returns.  Two signals are produced: the "
        "11-month (skipping the most recent 21 trading days) cumulative "
        "residual return scaled by residual vol (ext3_resmom_11m), and the "
        "analogous 6-month version (ext3_resmom_6m).  A raw cumulative "
        "idiosyncratic return over the 11-month window (ext3_resmom_raw) is "
        "also emitted as a complementary level feature.  Per-ticker proxy: "
        "all computation is causal and within-ticker."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext3_resmom_11m",   # 11-month resid-mom scaled by resid vol
        "ext3_resmom_6m",    # 6-month resid-mom scaled by resid vol
        "ext3_resmom_raw",   # raw cumulative idio return (11m window)
    ],
    "tags": ["momentum", "idiosyncratic", "market-model", "residual", "blitz2011"],
    "version": "1.0.0",
    "author": "Spec: Round-4 expansion (osap_idiovolaht); method: Blitz, Huij & Martens (2011)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REG_WIN   = 120   # rolling regression window (trading days)
_SKIP      = 21    # skip most-recent 21 days (≈1 month)
_MOM11_WIN = 231   # 11-month lookback in trading days (≈252-21)
_MOM6_WIN  = 105   # 6-month lookback in trading days (≈126-21)
_VOL_FLOOR = 1e-8  # prevent division by zero in vol scaling


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual momentum features; returns df with new columns appended."""
    n = len(df)

    # Initialise output arrays to NaN
    resmom_11m  = np.full(n, np.nan)
    resmom_6m   = np.full(n, np.nan)
    resmom_raw  = np.full(n, np.nan)

    if n < _REG_WIN + _MOM11_WIN + _SKIP + 2:
        # Not enough history for any valid bar; return early
        df["ext3_resmom_11m"] = resmom_11m
        df["ext3_resmom_6m"]  = resmom_6m
        df["ext3_resmom_raw"] = resmom_raw
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close; align to df by Date (merge_asof, backward)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spy_ser = _indexes.index_close("SPY")
            spy_df = spy_ser.reset_index()
            spy_df.columns = ["Date", "_spy_close"]
            spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        except Exception:
            spy_df = None

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])

    if spy_df is not None:
        work = pd.merge_asof(
            work.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original order
        work = work.set_index(work.index).reindex(
            work.sort_values("Date").index
        ).reset_index(drop=True)
        # Actually we need to keep the original row order aligned to df
        # Re-merge properly keeping df row order
        work2 = df[["Date"]].copy().reset_index(drop=True)
        work2["Date"] = pd.to_datetime(work2["Date"])
        work2 = pd.merge_asof(
            work2.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # work2 is sorted by Date; recover original index order via a positional map
        orig_order = df[["Date"]].copy().reset_index()
        orig_order["Date"] = pd.to_datetime(orig_order["Date"])
        work2 = work2.sort_values("Date")
        # build spy array aligned to original df row order
        spy_close_sorted = work2["_spy_close"].values  # sorted by date
        # We need them in the same date-sort order as df after sort
        # Simpler: re-join on Date directly preserving index
        tmp = df[["Date"]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        tmp = tmp.reset_index()          # col "index" = original row position
        tmp_sorted = tmp.sort_values("Date")
        tmp_sorted = pd.merge_asof(
            tmp_sorted,
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        tmp_sorted = tmp_sorted.set_index("index").sort_index()
        spy_arr = tmp_sorted["_spy_close"].values.astype(float)
        spy_ret = np.empty(n, dtype=float)
        spy_ret[0] = np.nan
        spy_ret[1:] = np.where(
            (spy_arr[:-1] > 0) & np.isfinite(spy_arr[:-1]) & np.isfinite(spy_arr[1:]),
            spy_arr[1:] / spy_arr[:-1] - 1.0,
            np.nan,
        )
    else:
        spy_ret = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 2. Stock daily log-returns (simple returns are fine for short windows)
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(float)
    stk_ret = np.empty(n, dtype=float)
    stk_ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
    stk_ret[1:] = np.where(np.isfinite(ratio) & (close[:-1] > 0), ratio - 1.0, np.nan)

    # ------------------------------------------------------------------
    # 3. Rolling 120-day OLS: stock_ret = alpha + beta * spy_ret + eps
    #    Closed-form rolling OLS using prefix sums.
    #    residual[t] = stk_ret[t] - (alpha_hat + beta_hat * spy_ret[t])
    # ------------------------------------------------------------------
    # We compute residuals bar-by-bar using a sliding window of length _REG_WIN.
    # To avoid O(n^2) loops we use vectorised prefix sums.

    W = _REG_WIN
    x = spy_ret   # market return
    y = stk_ret   # stock return

    # Replace NaN with 0 for rolling sums (will cause NaN beta when counts low)
    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    cnt = valid.astype(float)

    # Prefix sums for rolling windows
    def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(a)])
        out = cs[w:] - cs[:-w]
        return np.concatenate([np.full(w - 1, np.nan), out])

    n_valid  = _rolling_sum(cnt,      W)
    sum_x    = _rolling_sum(xv,       W)
    sum_y    = _rolling_sum(yv,       W)
    sum_xx   = _rolling_sum(xv * xv,  W)
    sum_xy   = _rolling_sum(xv * yv,  W)

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = n_valid * sum_xx - sum_x * sum_x
        beta  = np.where(np.abs(denom) > 1e-12,
                         (n_valid * sum_xy - sum_x * sum_y) / denom,
                         np.nan)
        alpha = np.where(np.isfinite(beta) & (n_valid > 0),
                         (sum_y - beta * sum_x) / n_valid,
                         np.nan)

    # Residual at each bar using the model estimated over the PAST W bars
    # (the window ending AT bar t includes bar t itself -- still causal because
    # we only use the residual of bar t after the regression is fitted on [t-W+1..t])
    with np.errstate(invalid="ignore"):
        resid = np.where(
            np.isfinite(alpha) & np.isfinite(beta) & valid,
            y - (alpha + beta * x),
            np.nan,
        )

    # ------------------------------------------------------------------
    # 4. Build cumulative residual returns and scaling vol
    #    For each bar t:
    #      window = resid[t - SKIP - MOM_WIN : t - SKIP]  (skip last SKIP days)
    #    vol = rolling std of residuals over the regression window
    # ------------------------------------------------------------------
    resid_sq = np.where(np.isfinite(resid), resid * resid, 0.0)
    resid_cnt_v = np.where(np.isfinite(resid), 1.0, 0.0)

    def _rolling_sum_1d(a: np.ndarray, w: int) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(a)])
        out = cs[w:] - cs[:-w]
        return np.concatenate([np.full(w - 1, np.nan), out])

    # Rolling residual vol (std) over W bars for scaling
    r_sum   = _rolling_sum_1d(np.where(np.isfinite(resid), resid, 0.0), W)
    r_sumsq = _rolling_sum_1d(resid_sq, W)
    r_cnt   = _rolling_sum_1d(resid_cnt_v, W)

    with np.errstate(invalid="ignore", divide="ignore"):
        r_mean = np.where(r_cnt > 0, r_sum / r_cnt, np.nan)
        r_var  = np.where(
            r_cnt > 1,
            np.maximum(r_sumsq / r_cnt - r_mean * r_mean, 0.0),
            np.nan,
        )
        resid_vol = np.sqrt(r_var)  # annualise below if desired (left daily)

    # Cumulative residual return over [t - SKIP - MOM_WIN, t - SKIP)
    # = sum of resid in that window.
    # Implement with prefix sums of resid and resid_cnt.

    resid_filled = np.where(np.isfinite(resid), resid, 0.0)
    cs_resid = np.concatenate([[0.0], np.cumsum(resid_filled)])
    cs_cnt   = np.concatenate([[0.0], np.cumsum(resid_cnt_v)])

    def _window_sum(cs: np.ndarray, start_offset: int, end_offset: int) -> np.ndarray:
        """
        At each bar t (0-based), sum cs[t - end_offset + 1 .. t - start_offset].
        start_offset >= 1 (skip), end_offset > start_offset.
        Result is NaN when indices are out of bounds.
        """
        out = np.full(n, np.nan)
        for t in range(end_offset, n):
            i_lo = t - end_offset     # inclusive lower index into original arrays
            i_hi = t - start_offset   # inclusive upper index
            if i_lo < 0 or i_hi < i_lo:
                continue
            out[t] = cs[i_hi + 1] - cs[i_lo]
        return out

    # Vectorised version (avoid O(n^2) loop using prefix sums)
    def _window_sum_vec(cs: np.ndarray, skip: int, window: int) -> np.ndarray:
        """
        For bar t: sum of array[t - skip - window + 1 .. t - skip] (inclusive).
        Equivalent to cs[t - skip + 1] - cs[t - skip - window + 1].
        Valid when t >= skip + window - 1.
        """
        hi = np.arange(n) - skip          # cs index for upper bound (exclusive)
        lo = np.arange(n) - skip - window  # cs index for lower bound (exclusive)
        out = np.full(n, np.nan)
        valid_idx = (hi >= 0) & (lo >= 0) & (hi <= n) & (lo <= n)
        out[valid_idx] = cs[hi[valid_idx] + 1] - cs[lo[valid_idx] + 1]
        return out

    # 11-month window
    cum11  = _window_sum_vec(cs_resid, _SKIP, _MOM11_WIN)
    cnt11  = _window_sum_vec(cs_cnt,   _SKIP, _MOM11_WIN)

    # 6-month window
    cum6   = _window_sum_vec(cs_resid, _SKIP, _MOM6_WIN)
    cnt6   = _window_sum_vec(cs_cnt,   _SKIP, _MOM6_WIN)

    # Scale by residual vol (at bar t - SKIP for lookahead-safety; use t-1 for simplicity)
    # We use the rolling resid_vol at bar (t - _SKIP) -- shift resid_vol forward by SKIP
    # which means resid_vol[t - SKIP] = resid_vol[t - _SKIP]
    resid_vol_lagged = np.full(n, np.nan)
    resid_vol_lagged[_SKIP:] = resid_vol[: n - _SKIP]

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_scale = np.where(
            np.isfinite(resid_vol_lagged) & (resid_vol_lagged > _VOL_FLOOR),
            resid_vol_lagged,
            np.nan,
        )
        # Require at least half the window to be valid observations
        resmom_11m = np.where(
            np.isfinite(cum11) & np.isfinite(vol_scale) & (cnt11 >= _MOM11_WIN * 0.5),
            cum11 / vol_scale,
            np.nan,
        )
        resmom_6m = np.where(
            np.isfinite(cum6) & np.isfinite(vol_scale) & (cnt6 >= _MOM6_WIN * 0.5),
            cum6 / vol_scale,
            np.nan,
        )
        resmom_raw = np.where(
            np.isfinite(cum11) & (cnt11 >= _MOM11_WIN * 0.5),
            cum11,
            np.nan,
        )

    # Guard inf
    resmom_11m = np.where(np.isfinite(resmom_11m), resmom_11m, np.nan)
    resmom_6m  = np.where(np.isfinite(resmom_6m),  resmom_6m,  np.nan)
    resmom_raw = np.where(np.isfinite(resmom_raw),  resmom_raw, np.nan)

    df["ext3_resmom_11m"] = resmom_11m
    df["ext3_resmom_6m"]  = resmom_6m
    df["ext3_resmom_raw"] = resmom_raw

    return df
