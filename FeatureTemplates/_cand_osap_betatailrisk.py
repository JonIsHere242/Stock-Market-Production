"""
osap_betatailrisk — Tail Risk Beta (Kelly & Jiang 2014, RFS)

Per-ticker proxy: the original measure requires a daily cross-sectional 5th-percentile
across all stocks, which is impossible inside a single-ticker compute() call.  We
approximate with SPY as a broad-market stand-in for the cross-sectional distribution.
On each calendar day we define the *tail threshold* as the rolling 5th-percentile of
SPY daily log-returns (252-day window), and tailEX as the average log-return excess on
days when SPY fell below that threshold (rolling 252-day).  BetaTailRisk is then the
slope of a rolling 1200-trading-day (~120-month) OLS regression of the stock's daily
log-return on tailEX.  A shorter 252-day rolling beta variant is also produced for
higher-frequency responsiveness.  This is a per-ticker PROXY — see Kelly & Jiang (2014)
for the true cross-sectional construction.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load _indexes helper ──────────────────────────────────────────────────────
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name": "osap_betatailrisk",
    "description": (
        "Tail-risk beta: rolling OLS slope of daily stock log-return on a "
        "daily tail-excess variable (tailEX), where tailEX is the mean of "
        "log(SPY_ret / SPY_5th_pct) on days the market fell into its own "
        "5th-percentile tail (rolling 252-day window).  Proxy for Kelly & "
        "Jiang (2014, RFS) BetaTailRisk; true version needs cross-sectional "
        "daily return panel. Positive beta = high tail-risk loading (expected "
        "negative premium per Kelly & Jiang Table 4A, sign=+1 long-short)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_betatailrisk_120m",   # 1200-day rolling beta (≈120-month, primary)
        "osap_betatailrisk_12m",    # 252-day rolling beta  (≈12-month, dynamic)
        "osap_betatailrisk_tailex", # the tailEX regressor itself (informative)
    ],
    "tags": ["risk", "tail", "beta", "market", "kelly_jiang"],
    "version": "1.0",
    "author": "Kelly, B. and Jiang, H. (2014). Tail risk and asset prices. "
              "Review of Financial Studies, 27(10), 2841-2871. "
              "Spec: Open Source Asset Pricing (osap_betatailrisk). "
              "Per-ticker proxy implementation.",
}


def _rolling_ols_slope(
    x: np.ndarray, y: np.ndarray, window: int
) -> np.ndarray:
    """
    Vectorised rolling OLS slope (beta) of y on x (with intercept).
    Returns array same length as x; leading values are NaN.
    Uses numpy sliding windows to avoid an O(n) Python loop.
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Use stride tricks for efficiency
    from numpy.lib.stride_tricks import sliding_window_view
    xw = sliding_window_view(x, window)   # shape (n-window+1, window)
    yw = sliding_window_view(y, window)

    xm = xw.mean(axis=1)
    ym = yw.mean(axis=1)
    xd = xw - xm[:, None]
    yd = yw - ym[:, None]

    denom = (xd * xd).sum(axis=1)
    numer = (xd * yd).sum(axis=1)

    # Guard zero-variance x (all tailEX values identical → NaN)
    slope = np.where(denom > 0, numer / denom, np.nan)
    out[window - 1:] = slope
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── 1. Fetch SPY close, compute daily log-returns ────────────────────────
    try:
        spy_close = _indexes.index_close("SPY")       # DatetimeIndex → Close
    except Exception:
        df["osap_betatailrisk_120m"]  = np.nan
        df["osap_betatailrisk_12m"]   = np.nan
        df["osap_betatailrisk_tailex"] = np.nan
        return df

    if spy_close is None or len(spy_close) == 0:
        df["osap_betatailrisk_120m"]  = np.nan
        df["osap_betatailrisk_12m"]   = np.nan
        df["osap_betatailrisk_tailex"] = np.nan
        return df

    spy_ret = np.log(spy_close / spy_close.shift(1))   # pd.Series, DatetimeIndex

    # ── 2. Build SPY tailEX: daily rolling (252-day) 5th-pct tail-excess ─────
    #   tailEX[t] = mean over past 252 days of log(spy_ret / p5) where spy_ret < p5
    #   Computed as a pandas rolling custom — but custom rolling is slow; use
    #   a vectorised approach with sliding_window_view.

    spy_ret_arr = spy_ret.values.astype(float)
    n_spy = len(spy_ret_arr)
    tail_window = 252

    tailex_arr = np.full(n_spy, np.nan)
    if n_spy >= tail_window:
        from numpy.lib.stride_tricks import sliding_window_view
        spy_wins = sliding_window_view(spy_ret_arr, tail_window)  # (M, 252)
        # p5 per window (ignoring NaN via nanpercentile-like approach)
        p5 = np.nanpercentile(spy_wins, 5, axis=1)               # (M,)
        # For each window, select days below p5 and average log(ret/p5)
        # Vectorised: mask then mean per row
        # log(ret/p5) = log(ret) - log(p5); but p5 may be negative → careful
        # Kelly & Jiang define tailEX = avg log(|ret/p5|) on tail days where
        # ret < p5 < 0 (tail is left tail, both negative).  We use:
        #   log(ret/p5) with p5 < 0, ret < p5 → ratio > 1 → positive tailEX
        # Guard: p5 must be < 0 for this to make sense
        valid_p5 = p5 != 0  # avoid /0

        # Broadcast comparison: (M, 252) < (M,1)
        tail_mask = spy_wins < p5[:, None]                        # (M, 252)

        # ratio = spy_wins / p5[:, None], guarded
        safe_p5 = np.where(valid_p5, p5, np.nan)
        ratio = spy_wins / safe_p5[:, None]                       # (M, 252)
        # log of ratio on tail days only; non-tail → NaN
        log_ratio = np.where(tail_mask & (ratio > 0), np.log(ratio), np.nan)

        # mean per row (ignoring NaN)
        with np.errstate(all="ignore"):
            tailex_wins = np.nanmean(log_ratio, axis=1)           # (M,)

        # Assign to positions [tail_window-1 ... n_spy-1]
        tailex_arr[tail_window - 1:] = tailex_wins

    # ── 3. Build SPY tailEX series aligned to a DatetimeIndex ────────────────
    spy_tailex_series = pd.Series(tailex_arr, index=spy_ret.index, name="tailex")

    # ── 4. Compute stock log-returns ─────────────────────────────────────────
    close = df["Close"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        stk_ret = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))
    stk_ret = np.concatenate([[np.nan], stk_ret])   # align with df rows

    # ── 5. Join tailEX onto df dates via merge_asof ──────────────────────────
    df_dates = pd.to_datetime(df["Date"])
    spy_tail_df = spy_tailex_series.reset_index()
    spy_tail_df.columns = ["Date", "tailex"]
    spy_tail_df["Date"] = pd.to_datetime(spy_tail_df["Date"])
    spy_tail_df = spy_tail_df.dropna(subset=["tailex"]).sort_values("Date")

    df_for_merge = pd.DataFrame({"Date": df_dates, "_orig_idx": np.arange(len(df))})
    merged = pd.merge_asof(
        df_for_merge.sort_values("Date"),
        spy_tail_df,
        on="Date",
        direction="backward",
    )
    # Restore original row order
    merged = merged.sort_values("_orig_idx")
    tailex_aligned = merged["tailex"].values.astype(float)

    # ── 6. Rolling OLS: stock_ret ~ tailEX ───────────────────────────────────
    # Replace inf in stk_ret
    stk_ret = np.where(np.isfinite(stk_ret), stk_ret, np.nan)
    tailex_aligned = np.where(np.isfinite(tailex_aligned), tailex_aligned, np.nan)

    # For OLS we need both non-NaN; set to nan where either is nan
    valid = np.isfinite(stk_ret) & np.isfinite(tailex_aligned)
    x_clean = np.where(valid, tailex_aligned, np.nan)
    y_clean = np.where(valid, stk_ret, np.nan)

    # Rolling slopes: 1200-day and 252-day
    # Interpolate NaN rows for the sliding window (keep them NaN in output)
    # We'll use a nan-aware rolling OLS via pandas rolling apply for correctness
    # but pandas rolling apply on 1200-day windows with ~700 rows is fine.
    s_x = pd.Series(x_clean)
    s_y = pd.Series(y_clean)

    def _ols_slope(xy_flat):
        half = len(xy_flat) // 2
        xx = xy_flat[:half]
        yy = xy_flat[half:]
        m = np.isfinite(xx) & np.isfinite(yy)
        if m.sum() < 20:
            return np.nan
        xm_ = xx[m]
        ym_ = yy[m]
        xd_ = xm_ - xm_.mean()
        denom_ = (xd_ * xd_).sum()
        if denom_ == 0:
            return np.nan
        return float((xd_ * (ym_ - ym_.mean())).sum() / denom_)

    # Use concat trick for pandas rolling apply (avoids closure over arrays)
    # pandas rolling on a single Series then call np inside is standard

    def _rolling_beta(x_arr, y_arr, win):
        n = len(x_arr)
        out = np.full(n, np.nan)
        if n < win:
            return out
        from numpy.lib.stride_tricks import sliding_window_view
        xw = sliding_window_view(x_arr, win)
        yw = sliding_window_view(y_arr, win)
        # per window
        for i in range(len(xw)):
            xw_i = xw[i]
            yw_i = yw[i]
            m = np.isfinite(xw_i) & np.isfinite(yw_i)
            if m.sum() < 20:
                continue
            xv = xw_i[m]
            yv = yw_i[m]
            xd_ = xv - xv.mean()
            den = (xd_ * xd_).sum()
            if den == 0:
                continue
            out[win - 1 + i] = float((xd_ * (yv - yv.mean())).sum() / den)
        return out

    # For 1200-day window: typically longer than the per-stock series (700 rows)
    # → will return all NaN, which is correct (need 100yr of data for full window).
    # Use 252-day window as the primary working feature and 1200-day where available.
    beta_1200 = _rolling_beta(x_clean, y_clean, win=1200)
    beta_252  = _rolling_beta(x_clean, y_clean, win=252)

    # ── 7. Assign columns ─────────────────────────────────────────────────────
    n = len(df)
    df["osap_betatailrisk_120m"]   = beta_1200[:n] if len(beta_1200) >= n else np.nan
    df["osap_betatailrisk_12m"]    = beta_252[:n]  if len(beta_252)  >= n else np.nan
    df["osap_betatailrisk_tailex"] = tailex_aligned[:n]

    return df
