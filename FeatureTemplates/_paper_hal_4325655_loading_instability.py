"""
Time-varying factor loading instability per ticker, derived from:
  "A penalized two-pass regression to predict stock returns with time-varying risk premia"
  (hal:4325655).

The paper develops a penalized two-pass regression where factor loadings (betas) are
allowed to vary over time driven by observable "time-variation drivers."  A key finding
is that the INSTABILITY of factor loadings — how much they drift — carries economic
information beyond the static (average) beta.  We proxy this per ticker as:
  - fli_vol_N: rolling std of sub-period betas estimated in non-overlapping sub-windows
    of size N//4 within a rolling N-day window.  High fli_vol = loadings are unstable.
  - fli_drift_N: rolling mean of |beta_t - beta_{t-1}| (absolute first difference of
    rolling short-window betas), measuring the pace of loading change.
  - fli_trend_N: OLS slope of the sequence of sub-window betas across time within the
    rolling window, indicating whether beta is trending up or down.
All distinct from standard beta (level), from beta spread (cross-sectional),
and from downside beta (return-conditioned).  Captures "loading regime shift" risk
that the static two-pass model cannot.
All rolling, causal, no lookahead.
"""

import numpy as np
import pandas as pd

try:
    from _indexes import index_close
    _SPY = index_close("SPY")
except Exception:
    _SPY = pd.Series(dtype="float64")

METADATA = {
    "name":        "_paper_hal_4325655_loading_instability",
    "description": "Time-varying factor loading instability: rolling beta drift, volatility, and trend vs SPY.",
    "requires":    ["Close"],
    "produces":    [
        "fli_vol_60",
        "fli_drift_60",
        "fli_trend_60",
        "fli_vol_120",
        "fli_drift_120",
        "fli_trend_120",
    ],
    "tags":        ["experimental", "beta", "regime", "instability"],
    "version":     "1.0",
    "author":      "paper-mining slate 2",
}


def _short_beta(ticker_ret: np.ndarray, spy_ret: np.ndarray, start: int, end: int) -> float:
    """OLS beta of ticker on SPY over [start, end) index slice."""
    x = spy_ret[start:end]
    y = ticker_ret[start:end]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xm = x[mask]; ym = y[mask]
    xvar = np.dot(xm - xm.mean(), xm - xm.mean())
    if xvar <= 0:
        return np.nan
    return np.dot(xm - xm.mean(), ym - ym.mean()) / xvar


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if _SPY.empty:
        for w in (60, 120):
            df[f"fli_vol_{w}"] = np.nan
            df[f"fli_drift_{w}"] = np.nan
            df[f"fli_trend_{w}"] = np.nan
        return df

    spy_ret = _SPY.pct_change()
    spy_df = spy_ret.rename("spy_ret").reset_index()
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_ret = df["Close"].pct_change().values

    if "Date" in df.columns:
        date_col = pd.to_datetime(df["Date"])
    else:
        date_col = pd.to_datetime(df.index)

    tmp = pd.DataFrame({"Date": date_col})
    tmp = pd.merge_asof(tmp.sort_values("Date"), spy_df.sort_values("Date"),
                        on="Date", direction="backward")
    spy_aligned = tmp["spy_ret"].values

    n = len(df)
    for window in (60, 120):
        sub_size = window // 4   # ~15 or 30 days per sub-window
        n_subs = 4               # 4 non-overlapping sub-windows inside rolling window

        vol_arr   = np.full(n, np.nan)
        drift_arr = np.full(n, np.nan)
        trend_arr = np.full(n, np.nan)

        for i in range(window - 1, n):
            betas = []
            for k in range(n_subs):
                s = i - window + 1 + k * sub_size
                e = s + sub_size
                b = _short_beta(ticker_ret, spy_aligned, s, e)
                betas.append(b)

            betas = np.array(betas, dtype=float)
            valid = np.isfinite(betas)
            if valid.sum() < 2:
                continue

            bv = betas[valid]
            vol_arr[i]   = bv.std()
            # drift: mean absolute change between consecutive valid betas
            diffs = np.abs(np.diff(betas[valid]))
            drift_arr[i] = diffs.mean() if len(diffs) > 0 else np.nan
            # trend: slope of betas across sub-window index
            idx = np.where(valid)[0].astype(float)
            if len(idx) >= 2:
                idx_c = idx - idx.mean()
                xvar = np.dot(idx_c, idx_c)
                if xvar > 0:
                    trend_arr[i] = np.dot(idx_c, bv - bv.mean()) / xvar

        df[f"fli_vol_{window}"]   = vol_arr
        df[f"fli_drift_{window}"] = drift_arr
        df[f"fli_trend_{window}"] = trend_arr

    return df
