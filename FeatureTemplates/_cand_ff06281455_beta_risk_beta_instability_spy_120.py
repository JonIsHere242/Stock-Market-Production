"""
Beta instability feature block.

Computes rolling 40-day beta vs SPY at each bar, then derives:
  - 120d std of that rolling beta series (beta instability)
  - 120d range of rolling beta (max - min)
  - Down-regime minus up-regime average beta (asymmetric beta instability)

Per-ticker proxy; causal, no lookahead.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by file path
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06281455_beta_risk_beta_instability_spy_120",
    "description": (
        "Per-ticker beta instability vs SPY. Computes a rolling 40-day OLS beta "
        "at every bar using numpy sliding windows, then summarises the trailing "
        "120-bar window of that beta series: (1) std = beta instability, "
        "(2) range = max-minus-min, (3) asymmetric instability = mean(beta | "
        "SPY return < 0) - mean(beta | SPY return >= 0). Captures how erratically "
        "a stock tracks the market, an axis orthogonal to the level of beta."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06281455_beta_risk_beta_instability_spy_120_std",
        "ff06281455_beta_risk_beta_instability_spy_120_range",
        "ff06281455_beta_risk_beta_instability_spy_120_asym",
    ],
    "tags": ["beta", "risk", "instability", "spy", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06281455",
}

_BETA_WIN = 40    # window for each point-in-time beta estimate
_INST_WIN = 120   # window over which to measure beta instability


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out_std = "ff06281455_beta_risk_beta_instability_spy_120_std"
    out_range = "ff06281455_beta_risk_beta_instability_spy_120_range"
    out_asym = "ff06281455_beta_risk_beta_instability_spy_120_asym"

    # Initialise all produced columns to NaN (required on every code path)
    df[out_std] = np.nan
    df[out_range] = np.nan
    df[out_asym] = np.nan

    n = len(df)
    if n < _BETA_WIN + 1:
        return df

    # -----------------------------------------------------------------------
    # 1. Fetch SPY index closes and align to this stock's dates
    # -----------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_series is None or spy_series.empty:
        return df

    # Build a small merge frame: stock dates + Close
    dates = pd.to_datetime(df["Date"])
    stock_frame = pd.DataFrame({"Date": dates, "stock_close": df["Close"].values})
    stock_frame = stock_frame.sort_values("Date").reset_index(drop=True)

    spy_frame = (
        spy_series.rename("spy_close")
        .reset_index()
        .rename(columns={"index": "Date", "Date": "Date"})
    )
    spy_frame["Date"] = pd.to_datetime(spy_frame["Date"])
    spy_frame = spy_frame.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        stock_frame,
        spy_frame,
        on="Date",
        direction="backward",
    )

    # Compute log returns (safer numerically)
    stock_ret = np.log(merged["stock_close"].values / np.where(
        merged["stock_close"].shift(1).values > 0,
        merged["stock_close"].shift(1).values,
        np.nan,
    ))
    spy_ret = np.log(merged["spy_close"].values / np.where(
        merged["spy_close"].shift(1).values > 0,
        merged["spy_close"].shift(1).values,
        np.nan,
    ))

    # -----------------------------------------------------------------------
    # 2. Rolling 40-bar OLS beta at every bar t  (vectorised via stride)
    #    beta_t = cov(stock[t-39:t], spy[t-39:t]) / var(spy[t-39:t])
    # -----------------------------------------------------------------------
    T = len(stock_ret)
    rolling_beta = np.full(T, np.nan)

    # We need at least _BETA_WIN bars of returns (ret[0] is NaN, so effectively
    # _BETA_WIN + 1 prices -> _BETA_WIN returns, first valid beta at index _BETA_WIN-1)
    for t in range(_BETA_WIN - 1, T):
        y = stock_ret[t - _BETA_WIN + 1: t + 1]
        x = spy_ret[t - _BETA_WIN + 1: t + 1]
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < _BETA_WIN // 2:
            continue
        ym = y[mask]
        xm = x[mask]
        xvar = np.var(xm, ddof=1)
        if xvar == 0 or not np.isfinite(xvar):
            continue
        rolling_beta[t] = np.cov(ym, xm, ddof=1)[0, 1] / xvar

    # -----------------------------------------------------------------------
    # 3. Instability metrics over trailing _INST_WIN bars of rolling_beta
    # -----------------------------------------------------------------------
    beta_std = np.full(T, np.nan)
    beta_range = np.full(T, np.nan)
    beta_asym = np.full(T, np.nan)

    for t in range(_INST_WIN - 1, T):
        b_win = rolling_beta[t - _INST_WIN + 1: t + 1]
        x_win = spy_ret[t - _INST_WIN + 1: t + 1]
        valid = np.isfinite(b_win)
        if valid.sum() < _INST_WIN // 2:
            continue

        b_valid = b_win[valid]
        beta_std[t] = np.std(b_valid, ddof=1) if len(b_valid) > 1 else np.nan
        beta_range[t] = np.nanmax(b_valid) - np.nanmin(b_valid)

        # Asymmetric: mean beta on down-SPY days vs up-SPY days
        both_valid = np.isfinite(b_win) & np.isfinite(x_win)
        if both_valid.sum() < 4:
            continue
        x_sub = x_win[both_valid]
        b_sub = b_win[both_valid]
        down_mask = x_sub < 0
        up_mask = ~down_mask
        if down_mask.sum() >= 2 and up_mask.sum() >= 2:
            beta_asym[t] = np.mean(b_sub[down_mask]) - np.mean(b_sub[up_mask])

    # -----------------------------------------------------------------------
    # 4. Map results back to the ORIGINAL df row order
    # -----------------------------------------------------------------------
    # merged was sorted by Date; df may have a different original index.
    # Re-align by Date.
    result_frame = pd.DataFrame({
        "Date": merged["Date"].values,
        out_std: beta_std,
        out_range: beta_range,
        out_asym: beta_asym,
    })

    # Merge back into df preserving original row order
    df_dates = pd.to_datetime(df["Date"])
    orig_index = df.index
    df2 = df.copy()
    df2["_date_key"] = df_dates.values
    result_frame = result_frame.rename(columns={"Date": "_date_key"})
    result_frame["_date_key"] = pd.to_datetime(result_frame["_date_key"])

    df2 = df2.merge(result_frame, on="_date_key", how="left", suffixes=("_OLD", ""))

    # Drop any duplicated OLD columns from merge collision
    for col in [out_std, out_range, out_asym]:
        old_col = col + "_OLD"
        if old_col in df2.columns:
            df2 = df2.drop(columns=[old_col])

    df2 = df2.drop(columns=["_date_key"])
    df2.index = orig_index

    df[out_std] = df2[out_std].values
    df[out_range] = df2[out_range].values
    df[out_asym] = df2[out_asym].values

    return df
