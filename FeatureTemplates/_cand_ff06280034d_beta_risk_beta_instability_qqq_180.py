"""
Beta instability vs QQQ: rolling 40d beta computed at each bar, then
180d std / range / regime-split of that beta series.

Per-ticker proxy (cross-sectional ranking not needed; instability is a
per-stock time-series property).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# -- load _indexes helper by file path --
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06280034d_beta_risk_beta_instability_qqq_180",
    "description": (
        "Computes a rolling 40-day OLS beta vs QQQ at every bar, then measures "
        "how unstable that beta is over a trailing 180-day window.  Produces: "
        "(1) the 180d std of the rolling beta (instability level), "
        "(2) the 180d range (max-min) of the rolling beta, "
        "(3) down-regime minus up-regime beta instability -- i.e. the 90d std "
        "of rolling beta on days when QQQ was down vs up, differenced.  "
        "Per-ticker causal proxy; no cross-sectional ranking required."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034d_beta_instability_std",
        "ff06280034d_beta_instability_range",
        "ff06280034d_beta_regime_asym",
    ],
    "tags": ["beta", "beta_risk", "instability", "qqq", "regime"],
    "version": "1.0.0",
    "author": "feature-factory/ff06280034d",
}

_BETA_WIN = 40    # window for rolling beta
_INST_WIN = 180   # window over which instability is measured
_REGIME_WIN = 90  # sub-window for regime-split instability


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs as NaN on every code path
    df["ff06280034d_beta_instability_std"] = np.nan
    df["ff06280034d_beta_instability_range"] = np.nan
    df["ff06280034d_beta_regime_asym"] = np.nan

    if len(df) < _BETA_WIN + 2:
        return df

    # --- fetch QQQ close aligned to this stock's dates ---
    try:
        qqq_series = _indexes.index_close("QQQ")  # pd.Series, DatetimeIndex
    except Exception:
        return df

    if qqq_series is None or len(qqq_series) == 0:
        return df

    # Build a small aligned frame; merge_asof requires sorted dates
    stock_dates = pd.to_datetime(df["Date"])
    qqq_df = qqq_series.rename("qqq_close").reset_index()
    qqq_df.columns = ["Date", "qqq_close"]
    qqq_df["Date"] = pd.to_datetime(qqq_df["Date"])
    qqq_df = qqq_df.sort_values("Date")

    tmp = pd.DataFrame({"Date": stock_dates, "stock_close": df["Close"].values})
    tmp = tmp.sort_values("Date")
    tmp = pd.merge_asof(tmp, qqq_df, on="Date", direction="backward")

    # Daily log-returns (length n-1, padded with leading NaN)
    stock_ret = np.log(tmp["stock_close"].values / np.where(
        tmp["stock_close"].shift(1).values == 0, np.nan, tmp["stock_close"].shift(1).values))
    qqq_ret = np.log(tmp["qqq_close"].values / np.where(
        tmp["qqq_close"].shift(1).values == 0, np.nan, tmp["qqq_close"].shift(1).values))

    n = len(tmp)
    rolling_beta = np.full(n, np.nan)

    # Vectorised rolling OLS beta: beta = cov(x,y)/var(x) over window
    # We use a pandas rolling trick for speed (avoids O(n^2) python loops)
    s_ret = pd.Series(stock_ret)
    q_ret = pd.Series(qqq_ret)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roll_cov = s_ret.rolling(_BETA_WIN).cov(q_ret)
        roll_var = q_ret.rolling(_BETA_WIN).var()

    # Guard: zero variance -> NaN
    denom = roll_var.values.copy()
    denom[denom == 0] = np.nan
    rolling_beta = roll_cov.values / denom  # shape (n,)

    # Replace inf/-inf
    rolling_beta = np.where(np.isfinite(rolling_beta), rolling_beta, np.nan)

    rb = pd.Series(rolling_beta)

    # (1) 180d std of rolling beta
    inst_std = rb.rolling(_INST_WIN).std()

    # (2) 180d range of rolling beta
    inst_max = rb.rolling(_INST_WIN).max()
    inst_min = rb.rolling(_INST_WIN).min()
    inst_range = inst_max - inst_min

    # (3) regime-asymmetry: std(rolling_beta | qqq_down, 90d) - std(rolling_beta | qqq_up, 90d)
    # We mask the rolling_beta series and compute rolling std on each regime
    qqq_down = (q_ret.values < 0).astype(float)   # 1 if QQQ down, 0 if up, nan if ret is nan
    qqq_down = np.where(np.isfinite(q_ret.values), qqq_down, np.nan)

    # For regime std we use an expanding-window approach over last _REGIME_WIN bars
    # Implemented via apply -- window small enough to be fast
    def _regime_asym(i: int) -> float:
        start = max(0, i - _REGIME_WIN + 1)
        betas = rolling_beta[start: i + 1]
        regimes = qqq_down[start: i + 1]
        valid = np.isfinite(betas) & np.isfinite(regimes)
        b_down = betas[valid & (regimes == 1)]
        b_up = betas[valid & (regimes == 0)]
        if len(b_down) < 5 or len(b_up) < 5:
            return np.nan
        return float(np.std(b_down, ddof=1) - np.std(b_up, ddof=1))

    # Use stride to keep fast: compute every bar (n <= ~700, loop is O(n*W) vectorised slices)
    regime_asym = np.array([_regime_asym(i) for i in range(n)], dtype=float)

    # Map results back to df (tmp was sorted by Date; df may be any order)
    # Use the original index of tmp to align
    out_idx = tmp.index  # same positional order as df after sort... need to reindex back

    # tmp was built from df.index reset; sort_values may reorder rows.
    # Safer: build a Date->value map and merge back via df's dates.
    date_to_inst_std = dict(zip(tmp["Date"].values, inst_std.values))
    date_to_inst_range = dict(zip(tmp["Date"].values, inst_range.values))
    date_to_regime_asym = dict(zip(tmp["Date"].values, regime_asym))

    dates_arr = pd.to_datetime(df["Date"]).values

    df["ff06280034d_beta_instability_std"] = [
        date_to_inst_std.get(d, np.nan) for d in dates_arr]
    df["ff06280034d_beta_instability_range"] = [
        date_to_inst_range.get(d, np.nan) for d in dates_arr]
    df["ff06280034d_beta_regime_asym"] = [
        date_to_regime_asym.get(d, np.nan) for d in dates_arr]

    # Replace any stray inf
    for col in METADATA["produces"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
