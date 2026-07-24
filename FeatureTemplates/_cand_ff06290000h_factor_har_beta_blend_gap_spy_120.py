from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load index helper
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_factor_har_beta_blend_gap_spy_120",
    "description": (
        "HAR-style multi-horizon beta blend vs SPY. Computes rolling OLS beta over "
        "22d, 66d, and 120d windows using cov/var. HAR blend = (beta22+beta66+beta120)/3. "
        "Primary feature = beta22 - HAR_blend (short-horizon deviation from multi-horizon "
        "consensus; positive = recently more correlated/volatile than long-run average). "
        "Also emits the HAR blend level and the rolling 22d beta alone. Per-ticker proxy; "
        "no cross-sectional information needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290000h_factor_har_beta_blend_gap_spy_120_gap",   # beta22 - HAR_blend
        "ff06290000h_factor_har_beta_blend_gap_spy_120_blend", # HAR blend level
        "ff06290000h_factor_har_beta_blend_gap_spy_120_b22",   # raw 22d beta
    ],
    "tags": ["factor", "beta", "har", "spy", "multi-horizon"],
    "version": "1.0.0",
    "author": "feature-factory ff06290000h",
}

_WINDOWS = [22, 66, 120]
_COL_GAP   = "ff06290000h_factor_har_beta_blend_gap_spy_120_gap"
_COL_BLEND = "ff06290000h_factor_har_beta_blend_gap_spy_120_blend"
_COL_B22   = "ff06290000h_factor_har_beta_blend_gap_spy_120_b22"

_VAR_FLOOR = 1e-12


def _rolling_beta(ret: np.ndarray, spy_ret: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling beta: cov(stock, spy) / var(spy) over `window` bars."""
    n = len(ret)
    out = np.full(n, np.nan)
    if n < window:
        return out
    for i in range(window - 1, n):
        r  = ret[i - window + 1 : i + 1]
        rs = spy_ret[i - window + 1 : i + 1]
        # demean
        rm  = r  - r.mean()
        rsm = rs - rs.mean()
        var_spy = (rsm * rsm).mean()
        if var_spy < _VAR_FLOOR:
            continue
        cov = (rm * rsm).mean()
        out[i] = cov / var_spy
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN upfront (every code path)
    df[_COL_GAP]   = np.nan
    df[_COL_BLEND] = np.nan
    df[_COL_B22]   = np.nan

    if len(df) < _WINDOWS[0] + 1:
        return df

    # -----------------------------------------------------------------------
    # Fetch SPY closes and align to df dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # merge_asof requires sorted datetime
    df_dates = pd.DataFrame({"Date": df["Date"].values})
    df_dates["Date"] = pd.to_datetime(df_dates["Date"])

    spy_df = spy_close.reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    df_dates = df_dates.sort_values("Date").reset_index(drop=True)
    merged = pd.merge_asof(df_dates, spy_df, on="Date", direction="backward")

    # If original df wasn't sorted, reorder merged to match df's row order
    # We need the SPY series aligned positionally with df rows
    orig_dates = pd.to_datetime(df["Date"].values)
    # Build a map from date -> spy_close value
    spy_map = merged.set_index("Date")["spy_close"]
    spy_aligned = spy_map.reindex(orig_dates).values.astype(float)

    if np.all(np.isnan(spy_aligned)):
        return df

    # Compute returns
    stock_ret = np.full(len(df), np.nan)
    close_arr = df["Close"].values.astype(float)
    stock_ret[1:] = np.where(
        close_arr[:-1] != 0,
        close_arr[1:] / close_arr[:-1] - 1.0,
        np.nan,
    )

    spy_ret = np.full(len(df), np.nan)
    spy_ret[1:] = np.where(
        spy_aligned[:-1] != 0,
        spy_aligned[1:] / spy_aligned[:-1] - 1.0,
        np.nan,
    )

    # Replace NaN with 0.0 only where BOTH series have valid data
    valid = ~np.isnan(stock_ret) & ~np.isnan(spy_ret)

    # Need at least the smallest window of valid paired returns
    if valid.sum() < _WINDOWS[0]:
        return df

    # Fill NaN positions with 0 for the rolling calc; result will be nan at those positions anyway
    sr = np.where(np.isnan(stock_ret), 0.0, stock_ret)
    sp = np.where(np.isnan(spy_ret), 0.0, spy_ret)

    # Compute per-window betas
    betas = {}
    for w in _WINDOWS:
        betas[w] = _rolling_beta(sr, sp, w)

    b22  = betas[22]
    b66  = betas[66]
    b120 = betas[120]

    # HAR blend: average of available non-nan windows per bar
    har_blend = np.full(len(df), np.nan)
    for i in range(len(df)):
        terms = []
        for w in _WINDOWS:
            v = betas[w][i]
            if not np.isnan(v):
                terms.append(v)
        if len(terms) > 0:
            har_blend[i] = np.mean(terms)

    gap = np.full(len(df), np.nan)
    for i in range(len(df)):
        if not np.isnan(b22[i]) and not np.isnan(har_blend[i]):
            gap[i] = b22[i] - har_blend[i]

    # Guard: no inf
    b22       = np.where(np.isinf(b22), np.nan, b22)
    har_blend = np.where(np.isinf(har_blend), np.nan, har_blend)
    gap       = np.where(np.isinf(gap), np.nan, gap)

    df[_COL_GAP]   = gap
    df[_COL_BLEND] = har_blend
    df[_COL_B22]   = b22

    return df
