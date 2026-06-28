"""
Multi-factor beta instability feature block.
Spec: ext4_factor_beta_instability (Round-5 expansion, extends ext3_beta_instability)
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

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
    "name": "ext4_factor_beta_instability",
    "description": (
        "Multi-factor rolling beta instability. Computes 40-day OLS rolling betas "
        "of the stock's returns against SPY (market), QQQ-SPY (growth factor), and "
        "IWM-SPY (size factor) using a vectorised sliding-window approach, then "
        "takes the 120-day rolling std of each beta series as a measure of structural "
        "instability. Produces size-beta instability, growth-beta instability, and "
        "their per-row maximum. Per-ticker OHLCV proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_factor_beta_instability_size",
        "ext4_factor_beta_instability_growth",
        "ext4_factor_beta_instability_max",
    ],
    "tags": ["beta", "instability", "factor", "size", "growth", "rolling"],
    "version": "1.0",
    "author": "Round-5 expansion spec (ext3_beta_instability), auto-implemented",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BETA_WIN = 40    # rolling window for OLS beta
_STD_WIN  = 120   # rolling window for beta std (instability)


def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorised rolling OLS beta of y on x (no intercept effect -- mean-centred
    inside the window).  Returns array same length as y, NaN for windows < `window`.
    Uses a strided sliding-window approach: O(n * window) but fully numpy, fast for
    window~40 on n~700.
    """
    n = len(y)
    betas = np.full(n, np.nan)
    if n < window:
        return betas

    # stride trick: shape (n-window+1, window)
    from numpy.lib.stride_tricks import sliding_window_view
    yw = sliding_window_view(y, window)   # shape (n-window+1, window)
    xw = sliding_window_view(x, window)

    # demean inside each window
    xm = xw - xw.mean(axis=1, keepdims=True)
    ym = yw - yw.mean(axis=1, keepdims=True)

    denom = (xm * xm).sum(axis=1)
    numer = (xm * ym).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(denom == 0, np.nan, numer / denom)

    betas[window - 1:] = b
    return betas


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    nan_col = pd.Series(np.nan, index=df.index)

    # Pull index close series; degrade to NaN columns on missing symbols
    try:
        spy_s  = _indexes.index_close("SPY")
        qqq_s  = _indexes.index_close("QQQ")
        iwm_s  = _indexes.index_close("IWM")
    except Exception:
        df["ext4_factor_beta_instability_size"]   = nan_col
        df["ext4_factor_beta_instability_growth"] = nan_col
        df["ext4_factor_beta_instability_max"]    = nan_col
        return df

    # Build a working frame with Date for merge_asof
    wdf = df[["Date"]].copy()
    wdf["_stock_close"] = df["Close"].values

    def _merge_index(s: pd.Series, col: str) -> None:
        """Merge an index Close series into wdf as col, backward-safe."""
        if s is None or len(s) == 0:
            wdf[col] = np.nan
            return
        tmp = pd.DataFrame({"Date": s.index, col: s.values})
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        wdf["Date"] = pd.to_datetime(wdf["Date"])
        merged = pd.merge_asof(
            wdf[["Date"]].reset_index(drop=True),
            tmp.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        wdf[col] = merged[col].values

    _merge_index(spy_s,  "_spy")
    _merge_index(qqq_s,  "_qqq")
    _merge_index(iwm_s,  "_iwm")

    # Daily log returns (shift(1) = look at yesterday's close -> no lookahead)
    def _ret(col: str) -> np.ndarray:
        c = wdf[col].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(
                (c[:-1] == 0) | np.isnan(c[:-1]) | np.isnan(c[1:]),
                np.nan,
                np.log(c[1:] / c[:-1]),
            )
        return np.concatenate([[np.nan], r])

    ret_stock = _ret("_stock_close")
    ret_spy   = _ret("_spy")
    ret_qqq   = _ret("_qqq")
    ret_iwm   = _ret("_iwm")

    # Factor returns (market-neutral spread)
    with np.errstate(invalid="ignore"):
        ret_growth = ret_qqq - ret_spy   # QQQ-SPY = growth factor
        ret_size   = ret_iwm - ret_spy   # IWM-SPY = size factor

    # Mask any NaN propagation from stock or factor side
    valid = ~(np.isnan(ret_stock) | np.isnan(ret_spy))
    ret_stock_m = np.where(valid, ret_stock, np.nan)
    ret_spy_m   = np.where(valid, ret_spy,   np.nan)
    ret_growth_m = np.where(valid & ~np.isnan(ret_growth), ret_growth, np.nan)
    ret_size_m   = np.where(valid & ~np.isnan(ret_size),   ret_size,   np.nan)

    n = len(df)
    betas_spy    = np.full(n, np.nan)
    betas_growth = np.full(n, np.nan)
    betas_size   = np.full(n, np.nan)

    # Rolling beta -- only compute where we have enough non-NaN data.
    # We fall back to a safe loop-free approach: fill NaN gaps with 0 for
    # the sliding window, then mask results where NaN ratio > 50%.
    def _safe_rolling_beta(y_raw: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        y = y_raw.copy()
        x = x_raw.copy()
        nan_mask = np.isnan(y) | np.isnan(x)
        y[nan_mask] = 0.0
        x[nan_mask] = 0.0
        betas = _rolling_beta(y, x, _BETA_WIN)

        # Invalidate windows with too many NaNs (>50% of the window)
        from numpy.lib.stride_tricks import sliding_window_view
        if n >= _BETA_WIN:
            nm_float = nan_mask.astype(float)
            nan_count_w = sliding_window_view(nm_float, _BETA_WIN).sum(axis=1)
            bad = nan_count_w > (_BETA_WIN * 0.5)
            betas[_BETA_WIN - 1:][bad] = np.nan
        return betas

    betas_spy    = _safe_rolling_beta(ret_stock_m, ret_spy_m)
    betas_growth = _safe_rolling_beta(ret_stock_m, ret_growth_m)
    betas_size   = _safe_rolling_beta(ret_stock_m, ret_size_m)

    # 120-day rolling std of each beta series = instability
    def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
        """Rolling std, NaN-aware, vectorised."""
        s = pd.Series(arr)
        return s.rolling(w, min_periods=max(10, w // 2)).std().values

    instab_spy    = _rolling_std(betas_spy,    _STD_WIN)  # noqa: unused, kept for extension
    instab_growth = _rolling_std(betas_growth, _STD_WIN)
    instab_size   = _rolling_std(betas_size,   _STD_WIN)

    with np.errstate(invalid="ignore"):
        instab_max = np.fmax(instab_size, instab_growth)  # fmax ignores NaN

    df["ext4_factor_beta_instability_size"]   = instab_size
    df["ext4_factor_beta_instability_growth"] = instab_growth
    df["ext4_factor_beta_instability_max"]    = instab_max

    return df
