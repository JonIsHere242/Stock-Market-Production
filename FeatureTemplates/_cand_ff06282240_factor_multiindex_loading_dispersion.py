"""
ff06282240_factor_multiindex_loading_dispersion
-----------------------------------------------
Cross-asset dispersion of simultaneous index loadings.

For each stock computes univariate beta to each of SPY, QQQ, IWM, DIA over a
trailing 126-day window (cov(ret_stock, ret_index) / var(ret_index)).

Produced columns:
  load_dispersion : std of the four betas (how differently the stock loads on
                    each equity index — a high value means the stock is more
                    sensitive to small-cap vs large-cap vs tech, etc.)
  load_range      : max - min of the four betas
  dispersion_chg  : load_dispersion minus its value 42 bars ago (trend in
                    multi-index exposure ambiguity)

Betas are computed on a stride-5 grid from the series start and forward-filled
for speed (requirement: < 100 ms per stock).

Cross-sectional context is lost (true multi-index loading is cross-sectional);
this is the closest faithful per-ticker proxy.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282240_factor_multiindex_loading_dispersion",
    "description": (
        "Cross-asset dispersion of per-ticker loadings on SPY, QQQ, IWM, DIA "
        "computed over a trailing 126-day window. load_dispersion = std of the "
        "four betas; load_range = max-min; dispersion_chg = 42-bar change in "
        "dispersion. Beta computation is strided (every 5th bar) and "
        "forward-filled for speed. Faithful per-ticker proxy of a normally "
        "cross-sectional multi-index loading concept."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282240_factor_multiindex_loading_dispersion_load_dispersion",
        "ff06282240_factor_multiindex_loading_dispersion_load_range",
        "ff06282240_factor_multiindex_loading_dispersion_dispersion_chg",
    ],
    "tags": ["factor", "beta", "dispersion", "multi-index", "regime"],
    "version": "1.0.0",
    "author": "feature-factory ff06282240",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 126
_STRIDE = 5
_CHG_LAG = 42
_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

_COL_DISP = "ff06282240_factor_multiindex_loading_dispersion_load_dispersion"
_COL_RANGE = "ff06282240_factor_multiindex_loading_dispersion_load_range"
_COL_CHG = "ff06282240_factor_multiindex_loading_dispersion_dispersion_chg"


# ---------------------------------------------------------------------------
# Helper: rolling scalar beta for two pre-aligned return arrays
# ---------------------------------------------------------------------------
def _rolling_beta(stock_ret: np.ndarray, idx_ret: np.ndarray, window: int) -> np.ndarray:
    """Return array of beta = cov(stock, idx) / var(idx) over trailing window."""
    n = len(stock_ret)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = stock_ret[t - window + 1 : t + 1]
        x = idx_ret[t - window + 1 : t + 1]
        mask = np.isfinite(s) & np.isfinite(x)
        if mask.sum() < window // 2:
            continue
        sm, xm = s[mask], x[mask]
        var_x = np.var(xm, ddof=1)
        if var_x == 0 or not np.isfinite(var_x):
            continue
        cov_sx = np.cov(sm, xm, ddof=1)[0, 1]
        out[t] = cov_sx / var_x
    return out


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs as NaN on every code path
    df[_COL_DISP] = np.nan
    df[_COL_RANGE] = np.nan
    df[_COL_CHG] = np.nan

    n = len(df)
    if n < _WINDOW + 1:
        return df

    # -----------------------------------------------------------------------
    # Build aligned index return arrays merged on Date (backward-safe)
    # -----------------------------------------------------------------------
    dates = df["Date"].values  # may be object or datetime64

    # Convert to DatetimeIndex for merge_asof
    stock_dates = pd.to_datetime(df["Date"])

    idx_rets: dict[str, np.ndarray] = {}
    for sym in _SYMBOLS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idx_close = _indexes.index_close(sym)
            if idx_close is None or len(idx_close) == 0:
                continue
            idx_df = (
                idx_close.rename("_close")
                .reset_index()
                .rename(columns={"Date": "Date"})
            )
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df = idx_df.sort_values("Date")
            merged = pd.merge_asof(
                pd.DataFrame({"Date": stock_dates}),
                idx_df,
                on="Date",
                direction="backward",
            )
            close_arr = merged["_close"].values.astype(float)
            # compute daily log return
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.full(n, np.nan)
                ret[1:] = np.log(close_arr[1:] / close_arr[:-1])
                # zero / negative prices produce nan — acceptable
            idx_rets[sym] = ret
        except Exception:
            continue

    present_syms = [s for s in _SYMBOLS if s in idx_rets]
    if len(present_syms) < 2:
        return df

    # -----------------------------------------------------------------------
    # Stock log return
    # -----------------------------------------------------------------------
    close_arr = df["Close"].values.astype(float)
    stock_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret[1:] = np.log(
            np.where(close_arr[:-1] > 0, close_arr[1:] / close_arr[:-1], np.nan)
        )

    # -----------------------------------------------------------------------
    # Compute betas on stride grid, forward-fill
    # -----------------------------------------------------------------------
    # Stride grid: positions where i % _STRIDE == 0 (from series start)
    stride_positions = np.arange(0, n, _STRIDE)

    betas_at_stride: dict[str, np.ndarray] = {}
    for sym in present_syms:
        full_beta = np.full(n, np.nan)
        ir = idx_rets[sym]
        # Only compute at stride positions
        for t in stride_positions:
            if t < _WINDOW - 1:
                continue
            s = stock_ret[t - _WINDOW + 1 : t + 1]
            x = ir[t - _WINDOW + 1 : t + 1]
            mask = np.isfinite(s) & np.isfinite(x)
            if mask.sum() < _WINDOW // 2:
                continue
            sm, xm = s[mask], x[mask]
            var_x = np.var(xm, ddof=1)
            if var_x == 0 or not np.isfinite(var_x):
                continue
            cov_sx = float(np.cov(sm, xm, ddof=1)[0, 1])
            full_beta[t] = cov_sx / var_x

        # Forward-fill the sparse beta series
        series = pd.Series(full_beta).ffill()
        betas_at_stride[sym] = series.values

    if len(betas_at_stride) < 2:
        return df

    # -----------------------------------------------------------------------
    # Stack betas -> [n, len(present_syms)] and derive dispersion metrics
    # -----------------------------------------------------------------------
    beta_mat = np.stack(
        [betas_at_stride[s] for s in present_syms], axis=1
    )  # shape (n, k)

    # Count valid (non-nan) betas per row
    valid_count = np.sum(np.isfinite(beta_mat), axis=1)

    load_dispersion = np.where(valid_count >= 2, np.nanstd(beta_mat, axis=1), np.nan)
    load_range = np.where(
        valid_count >= 2,
        np.nanmax(beta_mat, axis=1) - np.nanmin(beta_mat, axis=1),
        np.nan,
    )

    # 42-bar change in dispersion
    disp_series = pd.Series(load_dispersion)
    dispersion_chg = (disp_series - disp_series.shift(_CHG_LAG)).values

    df[_COL_DISP] = load_dispersion
    df[_COL_RANGE] = load_range
    df[_COL_CHG] = dispersion_chg

    return df
