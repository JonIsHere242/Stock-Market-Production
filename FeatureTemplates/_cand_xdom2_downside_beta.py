from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional helper: market index
# ---------------------------------------------------------------------------
try:
    _s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
    _indexes = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_indexes)
    _HAS_INDEXES = True
except Exception:
    _indexes = None
    _HAS_INDEXES = False

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "xdom2_downside_beta",
    "description": (
        "Rolling 120-day downside vs upside beta asymmetry relative to SPY. "
        "Downside beta = OLS slope of stock daily returns vs SPY returns on "
        "days when SPY return < 0; upside beta = same on SPY-up days. "
        "beta_asym = beta_down - beta_up. Downside-beta loading earns a risk "
        "premium (Ang, Chen, Xing 2006). Per-ticker implementation using "
        "index_close('SPY'); degrades to NaN if SPY unavailable."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_downside_beta_down_120",
        "xdom2_downside_beta_up_120",
        "xdom2_downside_beta_asym_120",
    ],
    "tags": ["beta", "downside-risk", "market-sensitivity", "cross-domain", "ang2006"],
    "version": "1.0",
    "author": "Ang, Chen, Xing (2006) — Journal of Political Economy; per-ticker OHLCV proxy",
}

# ---------------------------------------------------------------------------
# Rolling OLS slope helper (vectorised via numpy convolutions)
# ---------------------------------------------------------------------------
_WINDOW = 120
_MIN_OBS = 30  # minimum observations in each regime to emit a value


def _rolling_beta_regime(
    stock_ret: np.ndarray,
    mkt_ret: np.ndarray,
    mask: np.ndarray,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """
    For each bar t, compute OLS beta of stock_ret[t-window+1..t] vs
    mkt_ret[t-window+1..t] restricted to bars where mask[i] is True.

    Returns array of same length as stock_ret, NaN where insufficient data.
    """
    n = len(stock_ret)
    out = np.full(n, np.nan)

    # Precompute masked series (NaN elsewhere so sums work correctly)
    s_m = np.where(mask, stock_ret, np.nan)  # stock returns in regime
    x_m = np.where(mask, mkt_ret, np.nan)    # market returns in regime

    # Use pandas for efficient rolling ops
    s_ser = pd.Series(s_m)
    x_ser = pd.Series(x_m)
    cnt_ser = pd.Series(mask.astype(float))

    roll_n   = cnt_ser.rolling(window, min_periods=1).sum()
    roll_sx  = s_ser.rolling(window, min_periods=1).sum()
    roll_xx  = x_ser.rolling(window, min_periods=1).sum()
    roll_sxx = (s_ser * x_ser).rolling(window, min_periods=1).sum()
    roll_xxx = (x_ser * x_ser).rolling(window, min_periods=1).sum()

    n_arr  = roll_n.to_numpy()
    sx     = roll_sx.to_numpy()
    xx     = roll_xx.to_numpy()
    sxx    = roll_sxx.to_numpy()
    xxx    = roll_xxx.to_numpy()

    # OLS: beta = (n * Sigma(xy) - Sigma(x)*Sigma(y)) / (n * Sigma(x^2) - Sigma(x)^2)
    denom = n_arr * xxx - xx * xx
    numer = n_arr * sxx - xx * sx

    valid = (n_arr >= min_obs) & (denom != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(valid, numer / denom, np.nan)

    return beta


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_down = "xdom2_downside_beta_down_120"
    col_up   = "xdom2_downside_beta_up_120"
    col_asym = "xdom2_downside_beta_asym_120"

    n = len(df)

    # Initialise output columns with NaN
    df[col_down] = np.nan
    df[col_up]   = np.nan
    df[col_asym] = np.nan

    if n < _MIN_OBS + 1:
        return df

    if not _HAS_INDEXES or _indexes is None:
        return df

    # --- SPY daily returns ------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Align SPY to stock dates via backward merge_asof
    stock_dates = pd.to_datetime(df["Date"])
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    aligned = pd.merge_asof(
        pd.DataFrame({"Date": stock_dates}),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    spy_vals = aligned["spy_close"].to_numpy(dtype=float)

    # SPY returns (lag-1 safe: first entry NaN)
    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = np.where(
        spy_vals[:-1] > 0,
        spy_vals[1:] / spy_vals[:-1] - 1.0,
        np.nan,
    )

    # Stock returns
    close = df["Close"].to_numpy(dtype=float)
    stk_ret = np.full(n, np.nan)
    stk_ret[1:] = np.where(
        close[:-1] > 0,
        close[1:] / close[:-1] - 1.0,
        np.nan,
    )

    # Regime masks (exclude NaN positions)
    valid_both = np.isfinite(spy_ret) & np.isfinite(stk_ret)
    mask_down = valid_both & (spy_ret < 0.0)
    mask_up   = valid_both & (spy_ret >= 0.0)

    beta_down = _rolling_beta_regime(stk_ret, spy_ret, mask_down, _WINDOW, _MIN_OBS)
    beta_up   = _rolling_beta_regime(stk_ret, spy_ret, mask_up,   _WINDOW, _MIN_OBS)

    df[col_down] = beta_down
    df[col_up]   = beta_up

    with np.errstate(invalid="ignore"):
        asym = np.where(
            np.isfinite(beta_down) & np.isfinite(beta_up),
            beta_down - beta_up,
            np.nan,
        )
    df[col_asym] = asym

    return df
