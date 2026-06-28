"""
Idiosyncratic Information Ratio (ext3_idio_sharpe)

Per-ticker rolling alpha-to-idio-risk: residuals from a 120d market-model
regression vs SPY, then 60d mean/std of those residuals (idiosyncratic
information ratio), plus its 20d change.
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
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_idio_sharpe",
    "description": (
        "Idiosyncratic Information Ratio: residuals from a rolling 120-day "
        "OLS market model (stock return ~ SPY return) are computed, then the "
        "60-day rolling mean divided by rolling std gives the per-ticker "
        "alpha-to-idio-risk (idiosyncratic Sharpe / information ratio). "
        "Also produces a 20-day change (momentum of the idio IR). "
        "Extends/relates to osap_idiovolaht (addresses a new axis: ratio of "
        "persistent alpha to idiosyncratic risk, not just vol level). "
        "Per-ticker proxy -- cross-sectional ranking is done downstream."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_idio_sharpe_ir",       # 60d rolling mean(resid)/std(resid)
        "ext3_idio_sharpe_ir_chg20",  # 20d change of IR
    ],
    "tags": ["idiosyncratic", "information_ratio", "market_model", "residual", "risk"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_idiovolaht); spec ext3_idio_sharpe",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MARKET_WIN = 120   # days for rolling OLS market model
_IR_WIN = 60        # days for rolling mean/std of residuals
_CHG_WIN = 20       # days for change in IR


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute idiosyncratic information ratio per ticker.

    Steps
    -----
    1. Merge SPY daily close (backward-safe via merge_asof).
    2. Compute daily log-returns for both stock and SPY.
    3. Rolling 120-day OLS: estimate beta = cov(r_stock, r_spy) / var(r_spy).
       Alpha residual = r_stock - beta * r_spy  (intercept absorbed into residual mean).
    4. 60-day rolling mean / std of residuals → idiosyncratic IR.
    5. 20-day change of IR.
    """
    n = len(df)
    ir_out = np.full(n, np.nan)
    ir_chg_out = np.full(n, np.nan)

    # Need at least MARKET_WIN + 1 rows to do anything
    if n < _MARKET_WIN + 1:
        df["ext3_idio_sharpe_ir"] = np.nan
        df["ext3_idio_sharpe_ir_chg20"] = np.nan
        return df

    # -----------------------------------------------------------------------
    # 1. Get SPY close and merge onto df dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        df["ext3_idio_sharpe_ir"] = np.nan
        df["ext3_idio_sharpe_ir_chg20"] = np.nan
        return df

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original row order
    work = work.set_index(df.index)

    # -----------------------------------------------------------------------
    # 2. Log returns (shift(1) = previous bar, no lookahead)
    # -----------------------------------------------------------------------
    stock_ret = np.log(work["Close"] / work["Close"].shift(1))
    spy_ret = np.log(work["spy_close"] / work["spy_close"].shift(1))

    stock_arr = stock_ret.to_numpy(dtype=float)
    spy_arr = spy_ret.to_numpy(dtype=float)

    # -----------------------------------------------------------------------
    # 3 & 4. Rolling 120d OLS residuals, then 60d IR
    # -----------------------------------------------------------------------
    # We need at least MARKET_WIN rows of returns to estimate beta,
    # then IR_WIN rows of residuals.
    # Residual at bar t = stock_ret[t] - beta_t * spy_ret[t]
    # beta_t estimated on [t-MARKET_WIN+1 .. t] (rolling, causal).

    # Pre-compute rolling sums/cross-products for OLS (online update)
    # Use numpy sliding window for efficiency.
    W = _MARKET_WIN

    # Build residuals array using pandas rolling (efficient via vectorised ops)
    resid = np.full(n, np.nan)

    s_roll = pd.Series(stock_arr)
    m_roll = pd.Series(spy_arr)

    # Rolling covariance and variance using pandas (ddof=1 but consistent)
    roll_cov = s_roll.rolling(W, min_periods=W).cov(m_roll)
    roll_var = m_roll.rolling(W, min_periods=W).var(ddof=1)

    beta = np.where(
        (roll_var.to_numpy() != 0) & np.isfinite(roll_var.to_numpy()),
        roll_cov.to_numpy() / roll_var.to_numpy(),
        np.nan,
    )

    # Residual = stock_ret - beta * spy_ret  (rolling causal beta)
    resid = stock_arr - beta * spy_arr
    # Where beta is nan (first W-1 bars or missing SPY), residual is nan
    resid = np.where(np.isfinite(beta), resid, np.nan)

    # -----------------------------------------------------------------------
    # 5. 60d rolling IR = mean(resid) / std(resid)
    # -----------------------------------------------------------------------
    resid_s = pd.Series(resid)
    roll_mean = resid_s.rolling(_IR_WIN, min_periods=_IR_WIN).mean()
    roll_std = resid_s.rolling(_IR_WIN, min_periods=_IR_WIN).std(ddof=1)

    roll_mean_arr = roll_mean.to_numpy(dtype=float)
    roll_std_arr = roll_std.to_numpy(dtype=float)

    # Guard division
    valid_std = (roll_std_arr > 0) & np.isfinite(roll_std_arr)
    ir_arr = np.where(valid_std, roll_mean_arr / roll_std_arr, np.nan)

    # -----------------------------------------------------------------------
    # 6. 20d change of IR
    # -----------------------------------------------------------------------
    ir_s = pd.Series(ir_arr)
    ir_chg_arr = (ir_s - ir_s.shift(_CHG_WIN)).to_numpy(dtype=float)

    # Replace inf with nan (safety)
    ir_arr = np.where(np.isinf(ir_arr), np.nan, ir_arr)
    ir_chg_arr = np.where(np.isinf(ir_chg_arr), np.nan, ir_chg_arr)

    df["ext3_idio_sharpe_ir"] = ir_arr
    df["ext3_idio_sharpe_ir_chg20"] = ir_chg_arr
    return df
