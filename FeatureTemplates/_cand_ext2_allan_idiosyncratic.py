"""
_cand_ext2_allan_idiosyncratic.py

Idiosyncratic frequency stability (market-residual Allan deviation).

Allan deviation at tau=5 over an 80-day rolling window, applied to the
IDIOSYNCRATIC returns of the stock (stock return minus beta * SPY return),
where beta is estimated via trailing 120-day OLS vs SPY. Isolates stock-specific
noise regime from market noise. Also produces the ratio of idiosyncratic Allan
deviation to total-return Allan deviation.

Per-ticker proxy: beta is estimated per-ticker via a rolling 120d covariance
ratio (cov(r_stock, r_spy) / var(r_spy)), which is the standard single-factor
OLS beta.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: load _indexes module for SPY data
# ---------------------------------------------------------------------------
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_allan_idiosyncratic",
    "description": (
        "Idiosyncratic Allan deviation (tau=5, 80d rolling window). "
        "Computes stock returns minus a trailing 120d rolling OLS beta times SPY returns, "
        "then calculates Allan deviation (sigma_A = sqrt(0.5 * mean(diff(block_means)^2))) "
        "over non-overlapping 5-bar blocks within each 80-day window. "
        "Produces: (1) the idiosyncratic Allan dev itself, capturing stock-specific noise regime; "
        "(2) ratio of idiosyncratic to total-return Allan dev, indicating market vs idiosyncratic "
        "noise dominance. Beta estimated via rolling cov/var (causal, per-ticker OLS proxy). "
        "Uses _indexes (SPY). Degrades gracefully to NaN when SPY is unavailable or windows are too short."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_allan_idiosyncratic_adev",
        "ext2_allan_idiosyncratic_ratio",
    ],
    "tags": ["volatility", "allan_variance", "idiosyncratic", "market_residual", "noise_regime"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of rich winner vein (xdom_allan_variance); implemented as per-ticker proxy.",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TAU = 5          # block size (bars) for Allan deviation
_WIN_ADEV = 80    # rolling window for Allan deviation (bars)
_WIN_BETA = 120   # rolling window for beta estimation (bars)
_MIN_BLOCKS = 3   # minimum number of complete blocks to compute adev


def _rolling_allan_dev(returns: np.ndarray, win: int, tau: int) -> np.ndarray:
    """
    Compute rolling Allan deviation at a given tau (block size) over a window.

    For each position t, takes the last `win` returns, splits into
    non-overlapping `tau`-bar blocks, computes block means, then:
        sigma_A = sqrt(0.5 * mean(diff(block_means)^2))

    Returns array of same length as returns, with NaN where insufficient data.
    """
    n = len(returns)
    result = np.full(n, np.nan)
    n_blocks_full = win // tau  # number of complete blocks in the window

    if n_blocks_full < _MIN_BLOCKS:
        return result

    for t in range(win - 1, n):
        window = returns[t - win + 1 : t + 1]  # shape (win,)
        # Use only the last n_blocks_full * tau bars (drop partial leading block)
        usable = n_blocks_full * tau
        w = window[-usable:]
        # Reshape into blocks: (n_blocks_full, tau)
        blocks = w.reshape(n_blocks_full, tau)
        block_means = blocks.mean(axis=1)
        diffs = np.diff(block_means)
        if len(diffs) < 2:
            continue
        sigma_sq = 0.5 * np.mean(diffs ** 2)
        if sigma_sq > 0.0:
            result[t] = np.sqrt(sigma_sq)
        else:
            result[t] = 0.0

    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Stock returns (log returns for stationarity)
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    # Simple log returns; first value is NaN
    stock_ret = np.full(len(close), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = np.where(close[:-1] > 0.0, close[1:] / close[:-1], np.nan)
        stock_ret[1:] = np.log(ratio)

    # ------------------------------------------------------------------
    # 2. SPY returns (via _indexes helper)
    # ------------------------------------------------------------------
    spy_ret = np.full(len(close), np.nan)
    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close is not None and len(spy_close) > 0:
            spy_df = spy_close.rename("spy_close").reset_index()
            spy_df.columns = ["Date", "spy_close"]
            spy_df["Date"] = pd.to_datetime(spy_df["Date"])
            merged = pd.merge_asof(
                df[["Date"]].assign(Date=pd.to_datetime(df["Date"])).reset_index(drop=True),
                spy_df.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            sc = merged["spy_close"].values.astype(np.float64)
            # Log returns of SPY
            spy_ret_series = np.full(len(sc), np.nan)
            ratio_spy = np.where(sc[:-1] > 0.0, sc[1:] / sc[:-1], np.nan)
            spy_ret_series[1:] = np.log(ratio_spy)
            spy_ret = spy_ret_series
    except Exception:
        pass  # degrade to NaN for idiosyncratic (will just be NaN)

    # ------------------------------------------------------------------
    # 3. Rolling beta (120d) via cov(r_stock, r_spy) / var(r_spy)
    # ------------------------------------------------------------------
    n = len(close)
    beta = np.full(n, np.nan)
    spy_available = not np.all(np.isnan(spy_ret))

    if spy_available:
        sr = pd.Series(stock_ret)
        sp = pd.Series(spy_ret)
        # Rolling covariance and variance
        cov_rs = sr.rolling(_WIN_BETA, min_periods=_WIN_BETA // 2).cov(sp)
        var_sp = sp.rolling(_WIN_BETA, min_periods=_WIN_BETA // 2).var()
        # Guard division
        var_sp_arr = var_sp.values
        cov_arr = cov_rs.values
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(
                (var_sp_arr > 0.0) & np.isfinite(var_sp_arr) & np.isfinite(cov_arr),
                cov_arr / var_sp_arr,
                np.nan,
            )

    # ------------------------------------------------------------------
    # 4. Idiosyncratic returns = stock_ret - beta * spy_ret
    # ------------------------------------------------------------------
    if spy_available:
        idio_ret = np.where(
            np.isfinite(beta) & np.isfinite(spy_ret),
            stock_ret - beta * spy_ret,
            np.nan,
        )
    else:
        idio_ret = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 5. Allan deviation on idiosyncratic returns
    # ------------------------------------------------------------------
    adev_idio = _rolling_allan_dev(idio_ret, _WIN_ADEV, _TAU)

    # ------------------------------------------------------------------
    # 6. Allan deviation on total stock returns (for ratio)
    # ------------------------------------------------------------------
    adev_total = _rolling_allan_dev(stock_ret, _WIN_ADEV, _TAU)

    # ------------------------------------------------------------------
    # 7. Ratio: idiosyncratic / total (>1 means stock noisier than market-explained)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_arr = np.where(
            (adev_total > 0.0) & np.isfinite(adev_total) & np.isfinite(adev_idio),
            adev_idio / adev_total,
            np.nan,
        )

    # Replace any inf with nan
    adev_idio = np.where(np.isfinite(adev_idio), adev_idio, np.nan)
    ratio_arr = np.where(np.isfinite(ratio_arr), ratio_arr, np.nan)

    # ------------------------------------------------------------------
    # 8. Assign produced columns
    # ------------------------------------------------------------------
    df["ext2_allan_idiosyncratic_adev"] = adev_idio
    df["ext2_allan_idiosyncratic_ratio"] = ratio_arr

    return df
