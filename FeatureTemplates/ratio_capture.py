"""
ratio_capture.py — Price-ratio momentum, up/down capture ratios, and a
correlation-regime-change signal of the stock versus SPY.

  * Ratio ROC: rate-of-change of the stock/SPY price ratio over 5/20/60d. A
    rising ratio means the stock is winning the relative race regardless of the
    market's absolute direction.
  * Up/Down capture: the classic manager-skill statistic — average stock return
    on days the market rose (up-capture as a fraction of average market gain)
    and on days it fell (down-capture). The spread (up minus down) is high for
    stocks that participate in rallies but resist drawdowns.
  * Correlation regime change: 20d corr minus 60d corr vs SPY — positive means
    the stock is RE-coupling to the market (diversification benefit fading).

Inner-join on Date, log returns, df row order untouched. Lookahead-safe.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "ratio_capture",
    "description": (
        "Stock/SPY ratio ROC (5/20/60d), 60d up/down capture ratios and their "
        "spread, and 20d-minus-60d correlation regime change vs SPY."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "xas_ratio_roc_5",
        "xas_ratio_roc_20",
        "xas_ratio_roc_60",
        "xas_upcapture_60",
        "xas_downcapture_60",
        "xas_capture_spread_60",
        "xas_corr_regime_chg",
    ],
    "tags":    ["market_regime", "capture", "cross_asset"],
    "version": "1.0",
    "author": "feature-gen",
}

_CAP_WINDOW = 60
_CAP_MIN    = 30


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col in METADATA["produces"]:
        df[col] = np.nan

    df_dates = pd.to_datetime(df["Date"]).values

    stock_close = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    stock_ret = np.log(stock_close / stock_close.shift(1))

    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close.empty:
            return df
        spy_ret = np.log(spy_close / spy_close.shift(1))
    except Exception:
        return df

    # --- Price-ratio momentum (uses aligned levels, not returns) --------------
    s_close, m_close = stock_close.align(spy_close, join="inner")
    if len(s_close) < _CAP_MIN:
        return df
    ratio = s_close / m_close.replace(0, np.nan)

    def _roc(window: int) -> pd.Series:
        return ratio / ratio.shift(window) - 1.0

    ratio_roc_5  = _roc(5)
    ratio_roc_20 = _roc(20)
    ratio_roc_60 = _roc(60)

    # --- Up / down capture (uses aligned returns) -----------------------------
    s_ret, m_ret = stock_ret.align(spy_ret, join="inner")

    up_mask   = (m_ret > 0).astype("float64")
    down_mask = (m_ret < 0).astype("float64")

    def _masked_mean(values: pd.Series, mask: pd.Series) -> pd.Series:
        num = (values * mask).rolling(_CAP_WINDOW, min_periods=_CAP_MIN).sum()
        cnt = mask.rolling(_CAP_WINDOW, min_periods=_CAP_MIN).sum()
        return num / cnt.replace(0, np.nan)

    stock_up_mean   = _masked_mean(s_ret, up_mask)
    market_up_mean  = _masked_mean(m_ret, up_mask)
    stock_down_mean = _masked_mean(s_ret, down_mask)
    market_down_mean = _masked_mean(m_ret, down_mask)

    # Capture ratio = avg stock ret on up(down) market days / avg market ret.
    upcapture = stock_up_mean / market_up_mean.replace(0, np.nan)
    downcapture = stock_down_mean / market_down_mean.replace(0, np.nan)
    capture_spread = upcapture - downcapture

    # --- Correlation regime change: corr20 - corr60 --------------------------
    corr_20 = s_ret.rolling(20, min_periods=15).corr(m_ret)
    corr_60 = s_ret.rolling(60, min_periods=30).corr(m_ret)
    corr_regime_chg = (corr_20 - corr_60)

    # --- Assign back ----------------------------------------------------------
    df["xas_ratio_roc_5"]       = ratio_roc_5.reindex(df_dates).values
    df["xas_ratio_roc_20"]      = ratio_roc_20.reindex(df_dates).values
    df["xas_ratio_roc_60"]      = ratio_roc_60.reindex(df_dates).values
    df["xas_upcapture_60"]      = upcapture.reindex(df_dates).values
    df["xas_downcapture_60"]    = downcapture.reindex(df_dates).values
    df["xas_capture_spread_60"] = capture_spread.reindex(df_dates).values
    df["xas_corr_regime_chg"]   = corr_regime_chg.reindex(df_dates).values

    # Capture ratios can blow up when the market mean is tiny; clip hard.
    df["xas_upcapture_60"]      = df["xas_upcapture_60"].clip(-10, 10)
    df["xas_downcapture_60"]    = df["xas_downcapture_60"].clip(-10, 10)
    df["xas_capture_spread_60"] = df["xas_capture_spread_60"].clip(-20, 20)
    df["xas_corr_regime_chg"]   = df["xas_corr_regime_chg"].clip(-2, 2)

    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
