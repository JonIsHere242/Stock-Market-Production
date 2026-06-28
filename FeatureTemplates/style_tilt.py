"""
style_tilt.py — Growth/value and size style tilts of the stock measured against
QQQ (large-cap growth), IWM (small-cap), and SPY (broad), using STYLE SPREADS
rather than raw index dispersion (which is market-wide and cross-sectionally
inert).

  * Relative strength vs QQQ / IWM (20d excess log return): is the stock leading
    the growth basket or the small-cap basket?
  * Growth-value tilt: the stock's 60d beta to QQQ minus its 60d beta to IWM.
    Positive ⇒ the stock co-moves more with large-cap growth than with small-cap;
    negative ⇒ it leans small-cap. This is a genuinely per-stock contrast (a
    difference of two single-index betas), not a market-wide spread.
  * Style beta: rolling beta of the stock's return on the QQQ-minus-IWM return
    spread (the growth-minus-small factor). The sign/size says how much of the
    stock's variance is explained by the growth/size style factor — orthogonal
    to plain market beta.
  * Size beta: rolling beta on the IWM-minus-SPY spread (the small-minus-broad
    factor), isolating small-cap exposure.

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
    "name":        "style_tilt",
    "description": (
        "Style tilts vs QQQ/IWM/SPY: 20d relative strength vs QQQ and IWM, 60d "
        "growth-value tilt, and 60d betas on the growth-minus-small and "
        "small-minus-broad style-spread factors."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "xas_rs_qqq_20",
        "xas_rs_iwm_20",
        "xas_growth_value_tilt_60",
        "xas_style_beta_60",
        "xas_size_beta_60",
    ],
    "tags":    ["market_regime", "style", "cross_asset"],
    "version": "1.0",
    "author": "feature-gen",
}

_WINDOW = 60
_MIN    = 30


def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rolling_beta(y: pd.Series, x: pd.Series, window: int, minp: int) -> pd.Series:
    """beta of y on x = cov(y,x) / var(x), rolling."""
    cov = y.rolling(window, min_periods=minp).cov(x)
    var = x.rolling(window, min_periods=minp).var()
    return (cov / var).clip(-5, 5)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col in METADATA["produces"]:
        df[col] = np.nan

    df_dates = pd.to_datetime(df["Date"]).values

    stock_close = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    stock_ret = _log_ret(stock_close)

    try:
        spy_close = _indexes.index_close("SPY")
        qqq_close = _indexes.index_close("QQQ")
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df
    if spy_close.empty or qqq_close.empty or iwm_close.empty:
        return df

    spy_ret = _log_ret(spy_close)
    qqq_ret = _log_ret(qqq_close)
    iwm_ret = _log_ret(iwm_close)

    # --- Relative strength vs QQQ / IWM (20d excess log return) ---------------
    s_q, q_q = stock_ret.align(qqq_ret, join="inner")
    if len(s_q) >= 20:
        rs_qqq_20 = (s_q - q_q).rolling(20, min_periods=20).sum()
        df["xas_rs_qqq_20"] = rs_qqq_20.reindex(df_dates).values

    s_i, i_i = stock_ret.align(iwm_ret, join="inner")
    if len(s_i) >= 20:
        rs_iwm_20 = (s_i - i_i).rolling(20, min_periods=20).sum()
        df["xas_rs_iwm_20"] = rs_iwm_20.reindex(df_dates).values

    # --- Build a common frame for SPY/QQQ/IWM + stock on shared dates ---------
    panel = pd.concat(
        {
            "stock": stock_ret,
            "spy": spy_ret,
            "qqq": qqq_ret,
            "iwm": iwm_ret,
        },
        axis=1,
        join="inner",
    ).dropna()
    if len(panel) < _MIN:
        return df

    # Growth-value tilt: beta(stock, QQQ) - beta(stock, IWM).
    # A per-stock contrast of co-movement with growth vs small-cap baskets.
    beta_qqq = _rolling_beta(panel["stock"], panel["qqq"], _WINDOW, _MIN)
    beta_iwm = _rolling_beta(panel["stock"], panel["iwm"], _WINDOW, _MIN)
    growth_value_tilt = beta_qqq - beta_iwm
    df["xas_growth_value_tilt_60"] = growth_value_tilt.reindex(df_dates).values

    # Style-spread factors.
    growth_minus_small = panel["qqq"] - panel["iwm"]   # growth vs small factor
    small_minus_broad  = panel["iwm"] - panel["spy"]    # size factor

    style_beta = _rolling_beta(panel["stock"], growth_minus_small, _WINDOW, _MIN)
    size_beta  = _rolling_beta(panel["stock"], small_minus_broad,  _WINDOW, _MIN)

    df["xas_style_beta_60"] = style_beta.reindex(df_dates).values
    df["xas_size_beta_60"]  = size_beta.reindex(df_dates).values

    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).clip(-50, 50)

    return df
