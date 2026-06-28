"""
rgi_vix_idiosyncratic_coupling.py  --  Idiosyncratic vs systemic risk & VIX coupling.

Two distinct families:

1. IDIOSYNCRATIC-VS-SYSTEMIC RATIO
   The stock's own realized volatility relative to the level of the VIX.  A high
   ratio means the name is moving on its OWN news while the market is calm
   (idiosyncratic regime); a low ratio means most of its risk is the market's
   (systemic regime).  Built at 10d and 20d realized-vol horizons.  This is NOT a
   plain vol feature — it is vol *normalised by the systemic backdrop*.

2. VIX COUPLING ("vix-beta")
   Rolling 60d correlation and OLS beta of the stock's daily log return to the
   daily CHANGE in VIX.  Most equities fall when VIX jumps; the *strength* and
   *sign* of that coupling vary by name and regime and are orthogonal to plain
   return momentum.  We emit the 60d corr, the beta, and the SIGN of the beta
   (defensive names can have positive/zero VIX-beta).

All VIX is past-only (backward merge_asof). Returns are inner-joined on Date so we
never use a VIX move that did not actually trade with the stock.  df order untouched.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "rgi_vix_idiosyncratic_coupling",
    "description": "Stock realized-vol / VIX ratio (idiosyncratic vs systemic) and rolling 60d VIX-change beta/correlation with sign.",
    "requires":    ["Date", "Close"],
    "produces":    [
        "rgi_vix_idio_ratio_10",
        "rgi_vix_idio_ratio_20",
        "rgi_vix_corr_60",
        "rgi_vix_beta_60",
        "rgi_vix_beta_sign_60",
    ],
    "tags":        ["market_regime", "volatility", "beta", "rgi"],
    "version":     "1.0",
    "author":      "feature-gen",
}

_WINDOW = 60
_MIN_PERIODS = 30


def _aligned_vix(df: pd.DataFrame) -> pd.Series:
    vix_daily = _indexes.vix_daily_close()
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    if vix_daily.empty:
        return pd.Series(np.nan, index=df.index)
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    return pd.Series(merged["vix_close"].ffill().to_numpy(), index=df.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    vc = _aligned_vix(df)

    close = df["Close"]
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    new = {}

    # 1. Idiosyncratic-vs-systemic ratio.  Annualise stock vol so it is on a
    #    comparable scale to the VIX (VIX is an annualised vol in % points).
    for w in (10, 20):
        rv = log_ret.rolling(w, min_periods=max(5, w - 5)).std() * np.sqrt(252) * 100.0
        ratio = rv / vc.replace(0, np.nan)
        new[f"rgi_vix_idio_ratio_{w}"] = ratio.replace([np.inf, -np.inf], np.nan).clip(0.0, 20.0)

    # 2. VIX coupling: stock return vs VIX daily change, rolling 60d.
    #    Build a Date-indexed inner join so only shared trading days are paired.
    dates = pd.to_datetime(df["Date"])
    stock_ret = pd.Series(log_ret.to_numpy(), index=dates)

    try:
        vix_close_idx = _indexes.index_close("VIX")
    except Exception:
        vix_close_idx = pd.Series(dtype="float64")

    if not vix_close_idx.empty:
        vix_chg = vix_close_idx.diff()  # daily VIX change, DatetimeIndex
        a_stock, a_vix = stock_ret.align(vix_chg, join="inner")

        if len(a_stock) >= _MIN_PERIODS:
            cov = a_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).cov(a_vix)
            var = a_vix.rolling(_WINDOW, min_periods=_MIN_PERIODS).var()
            corr = a_stock.rolling(_WINDOW, min_periods=_MIN_PERIODS).corr(a_vix)

            beta = (cov / var.replace(0, np.nan)).clip(-5, 5)
            corr = corr.clip(-1, 1)

            # Reindex back to the ORIGINAL df dates (preserves df row order).
            df_dates = dates.values
            beta_aligned = beta.reindex(df_dates)
            corr_aligned = corr.reindex(df_dates)

            new["rgi_vix_corr_60"] = pd.Series(corr_aligned.values, index=df.index)
            new["rgi_vix_beta_60"] = pd.Series(beta_aligned.values, index=df.index)
            new["rgi_vix_beta_sign_60"] = pd.Series(
                np.sign(beta_aligned.values), index=df.index
            )
        else:
            new["rgi_vix_corr_60"] = pd.Series(np.nan, index=df.index)
            new["rgi_vix_beta_60"] = pd.Series(np.nan, index=df.index)
            new["rgi_vix_beta_sign_60"] = pd.Series(np.nan, index=df.index)
    else:
        new["rgi_vix_corr_60"] = pd.Series(np.nan, index=df.index)
        new["rgi_vix_beta_60"] = pd.Series(np.nan, index=df.index)
        new["rgi_vix_beta_sign_60"] = pd.Series(np.nan, index=df.index)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
