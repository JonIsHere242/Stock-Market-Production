"""
idio_skewness.py — Expected idiosyncratic-skewness proxy (Tier-2).

Boyer, Mitton & Vorkink (2010, RFS) "Expected Idiosyncratic Skewness." Stocks
with high *idiosyncratic* (firm-specific, lottery-like) skewness earn low
subsequent returns: investors over-pay for the small chance of a big upside, so
positively-skewed names UNDER-perform next period. The clean measure is the
skewness of returns AFTER stripping out systematic / trend components.

We have no market column inside a per-ticker block, so we form two complementary
de-trended residual streams and take their rolling skewness:
    1. de-meaned daily return  : r - rolling_mean(r)        (removes drift level)
    2. residual-from-own-MA     : r - r_predicted_by_MA      where the "expected"
       return is the change implied by a short EMA of Close (a cheap own-trend
       model). The residual is the firm-specific surprise.
We also keep the raw rolling skew (no de-trend) and an up/down skew asymmetry
(coskew-free) so the model can isolate the de-trended part.

Produces over a 60d window (BMV use ~1-5yr monthly; daily 60d is the trading
analogue) plus a 21d short horizon:
    lot_idioskew_demean_<w>  : rolling skew of de-meaned returns
    lot_idioskew_resid_<w>   : rolling skew of residual-from-own-EMA returns
    lot_rawskew_<w>          : rolling skew of raw returns (reference)
    lot_skew_asym_<w>        : (upside vol - downside vol)/total — sign of skew

Pure per-ticker, vectorised with pandas .rolling().skew() (C-implemented, not a
python callback).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(60, 40), (21, 15)]
_EMA_SPAN = 10

METADATA = {
    "name":        "idio_skewness",
    "description": "Expected idiosyncratic-skewness proxy: rolling skew of de-meaned and own-EMA-residual returns plus skew asymmetry at 60d/21d, per Boyer-Mitton-Vorkink 2010.",
    "requires":    ["Close"],
    "produces":    [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("lot_idioskew_demean", "lot_idioskew_resid",
                  "lot_rawskew", "lot_skew_asym")
    ],
    "tags":        ["tail", "lottery", "skewness", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (BMV 2010)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    r = close.pct_change()

    # Own-trend "expected" return: EMA-implied step. Residual = firm surprise.
    ema = close.ewm(span=_EMA_SPAN, min_periods=_EMA_SPAN, adjust=False).mean()
    expected_r = ema / ema.shift(1) - 1.0
    resid = r - expected_r

    for w, mp in _WINDOWS:
        demean = r - r.rolling(w, min_periods=mp).mean()

        df[f"lot_idioskew_demean_{w}"] = (
            demean.rolling(w, min_periods=mp).skew().clip(-15, 15).values
        )
        df[f"lot_idioskew_resid_{w}"] = (
            resid.rolling(w, min_periods=mp).skew().clip(-15, 15).values
        )
        df[f"lot_rawskew_{w}"] = (
            r.rolling(w, min_periods=mp).skew().clip(-15, 15).values
        )

        up_vol = r.clip(lower=0).rolling(w, min_periods=mp).std()
        dn_vol = (-r.clip(upper=0)).rolling(w, min_periods=mp).std()
        denom = (up_vol + dn_vol).replace(0, np.nan)
        df[f"lot_skew_asym_{w}"] = ((up_vol - dn_vol) / denom).clip(-1, 1).values

    return df
