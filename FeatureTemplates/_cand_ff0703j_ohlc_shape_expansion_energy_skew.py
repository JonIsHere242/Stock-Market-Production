from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703j_ohlc_shape_expansion_energy_skew",
    "description": (
        "Signed share of 'range-expansion energy' released on up-closing vs down-closing bars. "
        "Range R = High-Low (0->NaN). rmean20 = 20-bar rolling mean of R. Expansion magnitude "
        "e_t = max(R_t - rmean20_t, 0), the amount by which today's range exceeds its trailing "
        "20-day norm (0 when the bar is not expanding). Over a trailing 40-bar window, "
        "ees = (sum of e_t on up-closing bars - sum of e_t on down-closing bars) / (sum of e_t "
        "over all 40 bars), guarded to NaN when total expansion energy is zero; this is a -1..+1 "
        "signed share of where large-range 'breakout' energy landed. ees_slope = ees_t - ees_{t-20} "
        "tracks whether that skew is building toward the buy or sell side. Weighting by expansion "
        "magnitude (not just sign) distinguishes this from simple up/down-day-count bias features. "
        "Faithful per-ticker OHLC implementation of the spec; no cross-sectional or external data needed."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "ff0703j_ohlc_shape_expansion_energy_skew_ees",
        "ff0703j_ohlc_shape_expansion_energy_skew_ees_slope",
    ],
    "tags": ["ohlc_shape", "range", "expansion", "skew", "energy"],
    "version": "1.0",
    "author": "ff0703j spec batch (faithful implementation)",
}

_RMEAN_WIN = 20
_EES_WIN = 40
_SLOPE_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    ees = np.full(n, np.nan)
    ees_slope = np.full(n, np.nan)

    df["ff0703j_ohlc_shape_expansion_energy_skew_ees"] = ees
    df["ff0703j_ohlc_shape_expansion_energy_skew_ees_slope"] = ees_slope

    if n == 0:
        return df

    rng = df["High"] - df["Low"]
    rng_safe = rng.replace(0, np.nan)

    rmean20 = rng_safe.rolling(_RMEAN_WIN, min_periods=_RMEAN_WIN).mean()
    e = (rng_safe - rmean20).clip(lower=0.0)
    e = e.fillna(0.0)

    up = (df["Close"] > df["Open"]).astype(float)
    down = (df["Close"] < df["Open"]).astype(float)

    e_up = e * up
    e_down = e * down

    sum_up = e_up.rolling(_EES_WIN, min_periods=_EES_WIN).sum()
    sum_down = e_down.rolling(_EES_WIN, min_periods=_EES_WIN).sum()
    total = e.rolling(_EES_WIN, min_periods=_EES_WIN).sum()
    total_safe = total.replace(0, np.nan)

    ees_series = (sum_up - sum_down) / total_safe
    ees_slope_series = ees_series - ees_series.shift(_SLOPE_LAG)

    df["ff0703j_ohlc_shape_expansion_energy_skew_ees"] = ees_series
    df["ff0703j_ohlc_shape_expansion_energy_skew_ees_slope"] = ees_slope_series

    return df
