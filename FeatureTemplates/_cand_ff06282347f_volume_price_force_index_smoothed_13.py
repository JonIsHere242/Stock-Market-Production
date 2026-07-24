from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff06282347f_volume_price_force_index_smoothed_13",
    "description": (
        "Smoothed Force Index: raw force = (close_t - close_{t-1}) * volume_t, "
        "smoothed with a 13-day EMA, then normalized by trailing 60-day mean of "
        "|raw force| + epsilon to make it scale-free across tickers. "
        "Also produces a 3-day rate-of-change of the smoothed FI (slope/momentum "
        "of buying/selling conviction) and a bull/bear asymmetry ratio based on "
        "the fraction of positive smoothed FI readings over the trailing 20 bars."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282347f_volume_price_force_index_smoothed_13_fi",
        "ff06282347f_volume_price_force_index_smoothed_13_slope",
        "ff06282347f_volume_price_force_index_smoothed_13_bull_frac",
    ],
    "tags": ["volume", "price", "force_index", "ema", "momentum", "conviction"],
    "version": "1.0.0",
    "author": "feature-factory",
}

_COL_FI = "ff06282347f_volume_price_force_index_smoothed_13_fi"
_COL_SLOPE = "ff06282347f_volume_price_force_index_smoothed_13_slope"
_COL_BULL = "ff06282347f_volume_price_force_index_smoothed_13_bull_frac"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN upfront (required on every code path)
    df[_COL_FI] = np.nan
    df[_COL_SLOPE] = np.nan
    df[_COL_BULL] = np.nan

    if len(df) < 2:
        return df

    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n = len(close)

    # Raw force index: (close_t - close_{t-1}) * volume_t
    raw_force = np.empty(n, dtype=float)
    raw_force[0] = np.nan
    raw_force[1:] = (close[1:] - close[:-1]) * volume[1:]

    # 13-day EMA of raw force (pandas ewm, adjust=False = true recursive EMA)
    raw_s = pd.Series(raw_force)
    ema13 = raw_s.ewm(span=13, adjust=False, min_periods=13).mean().to_numpy()

    # Trailing 60-day mean of |raw force| for normalisation
    abs_raw = pd.Series(np.abs(raw_force))
    norm60 = abs_raw.rolling(window=60, min_periods=30).mean().to_numpy()

    # Normalised smoothed FI
    eps = 1e-10
    denom = np.where(norm60 == 0, np.nan, norm60) + eps
    fi_norm = ema13 / denom  # scale-free; inf-safe because denom >= eps

    # Guard inf / -inf
    fi_norm = np.where(np.isfinite(fi_norm), fi_norm, np.nan)

    df[_COL_FI] = fi_norm

    # 3-day rate-of-change of the normalised FI (slope / momentum of conviction)
    fi_s = pd.Series(fi_norm)
    slope = (fi_s - fi_s.shift(3)).to_numpy()
    df[_COL_SLOPE] = slope

    # Fraction of positive smoothed FI readings over trailing 20 bars
    # (bull/bear asymmetry proxy: > 0.5 means buying force dominates recently)
    positive = (fi_s > 0).astype(float)
    bull_frac = positive.rolling(window=20, min_periods=10).mean().to_numpy()
    df[_COL_BULL] = bull_frac

    return df
