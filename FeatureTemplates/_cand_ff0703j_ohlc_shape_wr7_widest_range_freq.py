"""
_cand_ff0703j_ohlc_shape_wr7_widest_range_freq.py -- WR7 (wide-range-7) frequency + close-location asymmetry.

METHOD
------
R_t = High_t - Low_t (true range proxy, OHLC only, no prior-close gap term).
W_t = 1 if R_t equals the max of R over the trailing 7 bars inclusive (t-6..t), else 0.
  This flags days where the current bar's range is a *local range peak* (a "widest range in 7" day).

LEVEL:
  wr7_freq_34 = mean(W) over the trailing 34 bars -> rate of local range peaks (frequency
  of range-expansion events). Distinct axis from raw volatility level (ATR-style features):
  this counts *events* (ties allowed -> more true near ties/plateaus), not magnitude.

ASYMMETRY:
  wr7_close_pos_asym_34 = mean, over the WR7==1 bars within the trailing 34, of the signed
  close-location   loc_t = 2*(Close_t - Low_t)/(High_t - Low_t) - 1  in [-1, 1].
  loc_t > 0 means the wide-range day closed in the upper half of its range (accumulation-like,
  buyers won the wide bar); loc_t < 0 means it closed low (distribution-like). Bars with
  High_t == Low_t are excluded from the average (loc undefined there); if no WR7 bar exists
  in the trailing 34-bar window the result is NaN.

Both columns are computed on a strict trailing/causal basis (rolling windows only, no
negative shifts, no anchoring to the end of the series), so they are lookahead-safe and
stable under truncation.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703j_ohlc_shape_wr7_widest_range_freq",
    "description": (
        "OHLC-shape feature: WR7 = indicator that today's High-Low range is the max of the "
        "trailing 7-bar range (a local range-expansion peak). wr7_freq_34 = trailing-34-bar "
        "frequency of WR7 events (range-peak rate, distinct from raw ATR/volatility level -- "
        "counts events not magnitude). wr7_close_pos_asym_34 = trailing-34-bar mean signed "
        "close-location (2*(Close-Low)/(High-Low)-1) averaged ONLY over the WR7 event bars, "
        "measuring whether wide-range days tend to close near their high (accumulation) or "
        "low (distribution). Faithful per-ticker implementation of the spec; fully causal "
        "trailing-window rolling computation, no cross-sectional or fitted components needed."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": ["ff0703j_wr7_freq_34", "ff0703j_wr7_close_pos_asym_34"],
    "tags": ["ohlc_shape", "range", "volatility", "asymmetry", "candidate"],
    "version": "1.0",
    "author": "feature-factory (ff0703j codegen)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df["ff0703j_wr7_freq_34"] = np.nan
    df["ff0703j_wr7_close_pos_asym_34"] = np.nan

    if n == 0:
        return df

    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    rng = high - low  # trailing/current-bar range, no lookahead

    rng_s = pd.Series(rng, index=df.index)

    # Trailing 7-bar (inclusive) rolling max of range -> causal, min_periods=1 so it is
    # well-defined from the first bar (early bars just have a shorter effective window).
    roll_max_7 = rng_s.rolling(window=7, min_periods=1).max()

    # WR7 indicator: current range equals the trailing-7 max. NaN-range days (shouldn't
    # happen for OHLC but guard anyway) never flag.
    with np.errstate(invalid="ignore"):
        wr7 = (rng_s == roll_max_7) & rng_s.notna()
    wr7_f = wr7.astype(np.float64)

    # LEVEL: frequency of WR7 events over the trailing 34 bars.
    freq_34 = wr7_f.rolling(window=34, min_periods=1).mean()
    df["ff0703j_wr7_freq_34"] = freq_34.to_numpy()

    # Signed close-location within the bar's range, guarded against High==Low.
    denom = high - low
    safe_denom = np.where(denom == 0, np.nan, denom)
    loc = 2.0 * (close - low) / safe_denom - 1.0
    loc = np.where(np.isfinite(loc), loc, np.nan)

    # Only keep close-location on WR7 event bars; non-event bars contribute NaN and are
    # excluded from the rolling mean via min_periods-based skipna behaviour of pandas rolling.
    wr7_np = wr7.to_numpy()
    loc_on_wr7 = np.where(wr7_np, loc, np.nan)
    loc_on_wr7_s = pd.Series(loc_on_wr7, index=df.index)

    # rolling().mean() over a window that may be all-NaN yields NaN naturally (skipna).
    asym_34 = loc_on_wr7_s.rolling(window=34, min_periods=1).mean()
    df["ff0703j_wr7_close_pos_asym_34"] = asym_34.to_numpy()

    return df
